"""Owned CUDA graphs for single-stream decoding with Titans fast-weight updates.

Graphs share one workspace and a private allocator pool. Every output needed by
the next step is copied into persistent buffers inside the graph. Before the
workspace changes owners, the old stream receives independent tensor storage.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from titans.methods.titans import MemoryState
from titans.runtime.state import StreamState

if TYPE_CHECKING:
    from titans.runtime.model import ContinualModel


class GraphDecoder:
    """Capture one graph per pending-write phase after the attention cache fills."""

    def __init__(self, model: "ContinualModel") -> None:
        """Create a lazy graph runner tied to immutable inference parameters.

        Args:
            model: Worker-owned model; discard this runner before changing its weights.
        """
        self.model = model
        self.graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        self.storage: StreamState | None = None
        self.active: StreamState | None = None
        self.ids: Tensor | None = None
        self.position: Tensor | None = None
        self.logits: Tensor | None = None
        self.hidden: Tensor | None = None
        self.stream: torch.cuda.Stream | None = None
        self.pool: tuple[int, int] | None = None
        self.replays = 0

    def deactivate(self) -> None:
        """Give the resident stream independent buffers before another operation uses them."""
        if self.active is not None:
            original = self.active
            snapshot = original._copy(detach=True)
            original._memory = snapshot._memory
            original._attention = snapshot._attention
            original._next_logits = snapshot._next_logits
            original._next_hidden = snapshot._next_hidden
            self.active = None

    def _eligible(self, ids: Tensor, state: StreamState) -> bool:
        """Check whether all shapes have reached the fixed one-token decode regime.

        Args:
            ids: Token IDs shaped [1, tokens].
            state: Current complete stream state.

        Returns:
            Whether graph replay can represent this forward without padding or masking changes.
        """
        config = self.model.config
        length = config.persistent_tokens + config.attention_window - 1
        return (
            config.cuda_graphs and ids.device.type == "cuda" and ids.numel() == 1
            and len(state._attention) == len(self.model.backbone.model.layers)
            and all(cache.keys.shape[-2] == length for cache in state._attention.values())
            and len({memory.pending.shape[1] for memory in state._memory.values()}) == 1
            and state._position >= config.persistent_tokens + config.attention_window
        )

    def _initialize(self, state: StreamState) -> None:
        """Allocate graph input/output buffers outside the shared private graph pool.

        Args:
            state: Full-window stream defining buffer shapes and dtypes.
        """
        self.storage = state._copy(detach=True)
        for memory in self.storage._memory.values():
            memory.pending = memory.pending.new_zeros(
                1, self.model.config.chunk_size, memory.pending.shape[-1],
            )
        weight = self.model.backbone.model.embed_tokens.weight
        self.ids = torch.zeros(1, 1, device=weight.device, dtype=torch.long)
        self.position = torch.zeros(1, device=weight.device, dtype=torch.long)
        self.logits = weight.new_empty(1, 1, self.model.backbone.config.vocab_size)
        self.hidden = weight.new_empty(1, 1, weight.shape[-1])
        self.stream = torch.cuda.Stream(device=weight.device)
        self.pool = torch.cuda.graph_pool_handle()

    def _load(self, state: StreamState) -> None:
        """Copy a newly selected stream into the stable-address graph workspace.

        Args:
            state: Independent stream whose values become the next graph inputs.
        """
        assert self.storage is not None and self.position is not None
        targets: list[Tensor] = []
        sources: list[Tensor] = []
        for index, target in self.storage._attention.items():
            source = state._attention[index]
            targets.extend((target.keys, target.values, target.positions))
            sources.extend((source.keys, source.values, source.positions))
        for index, target in self.storage._memory.items():
            source = state._memory[index]
            targets.extend((*target.weights, *target.momentum, target.conv_tail,
                            target.pending[:, :source.pending.shape[1]]))
            sources.extend((*source.weights, *source.momentum, source.conv_tail, source.pending))
        torch._foreach_copy_(targets, sources)
        self.position.fill_(state._position)

    def _view(self, phase: int) -> StreamState:
        """Build host metadata that views the persistent graph inputs for one phase.

        Args:
            phase: Number of pending writes before the next token.

        Returns:
            A temporary metadata wrapper; tensor storage remains owned by this runner.
        """
        assert self.storage is not None
        return StreamState(
            self.storage._owner, self.storage._revision,
            {index: MemoryState(memory.weights, memory.momentum,
                                memory.pending[:, :phase], memory.conv_tail)
             for index, memory in self.storage._memory.items()},
            dict(self.storage._attention),
            _position=self.model.config.persistent_tokens + self.model.config.attention_window,
        )

    def _step(self, phase: int, write: bool) -> None:
        """Run a capturable token update and copy outputs back to stable input addresses.

        Args:
            phase: Pending-write count used to specialize the chunk-commit branch.
            write: Whether memory weights and momentum may change.
        """
        assert self.ids is not None and self.position is not None
        assert self.storage is not None and self.logits is not None and self.hidden is not None
        state = self._view(phase)
        logits = self.model._forward_embeddings(
            self.model.backbone.model.embed_tokens(self.ids), state,
            write=write, last_only=True, positions=self.position,
        )
        targets: list[Tensor] = []
        sources: list[Tensor] = []
        for index, target in self.storage._attention.items():
            source = state._attention[index]
            targets.extend((target.keys, target.values, target.positions))
            sources.extend((source.keys, source.values, source.positions))
        for index, target in self.storage._memory.items():
            source = state._memory[index]
            targets.append(target.conv_tail)
            sources.append(source.conv_tail)
            if write:
                if phase + 1 == self.model.config.chunk_size:
                    targets.extend((*target.weights, *target.momentum))
                    sources.extend((*source.weights, *source.momentum))
                else:
                    targets.append(target.pending[:, :phase + 1])
                    sources.append(source.pending)
        assert state._next_hidden is not None
        targets.extend((self.logits, self.hidden))
        sources.extend((logits, state._next_hidden))
        torch._foreach_copy_(targets, sources)
        self.position.add_(1)

    def _buffers(self) -> list[Tensor]:
        """List persistent buffers for restoring state after capture warmup.

        Returns:
            All mutable graph input/output tensors, with no aliases between entries.
        """
        assert self.storage is not None and self.position is not None and self.ids is not None
        assert self.logits is not None and self.hidden is not None
        buffers = [self.position, self.ids, self.logits, self.hidden]
        for cache in self.storage._attention.values():
            buffers.extend((cache.keys, cache.values, cache.positions))
        for memory in self.storage._memory.values():
            buffers.extend((*memory.weights, *memory.momentum, memory.conv_tail, memory.pending))
        return buffers

    def _capture(self, phase: int, write: bool) -> torch.cuda.CUDAGraph:
        """Warm required kernels and capture a graph without advancing the real stream.

        Args:
            phase: Pending-write count before the token.
            write: Whether to capture a memory-writing or read-only step.

        Returns:
            A graph whose private temporaries may share a pool with other phase graphs.
        """
        assert self.stream is not None
        buffers = self._buffers()
        snapshot = [tensor.clone() for tensor in buffers]
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self._step(phase, write)
        current.wait_stream(self.stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
            self._step(phase, write)
        current.wait_stream(self.stream)
        torch._foreach_copy_(buffers, snapshot)
        return graph

    @torch.no_grad()
    def forward(self, ids: Tensor, state: StreamState, *, write: bool = True) -> Tensor:
        """Forward inference tokens, replaying a graph when shapes permit.

        Args:
            ids: Nonempty [1, tokens] token IDs on the model's device.
            state: Owned mutable stream. Copies of it remain independent of future replays.
            write: Whether this branch may update fast memory.

        Returns:
            The last token's logits [1, 1, vocabulary]. The returned buffer is
            borrowed until the next call; clone it when retaining logits.
        """
        if not self._eligible(ids, state):
            self.deactivate()
            for start in range(0, ids.shape[1], self.model.config.prefill_chunk_size):
                block = ids[:, start:start + self.model.config.prefill_chunk_size]
                self.model._forward_embeddings(
                    self.model.backbone.model.embed_tokens(block), state,
                    write=write, last_only=True,
                )
            state._tokens += ids.numel()
            assert state._next_logits is not None
            return state._next_logits
        if self.storage is None:
            self._initialize(state)
        if self.active is not state:
            self.deactivate()
            self._load(state)
            self.active = state
        phase = next(iter(state._memory.values())).pending.shape[1]
        key = (phase, write)
        if key not in self.graphs:
            self.graphs[key] = self._capture(phase, write)
        assert self.ids is not None and self.logits is not None
        self.ids.copy_(ids)
        self.graphs[key].replay()
        self.replays += 1
        next_phase = (phase + 1) % self.model.config.chunk_size if write else phase
        view = self._view(next_phase)
        state._memory, state._attention = view._memory, view._attention
        state._next_logits, state._next_hidden = self.logits, self.hidden
        state._position += 1
        state._tokens += 1
        return self.logits
