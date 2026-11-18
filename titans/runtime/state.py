"""Owned stream state; callers pass it between the four public operations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from torch import Tensor

from titans.core.protocols import MemoryState


@dataclass
class AttentionState:
    """Grouped-query KV tensors and their absolute rotary positions."""

    keys: Tensor
    values: Tensor
    positions: Tensor

    def copy(self, *, detach: bool = False) -> "AttentionState":
        """Return an independent copy, optionally dropping the autograd graph.

        Args:
            detach: Whether to sever past differentiation history.

        Returns:
            A cache that shares no mutable tensor storage with this one.
        """
        return AttentionState(
            self.keys.detach().clone() if detach else self.keys.clone(),
            self.values.detach().clone() if detach else self.values.clone(),
            self.positions.clone(),
        )


@dataclass
class StreamState:
    """Opaque, model-owned stream. Internal fields are not a serialization API.

    ``ingest`` consumes and mutates this object. ``answer`` reads it through an
    independent branch. Concurrent mutation of one stream is unsupported.
    """

    _owner: str
    _revision: int
    _memory: dict[int, MemoryState] = field(default_factory=dict)
    _attention: dict[int, AttentionState] = field(default_factory=dict)
    _position: int = 0
    _tokens: int = 0
    _documents: int = 0
    _next_logits: Tensor | None = None
    _valid: bool = True
    _next_hidden: Tensor | None = None

    _retrieval: dict[int, MemoryState] = field(default_factory=dict)
    _boundary_prediction: bool = False

    @property
    def memory(self) -> Mapping[int, MemoryState]:
        """Expose algorithm-owned state by layer without allowing layer reassignment."""
        return MappingProxyType(self._memory)

    @property
    def tokens_processed(self) -> int:
        """Return the number of non-prefix tokens committed to this branch."""
        return self._tokens

    @property
    def documents_seen(self) -> int:
        """Return the number of completely ingested documents, including repeats."""
        return self._documents

    def _copy(self, *, detach: bool = False) -> "StreamState":
        """Copy the full stream, preserving ownership and compatibility metadata.

        Args:
            detach: Whether copied tensors should retain differentiation history.

        Returns:
            An independent state, including pending writes and cached logits.
        """
        logits = self._next_logits
        if logits is not None:
            logits = logits.detach().clone() if detach else logits.clone()
        hidden = self._next_hidden
        if hidden is not None:
            hidden = hidden.detach().clone() if detach else hidden.clone()
        return StreamState(
            self._owner, self._revision,
            {index: memory.copy(detach=detach) for index, memory in self._memory.items()},
            {index: cache.copy(detach=detach) for index, cache in self._attention.items()},
            self._position, self._tokens, self._documents, logits, self._valid, hidden,
            _retrieval={i: m.copy(detach=detach) for i, m in self._retrieval.items()},
            _boundary_prediction=self._boundary_prediction,
        )

    def _detach(self) -> None:
        """Cut training history at a truncated-backpropagation boundary."""
        for memory in (*self._memory.values(), *self._retrieval.values()):
            memory.detach()
        for cache in self._attention.values():
            cache.keys = cache.keys.detach()
            cache.values = cache.values.detach()
        if self._next_logits is not None:
            self._next_logits = self._next_logits.detach()
        if self._next_hidden is not None:
            self._next_hidden = self._next_hidden.detach()
