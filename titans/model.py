"""A pretrained Llama with bounded attention and gated Titans memory branches."""

import json
import shutil
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, apply_rotary_pos_emb

from .memory import NeuralMemory
from .state import AttentionState, StreamState
from .types import (
    Answer,
    AnswerConfig,
    Document,
    Question,
    TitansConfig,
    TrainingConfig,
    TrainingEpisode,
    TrainingResult,
)

if TYPE_CHECKING:
    from .sglang_backend import SGLangBackend


class TitansModel(nn.Module):
    """Stable API for attaching, post-training and using a memory-equipped Llama.

    This is a MAG-inspired pretrained retrofit, with the paper's memory update
    rule. It is not a reproduction of the paper's from-scratch MAG architecture.
    A model instance supports one device and one stream per public call. The
    numerical memory core also supports batches of independent streams.
    """

    def __init__(
        self,
        backbone: LlamaForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        config: TitansConfig = TitansConfig(),
    ) -> None:
        """Attach initialized memory modules to an existing pretrained Llama.

        Args:
            backbone: An unquantized, single-device Hugging Face Llama model.
            tokenizer: Its tokenizer; vocabulary IDs must match the backbone.
            config: Memory placement, dimensions and attention window.

        Raises:
            ValueError: The architecture, device placement or RoPE mode is unsupported.
        """
        super().__init__()
        if not isinstance(backbone, LlamaForCausalLM):
            raise ValueError("Only LlamaForCausalLM backbones are currently supported")
        if max(config.memory_layers) >= len(backbone.model.layers):
            raise ValueError("memory_layers contains an index outside the backbone")
        if len({p.device for p in backbone.parameters()}) != 1:
            raise ValueError("Sharded device maps are not supported")
        rope = backbone.config.rope_scaling or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type not in ("default", "linear", "llama3"):
            raise ValueError("Only static default, linear and llama3 RoPE modes are supported")
        if getattr(backbone, "is_quantized", False):
            raise ValueError("Quantized backbones are not supported")
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.config = config
        parameter = backbone.model.embed_tokens.weight
        self.memories = nn.ModuleDict({
            str(index): NeuralMemory(backbone.config.hidden_size, config)
            for index in config.memory_layers
        }).to(device=parameter.device, dtype=parameter.dtype)
        self.persistent = nn.Parameter(torch.empty(
            1, config.persistent_tokens, backbone.config.hidden_size,
            device=parameter.device, dtype=parameter.dtype,
        ))
        nn.init.normal_(self.persistent, std=backbone.config.initializer_range)
        self.backbone.requires_grad_(False)
        self._owner = uuid.uuid4().hex
        self._revision = 0
        self._checkpoint_backbone = False
        self._sglang_backend: SGLangBackend | None = None
        self.eval()

    def _uses_sglang(self) -> bool:
        """Select SGLang for CUDA inference by default, retaining the CPU reference path.

        Returns:
            Whether public inference operations must run through SGLang.

        Raises:
            RuntimeError: SGLang was explicitly requested for a CPU model.
        """
        device = self.backbone.model.embed_tokens.weight.device
        if self.config.inference_backend == "sglang" and device.type != "cuda":
            raise RuntimeError("SGLang inference requires a CUDA model")
        return self.config.inference_backend == "sglang" or (
            self.config.inference_backend == "auto" and device.type == "cuda"
        )

    def _serving_backend(self) -> "SGLangBackend":
        """Start the SGLang worker lazily for the current trained-model revision.

        Returns:
            The owned local inference engine.
        """
        if self._sglang_backend is None:
            from .sglang_backend import SGLangBackend

            self._sglang_backend = SGLangBackend(self)
        return self._sglang_backend

    def close(self) -> None:
        """Release a running SGLang engine and invalidate its live state handles."""
        if self._sglang_backend is not None:
            try:
                self._sglang_backend.close()
            finally:
                self._sglang_backend = None
                self._revision += 1

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        config: TitansConfig = TitansConfig(),
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> "TitansModel":
        """Load a Llama checkpoint and attach new, untrained memory modules.

        Args:
            model_name_or_path: Hugging Face model ID or local backbone directory.
            config: Settings for newly initialized memory components.
            device: One device, such as ``cpu`` or ``cuda:0``.
            dtype: Backbone and learned-adapter storage dtype.
            local_files_only: Whether downloads are forbidden.

        Returns:
            A model in evaluation mode with a frozen backbone.
        """
        backbone = LlamaForCausalLM.from_pretrained(
            str(model_name_or_path), dtype=dtype, local_files_only=local_files_only,
            attn_implementation="sdpa",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_name_or_path), local_files_only=local_files_only,
        )
        return cls(backbone, tokenizer, config)

    def _validate_state(self, state: StreamState) -> None:
        """Reject foreign, stale, failed or differently placed streams.

        Args:
            state: A stream previously returned by this model.

        Raises:
            ValueError: The stream cannot safely be used by the current model.
        """
        if not state._valid:
            raise ValueError("This stream was invalidated by a failed ingest")
        if state._owner != self._owner or state._revision != self._revision:
            raise ValueError("Stream belongs to a different model or predates fit")
        device = self.backbone.model.embed_tokens.weight.device
        if any(memory.weights[0].device != device for memory in state._memory.values()):
            raise ValueError("Model moved devices; start a new stream")

    def _new_state(self) -> StreamState:
        """Initialize a stream and its learned persistent prefix.

        Returns:
            An empty document stream. Prefix tokens do not count toward the
            public token budget and do not write to the fast memory.
        """
        state = StreamState(self._owner, self._revision)
        for key, memory in self.memories.items():
            state._memory[int(key)] = memory.initial_state()
        if self.config.persistent_tokens:
            self._forward_embeddings(self.persistent, state, write=False, last_only=True)
        return state

    def _local_attention(
        self,
        layer: LlamaDecoderLayer,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: AttentionState | None,
        next_position: int,
    ) -> tuple[Tensor, AttentionState]:
        """Run causal SDPA while retaining only persistent and recent grouped KV.

        Args:
            layer: The host Llama decoder layer.
            hidden: Normalized layer input [1, tokens, hidden].
            positions: Absolute positions for this chunk, shaped [tokens].
            rotary: Cosine and sine tensors from the host's RoPE implementation.
            cache: Bounded cache from the preceding call, or None.
            next_position: Python integer position following this block; avoids
                device synchronization when selecting the retained cache.

        Returns:
            Attention output and a bounded cache for the next chunk.
        """
        attention = layer.self_attn
        key_positions = positions
        if cache is not None:
            key_positions = torch.cat((cache.positions, positions))
        use_fa3 = (not torch.is_grad_enabled() and hidden.device.type == "cuda"
                   and self.config.attention_backend in ("auto", "fa3"))
        allowed = None
        if not use_fa3:
            allowed = (key_positions[None, :] <= positions[:, None]) & (
                (key_positions[None, :] < self.config.persistent_tokens)
                | (key_positions[None, :] > positions[:, None] - self.config.attention_window)
            )
        if self.config.compile_kernels and not use_fa3:
            from .kernels import compiled_attention

            assert allowed is not None
            empty_shape = (hidden.shape[0], attention.k_proj.out_features // attention.head_dim,
                           0, attention.head_dim)
            old_keys = cache.keys if cache is not None else hidden.new_empty(empty_shape)
            old_values = cache.values if cache is not None else hidden.new_empty(empty_shape)
            output, key, value = compiled_attention(
                hidden, attention.q_proj.weight, attention.k_proj.weight,
                attention.v_proj.weight, attention.o_proj.weight,
                attention.q_proj.bias, attention.k_proj.bias,
                attention.v_proj.bias, attention.o_proj.bias,
                *rotary, old_keys, old_values, allowed, attention.head_dim, attention.scaling,
            )
        else:
            shape = (*hidden.shape[:-1], -1, attention.head_dim)
            query = attention.q_proj(hidden).view(shape).transpose(1, 2)
            key = attention.k_proj(hidden).view(shape).transpose(1, 2)
            value = attention.v_proj(hidden).view(shape).transpose(1, 2)
            if hidden.device.type == "cuda" and self.config.use_liger:
                from .gpu_ops import liger_rotary

                query, key = liger_rotary(query, key, *rotary)
            else:
                query, key = apply_rotary_pos_emb(query, key, *rotary)
            if cache is not None:
                key = torch.cat((cache.keys, key), -2)
                value = torch.cat((cache.values, value), -2)
            groups = query.shape[1] // key.shape[1]
            if use_fa3:
                from .gpu_ops import flash_attention3

                output = flash_attention3(
                    query, key, value, prefix_count=min(self.config.persistent_tokens, next_position),
                    window=self.config.attention_window, scale=attention.scaling,
                    prefix_only=next_position <= self.config.persistent_tokens,
                )
            else:
                output = F.scaled_dot_product_attention(
                    query, key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1),
                    attn_mask=allowed[None, None], dropout_p=0.0, scale=attention.scaling,
                )
            output = attention.o_proj(output.transpose(1, 2).reshape(*hidden.shape[:-1], -1))
        prefix_count = min(self.config.persistent_tokens, next_position)
        tail_count = min(self.config.attention_window - 1, key.shape[-2] - prefix_count)
        tail_start = key.shape[-2] - tail_count
        # Slices have known lengths on the host; boolean indexing here would
        # force a CUDA synchronization at every layer to discover output sizes.
        kept_keys = torch.cat((key[..., :prefix_count, :], key[..., tail_start:, :]), -2)
        kept_values = torch.cat((value[..., :prefix_count, :], value[..., tail_start:, :]), -2)
        kept_positions = torch.cat((key_positions[:prefix_count], key_positions[tail_start:]))
        return output, AttentionState(kept_keys, kept_values, kept_positions)

    def _forward_embeddings(
        self, embeddings: Tensor, state: StreamState, *, write: bool = True,
        last_only: bool = False, return_hidden: bool = False, positions: Tensor | None = None,
    ) -> Tensor:
        """Forward one bounded block, mutating its owned stream state.

        Args:
            embeddings: Embedded tokens [1, tokens, hidden].
            state: Mutable stream; callers enforce ownership before entry.
            write: Whether this block may update long-term memory.
            last_only: Whether to project only the final token to vocabulary logits.
            return_hidden: Whether to return normalized activations for fused Liger loss.
            positions: Optional GPU position buffer for the private CUDA graph runner.

        Returns:
            Logits [1, tokens, vocabulary] for next-token prediction.
        """
        if positions is None:
            positions = torch.arange(
                state._position, state._position + embeddings.shape[1], device=embeddings.device,
            )
        rotary = self.backbone.model.rotary_emb(embeddings, positions[None, :])
        hidden = embeddings
        use_liger = self.config.use_liger and hidden.device.type == "cuda"
        for index, layer in enumerate(self.backbone.model.layers):
            if use_liger:
                from .gpu_ops import liger_norm

                normalized = liger_norm(hidden, layer.input_layernorm.weight,
                                        layer.input_layernorm.variance_epsilon)
            elif self.config.compile_kernels:
                from .kernels import compiled_rms_norm

                normalized = compiled_rms_norm(hidden, layer.input_layernorm.weight,
                                               layer.input_layernorm.variance_epsilon)
            else:
                normalized = layer.input_layernorm(hidden)
            attended, cache = self._local_attention(
                layer, normalized, positions, rotary, state._attention.get(index),
                state._position + embeddings.shape[1],
            )
            state._attention[index] = cache
            if str(index) in self.memories:
                attended = attended + self.memories[str(index)](
                    normalized, state._memory[index], write=write,
                )
            hidden = hidden + attended
            if use_liger:
                normalized = liger_norm(hidden, layer.post_attention_layernorm.weight,
                                        layer.post_attention_layernorm.variance_epsilon)
            elif self.config.compile_kernels:
                normalized = compiled_rms_norm(hidden, layer.post_attention_layernorm.weight,
                                               layer.post_attention_layernorm.variance_epsilon)
            else:
                normalized = layer.post_attention_layernorm(hidden)
            if (use_liger or self.config.compile_kernels) and self.backbone.config.hidden_act == "silu":
                if use_liger:
                    from .gpu_ops import liger_mlp

                    mlp_forward = liger_mlp
                else:
                    from .kernels import compiled_llama_mlp

                    mlp_forward = compiled_llama_mlp

                mlp = layer.mlp
                arguments = (normalized, mlp.gate_proj.weight, mlp.up_proj.weight,
                             mlp.down_proj.weight, mlp.gate_proj.bias, mlp.up_proj.bias,
                             mlp.down_proj.bias)
                if self._checkpoint_backbone and torch.is_grad_enabled():
                    feedforward = checkpoint(mlp_forward, *arguments, use_reentrant=False)
                else:
                    feedforward = mlp_forward(*arguments)
            elif self._checkpoint_backbone and torch.is_grad_enabled():
                feedforward = checkpoint(layer.mlp, normalized, use_reentrant=False)
            else:
                feedforward = layer.mlp(normalized)
            hidden = hidden + feedforward
        if last_only:
            hidden = hidden[:, -1:]
        if use_liger:
            normalized = liger_norm(hidden, self.backbone.model.norm.weight,
                                    self.backbone.model.norm.variance_epsilon)
        elif self.config.compile_kernels:
            normalized = compiled_rms_norm(hidden, self.backbone.model.norm.weight,
                                           self.backbone.model.norm.variance_epsilon)
        else:
            normalized = self.backbone.model.norm(hidden)
        state._next_hidden = normalized[:, -1:].contiguous()
        logits = normalized if return_hidden else self.backbone.lm_head(normalized)
        state._position += embeddings.shape[1]
        state._next_logits = None if return_hidden else logits[:, -1:].contiguous()
        return logits

    def _forward_tokens(
        self, tokens: list[int], state: StreamState, *, write: bool = True, last_only: bool = False,
        return_hidden: bool = False,
    ) -> Tensor:
        """Forward a nonempty token block, splitting work into bounded chunks.

        Args:
            tokens: IDs in the backbone vocabulary.
            state: Owned state updated by the call.
            write: Whether long-term memory writes are enabled.
            last_only: Whether to return only the final token's logits.
            return_hidden: Whether to return normalized hidden states for fused training loss.

        Returns:
            Concatenated logits; public ingestion discards each chunk promptly.
        """
        if not tokens:
            raise ValueError("A forward block must contain at least one token")
        device = self.backbone.model.embed_tokens.weight.device
        outputs: list[Tensor] = []
        for start in range(0, len(tokens), self.config.prefill_chunk_size):
            ids = torch.tensor([tokens[start:start + self.config.prefill_chunk_size]], device=device)
            outputs.append(self._forward_embeddings(self.backbone.model.embed_tokens(ids), state,
                                                    write=write, last_only=last_only,
                                                    return_hidden=return_hidden))
        state._tokens += len(tokens)
        return outputs[-1] if last_only else torch.cat(outputs, 1)

    def _document_tokens(self, document: Document) -> list[int]:
        """Encode a document with explicit boundaries, without truncation.

        Args:
            document: Text and identifier to ingest, in arrival order.

        Returns:
            Tokens for the document header, body and trailing newline.
        """
        return self.tokenizer.encode(
            f"\nDocument {document.document_id}:\n{document.text}\n", add_special_tokens=False,
        )

    def _question_tokens(self, question: Question) -> list[int]:
        """Encode the same question prompt for training and inference.

        Args:
            question: The question text; its ID is metadata only.

        Returns:
            Tokens ending with the answer prefix.
        """
        return self.tokenizer.encode(f"\nQuestion: {question.text}\nAnswer:", add_special_tokens=False)

    @torch.no_grad()
    def ingest(self, document: Document, state: StreamState | None = None) -> StreamState:
        """Append a complete document, updating only that stream's temporary state.

        Args:
            document: Document to process; repeated IDs are new stream events.
            state: Owned stream to mutate, or None to start an independent stream.

        Returns:
            The supplied state, updated in place, or a new state.

        Raises:
            ValueError: State is stale, foreign or invalid. Any failure after
                processing starts invalidates that state; it must be discarded.
        """
        if self._uses_sglang():
            return self._serving_backend().ingest(document, state)
        if state is not None:
            self._validate_state(state)
        tokens = self._document_tokens(document)
        if state is None:
            state = self._new_state()
        try:
            for start in range(0, len(tokens), self.config.prefill_chunk_size):
                self._forward_tokens(tokens[start:start + self.config.prefill_chunk_size], state,
                                     last_only=True)
            state._documents += 1
        except Exception:
            state._valid = False
            raise
        return state

    def clone_state(self, state: StreamState) -> StreamState:
        """Create an independent inference branch, including all pending writes.

        Args:
            state: A compatible state owned by this model.

        Returns:
            A detached deep copy; no mutable tensor storage is shared.

        Raises:
            ValueError: The source state is stale, foreign or invalid.
        """
        self._validate_state(state)
        if state._remote_id is not None:
            return self._serving_backend().clone_state(state)
        return state._copy(detach=True)

    @torch.no_grad()
    def answer(
        self, question: Question, state: StreamState, config: AnswerConfig = AnswerConfig(),
    ) -> Answer:
        """Generate on a disposable branch, preserving source state and model weights.

        Args:
            question: Query about the documents already seen.
            state: Compatible source stream; never mutated, including on failure.
            config: Token limit, sampling and query-time memory policy.

        Returns:
            Generated text and stop metadata. ``[unknown]`` is the abstention
            convention; the library does not infer factual correctness.

        Raises:
            ValueError: The source stream is stale, foreign or invalid.
        """
        if self._uses_sglang():
            return self._serving_backend().answer(question, state, config)
        branch = self.clone_state(state)
        write = config.memory_policy == "temporary"
        prompt = self._question_tokens(question)
        for start in range(0, len(prompt), self.config.prefill_chunk_size):
            self._forward_tokens(prompt[start:start + self.config.prefill_chunk_size], branch, write=write,
                                 last_only=True)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        generated: list[int] = []
        eos = self.backbone.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        stopped_by_limit = True
        for _ in range(config.max_new_tokens):
            if branch._next_logits is None:
                raise ValueError("Tokenizer produced an empty question and no prefix exists")
            logits = branch._next_logits[0, -1].float()
            if config.temperature:
                probabilities = (logits / config.temperature).softmax(-1).cpu()
                token = int(torch.multinomial(probabilities, 1, generator=generator).item())
            else:
                token = int(logits.argmax().item())
            generated.append(token)
            if token in eos_ids:
                stopped_by_limit = False
                break
            self._forward_tokens([token], branch, write=write)
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return Answer(question.question_id, text, text.casefold() == "[unknown]",
                      len(generated), stopped_by_limit)

    def fit(self, episodes: Iterable[TrainingEpisode], config: TrainingConfig) -> TrainingResult:
        """Post-train memory and optionally the backbone on independent episodes.

        Args:
            episodes: One pass over document streams with optional QA supervision.
            config: Budget, truncation, objective and output checkpoint directory.

        Returns:
            Checkpoint location, actual work and measured training loss.

        Raises:
            ValueError: Episode supervision or configuration is invalid.
            FileExistsError: The requested checkpoint directory already exists.

        Existing streams become incompatible when fit starts. Training uses
        truncated backpropagation through memory writes; each episode resets
        state and each training question runs on a disposable branch.
        """
        from .training import fit_model

        return fit_model(self, episodes, config)

    def save_pretrained(self, path: str | Path) -> Path:
        """Atomically save a complete inference checkpoint without overwriting files.

        Args:
            path: A new output directory. Backbone weights are included for offline use.

        Returns:
            The completed checkpoint directory.

        Raises:
            FileExistsError: The destination already exists.
        """
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".titans-", dir=destination.parent))
        try:
            self.backbone.save_pretrained(temporary / "backbone", safe_serialization=True)
            self.tokenizer.save_pretrained(temporary / "backbone")
            adapter = {f"memories.{key}": value.detach().cpu().contiguous()
                       for key, value in self.memories.state_dict().items()}
            adapter["persistent"] = self.persistent.detach().cpu().contiguous()
            save_file(adapter, temporary / "memory.safetensors")
            (temporary / "titans.json").write_text(json.dumps({
                "format_version": 1, "revision": self._revision, "config": asdict(self.config),
            }, indent=2) + "\n")
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    @classmethod
    def load_pretrained(
        cls, path: str | Path, *, device: str = "cpu", dtype: torch.dtype = torch.float32,
    ) -> "TitansModel":
        """Restore trained weights from an offline checkpoint created by this library.

        Args:
            path: Complete Titans checkpoint directory.
            device: Single destination device.
            dtype: Backbone and learned-adapter dtype.

        Returns:
            An evaluation model with fresh stream ownership. Optimizer and live
            document states are deliberately not part of this checkpoint format.
        """
        source = Path(path)
        metadata = json.loads((source / "titans.json").read_text())
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported Titans checkpoint format")
        settings = metadata["config"]
        settings["memory_layers"] = tuple(settings["memory_layers"])
        model = cls.from_pretrained(source / "backbone", TitansConfig(**settings),
                                    device=device, dtype=dtype, local_files_only=True)
        adapter = load_file(source / "memory.safetensors")
        model.memories.load_state_dict({
            key.removeprefix("memories."): value
            for key, value in adapter.items() if key.startswith("memories.")
        }, strict=True)
        with torch.no_grad():
            model.persistent.copy_(adapter["persistent"])
        model._revision = int(metadata["revision"])
        return model
