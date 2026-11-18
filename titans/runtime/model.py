"""Fixed stream execution with selectable joint training and pluggable memory."""

import json
import shutil
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import asdict, fields, replace
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Self, final

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from titans.core.protocols import (
    ConnectionContext,
    ConnectionFactory,
    DataGenerator,
    MemoryArchitecture,
    MemoryFactory,
    Objective,
    UpdateRule,
)
from titans.core.types import (
    Answer,
    AnswerConfig,
    Document,
    Question,
    RuntimeConfig,
    TrainingConfig,
    TrainingEpisode,
    TrainingResult,
)
from titans.runtime.backbones import (
    LLAMA_TOKENIZER,
    PRETRAINED_BACKBONE,
    scratch_config,
    validate_backbone,
)
from titans.runtime.connections import ResidualConnection, residual_connection
from titans.runtime.state import AttentionState, StreamState

if TYPE_CHECKING:
    from titans.core.types import LearningResult
    from titans.serving.backend import SGLangBackend


class ContinualModel(nn.Module):
    """Fixed decoder runtime with causal, replaceable memory transitions.

    Ingestion order, bounded attention and branch ownership are fixed. A factory
    supplies the algorithm at each selected layer. Custom algorithms use the
    PyTorch runtime; specialized serving adapters declare their own support.
    """

    supports_sglang = False
    _config_type = RuntimeConfig

    def __init__(
        self,
        backbone: LlamaForCausalLM | Qwen2ForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        config: RuntimeConfig = RuntimeConfig(),
        *,
        memory_factory: MemoryFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Attach memory modules to one of the two fixed decoder architectures.

        Args:
            backbone: An unquantized, single-device model with a supported fixed configuration.
            tokenizer: Its tokenizer; vocabulary IDs must match the backbone.
            config: Memory placement and bounded backbone execution settings.
            memory_architecture: Architecture factory; defaults to Titans neural memory.
            connection_factory: Per-layer connection factory; defaults to causal MAC for ContinualModel.
            memory_factory: Constructs a fresh method module for each selected layer.

        Raises:
            ValueError: The architecture, device placement or RoPE mode is unsupported.
        """
        super().__init__()
        if not isinstance(backbone, (LlamaForCausalLM, Qwen2ForCausalLM)):
            raise ValueError("Only LlamaForCausalLM and Qwen2ForCausalLM backbones are supported")
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
        self.backbone_kind = validate_backbone(backbone.config)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.config = config
        parameter = backbone.model.embed_tokens.weight
        if memory_factory is not None and memory_architecture is not None:
            raise ValueError("Supply memory_architecture or legacy memory_factory, not both")
        if memory_factory is None and memory_architecture is None:
            from titans.methods.titans import TitansArchitecture, TitansConfig

            memory_config = config if isinstance(config, TitansConfig) else TitansConfig(
                compile_kernels=config.compile_kernels,
            )
            memory_architecture = TitansArchitecture(memory_config)
        architecture = memory_architecture or memory_factory
        assert architecture is not None
        if connection_factory is None:
            from titans.methods.titans import titans_mac_connection

            connection_factory = titans_mac_connection
        self.connections = nn.ModuleDict({
            str(index): connection_factory(index, backbone.config.hidden_size)
            for index in config.memory_layers
        }).to(device=parameter.device, dtype=parameter.dtype)
        if len({id(module) for module in self.connections.values()}) != len(config.memory_layers):
            raise ValueError("connection_factory must create independent modules per layer")
        from titans.methods.titans import TitansMACConnection

        self._mac_layers = {int(i) for i, c in self.connections.items()
                            if type(c) is TitansMACConnection}
        custom = [c for i, c in self.connections.items() if int(i) not in self._mac_layers]
        if any(c.max_context_tokens < 0 or c.max_context_tokens + 1 > config.attention_window
               for c in custom):
            raise ValueError("Connection context must fit inside attention_window")
        self._custom_context = any(c.max_context_tokens for c in custom)
        self.memories = nn.ModuleDict({
            str(index): architecture(backbone.config.hidden_size)
            for index in config.memory_layers
        }).to(device=parameter.device, dtype=parameter.dtype)
        if len({id(module) for module in self.memories.values()}) != len(config.memory_layers):
            raise ValueError("memory_factory must create independent modules for each layer")
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
        if not self.supports_sglang:
            if self.config.inference_backend == "sglang":
                raise ValueError("This memory method has no SGLang adapter; use torch")
            return False
        device = self.backbone.model.embed_tokens.weight.device
        if self.config.inference_backend == "sglang" and device.type != "cuda":
            raise RuntimeError("SGLang inference requires a CUDA model")
        if self.config.inference_backend != "auto":
            return self.config.inference_backend == "sglang"
        if (device.type != "cuda" or self.config.attention_backend == "sdpa"
                or self.backbone.dtype not in (torch.float16, torch.bfloat16)
                or find_spec("sglang") is None):
            return False
        from titans.optim.gpu_ops import fa3_device_unavailable

        return fa3_device_unavailable(device) is None

    def _serving_backend(self) -> "SGLangBackend":
        """Start the SGLang worker lazily for the current trained-model revision.

        Returns:
            The owned local inference engine.
        """
        if self._sglang_backend is None:
            from titans.serving.backend import SGLangBackend

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
        model_name_or_path: str | Path = PRETRAINED_BACKBONE,
        config: RuntimeConfig = RuntimeConfig(),
        *,
        memory_factory: MemoryFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
        connection_factory: ConnectionFactory | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> "ContinualModel":
        """Load the fixed Qwen checkpoint and attach new, untrained memory modules.

        Args:
            model_name_or_path: Must equal Qwen/Qwen2.5-1.5B-Instruct.
            config: Bounded runtime and placement settings.
            memory_architecture: Architecture factory; defaults to Titans neural memory.
            connection_factory: Per-layer connection factory; defaults to causal MAC for ContinualModel.
            memory_factory: Constructs the new method modules.
            device: One device, such as ``cpu`` or ``cuda:0``.
            dtype: Backbone and learned-adapter storage dtype.
            local_files_only: Whether downloads are forbidden.

        Returns:
            A model in evaluation mode with a frozen backbone.
        """
        if str(model_name_or_path) != PRETRAINED_BACKBONE:
            raise ValueError("Only " + PRETRAINED_BACKBONE + " is supported; use load_pretrained for library checkpoints")
        backbone = AutoModelForCausalLM.from_pretrained(
            str(model_name_or_path), dtype=dtype, local_files_only=local_files_only,
            attn_implementation="sdpa",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_name_or_path), local_files_only=local_files_only,
        )
        return cls(backbone, tokenizer, config, memory_factory=memory_factory,
                   memory_architecture=memory_architecture, connection_factory=connection_factory)

    @classmethod
    def from_scratch(
        cls, config: RuntimeConfig = RuntimeConfig(), *,
        memory_factory: MemoryFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
        connection_factory: ConnectionFactory | None = None,
        device: str = "cpu", dtype: torch.dtype = torch.float32,
        seed: int = 0, local_files_only: bool = False,
    ) -> Self:
        """Initialize the fixed Llama-style decoder and its memory from scratch.

        Args:
            config: Memory placement and execution settings; no decoder overrides.
            memory_factory: Legacy alias for memory_architecture.
            memory_architecture: Optional neural memory constructor.
            connection_factory: Optional backbone wiring; defaults to MAC.
            device: Single initialization and execution device.
            dtype: Floating-point parameter storage dtype.
            seed: Reproducible initialization seed, isolated from the caller's RNG.
            local_files_only: Require the fixed Llama 2 tokenizer in the local cache.

        Returns:
            An evaluation model whose fit default jointly trains backbone and phi.
        """
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER,
                                                  local_files_only=local_files_only)
        if len(tokenizer) != 32000:
            raise ValueError("Scratch training requires the fixed 32K Llama 2 tokenizer")
        target = torch.device(device)
        devices = ([target.index if target.index is not None else torch.cuda.current_device()]
                   if target.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            with torch.device(target):
                backbone = LlamaForCausalLM(scratch_config()).to(dtype=dtype)
                model = cls(backbone, tokenizer, config, memory_factory=memory_factory,
                            memory_architecture=memory_architecture,
                            connection_factory=connection_factory)
        return model

    def _validate_state(self, state: StreamState) -> None:
        """Reject foreign, stale, failed or differently placed streams.

        Args:
            state: A stream previously returned by this model.

        Raises:
            ValueError: The stream cannot safely be used by the current model.
        """
        if not state._valid:
            raise ValueError("This stream was invalidated by a failed ingest or learning update")
        if state._owner != self._owner or state._revision != self._revision:
            raise ValueError("Stream belongs to a different model or predates fit or a shared-parameter update")
        device = self.backbone.model.embed_tokens.weight.device
        if any(tensor.device != device for memory in state._memory.values()
               for tensor in memory.tensors()):
            raise ValueError("Model moved devices; start a new stream")

    def _new_state(self, *, prefix_hidden_only: bool = False) -> StreamState:
        """Initialize a stream and its learned persistent prefix.

        Args:
            prefix_hidden_only: Skip the prefix vocabulary projection when the caller
                consumes its hidden state or immediately replaces its prediction.

        Returns:
            An empty document stream. Prefix tokens do not count toward the
            public token budget and do not write to the fast memory.
        """
        state = StreamState(self._owner, self._revision)
        for key, memory in self.memories.items():
            state._memory[int(key)] = memory.initial_state()
        if self.config.persistent_tokens:
            self._forward_embeddings(self.persistent, state, write=False, last_only=True,
                                     return_hidden=prefix_hidden_only)
        return state

    def _local_attention(
        self,
        layer: LlamaDecoderLayer | Qwen2DecoderLayer,
        hidden: Tensor,
        positions: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: AttentionState | None,
        next_position: int,
        window: int | None = None,
    ) -> tuple[Tensor, AttentionState]:
        """Run causal SDPA while retaining only persistent and recent grouped KV.

        Args:
            layer: The host Llama or Qwen2 decoder layer.
            hidden: Normalized layer input [1, tokens, hidden].
            positions: Absolute positions for this chunk, shaped [tokens].
            rotary: Cosine and sine tensors from the host's RoPE implementation.
            cache: Bounded cache from the preceding call, or None.
            window: Local attention span in this layer's position units.
            next_position: Python integer position following this block; avoids
                device synchronization when selecting the retained cache.

        Returns:
            Attention output and a bounded cache for the next chunk.
        """
        attention = layer.self_attn
        key_positions = positions
        if cache is not None:
            key_positions = torch.cat((cache.positions, positions))
        window = self.config.attention_window if window is None else window
        backend = self.config.attention_backend
        prefix_count = min(self.config.persistent_tokens, next_position)
        prefix_only = next_position <= self.config.persistent_tokens

        def mask() -> Tensor:
            """Return the causal mask including globally visible persistent tokens."""
            return (key_positions[None, :] <= positions[:, None]) & (
                (key_positions[None, :] < self.config.persistent_tokens)
                | (key_positions[None, :] > positions[:, None] - window)
            )

        # Compiling the complete CUDA SDPA wrapper produced nonfinite Qwen
        # gradients in PyTorch 2.9.1. Native fused attention and Liger RoPE retain
        # their autograd rules; CPU attention and the memory core still use JIT.
        if self.config.compile_kernels and backend != "fa3" and hidden.device.type != "cuda":
            from titans.optim.kernels import compiled_attention

            empty_shape = (hidden.shape[0], attention.k_proj.out_features // attention.head_dim,
                           0, attention.head_dim)
            old_keys = cache.keys if cache is not None else hidden.new_empty(empty_shape)
            old_values = cache.values if cache is not None else hidden.new_empty(empty_shape)
            output, key, value = compiled_attention(
                hidden, attention.q_proj.weight, attention.k_proj.weight,
                attention.v_proj.weight, attention.o_proj.weight,
                attention.q_proj.bias, attention.k_proj.bias,
                attention.v_proj.bias, attention.o_proj.bias,
                *rotary, old_keys, old_values, mask(), attention.head_dim, attention.scaling,
            )
        else:
            shape = (*hidden.shape[:-1], -1, attention.head_dim)
            query = attention.q_proj(hidden).view(shape).transpose(1, 2)
            key = attention.k_proj(hidden).view(shape).transpose(1, 2)
            value = attention.v_proj(hidden).view(shape).transpose(1, 2)
            rotary = tuple(part.to(query.dtype) for part in rotary)
            if hidden.device.type == "cuda" and self.config.use_liger:
                from titans.optim.gpu_ops import liger_rotary

                query, key = liger_rotary(query, key, *rotary)
            else:
                query, key = apply_rotary_pos_emb(query, key, *rotary)
            if cache is not None:
                key = torch.cat((cache.keys, key), -2)
                value = torch.cat((cache.values, value), -2)
            groups = query.shape[1] // key.shape[1]
            use_fa3 = False
            if backend == "fa3" or (backend == "auto" and query.is_cuda):
                from titans.optim.gpu_ops import fa3_unavailable

                reason = fa3_unavailable(query, key, value, prefix_count, prefix_only)
                if reason and backend == "fa3":
                    raise RuntimeError(f"FA3 unavailable: {reason}. Use attention_backend='sdpa' or 'auto'.")
                use_fa3 = reason is None
            if use_fa3:
                from titans.optim.gpu_ops import flash_attention3

                try:
                    output = flash_attention3(
                        query, key, value, prefix_count=prefix_count,
                        window=window, scale=attention.scaling, prefix_only=prefix_only,
                    )
                except NotImplementedError as error:
                    if backend == "fa3":
                        raise RuntimeError(
                            "The FA3 kernel does not support this attention operation. "
                            "Use attention_backend='auto' or 'sdpa'."
                        ) from error
                    use_fa3 = False
            if not use_fa3:
                output = F.scaled_dot_product_attention(
                    query, key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1),
                    attn_mask=mask()[None, None], dropout_p=0.0, scale=attention.scaling,
                )
            output = attention.o_proj(output.transpose(1, 2).reshape(*hidden.shape[:-1], -1))
        tail_count = min(window - 1, key.shape[-2] - prefix_count)
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
        prefix_pass = state._position == 0 and embeddings.shape[1] == self.config.persistent_tokens
        segment = self.config.mac_segment_length
        offset = (state._position - self.config.persistent_tokens) % segment
        block_limit = embeddings.shape[1]
        if not prefix_pass:
            if self._custom_context:
                block_limit = 1
            elif self._mac_layers:
                block_limit = segment - offset
        if embeddings.shape[1] > block_limit:
            outputs = []
            start = 0
            while start < embeddings.shape[1]:
                end = min(start + block_limit, embeddings.shape[1])
                final_block = end == embeddings.shape[1]
                output = self._forward_embeddings(
                    embeddings[:, start:end], state, write=write, last_only=last_only,
                    return_hidden=return_hidden or (last_only and not final_block),
                )
                if not last_only or final_block:
                    outputs.append(output)
                start = end
                block_limit = 1 if self._custom_context else segment
            return outputs[-1] if last_only else torch.cat(outputs, dim=1)
        if positions is None:
            positions = torch.arange(
                state._position, state._position + embeddings.shape[1], device=embeddings.device,
            )
        rotary = self.backbone.model.rotary_emb(embeddings, positions[None, :])
        hidden = embeddings
        use_liger = self.config.use_liger and hidden.device.type == "cuda"
        for index, layer in enumerate(self.backbone.model.layers):
            if use_liger:
                from titans.optim.gpu_ops import liger_norm

                normalized = liger_norm(hidden, layer.input_layernorm.weight,
                                        layer.input_layernorm.variance_epsilon)
            elif self.config.compile_kernels:
                from titans.optim.kernels import compiled_rms_norm

                normalized = compiled_rms_norm(hidden, layer.input_layernorm.weight,
                                               layer.input_layernorm.variance_epsilon)
            else:
                normalized = layer.input_layernorm(hidden)
            context = ConnectionContext(index, hidden, normalized)
            connection = self.connections[str(index)] if str(index) in self.memories else None
            if prefix_pass and connection is not None and connection.max_context_tokens:
                connection = None
            plan = None
            if connection is not None:
                if index in self._mac_layers:
                    if index not in state._retrieval:
                        state._retrieval[index] = state._memory[index].copy()
                    plan = connection._prepare_segment(context, self.memories[str(index)],
                                                       state._retrieval[index])
                else:
                    plan = connection.prepare(context, self.memories[str(index)],
                                              state._memory[index], write=write)
                if (plan.memory_output.shape != hidden.shape
                        or plan.memory_output.device != hidden.device
                        or plan.memory_output.dtype != hidden.dtype):
                    raise ValueError("Memory read must preserve shape, device and dtype")
                extra = plan.attention_input.shape[1] - hidden.shape[1]
                limit = hidden.shape[1] if index in self._mac_layers else connection.max_context_tokens
                if (not 0 <= extra <= limit
                        or plan.attention_input.shape[0] != hidden.shape[0]
                        or plan.attention_input.shape[2:] != hidden.shape[2:]
                        or plan.attention_input.device != hidden.device
                        or plan.attention_input.dtype != hidden.dtype):
                    raise ValueError("Invalid connection attention input")
            else:
                extra = 0
            local_positions, local_rotary = positions, rotary
            old_cache = state._attention.get(index)
            next_position = state._position + embeddings.shape[1]
            if connection is not None and connection.max_context_tokens:
                # Paired MAC positions resume from the current segment offset.
                prefix = self.config.persistent_tokens
                count = embeddings.shape[1] + extra
                first = prefix + (2 * offset if index in self._mac_layers else 0)
                local_positions = torch.arange(first, first + count, device=hidden.device)
                local_rotary = self.backbone.model.rotary_emb(plan.attention_input,
                                                              local_positions[None, :])
                next_position = first + count
            attended, cache = self._local_attention(
                layer, plan.attention_input if plan is not None else normalized,
                local_positions, local_rotary, old_cache, next_position,
                window=2 * segment if index in self._mac_layers and not prefix_pass else None,
            )
            if connection is not None and connection.max_context_tokens:
                mac = index in self._mac_layers
                if not mac or offset + embeddings.shape[1] == segment:
                    n = self.config.persistent_tokens
                    cache = AttentionState(cache.keys[:, :, :n].clone(), cache.values[:, :, :n].clone(),
                                           cache.positions[:n].clone())
                    state._retrieval.pop(index, None)
                attended = attended[:, 1::2] if mac else attended[:, extra:]
            state._attention[index] = cache
            if connection is not None and plan is not None:
                hidden = connection.finish(context, attended, plan, self.memories[str(index)],
                                           state._memory[index], write=write)
                if (hidden.shape != context.hidden.shape or hidden.device != context.hidden.device
                        or hidden.dtype != context.hidden.dtype):
                    raise ValueError("Connection output must preserve shape, device and dtype")
            else:
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
                    from titans.optim.gpu_ops import liger_mlp

                    mlp_forward = liger_mlp
                else:
                    from titans.optim.kernels import compiled_llama_mlp

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
        state._boundary_prediction = False
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
            final_block = start + ids.shape[1] == len(tokens)
            output = self._forward_embeddings(self.backbone.model.embed_tokens(ids), state,
                                              write=write, last_only=last_only,
                                              return_hidden=return_hidden or (last_only and not final_block))
            if not last_only or final_block:
                outputs.append(output)
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

    @final
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
            state = self._new_state(prefix_hidden_only=bool(tokens))
        try:
            if tokens:
                self._forward_tokens(tokens, state, last_only=True)
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
        if prompt:
            self._forward_tokens(prompt, branch, write=write, last_only=True)
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

    def fit(
        self, episodes: Iterable[TrainingEpisode] | DataGenerator, config: TrainingConfig,
        *, loss: Objective | None = None, update: UpdateRule | None = None,
    ) -> TrainingResult:
        """Train memory alone or jointly with the selected backbone.

        Args:
            episodes: One pass over streams, or a generator called with config.seed.
            config: Budget, truncation, supervision routing and output checkpoint.
            loss: Arbitrary objective; defaults to token cross-entropy.
            update: Explicit checkpoint rule for phi, W, both, or joint theta/phi. When supplied,
                use document_qa routing; updates occur after complete QA checkpoints.

        Returns:
            Checkpoint location, actual work and measured training loss.

        Raises:
            ValueError: Episode supervision or configuration is invalid.
            FileExistsError: The requested checkpoint directory already exists.

        Existing streams become incompatible when fit starts. Training uses
        truncated backpropagation through memory writes; each episode resets
        state and each training question runs on a disposable branch.
        """
        from titans.training.trainer import fit_model

        return fit_model(self, episodes, config, loss=loss, update=update)

    def start_stream(self) -> StreamState:
        """Create a local stream; documents and questions never reset it implicitly.

        Returns:
            A fresh detached state carrying learned initialization and bounded caches.
        """
        if self._uses_sglang():
            raise ValueError("Explicit streams and feedback require inference_backend='torch'")
        with torch.no_grad():
            return self._new_state()

    def outer_parameters(self, *, train_backbone: bool) -> tuple[nn.Parameter, ...]:
        """Select outer-loop parameters without including active stream weights W.

        Args:
            train_backbone: Include theta along with phi for joint training.

        Returns:
            Unique optimizer leaves; tied embeddings occur only once.
        """
        parameters = self.shared_parameters() + (tuple(self.backbone.parameters()) if train_backbone else ())
        return tuple(dict.fromkeys(parameters))

    def shared_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return phi, excluding backbone parameters regardless of training mode."""
        return (tuple(self.memories.parameters()) + tuple(self.connections.parameters())
                + (self.persistent,))

    def learn(
        self, question: Question, target_answer: str, state: StreamState,
        *, loss: Objective | None = None, update: UpdateRule | None = None,
        retain_history: bool = False, train_backbone: bool = False,
    ) -> "LearningResult":
        """Adapt using explicitly supplied feedback, leaving teacher tokens off-stream.

        Args:
            question: Question whose answer has now become available.
            target_answer: Released supervision, never a hidden evaluation answer.
            state: Continuing stream receiving functional W replacements.
            loss: Arbitrary scalar objective; defaults to answer cross-entropy.
            update: Target/optimizer policy; defaults to W-only SGD, or joint SGD
                when train_backbone is enabled.
            retain_history: Preserve the current unroll for meta-gradients into phi.
            train_backbone: Expose theta to an explicit joint update policy.

        Returns:
            Loss and token accounting. Updating phi invalidates other live streams.
        """
        from titans.training.learning import learn

        return learn(self, question, target_answer, state, loss=loss, update=update,
                     retain_history=retain_history, train_backbone=train_backbone)

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
            adapter.update({f"connections.{key}": value.detach().cpu().contiguous()
                            for key, value in self.connections.state_dict().items()})
            adapter["persistent"] = self.persistent.detach().cpu().contiguous()
            save_file(adapter, temporary / "memory.safetensors")
            from titans.methods.titans import NeuralMemory

            memory_configs = {key: asdict(memory.config) for key, memory in self.memories.items()
                              if type(memory) is NeuralMemory}
            settings = asdict(self.config)
            chunks = {options["chunk_size"] for options in memory_configs.values()}
            if len(memory_configs) == len(self.memories) and len(chunks) == 1:
                settings["chunk_size"] = chunks.pop()
            (temporary / "titans.json").write_text(json.dumps({
                "format_version": 4,
                "mac_retrieval_policy": "shared_causal",
                "builtin_memory_configs": memory_configs,
                "connection_types": {key: type(module).__module__ + "." + type(module).__qualname__
                                     for key, module in self.connections.items()},
                "revision": self._revision, "config": settings,
            }, indent=2) + "\n")
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    @classmethod
    def load_pretrained(
        cls, path: str | Path, *, memory_factory: MemoryFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
        connection_factory: ConnectionFactory | None = None,
        config: RuntimeConfig | None = None, device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        """Restore trained weights from an offline checkpoint created by this library.

        Args:
            path: Complete checkpoint directory.
            memory_architecture: Architecture factory; defaults to Titans neural memory.
            connection_factory: Per-layer connection factory; defaults to causal MAC for ContinualModel.
            memory_factory: Explicit trusted method constructor; code is not loaded from metadata.
            config: Optional method-specific settings; otherwise restore runtime fields.
            device: Single destination device.
            dtype: Backbone and learned-adapter dtype.

        Returns:
            An evaluation model with fresh stream ownership. Optimizer and live
            document states are deliberately not part of this checkpoint format.
        """
        source = Path(path)
        metadata = json.loads((source / "titans.json").read_text())
        if metadata["format_version"] not in (1, 2, 3, 4):
            raise ValueError("Unsupported Titans checkpoint format")
        settings = metadata["config"]
        settings["memory_layers"] = tuple(settings["memory_layers"])
        if metadata["format_version"] < 4:
            settings["mac_segment_length"] = 1
            settings.setdefault("chunk_size", 1)
        elif metadata.get("mac_retrieval_policy") != "shared_causal":
            raise ValueError("Unsupported MAC checkpoint retrieval policy")
        runtime_fields = {field.name for field in fields(cls._config_type)}
        if config is None:
            config = cls._config_type(**{k: v for k, v in settings.items() if k in runtime_fields})
        else:
            config = replace(config, mac_segment_length=settings["mac_segment_length"])
        from titans.methods.titans import (
            TitansMACConnection,
            TitansMAGConnection,
            titans_mac_connection,
            titans_mag_connection,
        )

        if memory_architecture is None and memory_factory is None:
            from titans.methods.titans import NeuralMemory, TitansConfig

            saved = metadata.get("builtin_memory_configs", {})
            if metadata["format_version"] == 4 and set(saved) != {str(i) for i in config.memory_layers}:
                raise ValueError("Custom checkpoint requires an explicit memory_architecture")
            layer_settings = iter(saved[str(i)] if metadata["format_version"] == 4 else settings
                                  for i in config.memory_layers)

            def restore_memory(hidden_size: int) -> NeuralMemory:
                """Restore the next configured memory layer at the decoder's hidden width."""
                options = dict(next(layer_settings))
                options["memory_layers"] = tuple(options["memory_layers"])
                return NeuralMemory(hidden_size, TitansConfig(**options))

            memory_architecture = restore_memory

        # Older format-3 checkpoints recorded the original flat package paths.
        legacy_connections = {
            "titans.connections.ResidualConnection":
                ResidualConnection.__module__ + ".ResidualConnection",
            "titans.titans.TitansMACConnection":
                TitansMACConnection.__module__ + ".TitansMACConnection",
            "titans.titans.TitansMAGConnection":
                TitansMAGConnection.__module__ + ".TitansMAGConnection",
        }
        kinds = {key: legacy_connections.get(kind, kind)
                 for key, kind in metadata.get("connection_types", {}).items()}
        if connection_factory is None:
            names = set(kinds.values())
            if not names or names == {ResidualConnection.__module__ + ".ResidualConnection"}:
                connection_factory = residual_connection
            elif names == {TitansMACConnection.__module__ + ".TitansMACConnection"}:
                connection_factory = titans_mac_connection
            elif names == {TitansMAGConnection.__module__ + ".TitansMAGConnection"}:
                connection_factory = titans_mag_connection
            else:
                raise ValueError("Custom checkpoint requires an explicit connection_factory")
        backbone = AutoModelForCausalLM.from_pretrained(
            source / "backbone", dtype=dtype, local_files_only=True,
            attn_implementation="sdpa",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(source / "backbone", local_files_only=True)
        model = cls(backbone, tokenizer, config, memory_factory=memory_factory,
                    memory_architecture=memory_architecture, connection_factory=connection_factory)
        adapter = load_file(source / "memory.safetensors")
        model.memories.load_state_dict({
            key.removeprefix("memories."): value
            for key, value in adapter.items() if key.startswith("memories.")
        }, strict=True)
        actual_kinds = {key: type(module).__module__ + "." + type(module).__qualname__
                        for key, module in model.connections.items()}
        if kinds and actual_kinds != kinds:
            raise ValueError("Supplied connection architecture differs from the checkpoint")
        model.connections.load_state_dict({
            key.removeprefix("connections."): value
            for key, value in adapter.items() if key.startswith("connections.")
        }, strict=True)
        with torch.no_grad():
            model.persistent.copy_(adapter["persistent"])
        model._revision = int(metadata["revision"])
        return model
