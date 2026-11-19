"""Differentiable Titans updates, using weighted MLP gradient reductions.

See Behrouz et al., arXiv:2501.00663v1, equations 12--14 and section 3.2.
There is no autograd invocation or per-token parameter-gradient allocation in
the production update. Ordinary PyTorch operations retain outer gradients.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from titans.core.protocols import (
    BackboneConnection,
    ConnectionContext,
    ConnectionExecution,
    ConnectionFactory,
    ConnectionPlan,
    MemoryArchitecture,
    MemoryFactory,
    MemoryModule,
)
from titans.core.protocols import MemoryState as StateProtocol
from titans.core.types import RuntimeConfig
from titans.runtime.backbones import PRETRAINED_BACKBONE
from titans.runtime.connections import ResidualConnection
from titans.runtime.model import ContinualModel

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM, PreTrainedTokenizerBase, Qwen2ForCausalLM


@dataclass(frozen=True)
class TitansConfig(RuntimeConfig):
    """Memory and local-attention settings for the Titans method.

    Chunk size one implements tokenwise updates. Larger chunks use gradients
    evaluated at the chunk's initial weights and commit only complete chunks.
    Memory reads use the last committed weights. ``memory_layers`` is zero-based.
    Fast weights and their updates use float32 unless the whole model is float64.
    Fresh models use a residual GELU network with non-affine LayerNorm (eps=1e-5).
    An omitted hidden dimension expands the memory width by four. Legacy checkpoints
    retain their original SiLU network through memory_variant="legacy".
    """

    memory_dim: int = 128
    memory_hidden_dim: int | None = None
    memory_depth: int = 2
    chunk_size: int = 16
    conv_kernel: int = 3
    max_inner_learning_rate: float = 0.01
    gate_bias: float = -4.0
    use_liger: bool = True
    memory_variant: Literal["paper", "legacy"] = field(default="paper", kw_only=True)

    def __post_init__(self) -> None:
        """Validate the fixed runtime and this method's dimensions and update rates."""
        super().__post_init__()
        if self.memory_hidden_dim is None:
            object.__setattr__(self, "memory_hidden_dim", 4 * self.memory_dim)
        if self.memory_variant not in ("paper", "legacy"):
            raise ValueError("memory_variant must be paper or legacy")
        if self.memory_variant == "paper" and self.memory_dim < 2:
            raise ValueError("Paper memory requires memory_dim >= 2 for LayerNorm")
        if min(self.memory_dim, self.memory_hidden_dim, self.memory_depth,
               self.chunk_size, self.conv_kernel) < 1:
            raise ValueError("Memory dimensions, chunk size and kernel must be positive")
        if not 0 < self.max_inner_learning_rate <= 1:
            raise ValueError("max_inner_learning_rate must be in (0, 1]")
        if not math.isfinite(self.gate_bias):
            raise ValueError("gate_bias must be finite")


def mlp(weights: tuple[Tensor, ...], inputs: Tensor, *, paper: bool = False) -> Tensor:
    """Evaluate the selected batched fast-memory network.

    Args:
        weights: Matrices shaped [batch, output, input], one per layer.
        inputs: Keys or queries shaped [batch, tokens, dimension].
        paper: Use GELU and residual, non-affine LayerNorm with epsilon 1e-5.
            False retains the historical SiLU network.

    Returns:
        Predicted values with the same leading dimensions as inputs.
    """
    value = inputs
    for index, weight in enumerate(weights):
        value = value @ weight.transpose(-1, -2)
        if index + 1 < len(weights):
            value = F.gelu(value) if paper else F.silu(value)
    if paper:
        value = inputs + F.layer_norm(value, (value.shape[-1],), eps=1e-5)
    return value


def suffix_products(values: Tensor) -> Tensor:
    """Return products strictly to the right along the final axis, without division.

    Args:
        values: A nonempty [..., tokens] tensor; exact zeros are supported.

    Returns:
        A tensor of the same shape, ending in one.
    """
    reverse = values.flip(-1).cumprod(-1).flip(-1)
    return torch.cat((reverse[..., 1:], torch.ones_like(reverse[..., :1])), dim=-1)


def affine_scan(multiplier: Tensor, offset: Tensor) -> Tensor:
    """Solve x[i] = multiplier[i] * x[i-1] + offset[i], with x[-1] = 0.

    Args:
        multiplier: Transition coefficients shaped [..., tokens].
        offset: Additive terms with identical shape.

    Returns:
        All prefix states, using a division-free logarithmic-depth scan.
    """
    a, b = multiplier, offset
    stride = 1
    while stride < a.shape[-1]:
        b = torch.cat((b[..., :stride], b[..., stride:] + a[..., stride:] * b[..., :-stride]), -1)
        a = torch.cat((a[..., :stride], a[..., stride:] * a[..., :-stride]), -1)
        stride *= 2
    return b


def chunk_update(
    weights: tuple[Tensor, ...],
    momentum: tuple[Tensor, ...],
    keys: Tensor,
    values: Tensor,
    alpha: Tensor,
    eta: Tensor,
    theta: Tensor,
    *, paper: bool = False,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Apply equations 13--14 with gradients evaluated at chunk-start weights.

    Args:
        weights: MLP matrices [batch, output, input].
        momentum: Momentum matrices matching weights.
        keys: Chunk keys [batch, tokens, dimension].
        values: Targets with the same shape as keys.
        alpha: Forget gates [batch, tokens], in [0, 1].
        eta: Momentum gates [batch, tokens], in [0, 1].
        theta: Nonnegative inner learning rates [batch, tokens].
        paper: Differentiate the GELU/residual/LayerNorm network when True.
            LayerNorm is non-affine with epsilon 1e-5.

    Returns:
        Final weights and momentum. No input tensor is mutated. The loss is the
        sum of squared errors, not a mean over features. A one-token chunk is
        the exact tokenwise recurrence; larger chunks hold the gradient's
        evaluation point fixed as described in the paper's section 3.2.
    """
    if keys.shape[1] == 0:
        return weights, momentum
    beta = 1 - alpha
    beta_suffix = suffix_products(beta)
    reverse_eta = torch.cat((torch.zeros_like(eta[:, :1]), eta[:, 1:].flip(-1)), -1)
    h = affine_scan(reverse_eta, beta_suffix.flip(-1)).flip(-1)
    weight_coeff = theta * h
    momentum_coeff = theta * suffix_products(eta)
    old_weight_coeff = beta.prod(-1)[:, None, None]
    # Incoming momentum contributes at each write, then decays through later forget gates.
    old_momentum_coeff = (eta.cumprod(-1) * beta_suffix).sum(-1)[:, None, None]
    final_momentum_coeff = eta.prod(-1)[:, None, None]

    activations = [keys]
    preactivations: list[Tensor] = []
    for index, weight in enumerate(weights):
        z = activations[-1] @ weight.transpose(-1, -2)
        preactivations.append(z)
        if index + 1 < len(weights):
            z = F.gelu(z) if paper else F.silu(z)
        activations.append(z)
    if paper:
        centered = activations[-1] - activations[-1].mean(dim=-1, keepdim=True)
        inverse_std = (centered.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
        normalized = centered * inverse_std
        delta = 2 * (keys + normalized - values)
        delta = inverse_std * (
            delta - delta.mean(dim=-1, keepdim=True)
            - normalized * (delta * normalized).mean(dim=-1, keepdim=True)
        )
    else:
        delta = 2 * (activations[-1] - values)
    new_weights: list[Tensor] = []
    new_momentum: list[Tensor] = []
    for index in range(len(weights) - 1, -1, -1):
        inputs = activations[index]
        weight_gradient = delta.transpose(-1, -2) @ (weight_coeff[..., None] * inputs)
        momentum_gradient = delta.transpose(-1, -2) @ (momentum_coeff[..., None] * inputs)
        new_weights.append(
            old_weight_coeff * weights[index] + old_momentum_coeff * momentum[index]
            - weight_gradient
        )
        new_momentum.append(final_momentum_coeff * momentum[index] - momentum_gradient)
        if index:
            z = preactivations[index - 1]
            if paper:
                derivative = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
                derivative = derivative + z * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
            else:
                sigmoid = z.sigmoid()
                derivative = sigmoid * (1 + z * (1 - sigmoid))
            delta = (delta @ weights[index]) * derivative
    return tuple(reversed(new_weights)), tuple(reversed(new_momentum))


@dataclass
class MemoryState:
    """Private fast weights, momentum, incomplete write chunk and convolution tail."""

    weights: tuple[Tensor, ...]
    momentum: tuple[Tensor, ...]
    pending: Tensor
    conv_tail: Tensor

    def copy(self, *, detach: bool = False) -> "MemoryState":
        """Copy every mutable tensor, optionally removing its autograd history.

        Args:
            detach: Whether the copy starts a new differentiation window.

        Returns:
            An independently mutable state, including unfinished updates.
        """
        return MemoryState(
            tuple(t.detach().clone() if detach else t.clone() for t in self.weights),
            tuple(t.detach().clone() if detach else t.clone() for t in self.momentum),
            self.pending.detach().clone() if detach else self.pending.clone(),
            self.conv_tail.detach().clone() if detach else self.conv_tail.clone(),
        )

    def detach(self) -> None:
        """Discard past autograd graphs without allocating duplicate state buffers."""
        self.weights = tuple(t.detach() for t in self.weights)
        self.momentum = tuple(t.detach() for t in self.momentum)
        self.pending = self.pending.detach()
        self.conv_tail = self.conv_tail.detach()

    def tensors(self) -> tuple[Tensor, ...]:
        """Return weights, momentum, unfinished writes and convolution history."""
        return self.weights + self.momentum + (self.pending, self.conv_tail)

    def adaptable(self) -> Mapping[str, Tensor]:
        """Expose active neural weights W for optional QA-based adaptation."""
        return {str(index): value for index, value in enumerate(self.weights)}

    def replace(self, updates: Mapping[str, Tensor]) -> None:
        """Replace selected active weights without overwriting momentum or buffers.

        Args:
            updates: Weight indices as strings, with matching tensor shapes.

        Raises:
            ValueError: A name or tensor shape is incompatible with the memory.
        """
        weights = list(self.weights)
        for name, value in updates.items():
            if name not in self.adaptable() or value.shape != self.adaptable()[name].shape:
                raise ValueError("Invalid Titans memory replacement")
            weights[int(name)] = value
        self.weights = tuple(weights)


class NeuralMemory(MemoryModule):
    """Input projections, causal convolution, learned gates and deep fast memory."""

    def __init__(self, hidden_size: int, config: TitansConfig) -> None:
        """Allocate the trainable memory rule and initial fast weights.

        Args:
            hidden_size: Width of the host transformer's residual stream.
            config: Dimensions, chunk size and update-rate bound.
        """
        super().__init__()
        self.config = config
        width = config.memory_dim
        self.qkv = nn.Linear(hidden_size, 3 * width, bias=False)
        self.conv = nn.Conv1d(3 * width, 3 * width, config.conv_kernel, groups=3 * width)
        self.gates = nn.Linear(hidden_size, 3)
        self.output = nn.Linear(width, hidden_size, bias=False)
        self.mix_gate = nn.Linear(hidden_size, hidden_size)
        self.read_norm = nn.LayerNorm(width)
        sizes = [width] + [config.memory_hidden_dim] * (config.memory_depth - 1) + [width]
        self.initial_weights = nn.ParameterList(
            nn.Parameter(torch.empty(out_size, in_size))
            for in_size, out_size in zip(sizes[:-1], sizes[1:], strict=True)
        )
        for parameter in self.initial_weights:
            nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([-5.0, 2.0, -2.0]))
        nn.init.zeros_(self.mix_gate.weight)
        nn.init.constant_(self.mix_gate.bias, config.gate_bias)

    def initial_state(self, batch_size: int = 1) -> MemoryState:
        """Create a fresh differentiable stream from learned initial weights.

        Args:
            batch_size: Number of independent streams processed by the memory core.

        Returns:
            A state with zero momentum and empty pending updates.
        """
        parameter = self.initial_weights[0]
        dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        weights = tuple(p.to(dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
                        for p in self.initial_weights)
        pending = torch.empty(batch_size, 0, 2 * self.config.memory_dim + 3,
                              device=parameter.device, dtype=dtype)
        tail = torch.zeros(batch_size, 3 * self.config.memory_dim, self.config.conv_kernel - 1,
                           device=parameter.device, dtype=parameter.dtype)
        return MemoryState(weights, tuple(torch.zeros_like(w) for w in weights), pending, tail)

    def transition(self, hidden: Tensor, state: StateProtocol, *, write: bool = True) -> Tensor:
        """Read committed memory and optionally enqueue and commit causal writes.

        Args:
            hidden: Host activations [batch, tokens, hidden_size].
            state: Owned state, mutated in place. Callers branch before querying.
            write: Whether fast weights, momentum and pending writes may change.

        Returns:
            Gated memory output with the same shape and dtype as hidden. Reads
            within an unfinished chunk all use its initial weights; no current
            or future token's write is visible until the next chunk.
        """
        return self._read_and_update(hidden, state, write=write, post_write=False)

    def _read_and_update(self, hidden: Tensor, state: StateProtocol, *, write: bool,
                         post_write: bool) -> Tensor:
        """Project once and select the committed weights visible to each query.

        Args:
            hidden: Causal input activations in the host dtype.
            state: Owned memory and projection history.
            write: Whether to enqueue and commit writes.
            post_write: Whether a closing token reads its own chunk's commit.

        Returns:
            Memory outputs in the original host dtype.
        """
        with torch.autocast(hidden.device.type, enabled=False):
            return self._project_and_scan(hidden.to(self.qkv.weight.dtype), state,
                                          write=write, post_write=post_write).to(hidden.dtype)

    def _project_and_scan(self, hidden: Tensor, state: StateProtocol, *, write: bool,
                          post_write: bool) -> Tensor:
        """Scan projected tokens in chunks, retaining FP32 fast-weight arithmetic.

        Args:
            hidden: Activations in projection storage dtype.
            state: Mutable stream-local memory.
            write: Whether to update the pending buffer.
            post_write: Whether a closing query reads newly committed weights.

        Returns:
            Gated reads for all input positions.
        """
        if not isinstance(state, MemoryState):
            raise TypeError("Titans requires its own MemoryState")
        from titans.optim.kernels import (
            combine_read,
            compiled_chunk_update,
            compiled_combine_read,
            compiled_project_tokens,
            compiled_read,
            project_tokens,
        )

        project = compiled_project_tokens if self.config.compile_kernels else project_tokens
        retrieve = compiled_read if self.config.compile_kernels else mlp
        update = compiled_chunk_update if self.config.compile_kernels else chunk_update
        combine = compiled_combine_read if self.config.compile_kernels else combine_read
        query, key, value, gates, state.conv_tail = project(
            hidden, state.conv_tail, self.qkv.weight, self.conv.weight, self.conv.bias,
            self.gates.weight, self.gates.bias, self.config.max_inner_learning_rate,
        )
        outputs: list[Tensor] = []
        start = 0
        while start < hidden.shape[1]:
            count = min(self.config.chunk_size - state.pending.shape[1], hidden.shape[1] - start)
            end = start + count
            # Inductor may return transposed layouts. Canonicalize even singleton
            # strides so prefill/decode do not exhaust Dynamo's specialization cache.
            closing = write and count + state.pending.shape[1] == self.config.chunk_size
            before = end - int(post_write and closing)
            if before > start:
                queries = query[:, start:before].clone(memory_format=torch.contiguous_format)
                outputs.append(retrieve(state.weights, queries, paper=self.config.memory_variant == "paper"))
            if write:
                updates = torch.cat((key[:, start:end], value[:, start:end], gates[:, start:end]), -1)
                state.pending = torch.cat((state.pending, updates), 1)
                if state.pending.shape[1] == self.config.chunk_size:
                    width = self.config.memory_dim
                    k, v, g = state.pending.split((width, width, 3), -1)
                    state.weights, state.momentum = update(
                        state.weights, state.momentum, k, v, g[..., 0], g[..., 1], g[..., 2],
                        paper=self.config.memory_variant == "paper",
                    )
                    state.pending = state.pending[:, :0].clone()
                    if post_write:
                        last = query[:, end - 1:end].clone(memory_format=torch.contiguous_format)
                        outputs.append(retrieve(state.weights, last, paper=self.config.memory_variant == "paper"))
            start = end
        return combine(torch.cat(outputs, 1), hidden, self.read_norm.weight, self.read_norm.bias,
                       self.output.weight, self.mix_gate.weight, self.mix_gate.bias,
                       self.read_norm.eps)


def project_tokens(
    hidden: Tensor, tail: Tensor, qkv_weight: Tensor, conv_weight: Tensor, conv_bias: Tensor,
    gate_weight: Tensor, gate_bias: Tensor, max_learning_rate: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project QKV, apply causal convolution and normalize keys, queries and gates.

    Args:
        hidden: Host activations [batch, tokens, hidden].
        tail: Previous projected convolution context.
        qkv_weight: Combined input projection matrix.
        conv_weight: Depthwise convolution filters.
        conv_bias: Depthwise convolution biases.
        gate_weight: Learned forget, momentum and learning-rate projection.
        gate_bias: Bias for the three update gates.
        max_learning_rate: Upper bound for the inner update rate.

    Returns:
        Query, key, value, three gates, and the independent new convolution tail.
        Update operands use float32, or float64 when hidden uses float64.
    """
    projected = F.linear(hidden, qkv_weight)
    context = torch.cat((tail, projected.transpose(1, 2)), -1)
    mixed = F.silu(F.conv1d(context, conv_weight, conv_bias,
                           groups=conv_weight.shape[0]).transpose(1, 2))
    query, key, value = mixed.chunk(3, -1)
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    query = F.normalize(query.to(dtype), dim=-1, eps=1e-6)
    key = F.normalize(key.to(dtype), dim=-1, eps=1e-6)
    value = value.to(dtype)
    gates = F.linear(hidden, gate_weight, gate_bias).to(dtype).sigmoid()
    gates = gates * gates.new_tensor([1.0, 1.0, max_learning_rate])
    tail_length = conv_weight.shape[-1] - 1
    new_tail = context[..., -tail_length:].clone() if tail_length else context[..., :0].clone()
    return query, key, value, gates, new_tail



def combine_read(
    reads: Tensor, hidden: Tensor, norm_weight: Tensor, norm_bias: Tensor,
    output_weight: Tensor, gate_weight: Tensor, gate_bias: Tensor, norm_eps: float,
) -> Tensor:
    """Normalize and project a memory read, then apply the learned residual gate.

    Args:
        reads: Retrieved memory values [batch, tokens, memory_dimension].
        hidden: Original host activations, which determine the mix gate.
        norm_weight: Read normalization scales.
        norm_bias: Read normalization biases.
        output_weight: Projection into the host residual width.
        gate_weight: Mix-gate projection matrix.
        gate_bias: Mix-gate biases.
        norm_eps: Layer-normalization stabilizer.

    Returns:
        Gated memory contribution in the host's dtype and shape.
    """
    read = F.layer_norm(reads.to(hidden.dtype), (reads.shape[-1],), norm_weight, norm_bias, norm_eps)
    return F.linear(read, output_weight) * F.linear(hidden, gate_weight, gate_bias).sigmoid()



class TitansMAGConnection(ResidualConnection):
    """MAG topology: normalized parallel branches combined by a nonlinear gate.

    Implements the topology of Titans section 4.2, equations 26-28. RMS
    normalization and SiLU-product gating are explicit implementation choices;
    the paper permits nonlinear gating and does not provide complete code here.
    The host uses the fixed decoder, residual and FFN execution path.
    """

    def __init__(self, hidden_size: int) -> None:
        """Create trainable vector scales for the two branches.

        Args:
            hidden_size: Width of the backbone residual stream.
        """
        super().__init__()
        self.attention_scale = nn.Parameter(torch.ones(hidden_size))
        self.memory_scale = nn.Parameter(torch.ones(hidden_size))

    def combine(self, context: ConnectionContext, attention: Tensor,
                plan: ConnectionPlan) -> Tensor:
        """Gate normalized attention with normalized memory before adding the residual.

        Args:
            context: Original backbone residual.
            attention: Sliding-window attention output.
            plan: Read from the parallel memory branch.

        Returns:
            Residual plus the elementwise gated branch product.
        """
        shape = (attention.shape[-1],)
        short = F.rms_norm(attention, shape, self.attention_scale, eps=1e-6)
        long = F.rms_norm(plan.memory_output, shape, self.memory_scale, eps=1e-6)
        return context.hidden + short * F.silu(long)


def titans_mag_connection(layer_index: int, hidden_size: int) -> BackboneConnection:
    """Construct an independent MAG connection at a selected layer.

    Args:
        layer_index: Backbone layer index; available for layer-dependent policies.
        hidden_size: Residual stream width.

    Returns:
        A new trainable MAG connection.
    """
    return TitansMAGConnection(hidden_size)


class TitansMACConnection(ResidualConnection):
    """Memory as context, with one causal retrieval per arriving real token."""

    max_context_tokens = 1

    def execution(self, position: int, tokens: int, *, prefix_tokens: int,
                  segment_length: int, attention_window: int) -> ConnectionExecution:
        """Schedule causal retrieval/token pairs inside one MAC segment.

        Args:
            position: Absolute position including persistent-prefix tokens.
            tokens: Available real tokens.
            prefix_tokens: Persistent attention-prefix length.
            segment_length: Number of real tokens sharing the retrieval snapshot.
            attention_window: Ordinary decoder window, independent of MAC segments.

        Returns:
            Paired rotary positions, full segment attention, a shared read cursor,
            and prefix-only cache retention at the segment's end.
        """
        offset = (position - prefix_tokens) % segment_length
        size = min(tokens, segment_length - offset)
        return ConnectionExecution(
            block_size=size, extra_tokens=size, position_start=prefix_tokens + 2 * offset,
            attention_window=2 * segment_length, token_start=1, token_stride=2,
            reset_cache=offset + size == segment_length, snapshot_memory=True,
            retain_training_cache=True,
        )

    def prepare_block(self, context: ConnectionContext, memory: MemoryModule,
                      state: StateProtocol, *, retrieval: StateProtocol | None,
                      write: bool) -> ConnectionPlan:
        """Retrieve against the segment-start weights and advancing causal history.

        Args:
            context: Real-token activations for this part of the segment.
            memory: Built-in or supplied memory implementation.
            state: Live memory updated by finish.
            retrieval: Differentiable segment-start snapshot owned by the runtime.
            write: Whether finish may write live memory.

        Returns:
            Interleaved causal retrievals and real tokens.

        Raises:
            ValueError: The runtime omitted the requested retrieval cursor.
        """
        if retrieval is None:
            raise ValueError("MAC requires its segment retrieval snapshot")
        return self._prepare_segment(context, memory, retrieval)

    def prepare(self, context: ConnectionContext, memory: MemoryModule,
                state: StateProtocol, *, write: bool) -> ConnectionPlan:
        """Retrieve on a disposable branch; the live memory is written after attention.

        Args:
            context: One normalized real token and its original residual.
            memory: Memory architecture providing the retrieval/update operation.
            state: Live memory before this segment.
            write: Unused here; finish controls the attention-driven write.

        Returns:
            Retrieved context followed by the real token as the attention input.
        """
        return self._prepare_segment(context, memory, state.copy())

    def _prepare_segment(self, context: ConnectionContext, memory: MemoryModule,
                         cursor: StateProtocol) -> ConnectionPlan:
        """Interleave old-memory retrievals and real tokens in arrival order.

        Args:
            context: Decoder activations for this part of the segment.
            memory: Built-in or protocol-supplied memory module.
            cursor: Segment-start snapshot with its own advancing read history.

        Returns:
            Interleaved attention input and one retrieval per real token.
        """
        retrieved = memory(context.normalized, cursor, write=False)
        pairs = torch.stack((retrieved, context.normalized), dim=2).flatten(1, 2)
        return ConnectionPlan(pairs, retrieved)

    def finish(self, context: ConnectionContext, attention: Tensor, plan: ConnectionPlan,
               memory: MemoryModule, state: StateProtocol, *, write: bool) -> Tensor:
        """Write from attention, then gate it with a read from the resulting memory.

        Args:
            context: Original real-token residual.
            attention: Real-token output after attending to persistent/retrieved context.
            plan: Read-before-write retrieval produced by prepare.
            memory: Selected neural memory.
            state: Live state receiving the attention-driven update.
            write: False for isolated read-only answers and released-feedback loss branches.

        Returns:
            Residual plus attention multiplied by a SiLU memory gate.
        """
        if type(memory) is NeuralMemory:
            retrieved = memory._read_and_update(attention, state, write=write, post_write=True)
        else:
            reads = []
            for token in attention.split(1, dim=1):
                if write:
                    memory(token, state, write=True)
                reads.append(memory(token, state.copy(), write=False))
            retrieved = torch.cat(reads, dim=1)
        return self.combine(context, attention, ConnectionPlan(plan.attention_input, retrieved))

    def combine(self, context: ConnectionContext, attention: Tensor,
                plan: ConnectionPlan) -> Tensor:
        """Gate attention with the post-write retrieval; subclasses may replace fusion.

        Args:
            context: Original real-token residual.
            attention: Core attention output.
            plan: Post-write memory read provided by finish.

        Returns:
            The complete residual stream before the decoder FFN.
        """
        return context.hidden + attention * F.silu(plan.memory_output)


def titans_mac_connection(layer_index: int, hidden_size: int) -> BackboneConnection:
    """Build the default memory-as-context connection.

    Args:
        layer_index: Selected backbone placement.
        hidden_size: Residual width, supplied for custom connection factories.

    Returns:
        Independent causal MAC connection with one retrieval per arriving token.
    """
    return TitansMACConnection()


def titans(config: TitansConfig = TitansConfig()) -> MemoryFactory:
    """Instantiate the Titans method as a factory accepted by ContinualModel.

    Args:
        config: Titans dimensions, chunked inner updates and integration settings.

    Returns:
        A constructor creating an independent NeuralMemory at each selected layer.
    """
    return partial(NeuralMemory, config=config)


@dataclass(frozen=True)
class TitansArchitecture:
    """Default deep MLP memory architecture, with Titans projections and recurrence."""

    config: TitansConfig = TitansConfig()

    def __call__(self, hidden_size: int) -> NeuralMemory:
        """Build independent neural memory with configurable width, depth and chunk size.

        Args:
            hidden_size: Backbone residual width supplied by the runtime.

        Returns:
            A fresh registered memory module with the Titans transition.
        """
        return NeuralMemory(hidden_size, self.config)


class TitansModel(ContinualModel):
    """Convenience constructor for the Titans method on the fixed runtime."""


    _config_type = TitansConfig

    def __init__(
        self, backbone: "LlamaForCausalLM | Qwen2ForCausalLM",
        tokenizer: "PreTrainedTokenizerBase", config: TitansConfig = TitansConfig(),
        *, memory_factory: MemoryFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Attach Titans or an explicitly supplied compatible method factory.

        Args:
            backbone: Single-device model with one of the fixed decoder configurations.
            tokenizer: Tokenizer matching the backbone vocabulary.
            config: Runtime and Titans algorithm settings.
            memory_architecture: Optional replacement for the complete memory network.
            connection_factory: Optional connection; defaults to causal MAC.
            memory_factory: Optional constructor used by generic checkpoint restoration.
        """
        if memory_factory is not None and memory_architecture is not None:
            raise ValueError("Supply one memory architecture factory")
        super().__init__(backbone, tokenizer, config,
                         memory_architecture=memory_architecture or memory_factory or TitansArchitecture(config),
                         connection_factory=connection_factory or titans_mac_connection)

    @classmethod
    def from_scratch(
        cls, config: TitansConfig = TitansConfig(), *,
        memory_factory: MemoryFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
        connection_factory: ConnectionFactory | None = None,
        device: str = "cpu", dtype: torch.dtype = torch.float32,
        seed: int = 0, local_files_only: bool = False,
    ) -> Self:
        """Build the fixed 24-block, 1024-wide, 16-head scratch model with Titans.

        Args:
            config: Memory and runtime settings; the decoder architecture is fixed.
            memory_factory: Legacy memory constructor alias.
            memory_architecture: Optional replacement memory architecture.
            connection_factory: Optional replacement wiring; defaults to MAC.
            device: Single target device.
            dtype: Parameter storage dtype.
            seed: Isolated initialization seed.
            local_files_only: Require the tokenizer to be cached locally.

        Returns:
            Randomly initialized model with joint outer training as its fit default.
        """
        return super().from_scratch(
            config, memory_factory=memory_factory, memory_architecture=memory_architecture,
            connection_factory=connection_factory, device=device, dtype=dtype,
            seed=seed, local_files_only=local_files_only,
        )

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | Path = PRETRAINED_BACKBONE, config: TitansConfig = TitansConfig(),
        *, memory_factory: MemoryFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
        connection_factory: ConnectionFactory | None = None, device: str = "cpu",
        dtype: torch.dtype = torch.float32, local_files_only: bool = False,
    ) -> Self:
        """Load a pretrained backbone and attach untrained Titans modules.

        Args:
            model_name_or_path: Must equal the fixed Qwen model ID.
            config: Runtime and method settings.
            memory_architecture: Optional replacement for the complete memory network.
            connection_factory: Optional connection; defaults to causal MAC.
            memory_factory: Optional explicit constructor for checkpoint restoration.
            device: Single target device.
            dtype: Backbone and adapter parameter dtype.
            local_files_only: Whether downloads are forbidden.

        Returns:
            A frozen-backbone model with fresh stream ownership.
        """
        return super().from_pretrained(
            model_name_or_path, config, memory_factory=memory_factory,
            memory_architecture=memory_architecture, connection_factory=connection_factory,
            device=device, dtype=dtype, local_files_only=local_files_only,
        )

    @classmethod
    def load_pretrained(
        cls, path: str | Path, *, device: str = "cpu", dtype: torch.dtype = torch.float32,
        connection_factory: ConnectionFactory | None = None,
        memory_architecture: MemoryArchitecture | None = None,
    ) -> Self:
        """Restore a Titans checkpoint, including the previous version's format.

        Args:
            connection_factory: Explicit factory required for custom saved connections.
            memory_architecture: Explicit factory required for custom saved memory networks.
            path: Directory written by save_pretrained.
            device: Single target device.
            dtype: Backbone and adapter parameter dtype.

        Returns:
            Trained Titans parameters with fresh stream ownership.
        """
        return super().load_pretrained(path, memory_architecture=memory_architecture,
                                       connection_factory=connection_factory,
                                       device=device, dtype=dtype)
