"""Differentiable Titans updates, using weighted MLP gradient reductions.

See Behrouz et al., arXiv:2501.00663v1, equations 12--14 and section 3.2.
There is no autograd invocation or per-token parameter-gradient allocation in
the production update. Ordinary PyTorch operations retain outer gradients.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .types import TitansConfig


def mlp(weights: tuple[Tensor, ...], inputs: Tensor) -> Tensor:
    """Evaluate a bias-free batched MLP with SiLU hidden activations.

    Args:
        weights: Matrices shaped [batch, output, input], one per layer.
        inputs: Keys or queries shaped [batch, tokens, dimension].

    Returns:
        Predicted values with the same leading dimensions as inputs.
    """
    value = inputs
    for index, weight in enumerate(weights):
        value = value @ weight.transpose(-1, -2)
        if index + 1 < len(weights):
            value = F.silu(value)
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
    old_momentum_coeff = (eta[:, 0] * h[:, 0])[:, None, None]
    final_momentum_coeff = eta.prod(-1)[:, None, None]

    activations = [keys]
    preactivations: list[Tensor] = []
    for index, weight in enumerate(weights):
        z = activations[-1] @ weight.transpose(-1, -2)
        preactivations.append(z)
        activations.append(F.silu(z) if index + 1 < len(weights) else z)
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
            sigmoid = preactivations[index - 1].sigmoid()
            derivative = sigmoid * (1 + preactivations[index - 1] * (1 - sigmoid))
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


class NeuralMemory(nn.Module):
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

    def forward(self, hidden: Tensor, state: MemoryState, *, write: bool = True) -> Tensor:
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
        from .kernels import (
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
            read_queries = query[:, start:end].clone(memory_format=torch.contiguous_format)
            outputs.append(retrieve(state.weights, read_queries))
            if write:
                updates = torch.cat((key[:, start:end], value[:, start:end], gates[:, start:end]), -1)
                state.pending = torch.cat((state.pending, updates), 1)
                if state.pending.shape[1] == self.config.chunk_size:
                    width = self.config.memory_dim
                    k, v, g = state.pending.split((width, width, 3), -1)
                    state.weights, state.momentum = update(
                        state.weights, state.momentum, k, v, g[..., 0], g[..., 1], g[..., 2]
                    )
                    state.pending = state.pending[:, :0]
            start = end
        return combine(torch.cat(outputs, 1), hidden, self.read_norm.weight, self.read_norm.bias,
                       self.output.weight, self.mix_gate.weight, self.mix_gate.bias,
                       self.read_norm.eps)
