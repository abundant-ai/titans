"""JIT-compiled tensor kernels; Python stream ownership stays outside the graphs.

Inductor fuses pointwise work and generates device-specific code. Full-graph
compilation makes a graph break an error. CUDA graphs are deliberately disabled:
their output-buffer reuse needs separate validation for persistent stream state.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from .titans import chunk_update, combine_read, mlp, project_tokens


@torch.compile(fullgraph=True, dynamic=False)
def compiled_chunk_update(
    weights: tuple[Tensor, ...], momentum: tuple[Tensor, ...], keys: Tensor, values: Tensor,
    alpha: Tensor, eta: Tensor, theta: Tensor,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Compile the complete fixed-size write chunk, including gate scan and MLP gradients.

    Args:
        weights: Fast-weight matrices at the start of the chunk.
        momentum: Matching initial momentum matrices.
        keys: Projected keys [batch, chunk, memory_dimension].
        values: Target values matching the keys.
        alpha: Forget gates [batch, chunk].
        eta: Momentum gates [batch, chunk].
        theta: Inner learning rates [batch, chunk].

    Returns:
        Updated weights and momentum with the eager kernel's exact semantics.
    """
    return chunk_update(weights, momentum, keys, values, alpha, eta, theta)


@torch.compile(fullgraph=True, dynamic=True)
def compiled_read(weights: tuple[Tensor, ...], queries: Tensor) -> Tensor:
    """Compile deep-memory retrieval, supporting varying query-block lengths.

    Args:
        weights: Committed fast weights.
        queries: Normalized query vectors [batch, tokens, dimension].

    Returns:
        Retrieved value vectors; no state is mutated.
    """
    return mlp(weights, queries)




# SGLang 0.5.7 deterministic addmm omits alpha/beta keyword support.
# Inductor padding benchmarks pass those keywords; retain JIT without padding.
@torch.compile(fullgraph=True, dynamic=True, options={"shape_padding": False})
def compiled_project_tokens(
    hidden: Tensor, tail: Tensor, qkv_weight: Tensor, conv_weight: Tensor, conv_bias: Tensor,
    gate_weight: Tensor, gate_bias: Tensor, max_learning_rate: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """JIT the combined projection, convolution, normalization and gate path.

    Args:
        hidden: Host activations for this block.
        tail: Previous causal convolution tail.
        qkv_weight: Combined input projection matrix.
        conv_weight: Depthwise convolution filters.
        conv_bias: Depthwise convolution biases.
        gate_weight: Update-gate projection matrix.
        gate_bias: Update-gate bias.
        max_learning_rate: Inner learning-rate bound.

    Returns:
        Query, key, value, update gates, and the new convolution tail.
    """
    return project_tokens(hidden, tail, qkv_weight, conv_weight, conv_bias,
                          gate_weight, gate_bias, max_learning_rate)




@torch.compile(fullgraph=True, dynamic=True, options={"shape_padding": False})
def compiled_combine_read(
    reads: Tensor, hidden: Tensor, norm_weight: Tensor, norm_bias: Tensor,
    output_weight: Tensor, gate_weight: Tensor, gate_bias: Tensor, norm_eps: float,
) -> Tensor:
    """JIT the memory-output normalization, projection and residual gate.

    Args:
        reads: Retrieved memory values.
        hidden: Host activations used by the mix gate.
        norm_weight: Read normalization scales.
        norm_bias: Read normalization biases.
        output_weight: Projection into the host width.
        gate_weight: Mix-gate matrix.
        gate_bias: Mix-gate biases.
        norm_eps: Layer-normalization stabilizer.

    Returns:
        Gated memory output in the host dtype.
    """
    return combine_read(reads, hidden, norm_weight, norm_bias, output_weight,
                        gate_weight, gate_bias, norm_eps)


@torch.compile(fullgraph=True, dynamic=True)
def compiled_llama_mlp(
    hidden: Tensor, gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor,
    gate_bias: Tensor | None, up_bias: Tensor | None, down_bias: Tensor | None,
) -> Tensor:
    """Compile Llama's SiLU feed-forward block without wrapping or renaming its parameters.

    Args:
        hidden: Normalized residual activations.
        gate_weight: Gating projection matrix.
        up_weight: Intermediate projection matrix.
        down_weight: Output projection matrix.
        gate_bias: Optional gating bias.
        up_bias: Optional intermediate bias.
        down_bias: Optional output bias.

    Returns:
        The original Llama MLP output; autograd remains available for post-training.
    """
    activated = F.silu(F.linear(hidden, gate_weight, gate_bias))
    return F.linear(activated * F.linear(hidden, up_weight, up_bias), down_weight, down_bias)


@torch.compile(fullgraph=True, dynamic=True)
def compiled_rms_norm(hidden: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """Fuse Llama's float32 RMS normalization and learned scale.

    Args:
        hidden: Residual activations in the host dtype.
        weight: Learned per-feature normalization scale.
        epsilon: Host model's normalization stabilizer.

    Returns:
        Normalized activations with the same computation as LlamaRMSNorm.
    """
    value = hidden.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    return weight * value.to(hidden.dtype)


@torch.compile(fullgraph=True, dynamic=True)
def compiled_attention(
    hidden: Tensor, q_weight: Tensor, k_weight: Tensor, v_weight: Tensor, o_weight: Tensor,
    q_bias: Tensor | None, k_bias: Tensor | None, v_bias: Tensor | None, o_bias: Tensor | None,
    cosine: Tensor, sine: Tensor, old_keys: Tensor, old_values: Tensor, allowed: Tensor,
    head_dim: int, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """JIT QKV projection, rotary encoding, bounded-cache concatenation, SDPA and output.

    Args:
        hidden: Normalized host activations [batch, tokens, hidden].
        q_weight: Query projection matrix.
        k_weight: Grouped-key projection matrix.
        v_weight: Grouped-value projection matrix.
        o_weight: Output projection matrix.
        q_bias: Optional query bias.
        k_bias: Optional key bias.
        v_bias: Optional value bias.
        o_bias: Optional output bias.
        cosine: Host RoPE cosine values [batch, tokens, head_dim].
        sine: Host RoPE sine values matching cosine.
        old_keys: Previous grouped keys, possibly an empty cache.
        old_values: Previous grouped values matching old_keys.
        allowed: Boolean causal/local mask [query_tokens, key_tokens].
        head_dim: Width of each attention head.
        scale: Host attention's inverse-square-root scaling.

    Returns:
        Attention output and concatenated grouped KV. Cache retention happens
        outside this pure kernel, using fixed-length slices without GPU sync.
    """
    shape = (*hidden.shape[:-1], -1, head_dim)
    query = F.linear(hidden, q_weight, q_bias).view(shape).transpose(1, 2)
    key = F.linear(hidden, k_weight, k_bias).view(shape).transpose(1, 2)
    value = F.linear(hidden, v_weight, v_bias).view(shape).transpose(1, 2)
    query_rotation = torch.cat((-query[..., head_dim // 2:], query[..., :head_dim // 2]), -1)
    key_rotation = torch.cat((-key[..., head_dim // 2:], key[..., :head_dim // 2]), -1)
    query = query * cosine[:, None] + query_rotation * sine[:, None]
    key = key * cosine[:, None] + key_rotation * sine[:, None]
    key = torch.cat((old_keys, key), -2)
    value = torch.cat((old_values, value), -2)
    groups = query.shape[1] // key.shape[1]
    attended = F.scaled_dot_product_attention(
        query, key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1),
        attn_mask=allowed[None, None], dropout_p=0.0, scale=scale,
    )
    attended = attended.transpose(1, 2).reshape(*hidden.shape[:-1], -1)
    return F.linear(attended, o_weight, o_bias), key, value
