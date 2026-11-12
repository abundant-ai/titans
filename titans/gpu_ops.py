"""Optional CUDA kernels, imported only on the explicit accelerator paths."""

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.compile(fullgraph=True, dynamic=True)
def merge_persistent_attention(
    query: Tensor, prefix_keys: Tensor, prefix_values: Tensor,
    local_output: Tensor, local_lse: Tensor, scale: float,
) -> Tensor:
    """Merge global persistent tokens with local FA3 attention using exact partition sums.

    Args:
        query: Queries [batch, heads, query_tokens, head_dim].
        prefix_keys: Persistent grouped keys [batch, kv_heads, prefix_tokens, head_dim].
        prefix_values: Persistent values matching prefix_keys.
        local_output: Local FA3 output [batch, query_tokens, heads, head_dim].
        local_lse: Local attention log-partition [batch, heads, query_tokens].
        scale: Attention score scaling.

    Returns:
        Joint attention output in [batch, heads, query_tokens, head_dim] order.
        Softmax normalization is shared across both disjoint key partitions.
    """
    groups = query.shape[1] // prefix_keys.shape[1]
    scores = (query.float() @ prefix_keys.repeat_interleave(groups, 1).float().transpose(-1, -2))
    scores = scores * scale
    prefix_lse = scores.logsumexp(-1)
    prefix_output = scores.softmax(-1) @ prefix_values.repeat_interleave(groups, 1).float()
    prefix_fraction = (prefix_lse - local_lse).sigmoid()[..., None]
    local = local_output.transpose(1, 2).float()
    return (prefix_output * prefix_fraction + local * (1 - prefix_fraction)).to(query.dtype)


def flash_attention3(
    query: Tensor, keys: Tensor, values: Tensor, *, prefix_count: int,
    window: int, scale: float, prefix_only: bool,
) -> Tensor:
    """Run SGLang's FA3 kernel with bounded attention and globally visible prefix tokens.

    Args:
        query: Rotary-encoded queries [batch, heads, query_tokens, head_dim].
        keys: Concatenated grouped keys, with the persistent prefix first.
        values: Grouped values matching keys.
        prefix_count: Number of persistent prefix entries in keys.
        window: Number of ordinary tokens visible to each query, including itself.
        scale: Host attention scaling.
        prefix_only: Whether this call initializes the persistent prefix itself.

    Returns:
        Attention output in the same layout as query.

    Raises:
        RuntimeError: FA3 is requested without CUDA, BF16/FP16, or Hopper hardware.
    """
    if query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("FA3 requires CUDA BF16/FP16 tensors")
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        raise RuntimeError("This FA3 configuration is validated for Hopper H100/H800 GPUs")
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    q = query.transpose(1, 2).contiguous()
    if prefix_only:
        output = flash_attn_with_kvcache(
            q, keys.transpose(1, 2).contiguous(), values.transpose(1, 2).contiguous(),
            causal=True, softmax_scale=scale, ver=3,
        )
        return output.transpose(1, 2)
    result = flash_attn_with_kvcache(
        q, keys[:, :, prefix_count:].transpose(1, 2).contiguous(),
        values[:, :, prefix_count:].transpose(1, 2).contiguous(),
        causal=True, window_size=(window - 1, 0), softmax_scale=scale,
        return_softmax_lse=True, ver=3,
    )
    output, lse = result[0], result[1]
    if prefix_count:
        return merge_persistent_attention(query, keys[:, :, :prefix_count],
                                          values[:, :, :prefix_count], output, lse, scale)
    return output.transpose(1, 2)


def liger_norm(hidden: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """Use Liger's Llama-compatible RMSNorm without in-place gradient mutation.

    Args:
        hidden: CUDA residual activations.
        weight: Learned normalization scale.
        epsilon: Host normalization stabilizer.

    Returns:
        Normalized activations with a first-order autograd rule.
    """
    from liger_kernel.transformers.functional import liger_rms_norm

    return liger_rms_norm(hidden, weight, epsilon, casting_mode="llama", in_place=False)


def liger_rotary(query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor) -> tuple[Tensor, Tensor]:
    """Apply Liger's fused Llama RoPE kernel to newly projected query/key tensors.

    Args:
        query: Fresh queries [batch, query_heads, tokens, head_dim].
        key: Fresh grouped keys [batch, kv_heads, tokens, head_dim].
        cosine: Host rotary cosines [batch, tokens, head_dim].
        sine: Host rotary sines matching cosine.

    Returns:
        Rotary-encoded queries and keys. Cached source keys are never passed here.
    """
    from liger_kernel.transformers.functional import liger_rope

    return liger_rope(query, key, cosine, sine)


def liger_mlp(
    hidden: Tensor, gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor,
    gate_bias: Tensor | None, up_bias: Tensor | None, down_bias: Tensor | None,
) -> Tensor:
    """Use Liger's fused SwiGLU operation between the host's linear projections.

    Args:
        hidden: Normalized CUDA activations.
        gate_weight: Gate projection matrix.
        up_weight: Intermediate projection matrix.
        down_weight: Output projection matrix.
        gate_bias: Optional gate bias.
        up_bias: Optional intermediate bias.
        down_bias: Optional output bias.

    Returns:
        Feed-forward output, retaining gradients for post-training.
    """
    from liger_kernel.transformers.functional import liger_swiglu

    activated = liger_swiglu(F.linear(hidden, gate_weight, gate_bias),
                            F.linear(hidden, up_weight, up_bias))
    return F.linear(activated, down_weight, down_bias)


def liger_loss(weight: Tensor, hidden: Tensor, targets: Tensor) -> Tensor:
    """Fuse the vocabulary projection and summed cross entropy without dense logits.

    Args:
        weight: LM head matrix [vocabulary, hidden].
        hidden: Supervised prediction activations [tokens, hidden].
        targets: Aligned target IDs [tokens].

    Returns:
        Sum of token losses, ready for token-weighted gradient accumulation.
    """
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    return LigerFusedLinearCrossEntropyLoss(reduction="sum")(weight, hidden, targets)
