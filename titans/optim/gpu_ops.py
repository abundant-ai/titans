"""Optional CUDA kernels, imported only on the explicit accelerator paths."""

from torch import Tensor
from torch.nn import functional as F


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
