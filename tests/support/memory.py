"""Deterministic inputs for memory-update parity checks."""

import torch
from torch import Tensor


def case(
    depth: int, tokens: int, *, extreme_gates: bool = False,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Construct a reproducible double-precision case with nonzero momentum.

    Args:
        depth: Number of linear layers in the memory MLP.
        tokens: Number of tokens in its write chunk.
        extreme_gates: Whether to include exact zero and one gate values.

    Returns:
        Inputs accepted by both the optimized update and independent oracle.
    """
    generator = torch.Generator().manual_seed(42 + depth + tokens)
    dimensions = [5] + [7] * (depth - 1) + [5]
    weights = tuple(
        (torch.randn(2, out_size, in_size, generator=generator, dtype=torch.float64) * 0.2)
        .requires_grad_()
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:], strict=True)
    )
    momentum = tuple(torch.randn(w.shape, generator=generator, dtype=w.dtype).requires_grad_()
                     for w in weights)
    keys = torch.randn(2, tokens, 5, generator=generator, dtype=torch.float64).requires_grad_()
    values = torch.randn(2, tokens, 5, generator=generator, dtype=torch.float64).requires_grad_()
    alpha = torch.rand(2, tokens, generator=generator, dtype=torch.float64)
    eta = torch.rand(2, tokens, generator=generator, dtype=torch.float64)
    if extreme_gates:
        alpha = alpha.round()
        eta = eta.round()
    theta = torch.rand(2, tokens, generator=generator, dtype=torch.float64) * 0.01
    return weights, momentum, keys, values, alpha.requires_grad_(), eta.requires_grad_(), theta.requires_grad_()
