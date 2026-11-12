"""Slow autograd oracle; deliberately does not call the production memory code."""

import torch
from torch import Tensor
from torch.nn import functional as F


def reference_chunk(
    weights: tuple[Tensor, ...],
    momentum: tuple[Tensor, ...],
    keys: Tensor,
    values: Tensor,
    alpha: Tensor,
    eta: Tensor,
    theta: Tensor,
    *,
    create_graph: bool = True,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Execute equations 12--14 literally, at the section 3.2 chunk evaluation point.

    Args:
        weights: Initial MLP matrices with requires_grad enabled.
        momentum: Initial momentum tensors matching the matrices.
        keys: Batched token keys [batch, tokens, dimension].
        values: Target values with the same shape as keys.
        alpha: Forget coefficients [batch, tokens].
        eta: Momentum coefficients [batch, tokens].
        theta: Learning rates [batch, tokens].
        create_graph: Whether the reference must retain outer training gradients.

    Returns:
        Final weights and momentum, retaining the outer autograd graph.
    """
    current_weights, current_momentum = weights, momentum
    for token in range(keys.shape[1]):
        with torch.enable_grad():
            predicted = keys[:, token:token + 1]
            for index, weight in enumerate(weights):
                predicted = torch.bmm(predicted, weight.transpose(1, 2))
                if index + 1 < len(weights):
                    predicted = F.silu(predicted)
            loss = (predicted - values[:, token:token + 1]).square().sum()
            gradient = torch.autograd.grad(loss, weights, create_graph=create_graph)
        with torch.set_grad_enabled(create_graph):
            current_momentum = tuple(
                eta[:, token, None, None] * old - theta[:, token, None, None] * grad
                for old, grad in zip(current_momentum, gradient, strict=True)
            )
            current_weights = tuple(
                (1 - alpha[:, token, None, None]) * old + update
                for old, update in zip(current_weights, current_momentum, strict=True)
            )
    return current_weights, current_momentum
