"""Numerical checks of Titans recurrence, derivatives and stream chunking."""

import pytest
import torch
from torch import Tensor

from titans.titans import NeuralMemory, TitansConfig, chunk_update, mlp

from .reference import reference_chunk


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


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("tokens", [1, 2, 7, 32])
@pytest.mark.parametrize("extreme_gates", [False, True])
def test_update_matches_equations(depth: int, tokens: int, extreme_gates: bool) -> None:
    """Compare all final weights and momentum against literal autograd equations."""
    inputs = case(depth, tokens, extreme_gates=extreme_gates)
    actual = chunk_update(*inputs)
    expected = reference_chunk(*inputs)
    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_outer_derivatives_match_autograd(depth: int) -> None:
    """Check differentiation through the update, including its inner gradients."""
    inputs = case(depth, 7)
    leaves = inputs[0] + inputs[1] + inputs[2:]
    actual = chunk_update(*inputs)
    expected = reference_chunk(*inputs)
    loss_actual = sum(t.square().sum() for t in actual[0] + actual[1])
    loss_expected = sum(t.square().sum() for t in expected[0] + expected[1])
    gradients_actual = torch.autograd.grad(loss_actual, leaves, retain_graph=True)
    gradients_expected = torch.autograd.grad(loss_expected, leaves)
    for got, want in zip(gradients_actual, gradients_expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)


def test_finite_difference_gradient() -> None:
    """Check outer gradients numerically, independently of the analytic oracle."""
    inputs = case(2, 2)
    weights, momentum, keys, values, alpha, eta, theta = inputs

    def update_loss(a: Tensor, e: Tensor, t: Tensor, k: Tensor) -> Tensor:
        """Return a scalar updated-memory norm for finite-difference checking.

        Args:
            a: Forget gates.
            e: Momentum gates.
            t: Inner learning rates.
            k: Projected keys.

        Returns:
            Sum of squared updated weights.
        """
        updated, _ = chunk_update(weights, momentum, k, values, a, e, t)
        return sum(w.square().sum() for w in updated)

    assert torch.autograd.gradcheck(update_loss, (alpha, eta, theta, keys), fast_mode=True)


@pytest.mark.parametrize("chunk_size", [1, 4, 8])
def test_stream_split_preserves_memory(chunk_size: int) -> None:
    """Arbitrary caller boundaries must preserve reads, momentum and pending writes."""
    torch.manual_seed(3)
    memory = NeuralMemory(12, TitansConfig(memory_dim=5, memory_hidden_dim=7,
                                         chunk_size=chunk_size, compile_kernels=False)).double()
    hidden = torch.randn(2, 19, 12, dtype=torch.float64)
    whole = memory.initial_state(2)
    split = memory.initial_state(2)
    expected = memory(hidden, whole)
    actual = torch.cat([memory(hidden[:, :3], split), memory(hidden[:, 3:10], split),
                        memory(hidden[:, 10:11], split), memory(hidden[:, 11:], split)], 1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for got, want in zip(split.weights + split.momentum + (split.pending, split.conv_tail),
                         whole.weights + whole.momentum + (whole.pending, whole.conv_tail),
                         strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_future_tokens_cannot_change_earlier_reads() -> None:
    """Perturb the future within one update chunk and check earlier outputs."""
    torch.manual_seed(4)
    memory = NeuralMemory(8, TitansConfig(memory_dim=4, memory_hidden_dim=6, chunk_size=8,
                                         compile_kernels=False))
    hidden = torch.randn(1, 17, 8)
    changed = hidden.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 10
    first = memory(hidden, memory.initial_state())
    second = memory(changed, memory.initial_state())
    torch.testing.assert_close(first[:, :5], second[:, :5])


def test_read_only_leaves_writes_unchanged() -> None:
    """Read-only queries cannot modify weights, momentum or an incomplete chunk."""
    memory = NeuralMemory(8, TitansConfig(memory_dim=4, memory_hidden_dim=6, chunk_size=4,
                                         compile_kernels=False))
    state = memory.initial_state()
    memory(torch.randn(1, 3, 8), state)
    before = state.copy()
    memory(torch.randn(1, 9, 8), state, write=False)
    for got, want in zip(state.weights + state.momentum + (state.pending,),
                         before.weights + before.momentum + (before.pending,), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_overwrite_one_association() -> None:
    """A known linear-memory example must learn a value and then overwrite it."""
    key = torch.tensor([[[1.0, 0.0]]])
    weights = (torch.zeros(1, 2, 2),)
    momentum = (torch.zeros_like(weights[0]),)
    zero = torch.zeros(1, 1)
    rate = torch.full((1, 1), 0.5)
    first = torch.tensor([[[3.0, -1.0]]])
    second = torch.tensor([[[-2.0, 4.0]]])
    weights, momentum = chunk_update(weights, momentum, key, first, zero, zero, rate)
    torch.testing.assert_close(mlp(weights, key), first)
    weights, _ = chunk_update(weights, momentum, key, second, zero, zero, rate)
    torch.testing.assert_close(mlp(weights, key), second)
