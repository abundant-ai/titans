"""Complete stream snapshots and isolation assertions."""

import torch
from torch import Tensor

from titans import StreamState


def snapshot(state: StreamState) -> dict[str, Tensor]:
    """Copy all mutable tensors for source-preservation and isolation assertions.

    Args:
        state: A live stream returned by the model.

    Returns:
        Named tensor copies, including every pending update and cache.
    """
    tensors: dict[str, Tensor] = {}
    for index, memory in state._memory.items():
        for depth, weight in enumerate(memory.weights):
            tensors[f"{index}.w.{depth}"] = weight.detach().clone()
        for depth, momentum in enumerate(memory.momentum):
            tensors[f"{index}.s.{depth}"] = momentum.detach().clone()
        tensors[f"{index}.pending"] = memory.pending.detach().clone()
        tensors[f"{index}.conv"] = memory.conv_tail.detach().clone()
    for index, cache in state._attention.items():
        tensors[f"{index}.k"] = cache.keys.detach().clone()
        tensors[f"{index}.v"] = cache.values.detach().clone()
        tensors[f"{index}.positions"] = cache.positions.clone()
    if state._next_logits is not None:
        tensors["logits"] = state._next_logits.detach().clone()
    if state._next_hidden is not None:
        tensors["hidden"] = state._next_hidden.detach().clone()
    tensors["metadata"] = torch.tensor([state._position, state._tokens, state._documents,
                                        state._revision, state._valid])
    return tensors

def assert_same(actual: dict[str, Tensor], expected: dict[str, Tensor], *, exact: bool = True) -> None:
    """Compare a complete stream snapshot, including its structure.

    Args:
        actual: Observed tensor mapping.
        expected: Reference tensor mapping.
        exact: Whether to require bitwise equality instead of normal tolerances.
    """
    assert actual.keys() == expected.keys()
    for name in actual:
        if exact:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual[name], expected[name])
