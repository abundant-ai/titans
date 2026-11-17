"""A second method demonstrating the protocol with one linear associative matrix.

This is an interface example, not a claimed improvement over Titans. Its state
has no Titans momentum, pending chunks or convolution buffers.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from titans.core.protocols import MemoryModule, MemoryState


@dataclass
class LinearState:
    """One adaptable matrix per stream, with algorithm-local parameter names."""

    matrix: Tensor

    def copy(self, *, detach: bool = False) -> "LinearState":
        """Clone storage, optionally starting a fresh gradient history.

        Args:
            detach: Whether to drop the previous computation graph.

        Returns:
            Independent matrix storage with the requested differentiation policy.
        """
        return LinearState(self.matrix.detach().clone() if detach else self.matrix.clone())

    def detach(self) -> None:
        """Keep learned values while discarding the graph of prior writes."""
        self.matrix = self.matrix.detach()

    def tensors(self) -> tuple[Tensor, ...]:
        """Expose all state storage for placement checks and accounting."""
        return (self.matrix,)

    def adaptable(self) -> Mapping[str, Tensor]:
        """Expose the associative matrix as the method's only W target."""
        return {"matrix": self.matrix}

    def replace(self, updates: Mapping[str, Tensor]) -> None:
        """Replace W after an external learning step.

        Args:
            updates: Empty mapping or a shape-compatible matrix replacement.

        Raises:
            ValueError: An undeclared target or incompatible shape was supplied.
        """
        if not set(updates) <= {"matrix"}:
            raise ValueError("Unknown linear memory parameter")
        if "matrix" in updates:
            if updates["matrix"].shape != self.matrix.shape:
                raise ValueError("Linear memory shape changed")
            self.matrix = updates["matrix"]


class LinearMemory(MemoryModule):
    """Causal tokenwise delta updates with learned projections and write strength."""

    def __init__(self, hidden_size: int, width: int = 8) -> None:
        """Allocate learned initialization, projections and write gate.

        Args:
            hidden_size: Backbone residual width supplied by the fixed runtime.
            width: Size of the associative matrix's key/value dimensions.
        """
        super().__init__()
        self.projection = nn.Linear(hidden_size, 3 * width, bias=False)
        self.output = nn.Linear(width, hidden_size, bias=False)
        self.initial = nn.Parameter(torch.zeros(width, width))
        self.rate = nn.Parameter(torch.tensor(-2.0))

    def initial_state(self, batch_size: int = 1) -> LinearState:
        """Create independent active matrices from the learned initialization.

        Args:
            batch_size: Number of independent streams in this memory invocation.

        Returns:
            A differentiable copy of the shared initial matrix.
        """
        return LinearState(self.initial.unsqueeze(0).expand(batch_size, -1, -1).clone())

    def transition(self, hidden: Tensor, state: MemoryState, *, write: bool = True) -> Tensor:
        """Read before writing each token, retaining gradients through delta updates.

        Args:
            hidden: Causal backbone activations [batch, tokens, hidden_size].
            state: A LinearState owned by this stream.
            write: Whether to update the matrix after each read.

        Returns:
            Residual contributions matching the backbone's shape and dtype.

        Raises:
            TypeError: The state belongs to another method.
        """
        if not isinstance(state, LinearState):
            raise TypeError("LinearMemory requires LinearState")
        query, key, value = self.projection(hidden).chunk(3, dim=-1)
        query, key = F.normalize(query, dim=-1), F.normalize(key, dim=-1)
        outputs: list[Tensor] = []
        for index in range(hidden.shape[1]):
            outputs.append(torch.bmm(state.matrix, query[:, index, :, None]).squeeze(-1))
            if write:
                k = key[:, index]
                error = value[:, index] - torch.bmm(state.matrix, k[..., None]).squeeze(-1)
                state.matrix = state.matrix + self.rate.sigmoid() * error[..., None] * k[:, None]
        return self.output(torch.stack(outputs, dim=1))
