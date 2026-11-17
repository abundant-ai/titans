"""Replaceable learning policies; document ordering belongs to the runtime."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, Self

from torch import Tensor, nn

if TYPE_CHECKING:
    from .state import StreamState
    from .types import TrainingEpisode


class MemoryState(Protocol):
    """Algorithm-owned tensors, including every buffer needed to continue a stream."""

    def copy(self, *, detach: bool = False) -> Self:
        """Return independent storage, preserving the graph unless detach is true."""
        ...

    def detach(self) -> None:
        """Sever all history while preserving state values at a truncation boundary."""
        ...

    def tensors(self) -> tuple[Tensor, ...]:
        """Return all state tensors for device checks and resource accounting."""
        ...

    def adaptable(self) -> Mapping[str, Tensor]:
        """Return named floating-point memory tensors eligible for explicit learning."""
        ...

    def replace(self, updates: Mapping[str, Tensor]) -> None:
        """Replace named adaptable tensors without modifying other state buffers.

        Args:
            updates: A subset of the names returned by adaptable, with matching shapes.
        """
        ...


class MemoryModule(nn.Module, ABC):
    """A registered parameter set and a replaceable causal transition/read operation."""

    @abstractmethod
    def initial_state(self, batch_size: int = 1) -> MemoryState:
        """Create independent state from the learned initialization.

        Args:
            batch_size: Number of independent streams in the numerical memory core.

        Returns:
            State whose initial values may depend differentiably on shared parameters.
        """
        ...

    @abstractmethod
    def transition(self, hidden: Tensor, state: MemoryState, *, write: bool = True) -> Tensor:
        """Read memory and optionally update it, processing the block causally.

        Args:
            hidden: Backbone representations [batch, tokens, hidden_size].
            state: Owned algorithm state; updates must not mutate shared parameters.
            write: Whether to update learned memory. Read-side causal buffers may advance.

        Returns:
            A memory contribution with the input shape, device and dtype. Splitting a
            block across calls must preserve its numerical meaning. For meta-training,
            retain differentiation through transitions when gradients are enabled.
        """
        ...

    def forward(self, hidden: Tensor, state: MemoryState, *, write: bool = True) -> Tensor:
        """Dispatch through the transition while retaining PyTorch module hooks.

        Args:
            hidden: Backbone representations for the next causal block.
            state: State owned by the current stream.
            write: Whether memory writes are permitted.

        Returns:
            The method's residual contribution.
        """
        return self.transition(hidden, state, write=write)


class MemoryFactory(Protocol):
    """Construct a fresh registered module at each selected backbone layer."""

    def __call__(self, hidden_size: int) -> MemoryModule:
        """Build a module; the runtime subsequently places its parameters.

        Args:
            hidden_size: Backbone residual width.

        Returns:
            A new module instance, never shared implicitly between layers.
        """
        ...


@dataclass(frozen=True)
class LossContext:
    """Aligned predictions plus explicit shared and stream state for custom losses.

    Predictions are [targets, vocabulary]; targets are [targets]. During explicit
    QA learning, state is the pre-question stream, and the teacher-forced branch
    is disposable. A loss may use predictions, shared parameters, memory, or all.
    """

    predictions: Tensor
    targets: Tensor
    state: "StreamState"
    shared_parameters: tuple[nn.Parameter, ...]
    phase: Literal["document", "answer"]


@dataclass(frozen=True)
class LossValue:
    """Scalar differentiable loss sum and its positive normalization weight."""

    value: Tensor
    weight: float


class Objective(Protocol):
    """Compute an arbitrary scalar objective independently of its update targets."""

    def __call__(self, context: LossContext) -> LossValue:
        """Evaluate supervision or regularization using only the supplied context.

        Args:
            context: Predictions, labels and differentiable state at this checkpoint.

        Returns:
            Scalar loss and the normalization weight used across accumulated batches.
        """
        ...


class DataGenerator(Protocol):
    """Generate training episodes lazily without access to hidden evaluation data."""

    def __call__(self, seed: int) -> Iterable["TrainingEpisode"]:
        """Yield ordered document streams and checkpointed supervision.

        Args:
            seed: Explicit generation seed; no implicit global RNG contract.

        Returns:
            A possibly lazy, single-pass iterable. Only an episode boundary resets state.
        """
        ...


@dataclass(frozen=True)
class UpdateContext:
    """Normalized loss and permitted update targets for an explicit learning step.

    Shared parameters exclude the backbone. Memory entries are the adaptable
    tensors in the pre-question state, keyed by layer and algorithm-local name.
    """

    loss: Tensor
    shared_parameters: tuple[nn.Parameter, ...]
    memory_parameters: Mapping[tuple[int, str], Tensor]


@dataclass(frozen=True)
class UpdateResult:
    """Functional replacements for W, and whether shared parameters were changed."""

    memory: Mapping[tuple[int, str], Tensor]
    shared_changed: bool = False


class UpdateRule(Protocol):
    """Choose targets, gradients and optimizer behavior at a learning checkpoint."""

    def __call__(self, context: UpdateContext) -> UpdateResult:
        """Update shared parameters in place and return functional memory replacements.

        Args:
            context: One loss graph and all permitted phi/W targets. The backbone is absent.

        Returns:
            Replacements for any selected memory tensors, plus shared-change metadata.
            Custom rules may own optimizer state. They must report shared changes so
            the runtime invalidates other streams computed under previous parameters.
        """
        ...
