"""Replaceable learning policies; document ordering belongs to the runtime."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, Self

from torch import Tensor, nn

if TYPE_CHECKING:
    from titans.core.types import TrainingEpisode
    from titans.runtime.state import StreamState


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


class MemoryArchitecture(Protocol):
    """Construct a fresh registered module at each selected backbone layer."""

    def __call__(self, hidden_size: int) -> MemoryModule:
        """Build a module; the runtime subsequently places its parameters.

        Args:
            hidden_size: Backbone residual width.

        Returns:
            A new module instance, never shared implicitly between layers.
        """
        ...


MemoryFactory = MemoryArchitecture


@dataclass(frozen=True)
class ConnectionContext:
    """Read-only inputs at one selected layer; no backbone parameters are exposed."""

    layer_index: int
    hidden: Tensor
    normalized: Tensor


@dataclass(frozen=True)
class ConnectionPlan:
    """Attention input (optionally prepended context) and a real-token memory read."""

    attention_input: Tensor
    memory_output: Tensor


@dataclass(frozen=True)
class ConnectionExecution:
    """Connection-owned attention layout for the next causal block.

    block_size bounds real tokens. extra_tokens bounds virtual attention inputs.
    position_start overrides ordinary rotary positions when supplied. token_start
    and token_stride select real-token attention outputs; None skips the actual
    number of prepended inputs. reset_cache retains only persistent-prefix KV.
    snapshot_memory requests a differentiable read cursor that survives until
    reset_cache, independently of the live memory receiving writes.
    retain_training_cache preserves method-owned attention context when ordinary
    decoder KV is cleared between packed pretraining blocks.
    """

    block_size: int
    extra_tokens: int = 0
    position_start: int | None = None
    attention_window: int | None = None
    token_start: int | None = None
    token_stride: int = 1
    reset_cache: bool = False
    snapshot_memory: bool = False
    retain_training_cache: bool = False


class BackboneConnection(nn.Module, ABC):
    """Registered connection parameters; mutable temporal state belongs to MemoryState."""

    max_context_tokens: int = 0

    def execution(self, position: int, tokens: int, *, prefix_tokens: int,
                  segment_length: int, attention_window: int) -> ConnectionExecution:
        """Declare the next block's bounds and attention layout.

        The default preserves ordinary residual connections and legacy connections
        that prepend bounded context for each individual real token.

        Args:
            position: Absolute input position including the persistent prefix.
            tokens: Number of available real tokens.
            prefix_tokens: Persistent-prefix length.
            segment_length: Runtime segment setting available to segmented methods.
            attention_window: Ordinary decoder attention span.

        Returns:
            Execution requirements for this connection's next block.

        Raises:
            ValueError: The declared prepended context exceeds the attention span.
        """
        if not 0 <= self.max_context_tokens < attention_window:
            raise ValueError("Connection context must fit inside attention_window")
        if self.max_context_tokens:
            return ConnectionExecution(
                block_size=1, extra_tokens=self.max_context_tokens,
                position_start=prefix_tokens, reset_cache=True,
            )
        return ConnectionExecution(block_size=tokens)

    def prepare_block(self, context: ConnectionContext, memory: MemoryModule,
                      state: MemoryState, *, retrieval: MemoryState | None,
                      write: bool) -> ConnectionPlan:
        """Prepare attention using the connection's declared execution policy.

        Args:
            context: Current real-token representations.
            memory: Registered memory implementation.
            state: Live instance memory receiving writes.
            retrieval: Optional persistent read cursor requested by execution.
            write: Whether active-memory writes are enabled.

        Returns:
            Attention inputs and memory contribution. Existing implementations
            inherit dispatch to prepare with the live instance memory.
        """
        return self.prepare(context, memory, state, write=write)

    @abstractmethod
    def prepare(self, context: ConnectionContext, memory: MemoryModule,
                state: MemoryState, *, write: bool) -> ConnectionPlan:
        """Choose the memory inputs and the representations sent to attention.

        Args:
            context: Layer residual and normalized input, in chronological token order.
            memory: Architecture and transition supplied by the agent.
            state: Stream-owned memory; preserve graphs during training.
            write: Whether active memory may be updated.

        Returns:
            Attention input may prepend up to max_context_tokens virtual tokens.
            The memory output matches the real-token input shape; both preserve device/dtype.
            Implementations must be causal and invariant to splitting token blocks.
        """
        ...

    @abstractmethod
    def combine(self, context: ConnectionContext, attention: Tensor,
                plan: ConnectionPlan) -> Tensor:
        """Return the complete residual stream before the fixed decoder FFN.

        Args:
            context: Original layer inputs.
            attention: Frozen bounded attention output.
            plan: Values retained from prepare, with differentiable graph intact.

        Returns:
            Same-shape residual stream; may use learned fusion or additional projections.
        """
        ...

    def finish(self, context: ConnectionContext, attention: Tensor, plan: ConnectionPlan,
               memory: MemoryModule, state: MemoryState, *, write: bool) -> Tensor:
        """Finish the connection, optionally writing memory from attention outputs.

        Args:
            context: Original layer inputs.
            attention: Attention outputs corresponding to real tokens only.
            plan: Intermediate reads from prepare.
            memory: Registered memory architecture.
            state: Continuing algorithm state.
            write: Whether attention-driven writes are permitted.

        Returns:
            The residual stream before the fixed FFN. The default delegates to combine.
        """
        return self.combine(context, attention, plan)


class ConnectionFactory(Protocol):
    """Instantiate independent connection modules with layer-specific architecture."""

    def __call__(self, layer_index: int, hidden_size: int) -> BackboneConnection:
        """Create the connection for one configured memory layer.

        Args:
            layer_index: Backbone layer receiving the memory connection.
            hidden_size: Width of its residual stream.

        Returns:
            A fresh module; its parameters join phi and its weights are checkpointed.
        """
        ...


@dataclass(frozen=True)
class LossContext:
    """Aligned predictions plus explicit shared and stream state for custom losses.

    Predictions are [targets, vocabulary]; targets are [targets]. During explicit
    QA learning, state is the pre-question stream, and the teacher-forced branch
    is disposable. A loss may use predictions, shared parameters, memory, or all.
    Standard fit callbacks see the state after the forwarded block. The final
    label can be a lookahead target that has not entered that state. Callback
    grouping follows execution blocks and should not be used as an update clock.
    """

    predictions: Tensor
    targets: Tensor
    state: "StreamState"
    shared_parameters: tuple[nn.Parameter, ...]
    phase: Literal["document", "answer"]
    backbone_parameters: tuple[nn.Parameter, ...] = ()


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
            A possibly lazy, single-pass iterable. TrainingConfig.episode_memory
            chooses whether episode boundaries reset state or carry it forward.
        """
        ...


@dataclass(frozen=True)
class UpdateContext:
    """Normalized loss and permitted update targets for an explicit learning step.

    Shared parameters are phi; backbone_parameters exposes theta in joint mode.
    Memory entries are the adaptable
    tensors in the pre-question state, keyed by layer and algorithm-local name.
    """

    loss: Tensor
    shared_parameters: tuple[nn.Parameter, ...]
    memory_parameters: Mapping[tuple[int, str], Tensor]
    backbone_parameters: tuple[nn.Parameter, ...] = ()


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
            context: One loss graph and permitted phi/W targets. Theta is supplied
                separately only during joint training.

        Returns:
            Replacements for any selected memory tensors, plus shared-change metadata.
            Custom rules may own optimizer state. They must report shared changes so
            the runtime invalidates other streams computed under previous parameters.
        """
        ...
