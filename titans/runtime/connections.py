"""Backbone connection policies independent of the memory architecture."""

from torch import Tensor

from titans.core.protocols import (
    BackboneConnection,
    ConnectionContext,
    ConnectionPlan,
    MemoryModule,
    MemoryState,
)


class ResidualConnection(BackboneConnection):
    """The original pretrained retrofit, preserving its exact addition order."""

    def prepare(self, context: ConnectionContext, memory: MemoryModule,
                state: MemoryState, *, write: bool) -> ConnectionPlan:
        """Read/update memory on normalized input alongside attention.

        Args:
            context: Frozen backbone representations at this layer.
            memory: Independently supplied memory architecture.
            state: Owned mutable stream memory.
            write: Whether memory writes are allowed.

        Returns:
            Original attention input and the gated memory contribution.
        """
        return ConnectionPlan(context.normalized, memory(context.normalized, state, write=write))

    def combine(self, context: ConnectionContext, attention: Tensor,
                plan: ConnectionPlan) -> Tensor:
        """Combine the two branches with the original residual.

        Args:
            context: Original residual stream.
            attention: Output from bounded backbone attention.
            plan: Memory contribution computed by prepare.

        Returns:
            Residual plus attention plus memory, retaining original arithmetic order.
        """
        return context.hidden + (attention + plan.memory_output)


def residual_connection(layer_index: int, hidden_size: int) -> BackboneConnection:
    """Construct the original parameter-free pretrained connection.

    Args:
        layer_index: Selected backbone layer.
        hidden_size: Residual width, unused by this parameter-free policy.

    Returns:
        A new residual connection.
    """
    return ResidualConnection()
