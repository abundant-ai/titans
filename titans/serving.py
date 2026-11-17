"""State ownership shared by the SGLang adapter and CPU contract tests."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .state import StreamState

if TYPE_CHECKING:
    from .model import ContinualModel


@dataclass(frozen=True)
class RequestControl:
    """Private metadata transported with each local SGLang request."""

    operation: Literal["ingest", "answer", "clone", "release"]
    state_id: str
    source_id: str = ""
    write: bool = True
    document_complete: bool = False

    @classmethod
    def parse(cls, value: dict[str, object]) -> "RequestControl":
        """Validate metadata received at the SGLang process boundary.

        Args:
            value: Request metadata; extra SGLang bookkeeping keys are ignored.

        Returns:
            Typed state-control metadata.

        Raises:
            ValueError: Required fields have invalid values or types.
        """
        operation = value.get("operation")
        state_id = value.get("state_id")
        source_id = value.get("source_id", "")
        write = value.get("write", True)
        complete = value.get("document_complete", False)
        if operation not in ("ingest", "answer", "clone", "release"):
            raise ValueError("Unknown Titans serving operation")
        if not isinstance(state_id, str) or not state_id or not isinstance(source_id, str):
            raise ValueError("Titans state IDs must be strings")
        if not isinstance(write, bool) or not isinstance(complete, bool):
            raise ValueError("Titans write and document_complete flags must be booleans")
        if operation == "ingest":
            return cls("ingest", state_id, source_id, write, complete)
        if operation == "answer":
            return cls("answer", state_id, source_id, write, complete)
        if operation == "clone":
            return cls("clone", state_id, source_id, write, complete)
        return cls("release", state_id, source_id, write, complete)


class StateRegistry:
    """Worker-owned memory states; SGLang's token-prefix cache is not used for them."""

    def __init__(self, model: "ContinualModel") -> None:
        """Create an empty registry for one worker model.

        Args:
            model: Numerical model owning all worker states.
        """
        self.model = model
        self.states: dict[str, StreamState] = {}

    def prepare(self, control: RequestControl) -> StreamState | None:
        """Create, branch, look up or release a state according to a request.

        Args:
            control: Validated operation and state IDs.

        Returns:
            The owned state for ingestion/generation, or None for a control request.
        """
        if control.operation == "release":
            self.states.pop(control.state_id, None)
            return None
        if control.operation == "clone":
            if control.state_id in self.states:
                raise ValueError("Clone destination already exists")
            self.states[control.state_id] = self.model.clone_state(self.states[control.source_id])
            return None
        if control.state_id not in self.states:
            if control.operation == "answer":
                self.states[control.state_id] = self.model.clone_state(self.states[control.source_id])
            else:
                self.states[control.state_id] = self.model._new_state()
        state = self.states[control.state_id]
        self.model._validate_state(state)
        return state
