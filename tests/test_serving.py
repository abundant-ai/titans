"""CPU verification of the worker-side state protocol, independent of GPU libraries."""

import torch

from tests.support.state import assert_same, snapshot
from titans import Document, TitansModel
from titans.serving.registry import RequestControl, StateRegistry


def test_serving_branches_preserve_fast_weights_and_pending_updates(tiny_model: TitansModel) -> None:
    """Server-side answer/clone operations must preserve all source-state tensors."""
    registry = StateRegistry(tiny_model)
    with torch.no_grad():
        source = registry.prepare(RequestControl("ingest", "documents"))
        assert source is not None
        tiny_model.ingest(Document("a", "Alice lives in Paris"), source)
        before = snapshot(source)
        answer = registry.prepare(RequestControl("answer", "answer-1", "documents"))
        assert answer is not None
        tiny_model._forward_tokens([7, 8, 9, 10, 11], answer)
        assert_same(snapshot(source), before)
        registry.prepare(RequestControl("clone", "fork", "documents"))
        tiny_model._forward_tokens([14, 15, 16], registry.states["fork"])
        assert_same(snapshot(source), before)
        registry.prepare(RequestControl("release", "answer-1"))
        assert set(registry.states) == {"documents", "fork"}


def test_request_metadata_validation() -> None:
    """Valid engine metadata is narrowed to a typed control record."""
    value = RequestControl.parse({"operation": "answer", "state_id": "q", "source_id": "d",
                                  "write": False, "__req__": object()})
    assert value == RequestControl("answer", "q", "d", write=False)
