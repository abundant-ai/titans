"""Compiled-path checks, including gradients and persistent-state ownership."""

from dataclasses import replace
from pathlib import Path

import torch

from titans import (
    AnswerConfig,
    Document,
    Question,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
)
from titans.kernels import compiled_chunk_update

from .reference import reference_chunk
from .test_memory import case
from .test_model import assert_same, snapshot


def test_compiled_update_and_outer_gradients() -> None:
    """Compiled final states and all outer derivatives must match literal autograd."""
    inputs = case(2, 7)
    leaves = inputs[0] + inputs[1] + inputs[2:]
    actual = compiled_chunk_update(*inputs)
    expected = reference_chunk(*inputs)
    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)
    actual_gradients = torch.autograd.grad(sum(t.square().sum() for t in actual[0] + actual[1]),
                                         leaves, retain_graph=True)
    expected_gradients = torch.autograd.grad(sum(t.square().sum() for t in expected[0] + expected[1]),
                                           leaves)
    for got, want in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-8)


def test_compiled_model_and_training_match_eager(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Compare compiled/eager forwards and optimizer results with activation checkpointing."""
    compiled = TitansModel(tiny_model.backbone, tiny_model.tokenizer,
                           replace(tiny_model.config, compile_kernels=True))
    compiled.load_state_dict(tiny_model.state_dict())
    tokens = [7, 10, 11, 12, 13, 8, 10, 11, 12, 14, 17]
    with torch.no_grad():
        expected = tiny_model._forward_tokens(tokens, tiny_model._new_state())
        actual = compiled._forward_tokens(tokens, compiled._new_state())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    episode = TrainingEpisode("one", (Document("a", "Alice lives in Paris"),), (
        TrainingQuestion(1, Question("q", "Where Alice lives ?"), "Paris"),
    ))
    eager_result = tiny_model.fit([episode], TrainingConfig(
        tmp_path / "eager", max_tokens=32, sequence_length=32, activation_checkpointing=False,
    ))
    compiled_result = compiled.fit([episode], TrainingConfig(
        tmp_path / "compiled", max_tokens=32, sequence_length=32, activation_checkpointing=True,
    ))
    assert eager_result.optimizer_steps == compiled_result.optimizer_steps == 1
    for expected_parameter, actual_parameter in zip(tiny_model.memories.parameters(),
                                                    compiled.memories.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-4, atol=2e-6)
    state = compiled.ingest(Document("a", "Alice lives in Paris"))
    before = snapshot(state)
    compiled.answer(Question("q", "Where ?"), state, AnswerConfig(max_new_tokens=2))
    assert_same(snapshot(state), before)
