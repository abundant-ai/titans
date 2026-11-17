"""Public protocol, pretrained integration, training and checkpoint contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

from titans import (
    AnswerConfig,
    Document,
    Question,
    StreamState,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
)


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


@pytest.mark.parametrize("policy", ["read_only", "temporary"])
def test_answer_preserves_source_and_weights(tiny_model: TitansModel, policy: str) -> None:
    """Both answer policies preserve every source tensor and every trained parameter."""
    state = tiny_model.ingest(Document("a", "Alice lives in Paris"))
    before = snapshot(state)
    weights = {name: p.detach().clone() for name, p in tiny_model.named_parameters()}
    config = AnswerConfig(max_new_tokens=4, memory_policy="read_only" if policy == "read_only"
                          else "temporary")
    first = tiny_model.answer(Question("q", "Where Alice lives ?"), state, config)
    second = tiny_model.answer(Question("q", "Where Alice lives ?"), state, config)
    assert first == second
    assert_same(snapshot(state), before)
    for name, parameter in tiny_model.named_parameters():
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)


def test_answer_failure_preserves_source(tiny_model: TitansModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception after branch mutation must leave the document stream intact."""
    state = tiny_model.ingest(Document("a", "Alice lives in Paris"))
    before = snapshot(state)
    original = tiny_model._forward_tokens

    def fail_after_forward(
        tokens: list[int], branch: StreamState, *, write: bool = True, last_only: bool = False,
    ) -> Tensor:
        """Mutate the query branch and then simulate a decoding failure.

        Args:
            tokens: Query tokens.
            branch: Owned disposable query branch.
            write: Whether to update long-term memory.
            last_only: Whether to return only final-token logits.

        Raises:
            RuntimeError: Always, after the real forward completes.
        """
        original(tokens, branch, write=write, last_only=last_only)
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(tiny_model, "_forward_tokens", fail_after_forward)
    with pytest.raises(RuntimeError, match="simulated failure"):
        tiny_model.answer(Question("q", "Where ?"), state)
    assert_same(snapshot(state), before)


def test_clone_copies_incomplete_updates_and_caches(tiny_model: TitansModel) -> None:
    """Mutating cloned tensors must not touch the source's storage."""
    state = tiny_model.ingest(Document("a", "Alice lives in Paris"))
    before = snapshot(state)
    clone = tiny_model.clone_state(state)
    assert_same(snapshot(clone), before)
    for memory in clone._memory.values():
        for tensor in memory.weights + memory.momentum + (memory.pending, memory.conv_tail):
            tensor.add_(7)
    for cache in clone._attention.values():
        cache.keys.add_(7)
        cache.values.add_(7)
        cache.positions.add_(7)
    if clone._next_logits is not None:
        clone._next_logits.add_(7)
    if clone._next_hidden is not None:
        clone._next_hidden.add_(7)
    assert_same(snapshot(state), before)


def test_separate_streams_and_repeated_documents(tiny_model: TitansModel) -> None:
    """Starting a new stream resets memory; reusing an ID still ingests new text."""
    document = Document("a", "Alice lives in Paris")
    first = tiny_model.ingest(document)
    expected = snapshot(first)
    assert tiny_model.ingest(Document("a", "Alice lives in Rome now"), first) is first
    assert first.documents_seen == 2
    independent = tiny_model.ingest(document)
    assert_same(snapshot(independent), expected)


def test_cache_and_memory_are_bounded(tiny_model: TitansModel) -> None:
    """All layers retain only the configured local window and persistent prefix."""
    state = None
    for _ in range(40):
        state = tiny_model.ingest(Document("a", "Alice lives in Paris"), state)
    assert state is not None
    for cache in state._attention.values():
        assert cache.keys.shape[-2] <= 2 + 8 - 1
        assert cache.values.shape[-2] == cache.positions.numel()
    for memory in state._memory.values():
        assert memory.pending.shape[1] < 4
        assert memory.conv_tail.shape[-1] == 2
        assert all(not tensor.requires_grad for tensor in memory.weights)
    assert state.tokens_processed > 200


def test_model_chunk_boundaries_and_causality(tiny_model: TitansModel) -> None:
    """Whole/split calls agree, and changing future tokens cannot change the prefix."""
    tokens = [7, 10, 11, 12, 13, 8, 10, 11, 12, 14, 17]
    with torch.no_grad():
        whole = tiny_model._new_state()
        split = tiny_model._new_state()
        expected = tiny_model._forward_tokens(tokens, whole)
        actual = torch.cat((tiny_model._forward_tokens(tokens[:3], split),
                            tiny_model._forward_tokens(tokens[3:9], split),
                            tiny_model._forward_tokens(tokens[9:], split)), 1)
        changed = tiny_model._forward_tokens(tokens[:5] + [20] * 6, tiny_model._new_state())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(changed[:, :5], expected[:, :5], rtol=1e-5, atol=1e-7)
    assert_same(snapshot(split), snapshot(whole), exact=False)


def test_attention_matches_stock_backbone_when_window_covers_input(tiny_model: TitansModel) -> None:
    """Zeroing the memory output must recover the host model's causal attention."""
    model = TitansModel(tiny_model.backbone, tiny_model.tokenizer,
                        replace(tiny_model.config, persistent_tokens=0, attention_window=64))
    with torch.no_grad():
        for memory in model.memories.values():
            memory.output.weight.zero_()
        tokens = [7, 10, 11, 12, 13, 8, 10, 11, 12, 14, 17]
        expected = model.backbone(torch.tensor([tokens]), use_cache=False).logits
        actual = model._forward_tokens(tokens, model._new_state())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("objective", ["language_modeling", "document_qa", "mixed"])
def test_fit_updates_memory_freezes_backbone_and_invalidates_streams(
    tiny_model: TitansModel, tmp_path: Path, objective: str,
) -> None:
    """Exercise actual LM/QA post-training, including an independent answer branch."""
    state = tiny_model.ingest(Document("a", "Alice lives in Paris"))
    backbone_before = {name: p.detach().clone() for name, p in tiny_model.backbone.named_parameters()}
    memory_before = {name: p.detach().clone() for name, p in tiny_model.memories.named_parameters()}
    episode = TrainingEpisode("one", (Document("a", "Alice lives in Paris"),), (
        TrainingQuestion(1, Question("q", "Where Alice lives ?"), "Paris"),
    ))
    selected = "language_modeling" if objective == "language_modeling" else (
        "document_qa" if objective == "document_qa" else "mixed"
    )
    result = tiny_model.fit([episode] * 2, TrainingConfig(
        tmp_path / "checkpoint", max_tokens=100, sequence_length=32,
        tokens_per_step=20, objective=selected, learning_rate=1e-3,
    ))
    assert result.optimizer_steps > 0
    assert result.mean_loss is not None and result.mean_loss > 0
    assert result.stop_reason == "data_exhausted"
    assert (result.checkpoint_path / "memory.safetensors").exists()
    assert not tiny_model.training
    for name, parameter in tiny_model.backbone.named_parameters():
        torch.testing.assert_close(parameter, backbone_before[name], rtol=0, atol=0)
    assert any(not torch.equal(parameter, memory_before[name])
               for name, parameter in tiny_model.memories.named_parameters())
    # QA must teach the write rule, not merely the output gate.
    assert not torch.equal(tiny_model.memories["0"].gates.weight, memory_before["0.gates.weight"])
    with pytest.raises(ValueError, match="predates fit"):
        tiny_model.answer(Question("q", "Where ?"), state)
    with pytest.raises(ValueError, match="predates fit"):
        tiny_model.clone_state(state)


def test_fit_exact_budget_and_partial_accumulation(tiny_model: TitansModel, tmp_path: Path) -> None:
    """A short budget is consumed exactly and its partial gradient batch is applied."""
    episode = TrainingEpisode("a", (Document("a", "Alice lives in Paris " * 30),))
    result = tiny_model.fit([episode], TrainingConfig(
        tmp_path / "small", max_tokens=17, sequence_length=5, tokens_per_step=100,
        objective="language_modeling",
    ))
    assert result.tokens_processed == 17
    assert result.optimizer_steps == 1
    assert result.stop_reason == "token_budget"


def test_backbone_unfreezing_is_rejected() -> None:
    """The fixed runtime contract rejects attempts to update backbone weights."""
    with pytest.raises(ValueError, match="always freezes"):
        TrainingConfig(Path("unused"), max_tokens=20, freeze_backbone=False)


def test_checkpoint_roundtrip(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Saving/reloading must reproduce ingestion, state contents and generation offline."""
    path = tiny_model.save_pretrained(tmp_path / "saved")
    loaded = TitansModel.load_pretrained(path)
    document = Document("a", "Alice lives in Paris")
    first = tiny_model.ingest(document)
    second = loaded.ingest(document)
    assert_same(snapshot(second), snapshot(first))
    question = Question("q", "Where Alice lives ?")
    config = AnswerConfig(max_new_tokens=4)
    assert tiny_model.answer(question, first, config) == loaded.answer(question, second, config)
    with pytest.raises(ValueError, match="different model"):
        loaded.clone_state(first)
    with pytest.raises(FileExistsError):
        tiny_model.save_pretrained(path)


def test_loading_existing_backbone_entry_point(tiny_model: TitansModel, tmp_path: Path) -> None:
    """The public attach entry point must load a real local HF checkpoint and tokenizer."""
    path = tmp_path / "backbone"
    tiny_model.backbone.save_pretrained(path)
    tiny_model.tokenizer.save_pretrained(path)
    loaded = TitansModel.from_pretrained(path, tiny_model.config, local_files_only=True)
    assert type(loaded.backbone) is type(tiny_model.backbone)
    assert loaded.ingest(Document("a", "Alice lives in Paris")).documents_seen == 1


def test_seeded_answers_leave_global_rng_unchanged(tiny_model: TitansModel) -> None:
    """Sampling must be reproducible without consuming the caller's global RNG."""
    state = tiny_model.ingest(Document("a", "Alice lives in Paris"))
    rng = torch.random.get_rng_state().clone()
    config = AnswerConfig(max_new_tokens=4, temperature=0.7, seed=123)
    first = tiny_model.answer(Question("q", "Where ?"), state, config)
    second = tiny_model.answer(Question("q", "Where ?"), state, config)
    assert first == second
    assert torch.equal(rng, torch.random.get_rng_state())


def test_existing_output_rejected_before_training(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Output conflicts must not invalidate streams or mutate trained parameters."""
    state = tiny_model.ingest(Document("a", "Alice lives in Paris"))
    with pytest.raises(FileExistsError):
        tiny_model.fit([], TrainingConfig(tmp_path, max_tokens=4))
    tiny_model.clone_state(state)
