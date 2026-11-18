"""Regression checks for episode ownership, causal supervision and prefix gradients."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from titans import (
    Document,
    GradientUpdate,
    Question,
    TitansConfig,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
)
from titans.core.protocols import LossContext, LossValue
from titans.training.learning import cross_entropy
from titans.training.trainer import _TrainingRun


@pytest.mark.parametrize("mode", ["reset", "carry"])
def test_episode_state_policy(tiny_model: TitansModel, tmp_path: Path, mode: str) -> None:
    """Compare episode memory to independent uninterrupted document ingestion.

    Args:
        tiny_model: Offline real decoder with memory.
        tmp_path: Unused optimizer destination.
        mode: Episode memory policy.
    """
    config = TrainingConfig(tmp_path / "unused", max_tokens=100, sequence_length=5,
                            objective="document_qa", episode_memory=mode)
    run = _TrainingRun(tiny_model, config, None)
    documents = (Document("one", "Alice lives in Paris"),
                 Document("two", "Alice lives in Rome"))
    run.episode(TrainingEpisode("a", documents[:1]))
    first_state = run.state
    run.episode(TrainingEpisode("b", documents[1:]))
    state = run.state
    assert state is not None
    assert (state is first_state) == (mode == "carry")
    expected = tiny_model.start_stream()
    with torch.no_grad():
        for doc in documents if mode == "carry" else documents[1:]:
            tiny_model.ingest(doc, expected)
    assert state.documents_seen == expected.documents_seen
    assert state.tokens_processed == expected.tokens_processed
    for layer, memory in state.memory.items():
        for actual, reference in zip(memory.tensors(), expected.memory[layer].tensors()):
            torch.testing.assert_close(actual, reference)
            assert actual.grad_fn is None
    assert run.steps == 0


@pytest.mark.parametrize("mode", ["reset", "carry"])
@pytest.mark.parametrize("custom_update", [False, True])
def test_fit_checkpoint_counts(tiny_model: TitansModel, tmp_path: Path,
                              mode: str, custom_update: bool) -> None:
    """Keep checkpoint counts local while carry mode preserves prior documents.

    Args:
        tiny_model: Offline decoder.
        tmp_path: Checkpoint destination.
        mode: Episode memory policy.
        custom_update: Exercise explicit W updates as well as standard AdamW.
    """
    seen = []

    def objective(context: LossContext) -> LossValue:
        """Observe branch provenance and compute summed answer cross-entropy.

        Args:
            context: Predictions and the current objective state.

        Returns:
            Summed loss and target count.
        """
        seen.append(context.state.documents_seen)
        return cross_entropy(context)

    episodes = [
        TrainingEpisode("a", (Document("d", "Alice lives in Paris"),)),
        TrainingEpisode("b", (), (
            TrainingQuestion(0, Question("q", "Where Alice lives ?"), "Paris"),
        )),
    ]
    config = TrainingConfig(tmp_path / "checkpoint", max_tokens=100, sequence_length=32,
                            objective="document_qa", episode_memory=mode)
    result = tiny_model.fit(
        iter(episodes), config, loss=objective,
        update=GradientUpdate("memory") if custom_update else None,
    )
    assert seen and set(seen) == {int(mode == "carry")}
    assert result.optimizer_steps > 0
    if mode == "carry":
        assert result.final_state is not None
        assert result.final_state.documents_seen == 1


@pytest.mark.parametrize("window", [1, 4])
@pytest.mark.parametrize("joint", [False, True])
def test_qa_boundary_receives_gradient(tiny_model: TitansModel, tmp_path: Path,
                                      window: int, joint: bool) -> None:
    """Verify learning from a QA target whose prediction falls exactly at a cut.

    Args:
        tiny_model: Offline decoder.
        tmp_path: Unused checkpoint destination.
        window: Unroll length, including the formerly degenerate one-token case.
        joint: Whether backbone parameters train too.
    """
    tiny_model.backbone.requires_grad_(joint)
    run = _TrainingRun(tiny_model, TrainingConfig(
        tmp_path / "unused", max_tokens=window + 1, sequence_length=window,
        tokens_per_step=100, objective="document_qa", freeze_backbone=not joint,
    ), None)
    state = tiny_model._new_state()
    run.consume([7] * window, state, supervise=False, live_states=(state,),
                next_target=13, next_phase="answer")
    assert run.total_targets == 1
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in run.parameters)
    run.consume([13], state, supervise=True, live_states=(state,), phase="answer")
    assert run.total_targets == 1


def test_prefix_learns_after_detach(tiny_model: TitansModel, tmp_path: Path) -> None:
    """A loss in a later window must reach refreshed persistent embeddings.

    Args:
        tiny_model: Frozen-backbone model with learned prefix.
        tmp_path: Unused checkpoint destination.
    """
    run = _TrainingRun(tiny_model, TrainingConfig(
        tmp_path / "unused", max_tokens=5, sequence_length=2, tokens_per_step=100,
        objective="document_qa",
    ), None)
    state = tiny_model._new_state()
    run.consume([7, 8], state, supervise=False, live_states=(state,))
    assert tiny_model.persistent.grad is None
    run.consume([9, 10], state, supervise=False, live_states=(state,),
                next_target=13, next_phase="answer")
    assert tiny_model.persistent.grad is not None
    assert torch.count_nonzero(tiny_model.persistent.grad)


@pytest.mark.parametrize("budget", [1, 4, 11, 30])
def test_lm_target_accounting(tiny_model: TitansModel, tmp_path: Path, budget: int) -> None:
    """Lookahead scores exactly the forwarded labels, without entering the state early.

    Args:
        tiny_model: Offline decoder.
        tmp_path: Unused checkpoint destination.
        budget: Forward token budget, including cuts before the input finishes.
    """
    labels = []
    observed_positions = []

    def objective(context: LossContext) -> LossValue:
        """Record targets and state positions for causal-label checks.

        Args:
            context: LM predictions with a possibly unforwarded final label.

        Returns:
            Summed cross-entropy and target count.
        """
        labels.extend(context.targets.tolist())
        observed_positions.append(context.state.tokens_processed)
        return cross_entropy(context)

    run = _TrainingRun(tiny_model, TrainingConfig(
        tmp_path / "unused", max_tokens=budget, sequence_length=4,
        tokens_per_step=100, objective="language_modeling",
    ), objective)
    state = tiny_model._new_state()
    tokens = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    run.consume(tokens, state, supervise=True, live_states=(state,))
    assert labels == tokens[:budget]  # The persistent prefix predicts the first token.
    assert run.total_targets == min(budget, len(tokens))
    assert state.tokens_processed == min(budget, len(tokens))
    assert observed_positions[0] <= 4


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_network_roundtrip(tiny_model: TitansModel, tmp_path: Path, version: int) -> None:
    """Missing architecture metadata must restore the historical SiLU memory.

    Args:
        tiny_model: Offline decoder.
        tmp_path: Saved checkpoint location.
        version: Historical metadata version to exercise.
    """
    from titans import residual_connection

    model = TitansModel(copy.deepcopy(tiny_model.backbone), tiny_model.tokenizer,
                        replace(tiny_model.config, memory_variant="legacy"),
                        connection_factory=residual_connection)
    path = model.save_pretrained(tmp_path / "legacy")
    metadata = json.loads((path / "titans.json").read_text())
    metadata["format_version"] = version
    metadata["config"].pop("memory_variant")
    for settings in metadata["builtin_memory_configs"].values():
        settings.pop("memory_variant")
    if version < 4:
        metadata.pop("builtin_memory_configs")
        metadata.pop("mac_retrieval_policy")
    if version < 3:
        metadata.pop("connection_types")
    (path / "titans.json").write_text(json.dumps(metadata))
    restored = TitansModel.load_pretrained(path)
    assert restored.memories["0"].config.memory_variant == "legacy"
    with torch.no_grad():
        inputs = [7, 8, 9, 10, 11]
        torch.testing.assert_close(
            model._forward_tokens(inputs, model.start_stream()),
            restored._forward_tokens(inputs, restored.start_stream()),
        )


def test_clear_configuration_defaults() -> None:
    """Episode isolation is the default and fresh memory uses fourfold expansion."""
    assert TrainingConfig(Path("unused"), max_tokens=1).episode_memory == "reset"
    with pytest.raises(ValueError, match="episode_memory"):
        TrainingConfig(Path("unused"), max_tokens=1, episode_memory="shared")
    config = TitansConfig(memory_dim=19)
    assert config.memory_hidden_dim == 76 and config.memory_variant == "paper"
    with pytest.raises(ValueError, match="LayerNorm"):
        TitansConfig(memory_dim=1)


@pytest.mark.parametrize("budget", [8, 15, 30, 100])
def test_mixed_labels_across_branches(tiny_model: TitansModel, tmp_path: Path,
                                    budget: int) -> None:
    """Match labels to a literal document/QA schedule under a shared token budget.

    Args:
        tiny_model: Offline decoder and tokenizer.
        tmp_path: Unused checkpoint destination.
        budget: Budget that can stop in a document, prompt or answer.
    """
    from titans.training.learning import feedback_tokens

    observed = {"document": [], "answer": []}

    def objective(context: LossContext) -> LossValue:
        """Collect routed labels and compute cross-entropy.

        Args:
            context: Block predictions with its explicit supervision phase.

        Returns:
            Summed cross-entropy and target count.
        """
        observed[context.phase].extend(context.targets.tolist())
        return cross_entropy(context)

    documents = (Document("a", "Alice lives in Paris"),
                 Document("b", "Alice lives in Rome"))
    question = Question("q", "Where Alice lives ?")
    prompt, answer = feedback_tokens(tiny_model, question, "Paris")
    first = tiny_model._document_tokens(documents[0])
    second = tiny_model._document_tokens(documents[1])
    remaining = budget
    expected = {"document": [], "answer": []}
    for tokens, phase in ((first, "document"), (prompt, None),
                          (answer, "answer"), (second, "document")):
        size = min(remaining, len(tokens))
        if phase:
            expected[phase].extend(tokens[:size])
        remaining -= size
    run = _TrainingRun(tiny_model, TrainingConfig(
        tmp_path / "unused", max_tokens=budget, sequence_length=1,
        tokens_per_step=4, objective="mixed",
    ), objective)
    run.episode(TrainingEpisode("s", documents, (
        TrainingQuestion(1, question, "Paris"),
    )))
    assert observed == expected
    assert run.tokens == min(budget, len(first) + len(prompt) + len(answer) + len(second))
