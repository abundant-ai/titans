"""Behavioral checks for method, data, loss and update-policy replacement."""

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch

from examples.linear_memory import LinearMemory, LinearState
from titans import (
    ContinualModel,
    Document,
    GradientUpdate,
    LossContext,
    LossValue,
    Question,
    RuntimeConfig,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
    UpdateContext,
    UpdateResult,
    cross_entropy,
    residual_connection,
)


@pytest.fixture
def linear_model(tiny_model: TitansModel) -> ContinualModel:
    """Attach a structurally different method to each real tiny HF backbone.

    Args:
        tiny_model: Offline Llama/Qwen fixture; its original memory is not reused.

    Returns:
        Generic runtime with a single linear matrix and no Titans-specific state.
    """
    return ContinualModel(tiny_model.backbone, tiny_model.tokenizer,
                          RuntimeConfig(attention_window=8, persistent_tokens=0,
                                        compile_kernels=False, prefill_chunk_size=3),
                          memory_factory=LinearMemory, connection_factory=residual_connection)


def test_alternative_transition_order_and_continuity(linear_model: ContinualModel) -> None:
    """A non-Titans state survives split ingestion, cloning and isolated predictions."""
    with torch.no_grad():
        first, second = linear_model.start_stream(), linear_model.start_stream()
        tokens = [7, 10, 11, 12, 13, 8, 10, 11, 12, 14, 17]
        expected = linear_model._forward_tokens(tokens, first)
        actual = torch.cat((linear_model._forward_tokens(tokens[:4], second),
                            linear_model._forward_tokens(tokens[4:], second)), dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert isinstance(first._memory[0], LinearState)
    before = first._memory[0].matrix.clone()
    linear_model.answer(Question("q", "Where Alice lives ?"), first)
    torch.testing.assert_close(before, first._memory[0].matrix, atol=0, rtol=0)
    fork = linear_model.clone_state(first)
    linear_model.ingest(Document("later", "Alice lives in Rome"), fork)
    assert not torch.equal(fork._memory[0].matrix, before)
    torch.testing.assert_close(before, first._memory[0].matrix, atol=0, rtol=0)


@pytest.mark.parametrize("targets", ["phi", "memory", "both"])
def test_feedback_target_selection(linear_model: ContinualModel, targets: str) -> None:
    """Changing phi, W or both must leave B and committed document history untouched."""
    state = linear_model.ingest(Document("a", "Alice lives in Paris"))
    unrelated = linear_model.clone_state(state)
    backbone = [p.detach().clone() for p in linear_model.backbone.parameters()]
    shared = [p.detach().clone() for p in linear_model.shared_parameters()]
    memory = state._memory[0].tensors()[0].clone()
    cache = {index: value.copy() for index, value in state._attention.items()}
    tokens, documents = state.tokens_processed, state.documents_seen
    selected = "phi" if targets == "phi" else "memory" if targets == "memory" else "both"
    result = linear_model.learn(Question("q", "Where Alice lives ?"), "Rome", state,
                                update=GradientUpdate(selected, learning_rate=0.1))
    assert result.tokens_processed > 0
    assert result.shared_changed == (targets != "memory")
    assert state.tokens_processed == tokens and state.documents_seen == documents
    for actual, expected in zip(linear_model.backbone.parameters(), backbone, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not actual.requires_grad
    shared_changed = any(not torch.equal(p, old) for p, old in
                         zip(linear_model.shared_parameters(), shared, strict=True))
    assert shared_changed == (targets != "memory")
    assert (not torch.equal(memory, state._memory[0].tensors()[0])) == (targets != "phi")
    for index, old in cache.items():
        torch.testing.assert_close(state._attention[index].keys, old.keys, atol=0, rtol=0)
    linear_model.clone_state(state)
    if targets != "memory":
        with pytest.raises(ValueError, match="predates fit"):
            linear_model.clone_state(unrelated)
    else:
        linear_model.clone_state(unrelated)


def test_custom_objective_can_depend_only_on_memory(linear_model: ContinualModel) -> None:
    """An objective independent of QA cross-entropy can drive W toward its own target."""
    state = linear_model.start_stream()

    def objective(context: LossContext) -> LossValue:
        """Penalize distance to a synthetic target matrix, ignoring answer logits.

        Args:
            context: Differentiable pre-question state supplied by the runtime.

        Returns:
            Scalar memory penalty with unit normalization.
        """
        matrix = context.state.memory[0].adaptable()["matrix"]
        return LossValue((matrix - 1).square().sum(), 1.0)

    before = float((state._memory[0].tensors()[0] - 1).square().sum())
    linear_model.learn(Question("q", "Where ?"), "Rome", state, loss=objective,
                       update=GradientUpdate("memory", learning_rate=0.1))
    after = float((state._memory[0].tensors()[0] - 1).square().sum())
    assert after < before


def test_generated_checkpoint_training_retains_write_gradients(
    linear_model: ContinualModel, tmp_path: Path,
) -> None:
    """A supplied generator, objective and joint update learn through document writes."""
    seeds: list[int] = []
    calls: list[str] = []

    def generate(seed: int) -> Iterable[TrainingEpisode]:
        """Yield one stream with two temporally separated supervision checkpoints.

        Args:
            seed: Seed supplied by the fit configuration.

        Yields:
            One episode with no question/answer text in its document payloads.
        """
        seeds.append(seed)
        yield TrainingEpisode("stream", (
            Document("a", "Alice lives in Paris"), Document("b", "Alice lives in Rome"),
        ), (TrainingQuestion(1, Question("q1", "Where Alice lives ?"), "Paris"),
            TrainingQuestion(2, Question("q2", "Where Alice lives ?"), "Rome")))

    def objective(context: LossContext) -> LossValue:
        """Record dispatch and use a rescaled token objective.

        Args:
            context: Aligned answer predictions and original document state.

        Returns:
            Scaled answer loss, preserving the target-count denominator.
        """
        calls.append(context.phase)
        value = cross_entropy(context)
        return LossValue(2 * value.value, value.weight)

    before = linear_model.memories["0"].rate.detach().clone()
    result = linear_model.fit(generate, TrainingConfig(
        tmp_path / "fit", max_tokens=200, sequence_length=128, seed=42,
        objective="document_qa",
    ), loss=objective, update=GradientUpdate("both", learning_rate=0.1))
    assert seeds == [42] and calls == ["answer", "answer"]
    assert result.optimizer_steps == 2
    assert result.final_state is not None and result.final_state.documents_seen == 2
    assert not torch.equal(before, linear_model.memories["0"].rate)
    linear_model.clone_state(result.final_state)


def test_generic_checkpoint_requires_explicit_factory(
    linear_model: ContinualModel, tmp_path: Path,
) -> None:
    """Restored custom methods reproduce logits without importing code from metadata."""
    path = linear_model.save_pretrained(tmp_path / "custom")
    restored = ContinualModel.load_pretrained(path, memory_factory=LinearMemory, connection_factory=residual_connection)
    with torch.no_grad():
        expected = linear_model._forward_tokens([7, 10, 13], linear_model.start_stream())
        actual = restored._forward_tokens([7, 10, 13], restored.start_stream())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_loss_leaves_stream_unchanged(linear_model: ContinualModel) -> None:
    """An invalid objective must fail before applying state or parameter changes."""
    state = linear_model.ingest(Document("a", "Alice lives in Paris"))
    before = state._memory[0].tensors()[0].clone()

    def invalid(context: LossContext) -> LossValue:
        """Return an invalid normalization weight for boundary validation.

        Args:
            context: Loss inputs supplied by the runtime.

        Returns:
            A scalar with an invalid zero denominator.
        """
        return LossValue(context.predictions.sum(), 0)

    with pytest.raises(ValueError, match="positive weight"):
        linear_model.learn(Question("q", "Where ?"), "Rome", state, loss=invalid)
    linear_model.clone_state(state)
    torch.testing.assert_close(before, state._memory[0].tensors()[0], atol=0, rtol=0)


def test_update_rule_is_a_callable_protocol(linear_model: ContinualModel) -> None:
    """A user function can select a single W tensor and implement its own solver."""
    state = linear_model.ingest(Document("a", "Alice lives in Paris"))
    before = state.memory[0].tensors()[0].clone()

    def update(context: UpdateContext) -> UpdateResult:
        """Take an unclipped gradient step on just the matrix target.

        Args:
            context: Normalized loss and named permitted tensors.

        Returns:
            One functional W replacement, leaving shared parameters unchanged.
        """
        matrix = context.memory_parameters[0, "matrix"]
        gradient, = torch.autograd.grad(context.loss, (matrix,))
        return UpdateResult({(0, "matrix"): matrix - 0.1 * gradient})

    result = linear_model.learn(Question("q", "Where ?"), "Rome", state, update=update)
    assert not result.shared_changed
    assert not torch.equal(before, state.memory[0].tensors()[0])


def test_truncation_severs_old_write_credit(linear_model: ContinualModel, tmp_path: Path) -> None:
    """A QA loss after detachment can train reading but cannot reach the old write gate."""
    before = linear_model.memories["0"].rate.detach().clone()
    episode = TrainingEpisode("one", (Document("a", "Alice lives in Paris " * 20),), (
        TrainingQuestion(1, Question("q", "Where ?"), "Rome"),
    ))
    result = linear_model.fit([episode], TrainingConfig(
        tmp_path / "truncated", max_tokens=200, sequence_length=8, objective="document_qa",
    ), update=GradientUpdate("phi", learning_rate=0.1))
    assert result.optimizer_steps == 1
    torch.testing.assert_close(before, linear_model.memories["0"].rate, atol=0, rtol=0)


def test_custom_loss_default_trainer_dispatch(linear_model: ContinualModel, tmp_path: Path) -> None:
    """Accumulated training calls the custom objective for both document and QA targets."""
    phases: set[str] = set()

    def objective(context: LossContext) -> LossValue:
        """Record the supervision phase and compute the normal token loss.

        Args:
            context: Aligned document or answer token predictions.

        Returns:
            Summed cross-entropy with token-count normalization.
        """
        phases.add(context.phase)
        return cross_entropy(context)

    episode = TrainingEpisode("one", (Document("a", "Alice lives in Paris"),), (
        TrainingQuestion(1, Question("q", "Where ?"), "Rome"),
    ))
    result = linear_model.fit([episode], TrainingConfig(
        tmp_path / "accumulated", max_tokens=100, sequence_length=64,
    ), loss=objective)
    assert phases == {"document", "answer"} and result.optimizer_steps == 1


def test_titans_feedback_changes_only_active_weights(tiny_model: TitansModel) -> None:
    """Explicit W adaptation also works with Titans' pending writes and momentum."""
    state = tiny_model.ingest(Document("a", "Alice lives in Paris"))
    before = {layer: memory.copy(detach=True) for layer, memory in state.memory.items()}
    shared = [p.detach().clone() for p in tiny_model.shared_parameters()]
    tiny_model.learn(Question("q", "Where ?"), "Rome", state,
                     update=GradientUpdate("memory", learning_rate=0.1))
    assert any(not torch.equal(old, state.memory[layer].adaptable()[name])
               for layer, memory in before.items() for name, old in memory.adaptable().items())
    for actual, old in zip(tiny_model.shared_parameters(), shared, strict=True):
        torch.testing.assert_close(actual, old, atol=0, rtol=0)
