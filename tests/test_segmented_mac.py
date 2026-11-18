"""Numerical regression tests for segmented MAC and truncated training."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from titans import (
    AnswerConfig,
    ContinualModel,
    Document,
    Question,
    RuntimeConfig,
    TitansArchitecture,
    TitansConfig,
    TitansModel,
    TrainingConfig,
)
from titans.methods.titans import NeuralMemory, mlp
from titans.runtime.state import StreamState
from titans.training.trainer import _TrainingRun


def assert_stream(actual: StreamState, expected: StreamState) -> None:
    """Compare the numerical contents of two streams, including the retrieval cursor.

    Args:
        actual: Stream produced by block execution.
        expected: Stream produced by the reference execution.
    """
    assert (actual._position, actual._tokens, actual._documents) == (
        expected._position, expected._tokens, expected._documents)
    for states, reference in ((actual.memory, expected.memory),
                              (actual._retrieval, expected._retrieval)):
        assert states.keys() == reference.keys()
        for layer, state in states.items():
            for a, b in zip(state.tensors(), reference[layer].tensors(), strict=True):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for layer, cache in actual._attention.items():
        other = expected._attention[layer]
        for a, b in ((cache.keys, other.keys), (cache.values, other.values),
                     (cache.positions, other.positions)):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual._next_logits, expected._next_logits, atol=2e-6, rtol=2e-5)


@torch.no_grad()
def test_segment_stream(tiny_model: TitansModel) -> None:
    """Compare arbitrary block boundaries, branch policies and vocabulary projections.

    Args:
        tiny_model: Offline Llama or Qwen decoder and tokenizer.
    """
    config = replace(tiny_model.config, memory_layers=(0,), prefill_chunk_size=256, chunk_size=16)
    model = TitansModel(tiny_model.backbone, tiny_model.tokenizer, config)
    ids = [7 + i % 12 for i in range(513)]
    block, single = model.start_stream(), model.start_stream()
    for boundary in (15, 16, 17, 255, 256, 257, 511, 512, 513):
        start = block._tokens
        expected = torch.cat([model._forward_tokens([t], single) for t in ids[start:boundary]], 1)
        actual = model._forward_tokens(ids[start:boundary], block)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert_stream(block, single)
        count = boundary % 512
        assert block._attention[0].keys.shape[-2] == config.persistent_tokens + 2 * count
        assert block.memory[0].pending.shape[1] == boundary % 16
        if count:
            assert block._retrieval[0].pending.shape[1] == 0
        if boundary == 17:
            for policy in ("temporary", "read_only"):
                model.answer(Question("q", "Where ?"), block, AnswerConfig(
                    max_new_tokens=2, memory_policy=policy))
                assert_stream(block, single)
    # Use two independent calls from fresh state with different future tokens.
    first, second = model.start_stream(), model.start_stream()
    original = model._forward_tokens(ids[:33], first)
    changed = model._forward_tokens(ids[:15] + [4] * 18, second)
    torch.testing.assert_close(original[:, :15], changed[:, :15], atol=0, rtol=0)

    fresh = model.start_stream()
    head_rows = []
    hook = model.backbone.lm_head.register_forward_hook(
        lambda module, args, output: head_rows.append(output.shape[1]))
    try:
        model._forward_tokens(ids, fresh, last_only=True)
        assert head_rows == [1]
        head_rows.clear()
        model.ingest(Document("new", "Alice lives in Paris " * 150))
    finally:
        hook.remove()
    assert head_rows == [1]
    assert_stream(fresh, single)


@pytest.mark.parametrize("count", [3, 4, 5])
def test_post_write_projection(tiny_model: TitansModel, count: int) -> None:
    """Check each query, its causal history and outer derivatives against literal token updates.

    Args:
        tiny_model: Fixture supplying the memory dimensions.
        count: Number of attention outputs around the four-token commit.
    """
    memory = NeuralMemory(16, replace(tiny_model.config, chunk_size=4)).double()
    inputs = torch.randn(1, count, 16, dtype=torch.float64, requires_grad=True)
    actual_state, expected_state = memory.initial_state(), memory.initial_state()
    actual = memory._read_and_update(inputs, actual_state, write=True, post_write=True)
    reads = []
    for token in inputs.split(1, dim=1):
        before = expected_state.copy()
        memory(token, expected_state, write=True)
        # Independently form the current query from the PRE-write tail and the
        # current projected input. Read only the weights committed at this token.
        history = torch.cat((before.conv_tail, memory.qkv(token).transpose(1, 2)), dim=-1)
        query = F.silu(memory.conv(history).transpose(1, 2)).chunk(3, -1)[0]
        query = F.normalize(query, dim=-1, eps=1e-6)
        read = mlp(expected_state.weights, query)
        reads.append(memory.output(memory.read_norm(read)) * memory.mix_gate(token).sigmoid())
    expected = torch.cat(reads, dim=1)
    torch.testing.assert_close(actual, expected)
    params = (inputs, *memory.initial_weights, memory.qkv.weight, memory.conv.weight)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), params),
                    torch.autograd.grad(expected.square().sum(), params), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("joint", [True, False])
def test_detached_fused_boundary(tiny_model: TitansModel, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch, joint: bool) -> None:
    """Score a one-token answer after an actual optimizer step in both loss paths.

    Args:
        tiny_model: Offline decoder used to initialize both runs.
        tmp_path: Temporary path for the training configuration.
        monkeypatch: Injects a dense equivalent of the fused GPU objective.
        joint: Whether the backbone also receives outer gradients.
    """
    def dense_loss(weight: Tensor, hidden: Tensor, targets: Tensor) -> Tensor:
        """Stand in for the GPU-only fused kernel with the equivalent CPU objective."""
        return F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction="sum")

    monkeypatch.setattr("titans.optim.gpu_ops.liger_loss", dense_loss)
    config = TrainingConfig(tmp_path / "unused", max_tokens=8, sequence_length=4,
                            tokens_per_step=4, freeze_backbone=not joint)
    models = [tiny_model, copy.deepcopy(tiny_model)]
    reports = []
    for fused, model in enumerate(models):
        model.backbone.requires_grad_(joint)
        run = _TrainingRun(model, config, None)
        run.use_liger = bool(fused)
        state = model._new_state()
        run.consume([6, 9, 12], state, supervise=True, live_states=(state,))
        predicted = model._forward_tokens([8], state._copy(), last_only=True).detach()
        run.consume([8], state, supervise=True, live_states=(state,))
        assert run.steps == 1 and state._boundary_prediction
        torch.testing.assert_close(state._next_logits, predicted)
        start_loss = run.total_loss
        run.consume([10], state, supervise=True, live_states=(state,), phase="answer")
        expected = F.cross_entropy(predicted[0].float(), torch.tensor([10])).item()
        assert run.total_loss - start_loss == pytest.approx(expected)
        gradients = torch.autograd.grad(sum(run.losses), run.parameters, allow_unused=True,
                                        retain_graph=True)
        assert all(g is None or not torch.count_nonzero(g) for g in gradients)
        run.flush((state,), final=True)
        reports.append((gradients, run.total_loss, run.total_targets, run.steps))
    assert reports[0][1:] == pytest.approx(reports[1][1:])
    for a, b in zip(reports[0][0], reports[1][0], strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_checkpoint_defaults(tiny_model: TitansModel, tmp_path: Path, version: int) -> None:
    """Restore numeric defaults from both constructors, including older missing chunk metadata.

    Args:
        tiny_model: Offline decoder and tokenizer.
        tmp_path: Temporary checkpoint directory.
        version: Saved format to reconstruct.
    """
    model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer,
                           replace(tiny_model.config, chunk_size=16))
    path = model.save_pretrained(tmp_path / "checkpoint")
    metadata = json.loads((path / "titans.json").read_text())
    assert metadata["builtin_memory_configs"]["0"]["chunk_size"] == 16
    if version < 4:
        metadata["format_version"] = version
        metadata.pop("builtin_memory_configs")
        metadata.pop("mac_retrieval_policy")
        metadata["config"].pop("mac_segment_length")
        if version == 2:
            metadata["config"]["chunk_size"] = 7
        else:
            metadata["config"].pop("chunk_size")
        if version < 3:
            metadata.pop("connection_types")
        (path / "titans.json").write_text(json.dumps(metadata))
    for cls in (ContinualModel, TitansModel):
        restored = cls.load_pretrained(path)
        assert restored.config.mac_segment_length == (512 if version == 4 else 1)
        assert restored.memories["0"].config.chunk_size == {1: 1, 2: 7, 3: 1, 4: 16}[version]
    override = replace(tiny_model.config, mac_segment_length=37, attention_backend="sdpa")
    restored = ContinualModel.load_pretrained(path, config=override)
    assert restored.config.mac_segment_length == (512 if version == 4 else 1)
    assert TitansConfig().chunk_size == 16


def test_segment_snapshot_and_detach(tiny_model: TitansModel) -> None:
    """Check frozen retrieval weights, differentiable snapshots and a detach inside a segment.

    Args:
        tiny_model: Offline decoder providing the backbone and tokenizer.
    """
    config = replace(tiny_model.config, mac_segment_length=8, chunk_size=3, memory_layers=(0,))
    model = TitansModel(tiny_model.backbone, tiny_model.tokenizer, config)
    state = model._new_state()
    original = tuple(w.detach().clone() for w in state.memory[0].weights)
    model._forward_tokens([7, 8, 9, 10], state)
    cursor = state._retrieval[0]
    for initial, snap in zip(original, cursor.weights, strict=True):
        torch.testing.assert_close(initial, snap, rtol=0, atol=0)
    assert not torch.equal(state.memory[0].weights[0], cursor.weights[0])
    derivative, = torch.autograd.grad(cursor.weights[0].sum(), model.memories["0"].initial_weights[0])
    torch.testing.assert_close(derivative, torch.ones_like(derivative))
    reference = state._copy()
    state._detach()
    assert not any(t.requires_grad for t in state._retrieval[0].tensors())
    with torch.no_grad():
        for chunk in ([11, 12, 13, 14], [15]):
            model._forward_tokens(chunk, state)
            model._forward_tokens(chunk, reference)
            assert_stream(state, reference)
            if state.tokens_processed == 8:
                incoming = tuple(w.clone() for w in state.memory[0].weights)
                assert not state._retrieval
        for a, b in zip(incoming, state._retrieval[0].weights, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert state.memory[0].pending.shape[1] == 0


def test_explicit_fa3_rejects_unsupported(tiny_model: TitansModel) -> None:
    """Require an actionable error for explicit FA3 while auto accepts the same CPU input.

    Args:
        tiny_model: CPU decoder with a configurable attention backend.
    """
    tiny_model.config = replace(tiny_model.config, attention_backend="fa3")
    with torch.no_grad(), pytest.raises(RuntimeError, match="CUDA.*sdpa"):
        tiny_model._forward_tokens([7], tiny_model._new_state())
    assert RuntimeConfig().attention_backend == TitansConfig().attention_backend == "auto"
    with torch.no_grad():
        tiny_model.config = replace(tiny_model.config, attention_backend=RuntimeConfig().attention_backend)
        automatic = tiny_model._forward_tokens([7, 8, 9], tiny_model._new_state())
        tiny_model.config = replace(tiny_model.config, attention_backend="sdpa")
        reference = tiny_model._forward_tokens([7, 8, 9], tiny_model._new_state())
    torch.testing.assert_close(automatic, reference, rtol=0, atol=0)


def test_generic_checkpoint_records_effective_chunk(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Restore a nondefault write chunk supplied by an architecture factory.

    Args:
        tiny_model: Real offline decoder and tokenizer.
        tmp_path: Isolated checkpoint destination.
    """
    config = RuntimeConfig(compile_kernels=False)
    model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer, config)
    assert model.memories["0"].config.chunk_size == 16
    model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer, config,
                           memory_architecture=TitansArchitecture(replace(tiny_model.config, chunk_size=7)))
    path = model.save_pretrained(tmp_path / "factory")
    assert json.loads((path / "titans.json").read_text())["config"]["chunk_size"] == 7
    restored = TitansModel.load_pretrained(path)
    assert restored.config.chunk_size == restored.memories["0"].config.chunk_size == 7
    with torch.no_grad():
        ids = list(range(12))
        torch.testing.assert_close(model._forward_tokens(ids, model.start_stream()),
                                   restored._forward_tokens(ids, restored.start_stream()))
