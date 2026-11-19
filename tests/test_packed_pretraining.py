"""Single-stream packing, causal continuation and joint FP32 optimizer checks."""

import copy
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from examples.linear_memory import LinearMemory
from titans import (
    ConnectionExecution,
    ContinualModel,
    ResidualConnection,
    TitansModel,
    pretraining_batches,
)
from titans.runtime.state import StreamState
from titans.training.pretraining import _block_loss, _prepare_next_block, fp32_execution


class SmallBlockConnection(ResidualConnection):
    """Independent connection exercising its own execution boundary."""

    def execution(self, position: int, tokens: int, *, prefix_tokens: int,
                  segment_length: int, attention_window: int) -> ConnectionExecution:
        """Limit temporary attention tensors while retaining ordinary causal KV.

        Args:
            position: Current absolute position.
            tokens: Available real tokens.
            prefix_tokens: Persistent-prefix length.
            segment_length: Host segment setting, unused by this connection.
            attention_window: Ordinary sliding-attention span.

        Returns:
            A block of at most 64 real tokens with ordinary attention layout.
        """
        return ConnectionExecution(block_size=min(tokens, 64))


def small_connection(layer_index: int, hidden_size: int) -> SmallBlockConnection:
    """Create an independent residual connection for the generic-method checks.

    Args:
        layer_index: Decoder placement selected by the runtime.
        hidden_size: Backbone width supplied by the factory protocol.

    Returns:
        A fresh connection with its own execution declaration.
    """
    return SmallBlockConnection()


@pytest.fixture(params=["titans", "linear"])
def packed_model(tiny_model: TitansModel, request: pytest.FixtureRequest) -> ContinualModel:
    """Build small stable methods for long-block optimizer parity on CPU.

    Args:
        tiny_model: Offline decoder and tokenizer fixture.
        request: Method selection for exercising algorithm independence.

    Returns:
        An FP32 model using the scratch training policy in this offline test.
    """
    config = replace(tiny_model.config, memory_layers=(0,), memory_variant="legacy",
                     chunk_size=16, prefill_chunk_size=1)
    if request.param == "titans":
        model = TitansModel(tiny_model.backbone, tiny_model.tokenizer, config)
    else:
        model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer, config,
                               memory_architecture=LinearMemory, connection_factory=small_connection)
    model.backbone_kind = "scratch"
    return model


def assert_memory(actual: StreamState, expected: StreamState) -> None:
    """Compare all continuing memory buffers, retrievals, KV and positions.

    Args:
        actual: Instance memory produced by pretraining.
        expected: Independently advanced reference instance memory.
    """
    assert actual._position == expected._position
    assert actual.tokens_processed == expected.tokens_processed
    for current, reference in ((actual.memory, expected.memory),
                                (actual._retrieval, expected._retrieval)):
        assert current.keys() == reference.keys()
        for index, memory in current.items():
            for value, target in zip(memory.tensors(), reference[index].tensors(), strict=True):
                torch.testing.assert_close(value, target, atol=3e-6, rtol=3e-5)
    for index, cache in actual._attention.items():
        other = expected._attention[index]
        for value, target in ((cache.keys, other.keys), (cache.values, other.values),
                              (cache.positions, other.positions)):
            torch.testing.assert_close(value, target, atol=3e-6, rtol=3e-5)


def test_packing_document_and_batch_boundaries(tiny_model: TitansModel) -> None:
    """Preserve all shifted targets and EOS separators across the 4096 boundary.

    Args:
        tiny_model: Offline tokenizer with a known EOS token.
    """
    documents = ["Alice " * 4094, "lives in Paris", "Rome"]
    expected = []
    for document in documents:
        expected.extend(tiny_model.tokenizer.encode(document, add_special_tokens=False))
        expected.append(tiny_model.tokenizer.eos_token_id)
    batches = list(pretraining_batches(iter(documents), tiny_model.tokenizer))
    assert len(batches) == 2 and batches[0][0].shape == (1, 4096)
    assert torch.cat([x for x, _ in batches], 1).tolist()[0] == expected[:-1]
    assert torch.cat([y for _, y in batches], 1).tolist()[0] == expected[1:]
    assert batches[0][1][0, -1] == batches[1][0][0, 0]


@pytest.mark.parametrize("compiled", [False, True])
def test_native_normalization_in_optimized_runtime(tiny_model: TitansModel, compiled: bool) -> None:
    """Keep all decoder RMSNorm calls on the native modules in both kernel modes.

    Args:
        tiny_model: Offline Llama or Qwen decoder with the memory connection.
        compiled: Whether the remaining supported tensor operations use compilation.
    """
    model = tiny_model
    model.config = replace(model.config, use_liger=True, compile_kernels=compiled)
    calls: list[torch.nn.Module] = []

    def observe(module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> None:
        """Record native normalization calls without altering their inputs."""
        calls.append(module)

    norms = [norm for layer in model.backbone.model.layers
             for norm in (layer.input_layernorm, layer.post_attention_layernorm)]
    norms.append(model.backbone.model.norm)
    handles = [norm.register_forward_pre_hook(observe) for norm in norms]
    try:
        with torch.no_grad():
            state = model._new_state()
            calls.clear()
            model._forward_tokens([7, 8, 9], state)
        assert set(calls) == set(norms)
    finally:
        for handle in handles:
            handle.remove()


def test_pretrain_matches_two_manual_steps(packed_model: ContinualModel, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare two optimizer steps and every carried buffer with a direct reference.

    Args:
        packed_model: Titans MAC or an independent residual-memory implementation.
        tmp_path: Isolated checkpoint destination.
        monkeypatch: Observe normalized gradients immediately before clipping.
    """
    model = packed_model
    reference = copy.deepcopy(model)
    # prefill_chunk_size differs and must have no influence on this training path.
    reference.config = replace(reference.config, prefill_chunk_size=257)
    ids = (torch.arange(4096 + 17 + 1) % 20 + 3).reshape(1, -1)
    ids[:, 2000] = model.tokenizer.eos_token_id
    batches = [(ids[:, :4096], ids[:, 1:4097]), (ids[:, 4096:-1], ids[:, 4097:])]
    reference.train()
    reference.requires_grad_(True)
    parameters = reference.outer_parameters(train_backbone=True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    state = reference._new_state(prefix_hidden_only=True)
    total_loss = 0.0
    gradients: list[list[Tensor | None]] = []
    for inputs, labels in batches:
        if gradients:
            prefix = reference._new_state(prefix_hidden_only=True)
            for index, cache in state._attention.items():
                # These fixtures reach a completed MAC segment at token 4096,
                # so every layer starts with only its freshly computed prefix.
                fresh = prefix._attention[index]
                cache.keys, cache.values, cache.positions = fresh.keys, fresh.values, fresh.positions
        optimizer.zero_grad(set_to_none=True)
        logits = reference._forward_embeddings(reference.backbone.model.embed_tokens(inputs), state)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        total_loss += loss.item() * labels.numel()
        loss.backward()
        assert reference.persistent.grad is not None and reference.persistent.grad.abs().sum() > 0
        gradients.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        state._tokens += inputs.numel()
        state._detach()
    clip = torch.nn.utils.clip_grad_norm_
    observed = 0
    optimizer_steps = 0
    adam_step = torch.optim.AdamW.step

    def checked_step(optimizer: torch.optim.AdamW, closure: object = None) -> None:
        """Check that one global clipping operation precedes each CPU AdamW step.

        Args:
            optimizer: Joint backbone and method optimizer under test.
            closure: Unused by the fixed pretraining loop; must be absent.
        """
        nonlocal optimizer_steps
        assert closure is None and observed == optimizer_steps + 1
        gradients = [p.grad for group in optimizer.param_groups for p in group["params"]
                     if p.grad is not None]
        norm = torch.stack([g.norm() for g in gradients]).norm()
        assert norm <= 1.0 + 1e-6
        adam_step(optimizer)
        optimizer_steps += 1

    def checked_clip(parameters: Iterable[Tensor], max_norm: float, *,
                     error_if_nonfinite: bool = False) -> Tensor:
        """Compare gradients before clipping can obscure normalization errors.

        Args:
            parameters: Actual joint outer-optimizer parameters.
            max_norm: Configured clipping threshold.
            error_if_nonfinite: Whether nonfinite norms must fail.

        Returns:
            Norm returned by native PyTorch clipping.
        """
        nonlocal observed
        parameters = tuple(parameters)
        assert max_norm == 1.0 and error_if_nonfinite
        for parameter, expected in zip(parameters, gradients[observed], strict=True):
            torch.testing.assert_close(parameter.grad, expected, atol=3e-6, rtol=3e-5)
        observed += 1
        return clip(parameters, max_norm, error_if_nonfinite=error_if_nonfinite)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", checked_clip)
    monkeypatch.setattr(torch.optim.AdamW, "step", checked_step)
    previous = model.start_stream()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model.pretrain(iter(batches), output_dir=tmp_path / "trained")
    assert result.optimizer_steps == 2 and result.tokens_processed == 4113
    assert observed == optimizer_steps == 2
    assert result.mean_loss == pytest.approx(total_loss / 4113, rel=2e-6)
    assert result.checkpoint_path.is_dir()
    assert result.final_state is not None
    assert_memory(result.final_state, state)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        assert actual.dtype == torch.float32
    assert not model.training and not any(p.requires_grad for p in model.backbone.parameters())
    for memory in (*result.final_state.memory.values(), *result.final_state._retrieval.values()):
        assert all(t.grad_fn is None for t in memory.tensors())
    assert all(c.keys.grad_fn is None for c in result.final_state._attention.values())
    model.clone_state(result.final_state)
    with pytest.raises(ValueError, match="predates fit"):
        model.clone_state(previous)


def test_pretraining_boundary_preserves_memory(tiny_model: TitansModel) -> None:
    """Reset ordinary KV while preserving partial MAC segments and memory writes.

    Args:
        tiny_model: Offline decoder with two layers and learned persistent tokens.
    """
    model = TitansModel(tiny_model.backbone, tiny_model.tokenizer, replace(
        tiny_model.config, memory_layers=(0,), chunk_size=16, mac_segment_length=512,
    ))
    model.requires_grad_(True)
    state = model._new_state(prefix_hidden_only=True)
    model._forward_tokens([7, 8, 9, 10] * 4 + [11], state)
    state._detach()
    old = state._copy(detach=True)
    count = model.config.persistent_tokens
    with torch.no_grad():
        model.persistent.add_(0.01)
    _prepare_next_block(model, state)
    assert (state._position, state._tokens, state._documents) == (old._position, old._tokens, old._documents)
    assert state.memory[0].pending.shape[1] == 1
    for current, previous in ((state.memory, old.memory), (state._retrieval, old._retrieval)):
        for value, expected in zip(current[0].tensors(), previous[0].tensors(), strict=True):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
    assert state._attention[1].keys.shape[-2] == count
    assert state._attention[0].keys.shape[-2] == count + 34
    torch.testing.assert_close(state._attention[0].keys[:, :, count:],
                               old._attention[0].keys[:, :, count:], atol=0, rtol=0)
    fresh = model._new_state(prefix_hidden_only=True)
    for index, cache in state._attention.items():
        torch.testing.assert_close(cache.keys[:, :, :count], fresh._attention[index].keys, atol=0, rtol=0)
    assert state._attention[1].keys.grad_fn is not None
    logits = model._forward_tokens([12, 13, 14], state)
    gradient, = torch.autograd.grad(logits.square().sum(), model.persistent)
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_split_stream_and_future_causality(packed_model: ContinualModel) -> None:
    """Verify caller blocks, document separators and future inputs preserve causal meaning.

    Args:
        packed_model: Memory method and decoder running in FP32.
    """
    model = packed_model
    ids = (torch.arange(4098) % 20 + 3).reshape(1, -1)
    ids[:, 255] = model.tokenizer.eos_token_id
    with torch.no_grad():
        continuous, split = model._new_state(), model._new_state()
        expected = model._forward_embeddings(model.backbone.model.embed_tokens(ids), continuous)
        outputs = []
        start = 0
        for end in (15, 16, 17, 255, 256, 257, 511, 512, 513, 4096, 4098):
            outputs.append(model._forward_embeddings(model.backbone.model.embed_tokens(ids[:, start:end]), split))
            split._detach()
            start = end
        torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-5, rtol=1e-4)
        assert_memory(split, continuous)
        changed = ids.clone()
        changed[:, 33:] = 4
        altered = model._forward_embeddings(model.backbone.model.embed_tokens(changed), model._new_state())
        torch.testing.assert_close(altered[:, :33], expected[:, :33], atol=0, rtol=0)


def test_memory_write_gradient_and_detachment(packed_model: ContinualModel) -> None:
    """A late memory loss reaches early writes within a block and stops at detach.

    Args:
        packed_model: Differentiable memory method under the generic runtime.
    """
    model = packed_model
    early = torch.randn(1, 17, 16, requires_grad=True)
    late = torch.randn(1, 17, 16, requires_grad=True)
    state = model._new_state()
    model._forward_embeddings(early, state)
    model._forward_embeddings(late, state)
    loss = sum(t.square().sum() for t in state.memory[0].adaptable().values())
    gradient, = torch.autograd.grad(loss, early)
    assert gradient[:, 0].abs().sum() > 0 and gradient.isfinite().all()
    state._detach()
    model._forward_embeddings(late, state)
    loss = sum(t.square().sum() for t in state.memory[0].adaptable().values())
    detached, = torch.autograd.grad(loss, early, allow_unused=True)
    assert detached is None


@pytest.mark.parametrize("error", ["batch", "length", "dtype", "range", "shift", "empty", "short"])
def test_invalid_pretraining_input(tiny_model: TitansModel, tmp_path: Path, error: str) -> None:
    """Reject invalid packed data and restore training flags on failure.

    Args:
        tiny_model: Offline model assigned the scratch policy.
        tmp_path: Unused output destination.
        error: Invalid shape, target relationship or block schedule to exercise.
    """
    tiny_model.backbone_kind = "scratch"
    ids = torch.tensor([[7, 8, 9, 10]])
    inputs, labels = ids[:, :-1], ids[:, 1:]
    if error == "batch":
        inputs, labels = inputs.expand(2, -1), labels.expand(2, -1)
    elif error == "length":
        inputs = labels = torch.zeros(1, 4097, dtype=torch.long)
    elif error == "dtype":
        inputs = inputs.float()
    elif error == "range":
        labels = labels + 100
    elif error == "shift":
        labels = inputs
    batches = [] if error == "empty" else [(inputs, labels)]
    if error == "short":
        batches.append((inputs, labels))
    with pytest.raises(ValueError):
        tiny_model.pretrain(batches, output_dir=tmp_path / error)
    assert not (tmp_path / error).exists()
    assert not tiny_model.training
    assert not any(p.requires_grad for p in tiny_model.backbone.parameters())


def test_pretraining_failures_and_precision_scope(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Reject incompatible starts and restore FP32 scopes after an exception.

    Args:
        tiny_model: Initially pretrained-policy offline fixture.
        tmp_path: Existing and prospective checkpoint destinations.
    """
    with pytest.raises(ValueError, match="scratch"):
        tiny_model.pretrain([], output_dir=tmp_path / "pretrained")
    tiny_model.backbone_kind = "scratch"
    with pytest.raises(FileExistsError):
        tiny_model.pretrain([], output_dir=tmp_path)
    with pytest.raises(ValueError, match="learning_rate"):
        tiny_model.pretrain([], output_dir=tmp_path / "rate", learning_rate=float("nan"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="probe"):
            with fp32_execution(torch.device("cpu")):
                assert not torch.is_autocast_enabled("cpu")
                raise RuntimeError("probe")
        assert torch.is_autocast_enabled("cpu")


def test_labels_only_score_predictions(packed_model: ContinualModel) -> None:
    """Changing a lookahead label leaves all forwarded memory and KV identical.

    Args:
        packed_model: Algorithm supplied to the shared block-loss implementation.
    """
    model = packed_model
    ids = (torch.arange(34) % 20 + 3).reshape(1, -1)
    first, second = model._new_state(), model._new_state()
    labels = ids[:, 1:].clone()
    _block_loss(model, ids[:, :-1], labels, first)
    labels[:, -1] = 2
    _block_loss(model, ids[:, :-1], labels, second)
    assert_memory(first, second)


def test_loader_failure_invalidates_continuation(packed_model: ContinualModel, tmp_path: Path) -> None:
    """A loader failure after one update produces no success checkpoint.

    Args:
        packed_model: Scratch model for a short first update.
        tmp_path: Output location that must remain absent on failure.
    """
    def batches() -> Iterator[tuple[Tensor, Tensor]]:
        """Yield a batch and fail while fetching the next one.

        Yields:
            One valid short batch before a simulated loader exception.
        """
        yield torch.tensor([[7, 8, 9]]), torch.tensor([[8, 9, 10]])
        raise RuntimeError("loader failed")

    with pytest.raises(RuntimeError, match="loader failed"):
        packed_model.pretrain(batches(), output_dir=tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not packed_model.training
