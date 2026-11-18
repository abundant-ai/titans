"""Manually invoked CUDA inference checks; excluded from default pytest and CI."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from examples.linear_memory import LinearMemory
from tests.support.memory import case
from titans import (
    AnswerConfig,
    ConnectionFactory,
    ContinualModel,
    Document,
    Question,
    RuntimeConfig,
    TitansConfig,
    TitansModel,
    residual_connection,
    titans,
    titans_mac_connection,
)
from titans.methods.titans import NeuralMemory, chunk_update
from titans.optim.gpu_ops import liger_norm, liger_rotary
from titans.optim.kernels import compiled_chunk_update

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")


@pytest.mark.parametrize("connection", [titans_mac_connection, residual_connection], ids=["mac", "residual"])
@pytest.mark.parametrize("method", ["titans", "linear"])
@torch.no_grad()
def test_generic_cuda_inference_and_restore(
    tiny_model: TitansModel, method: str, connection: ConnectionFactory, tmp_path: Path,
) -> None:
    """Exercise both method protocols with streaming, isolated QA and checkpoint restore.

    Args:
        tiny_model: Offline Llama/Qwen architecture and tokenizer fixture.
        method: Titans or a structurally independent linear memory method.
        connection: Built-in wiring used by both the original and restored stream.
        tmp_path: Temporary directory for an untrained checkpoint round trip.
    """
    backbone = type(tiny_model.backbone)(type(tiny_model.backbone.config)(
        vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, eos_token_id=2,
    )).to(device="cuda", dtype=torch.bfloat16)
    settings = RuntimeConfig(memory_layers=(0, 1), attention_window=16, persistent_tokens=2,
                             compile_kernels=False, use_liger=True, prefill_chunk_size=16)
    factory = (titans(TitansConfig(memory_dim=8, memory_hidden_dim=16, chunk_size=4,
                                  compile_kernels=False)) if method == "titans" else LinearMemory)
    model = ContinualModel(backbone, tiny_model.tokenizer, settings, memory_factory=factory,
                           connection_factory=connection)
    parameters = [parameter.clone() for parameter in model.parameters()]
    state = model.start_stream()
    for index in range(3):
        model.ingest(Document(str(index), "Alice lives in Paris " * 8), state)
    assert state.documents_seen == 3 and state.tokens_processed > settings.attention_window
    source = model.clone_state(state)
    prediction = model.answer(Question("q", "Where Alice lives ?"), state,
                              AnswerConfig(max_new_tokens=3))
    assert isinstance(prediction.text, str)
    assert state.tokens_processed == source.tokens_processed
    for layer, component in state.memory.items():
        for actual, expected in zip(component.tensors(), source.memory[layer].tensors(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert bool(torch.isfinite(actual).all().item())
    for layer, cache in state._attention.items():
        retained = (2 * (state.tokens_processed % settings.mac_segment_length)
                    if connection is titans_mac_connection else settings.attention_window - 1)
        assert cache.keys.shape[2] == settings.persistent_tokens + retained
        torch.testing.assert_close(cache.keys, source._attention[layer].keys, rtol=0, atol=0)
    fork = model.clone_state(state)
    model.ingest(Document("new", "Alice lives in Rome"), fork)
    assert fork.documents_seen == 4 and state.documents_seen == 3
    assert any(not torch.equal(actual, old)
               for layer, component in fork.memory.items()
               for actual, old in zip(component.tensors(), state.memory[layer].tensors(), strict=True))
    for actual, expected in zip(model.parameters(), parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    checkpoint = model.save_pretrained(tmp_path / "untrained")
    restored = ContinualModel.load_pretrained(checkpoint, memory_factory=factory,
                                              device="cuda", dtype=torch.bfloat16)
    expected = model._forward_tokens([7, 10, 13], model.start_stream())
    actual = restored._forward_tokens([7, 10, 13], restored.start_stream())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_liger_norm_preserves_output_and_input() -> None:
    """The Liger normalization path must not modify shared residual storage."""
    hidden = torch.randn(1, 7, 128, device="cuda", dtype=torch.float32)
    weight = torch.randn(128, device="cuda")
    original = hidden.detach().clone()
    actual = liger_norm(hidden, weight, 1e-5)
    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)


@torch.no_grad()
def test_liger_rotary_matches_llama() -> None:
    """Fused grouped-query rotary outputs must match the host operation."""
    query = torch.randn(1, 4, 7, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 7, 64, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(1, 7, 32, device="cuda")
    angles = torch.cat((angles, angles), -1)
    cosine, sine = angles.cos().bfloat16(), angles.sin().bfloat16()
    expected = apply_rotary_pos_emb(query.clone(), key.clone(), cosine, sine)
    actual = liger_rotary(query, key, cosine, sine)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("paper", [False, True], ids=["legacy", "paper"])
@torch.no_grad()
def test_cuda_compiled_memory_matches_eager(paper: bool) -> None:
    """Compare compiled test-time transitions for both supported memory networks.

    Args:
        paper: Use the residual GELU/LayerNorm network instead of legacy SiLU.
    """
    raw = case(2, 7)
    weights = tuple(w.detach().float().cuda() for w in raw[0])
    momentum = tuple(w.detach().float().cuda() for w in raw[1])
    operands = tuple(t.detach().float().cuda() for t in raw[2:])
    actual = compiled_chunk_update(weights, momentum, *operands, paper=paper)
    expected = chunk_update(weights, momentum, *operands, paper=paper)
    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("variant", ["legacy", "paper"])
@torch.no_grad()
def test_qwen_memory_jit_matches_eager(dtype: torch.dtype, variant: str) -> None:
    """Compare Qwen-width prefill and decode against eager and an FP32 reference.

    BF16 projection kernels can round differently before their outputs enter the
    FP32 memory update. The residual LayerNorm network amplifies those differences
    over repeated writes. Keep strict FP32 checks and compare both BF16 paths to
    the same full-precision calculation with identical stored weights and inputs.

    Args:
        dtype: Projection storage and input dtype; fast weights remain FP32.
        variant: Legacy SiLU or paper residual GELU/LayerNorm memory network.
    """
    torch.compiler.reset()
    torch.manual_seed(38)
    settings = TitansConfig(memory_dim=64, memory_hidden_dim=128, compile_kernels=True,
                            memory_variant=variant)
    actual = NeuralMemory(1536, settings).to(device="cuda", dtype=dtype)
    expected = NeuralMemory(1536, replace(settings, compile_kernels=False)).to(
        device="cuda", dtype=dtype,
    )
    actual.gates.weight.normal_(std=0.01)
    actual.mix_gate.weight.normal_(std=0.01)
    expected.load_state_dict(actual.state_dict())
    reference = copy.deepcopy(expected).float()
    try:
        first, second, third = actual.initial_state(), expected.initial_state(), reference.initial_state()
        for count in (4, 256, 1, 1):
            hidden = torch.randn(1, count, 1536, device="cuda", dtype=dtype)
            got, want = actual(hidden, first), expected(hidden, second)
            exact = reference(hidden.float(), third)
            if dtype == torch.float32:
                torch.testing.assert_close(got, want, rtol=0.002, atol=0.0002)
                for a, b in zip(first.tensors(), second.tensors(), strict=True):
                    torch.testing.assert_close(a, b, rtol=0.002, atol=0.0002)
            else:
                torch.testing.assert_close(got, want, rtol=0.04, atol=0.002)
                # Two percent is a norm-wise BF16 error budget, not a per-element
                # relative error at near-zero weights. The FP32 and fixed-operand
                # checks above separately constrain the transition mathematics.
                for output in (got, want):
                    assert (output.float() - exact).norm() <= 0.02 * exact.norm().clamp_min(1e-12)
                for state in (first, second):
                    for a, b in zip(state.tensors(), third.tensors(), strict=True):
                        assert a.isfinite().all()
                        assert (a.float() - b).norm() <= 0.02 * b.norm().clamp_min(1e-12)
                for a, b in zip(first.weights, second.weights, strict=True):
                    assert (a - b).abs().max() < 0.001
    finally:
        torch.compiler.reset()
