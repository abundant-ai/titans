"""Manually invoked CUDA inference checks; excluded from default pytest and CI."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from examples.linear_memory import LinearMemory
from tests.support.memory import case
from titans import (
    AnswerConfig,
    ContinualModel,
    Document,
    Question,
    RuntimeConfig,
    TitansConfig,
    TitansModel,
    titans,
)
from titans.methods.titans import chunk_update
from titans.optim.cuda_graphs import GraphDecoder
from titans.optim.gpu_ops import flash_attention3, liger_norm, liger_rotary
from titans.optim.kernels import compiled_chunk_update
from titans.runtime.connections import residual_connection
from titans.runtime.state import StreamState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")


@pytest.mark.parametrize("method", ["titans", "linear"])
@torch.no_grad()
def test_generic_cuda_inference_and_restore(
    tiny_model: TitansModel, method: str, tmp_path: Path,
) -> None:
    """Exercise both method protocols with streaming, isolated QA and checkpoint restore.

    Args:
        tiny_model: Offline Llama/Qwen architecture and tokenizer fixture.
        method: Titans or a structurally independent linear memory method.
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
    model = ContinualModel(backbone, tiny_model.tokenizer, settings, memory_factory=factory)
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
        expected_pairs = 2 * (state.tokens_processed % settings.mac_segment_length)
        assert cache.keys.shape[2] == settings.persistent_tokens + expected_pairs
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


@pytest.mark.parametrize("prefix", [0, 4])
@pytest.mark.parametrize("heads,kv_heads,head_dim", [(4, 2, 64), (12, 2, 128)])
def test_fa3_local_attention_matches_sdpa(
    prefix: int, heads: int, kv_heads: int, head_dim: int,
) -> None:
    """Compare FA3 with a causal mask for Llama and Qwen attention dimensions.

    Args:
        prefix: Number of persistent tokens.
        heads: Query head count.
        kv_heads: Key/value head count.
        head_dim: Channels per head, including Qwen2.5's 128.
    """
    torch.manual_seed(24)
    query = torch.randn(1, heads, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(1, kv_heads, prefix + 7 + 5, head_dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    qpos = torch.arange(prefix + 7, prefix + 12, device="cuda")
    kpos = torch.arange(prefix + 12, device="cuda")
    mask = (kpos[None] <= qpos[:, None]) & ((kpos[None] < prefix) | (kpos[None] > qpos[:, None] - 8))
    with torch.no_grad():
        expected = F.scaled_dot_product_attention(query, keys.repeat_interleave(heads // kv_heads, 1),
                                                  values.repeat_interleave(heads // kv_heads, 1),
                                                  attn_mask=mask[None, None], scale=head_dim**-0.5)
        actual = flash_attention3(query, keys, values, prefix_count=prefix,
                                  window=8, scale=head_dim**-0.5, prefix_only=False)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)


def test_fa3_persistent_initialization_is_causal() -> None:
    """The learned prefix's own forward pass must retain causal attention."""
    query = torch.randn(1, 4, 4, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 4, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    with torch.no_grad():
        expected = F.scaled_dot_product_attention(query, key.repeat_interleave(2, 1),
                                                  value.repeat_interleave(2, 1), is_causal=True)
        actual = flash_attention3(query, key, value, prefix_count=4, window=8,
                                  scale=0.125, prefix_only=True)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)


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


@torch.no_grad()
def test_cuda_compiled_memory_matches_eager() -> None:
    """Compare compiled test-time memory transitions with their eager implementation."""
    raw = case(2, 7)
    weights = tuple(w.detach().float().cuda() for w in raw[0])
    momentum = tuple(w.detach().float().cuda() for w in raw[1])
    operands = tuple(t.detach().float().cuda() for t in raw[2:])
    actual = compiled_chunk_update(weights, momentum, *operands)
    expected = chunk_update(weights, momentum, *operands)
    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def assert_stream_matches(actual: StreamState, expected: StreamState) -> None:
    """Compare all mutable graph state with the ordinary numerical forward path.

    Args:
        actual: Stream advanced by CUDA graphs.
        expected: Independently advanced reference stream.
    """
    assert actual._position == expected._position
    assert actual._tokens == expected._tokens
    assert actual._documents == expected._documents
    for index, memory in actual._memory.items():
        reference = expected._memory[index]
        for first, second in zip(
            (*memory.weights, *memory.momentum, memory.pending, memory.conv_tail),
            (*reference.weights, *reference.momentum, reference.pending, reference.conv_tail),
            strict=True,
        ):
            torch.testing.assert_close(first, second, rtol=2e-4, atol=2e-5)
    for index, cache in actual._attention.items():
        reference_cache = expected._attention[index]
        torch.testing.assert_close(cache.keys, reference_cache.keys, rtol=0, atol=0)
        torch.testing.assert_close(cache.values, reference_cache.values, rtol=0, atol=0)
        torch.testing.assert_close(cache.positions, reference_cache.positions, rtol=0, atol=0)
    torch.testing.assert_close(actual._next_logits, expected._next_logits, rtol=0, atol=0)


@pytest.mark.parametrize("prefix", [0, 2])
@torch.no_grad()
def test_cuda_graphs_preserve_streams_and_chunk_commits(tiny_model: TitansModel, prefix: int) -> None:
    """Replay all write phases and read-only branches, including stream switches and fallback.

    Args:
        tiny_model: Offline tokenizer and small memory settings.
        prefix: Number of globally visible persistent tokens.
    """
    torch.compiler.reset()
    backbone = type(tiny_model.backbone)(type(tiny_model.backbone.config)(
        vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, eos_token_id=2,
    )).to(device="cuda", dtype=torch.bfloat16)
    config = replace(tiny_model.config, compile_kernels=True, use_liger=True,
                     persistent_tokens=prefix, inference_backend="torch", attention_backend="fa3")
    model = TitansModel(backbone, tiny_model.tokenizer, config,
                        connection_factory=residual_connection)
    decoder = GraphDecoder(model)
    state = model._new_state()
    reference = state._copy(detach=True)
    initial = [7, 12, 5, 9, 3, 1, 10, 8, 15, 7, 6]
    decoder.forward(torch.tensor([initial], device="cuda"), state)
    model._forward_tokens(initial, reference, last_only=True)
    source = state._copy(detach=True)
    untouched = source._copy(detach=True)
    alternate = state._copy(detach=True)
    alternate_reference = reference._copy(detach=True)
    for step in range(32):
        write = step % 7 != 0
        selected, expected = (state, reference) if step % 3 else (alternate, alternate_reference)
        token = 7 + step % 10
        decoder.forward(torch.tensor([[token]], device="cuda"), selected, write=write)
        model._forward_tokens([token], expected, write=write, last_only=True)
        assert_stream_matches(selected, expected)
        assert_stream_matches(source, untouched)
    # A prefill after graph replay must stop borrowing workspace storage.
    decoder.forward(torch.tensor([[3, 8, 11]], device="cuda"), state)
    model._forward_tokens([3, 8, 11], reference, last_only=True)
    assert_stream_matches(state, reference)
    assert_stream_matches(alternate, alternate_reference)
    assert decoder.replays == 32
    assert all((phase, True) in decoder.graphs for phase in range(config.chunk_size))


@torch.no_grad()
def test_qwen_memory_jit_with_sglang_determinism() -> None:
    """Compile real Qwen-width memory under SGLang's deterministic operator replacements.

    Small gate projections previously triggered Inductor's addmm padding benchmark,
    whose alpha/beta keywords are unsupported by SGLang 0.5.7. Cover prefix,
    prefill and decode shapes while retaining compiled kernels and seeded serving.
    """
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )

    from titans import TitansConfig
    from titans.methods.titans import NeuralMemory

    torch.compiler.reset()
    torch.manual_seed(38)
    settings = TitansConfig(memory_dim=64, memory_hidden_dim=128, compile_kernels=True)
    actual = NeuralMemory(1536, settings).to(device="cuda", dtype=torch.bfloat16)
    expected = NeuralMemory(1536, replace(settings, compile_kernels=False)).to(
        device="cuda", dtype=torch.bfloat16,
    )
    actual.gates.weight.normal_(std=0.01)
    actual.mix_gate.weight.normal_(std=0.01)
    expected.load_state_dict(actual.state_dict())
    try:
        enable_batch_invariant_mode()
        first, second = actual.initial_state(), expected.initial_state()
        for count in (4, 256, 1, 1):
            hidden = torch.randn(1, count, 1536, device="cuda", dtype=torch.bfloat16)
            got, want = actual(hidden, first), expected(hidden, second)
            torch.testing.assert_close(got, want, rtol=0.04, atol=0.002)
            for got_weight, want_weight in zip(first.weights, second.weights, strict=True):
                torch.testing.assert_close(got_weight, want_weight, rtol=0.002, atol=0.0002)
    finally:
        disable_batch_invariant_mode()
        torch.compiler.reset()
