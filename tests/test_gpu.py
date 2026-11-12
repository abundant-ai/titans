"""Hardware checks for FA3 and Liger, skipped explicitly on non-CUDA machines."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from titans import (
    Document,
    Question,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
)
from titans.cuda_graphs import GraphDecoder
from titans.gpu_ops import flash_attention3, liger_loss, liger_norm, liger_rotary
from titans.kernels import compiled_chunk_update
from titans.state import StreamState

from .reference import reference_chunk
from .test_memory import case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA Hopper GPU")


@pytest.mark.parametrize("prefix", [0, 4])
def test_fa3_local_attention_matches_sdpa(prefix: int) -> None:
    """FA3 plus persistent-prefix merging must equal the explicit causal/local mask."""
    torch.manual_seed(24)
    query = torch.randn(1, 4, 5, 64, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(1, 2, prefix + 7 + 5, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    qpos = torch.arange(prefix + 7, prefix + 12, device="cuda")
    kpos = torch.arange(prefix + 12, device="cuda")
    mask = (kpos[None] <= qpos[:, None]) & ((kpos[None] < prefix) | (kpos[None] > qpos[:, None] - 8))
    with torch.no_grad():
        expected = F.scaled_dot_product_attention(query, keys.repeat_interleave(2, 1),
                                                  values.repeat_interleave(2, 1),
                                                  attn_mask=mask[None, None], scale=0.125)
        actual = flash_attention3(query, keys, values, prefix_count=prefix,
                                  window=8, scale=0.125, prefix_only=False)
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


def test_liger_fused_loss_and_gradients() -> None:
    """Fused vocabulary loss must preserve its sum reduction and both input gradients."""
    torch.manual_seed(25)
    hidden = torch.randn(19, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(128, (19,), device="cuda")
    actual = liger_loss(weight, hidden, targets)
    expected = F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction="sum")
    actual_grad = torch.autograd.grad(actual, (hidden, weight))
    expected_grad = torch.autograd.grad(expected, (hidden, weight))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.1)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want, rtol=0.06, atol=0.06)


def test_liger_norm_preserves_gradients_and_input() -> None:
    """The Liger normalization path must not modify shared residual storage."""
    hidden = torch.randn(1, 7, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(128, device="cuda", requires_grad=True)
    original = hidden.detach().clone()
    actual = liger_norm(hidden, weight, 1e-5)
    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-5) * weight
    got = torch.autograd.grad(actual.square().sum(), (hidden, weight))
    want = torch.autograd.grad(expected.square().sum(), (hidden, weight))
    for first, second in zip(got, want, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)


def test_liger_rotary_matches_llama() -> None:
    """Fused grouped-query rotary encoding must match the host's explicit operation."""
    query = torch.randn(1, 4, 7, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 7, 64, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(1, 7, 32, device="cuda")
    angles = torch.cat((angles, angles), -1)
    cosine, sine = angles.cos().bfloat16(), angles.sin().bfloat16()
    expected = apply_rotary_pos_emb(query.clone(), key.clone(), cosine, sine)
    actual = liger_rotary(query, key, cosine, sine)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)


def test_cuda_compiled_memory_matches_equations() -> None:
    """Verify final states and training gradients from the actual CUDA JIT memory kernel."""
    raw = case(2, 7)
    weights = tuple(w.detach().float().cuda().requires_grad_() for w in raw[0])
    momentum = tuple(w.detach().float().cuda().requires_grad_() for w in raw[1])
    operands = tuple(t.detach().float().cuda().requires_grad_() for t in raw[2:])
    actual = compiled_chunk_update(weights, momentum, *operands)
    expected = reference_chunk(weights, momentum, *operands)
    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    inputs = weights + momentum + operands
    got_grad = torch.autograd.grad(sum(w.square().sum() for w in actual[0] + actual[1]), inputs)
    want_grad = torch.autograd.grad(sum(w.square().sum() for w in expected[0] + expected[1]), inputs)
    for got, want in zip(got_grad, want_grad, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-5)


def test_cuda_liger_posttraining_updates_write_rule(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Exercise the complete BF16 post-training path through Liger and compiled memory."""
    backbone = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, eos_token_id=2,
    )).to(device="cuda", dtype=torch.bfloat16)
    config = replace(tiny_model.config, compile_kernels=True, use_liger=True,
                     inference_backend="torch", attention_backend="auto")
    model = TitansModel(backbone, tiny_model.tokenizer, config)
    before = model.memories["0"].gates.weight.detach().clone()
    episode = TrainingEpisode("one", (Document("a", "Alice lives in Paris"),), (
        TrainingQuestion(1, Question("q", "Where Alice lives ?"), "Paris"),
    ))
    result = model.fit([episode], TrainingConfig(tmp_path / "cuda", max_tokens=64,
                                               sequence_length=32, tokens_per_step=32))
    assert result.optimizer_steps == 1
    assert not torch.equal(before, model.memories["0"].gates.weight)


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
    backbone = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, eos_token_id=2,
    )).to(device="cuda", dtype=torch.bfloat16)
    config = replace(tiny_model.config, compile_kernels=True, use_liger=True,
                     persistent_tokens=prefix, inference_backend="torch", attention_backend="fa3")
    model = TitansModel(backbone, tiny_model.tokenizer, config)
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
