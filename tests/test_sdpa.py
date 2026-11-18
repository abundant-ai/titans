"""The cached decoder must implement the same softmax as attention over the full history."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from titans import TitansModel


@pytest.mark.parametrize("prefix,window", [(0, 1), (2, 1), (2, 4), (2, 16)])
def test_cached_sdpa_matches_explicit_softmax(tiny_model: TitansModel, prefix: int, window: int) -> None:
    """Compare outputs, input gradients and Q/K/V gradients across unequal query and cache lengths.

    Args:
        tiny_model: Offline Llama or Qwen with grouped KV heads and two prefix positions.
        prefix: Number of globally visible prefix positions, including zero.
        window: Local span, ranging from current-token-only to full causal attention.
    """
    tiny_model.config = replace(tiny_model.config, persistent_tokens=prefix)
    layer = tiny_model.backbone.model.layers[0].double().requires_grad_(True)
    attn = layer.self_attn
    length = prefix + 10
    positions = torch.arange(length)
    hidden = torch.randn(1, length, attn.q_proj.in_features, dtype=torch.double, requires_grad=True)
    # Nonzero angles also exercise the grouped keys retained from previous forwards.
    angles = positions[:, None] * torch.arange(1, attn.head_dim // 2 + 1, dtype=torch.double) / 9
    angles = angles.repeat(1, 2).unsqueeze(0)
    cos, sin = angles.cos(), angles.sin()
    blocks = [prefix, 3, 1, 5, 1]
    cache = None
    result = []
    start = 0
    for n in filter(None, blocks):
        end = start + n
        out, cache = tiny_model._local_attention(
            layer, hidden[:, start:end], positions[start:end], (cos[:, start:end], sin[:, start:end]),
            cache, end, window=window,
        )
        result.append(out)
        start = end
    actual = torch.cat(result, dim=1)

    shape = (1, length, -1, attn.head_dim)
    q, k = apply_rotary_pos_emb(attn.q_proj(hidden).view(shape).transpose(1, 2),
                                attn.k_proj(hidden).view(shape).transpose(1, 2), cos, sin)
    v = attn.v_proj(hidden).view(shape).transpose(1, 2)
    kv_head = torch.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
    rows = []
    for i in range(length):
        visible = [j for j in range(i + 1) if j < prefix or i - j < window]
        keys = k[:, kv_head][:, :, visible]
        values = v[:, kv_head][:, :, visible]
        scores = (q[:, :, i:i + 1] * keys).sum(-1) * attn.scaling
        rows.append((scores.softmax(-1).unsqueeze(-1) * values).sum(2))
    expected = attn.o_proj(torch.stack(rows, dim=1).reshape_as(hidden))

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    targets = (hidden, *attn.parameters())
    upstream = torch.randn_like(expected)
    for observed, reference in zip(torch.autograd.grad(actual, targets, upstream),
                                  torch.autograd.grad(expected, targets, upstream), strict=True):
        torch.testing.assert_close(observed, reference, atol=1e-11, rtol=1e-9)


def test_bf16_checkpoint_preserves_rotary_frequencies(tiny_model: TitansModel, tmp_path: Path) -> None:
    """A BF16-cast backbone must use identical rotary phases after checkpoint restoration.

    Args:
        tiny_model: Offline Llama or Qwen instance with FP32 rotary frequencies.
        tmp_path: Temporary directory for a checkpoint and its legacy equivalent.
    """
    canonical = tiny_model.backbone.model.rotary_emb.inv_freq.detach().clone()
    tiny_model.to(dtype=torch.bfloat16)
    path = tiny_model.save_pretrained(tmp_path / "checkpoint")
    rounded = tiny_model.backbone.model.rotary_emb.inv_freq.float()
    assert not torch.equal(rounded, canonical)
    restored = TitansModel.load_pretrained(path, dtype=torch.bfloat16)
    torch.testing.assert_close(restored.backbone.model.rotary_emb.inv_freq.float(), rounded, atol=0, rtol=0)

    adapter = load_file(path / "memory.safetensors")
    del adapter["rotary_inv_freq"]
    save_file(adapter, path / "memory.safetensors")
    legacy = TitansModel.load_pretrained(path, dtype=torch.bfloat16)
    torch.testing.assert_close(legacy.backbone.model.rotary_emb.inv_freq.float(), canonical, atol=0, rtol=0)
