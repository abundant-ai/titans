"""Offline fixtures; no model downloads or accelerator are required."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from titans import TitansConfig, TitansModel, residual_connection


@pytest.fixture(params=["llama", "qwen2"])
def tiny_model(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> TitansModel:
    """Build both supported architectures with a deterministic local tokenizer.

    Args:
        request: Architecture parameter supplied by pytest.
        monkeypatch: Permit small offline decoders only inside tests.

    Returns:
        A real HF backbone integration with random weights and two memory branches.
    """
    def test_backbone_kind(config: LlamaConfig | Qwen2Config) -> str:
        """Keep offline fixtures in frozen-backbone mode.

        Args:
            config: Small test-only decoder configuration.

        Returns:
            The pretrained training policy for existing regression tests.
        """
        return "qwen"

    monkeypatch.setattr("titans.runtime.model.validate_backbone", test_backbone_kind)
    torch.set_num_threads(1)
    torch.manual_seed(12)
    words = ["[PAD]", "[UNK]", "[EOS]", "Document", ":", "Question", "Answer",
             "a", "b", "c", "Alice", "lives", "in", "Paris", "Rome", "Where", "?",
             "now", "old", "new", "one", "two", "three", "unknown", "[", "]"]
    vocabulary = {word: index for index, word in enumerate(words)}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]",
                                        unk_token="[UNK]", eos_token="[EOS]")
    config_class = LlamaConfig if request.param == "llama" else Qwen2Config
    model_class = LlamaForCausalLM if request.param == "llama" else Qwen2ForCausalLM
    backbone = model_class(config_class(
        vocab_size=len(words), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        eos_token_id=2, pad_token_id=0, attention_dropout=0.0,
        tie_word_embeddings=request.param == "qwen2",
    ))
    if request.param == "qwen2":
        # Nonzero biases make dropping Qwen Q/K/V biases observable in parity checks.
        with torch.no_grad():
            for layer in backbone.model.layers:
                for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                                   layer.self_attn.v_proj):
                    projection.bias.normal_(std=0.1)
    return TitansModel(backbone, tokenizer, TitansConfig(
        memory_dim=4, memory_hidden_dim=6, memory_layers=(0, 1), chunk_size=4,
        attention_window=8, persistent_tokens=2,
        compile_kernels=False,
    ), connection_factory=residual_connection)
