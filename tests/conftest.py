"""Offline fixtures; no model downloads or accelerator are required."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from titans import TitansConfig, TitansModel


@pytest.fixture
def tiny_model() -> TitansModel:
    """Build a deterministic two-layer Llama and a small local word tokenizer.

    Returns:
        A real HF Llama integration with random weights and two memory branches.
    """
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
    backbone = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(words), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        eos_token_id=2, pad_token_id=0, attention_dropout=0.0,
    ))
    return TitansModel(backbone, tokenizer, TitansConfig(
        memory_dim=4, memory_hidden_dim=6, memory_layers=(0, 1), chunk_size=4,
        attention_window=8, persistent_tokens=2,
        compile_kernels=False,
    ))
