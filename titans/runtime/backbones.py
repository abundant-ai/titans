"""Fixed backbone choices; memory algorithms and connections are independent."""

from typing import Literal

from transformers import LlamaConfig, PretrainedConfig

PRETRAINED_BACKBONE = "Qwen/Qwen2.5-1.5B-Instruct"
LLAMA_TOKENIZER = "NousResearch/Llama-2-7b-hf"
BackboneKind = Literal["scratch", "qwen"]


def scratch_config() -> LlamaConfig:
    """Return the fixed 24-block Llama-style decoder configuration.

    The paper's 340M label identifies a scale, not an exact parameter budget for
    every memory plugin. FFN width uses the Llama multiple-of-256 convention.

    Returns:
        A fresh configuration with no caller-supplied architecture overrides.
    """
    return LlamaConfig(
        vocab_size=32000, hidden_size=1024, intermediate_size=2816,
        num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16,
        hidden_act="silu", max_position_embeddings=4096, rms_norm_eps=1e-5,
        rope_theta=10000.0, rope_scaling=None, attention_bias=False, mlp_bias=False,
        attention_dropout=0.0, tie_word_embeddings=False, initializer_range=0.02,
        bos_token_id=1, eos_token_id=2, pad_token_id=None,
    )


def validate_backbone(config: PretrainedConfig) -> BackboneKind:
    """Reject decoder architecture overrides, including on checkpoint restoration.

    Args:
        config: Actual backbone configuration before attaching memory.

    Returns:
        The permitted backbone family, which determines the default training mode.

    Raises:
        ValueError: The configuration differs from either fixed decoder.
    """
    fields = (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "hidden_act",
        "max_position_embeddings", "rms_norm_eps", "rope_theta", "rope_scaling",
        "attention_dropout", "tie_word_embeddings", "initializer_range",
    )
    if config.model_type == "llama":
        expected = scratch_config()
        fields += ("attention_bias", "mlp_bias")
        if (all(getattr(config, key, None) == getattr(expected, key) for key in fields)
                and getattr(config, "head_dim", 64) == 64):
            return "scratch"
    if config.model_type == "qwen2":
        expected_values = dict(zip(fields, (
            151936, 1536, 8960, 28, 12, 2, "silu", 32768, 1e-6, 1000000.0,
            None, 0.0, True, 0.02,
        ), strict=True))
        if (all(getattr(config, key, None) == value for key, value in expected_values.items())
                and not getattr(config, "use_sliding_window", False)
                and getattr(config, "head_dim", 128) == 128):
            return "qwen"
    raise ValueError("Backbone must be the fixed scratch Llama decoder or " + PRETRAINED_BACKBONE)
