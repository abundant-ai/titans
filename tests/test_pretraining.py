"""Joint outer learning, fixed backbone selection and checkpoint contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM, Qwen2Config

from titans import (
    Document,
    GradientUpdate,
    Question,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
    titans_mac_connection,
)
from titans.runtime.backbones import scratch_config, validate_backbone


def test_fixed_backbone_configuration() -> None:
    """Validate exact decoder settings without allocating hundreds of millions of weights."""
    config = scratch_config()
    assert (config.num_hidden_layers, config.hidden_size, config.num_attention_heads) == (24, 1024, 16)
    assert (config.intermediate_size, config.num_key_value_heads, config.vocab_size) == (2816, 16, 32000)
    assert validate_backbone(config) == "scratch"
    config.num_hidden_layers = 12
    with pytest.raises(ValueError, match="fixed scratch"):
        validate_backbone(config)
    with pytest.raises(ValueError, match="fixed scratch"):
        validate_backbone(Qwen2Config(hidden_size=128))
    with torch.device("meta"):
        backbone = LlamaForCausalLM(scratch_config())
    assert sum(p.numel() for p in backbone.parameters()) == 373867520


@pytest.mark.parametrize("model_id", ["meta-llama/Llama-2-7b-hf", "Qwen/Qwen2.5-0.5B-Instruct", "/tmp/backbone"])
def test_pretrained_selection_is_locked(model_id: str) -> None:
    """Reject unsupported names before any download.

    Args:
        model_id: A disallowed pretrained model or arbitrary local directory.
    """
    with pytest.raises(ValueError, match="Only Qwen"):
        TitansModel.from_pretrained(model_id, local_files_only=True)


def test_inner_write_has_backbone_gradient(tiny_model: TitansModel) -> None:
    """Memory writes retain a gradient path into the backbone's token features.

    Args:
        tiny_model: Offline test-only decoder with trainable memory.
    """
    tiny_model.backbone.requires_grad_(True)
    state = tiny_model._new_state()
    tiny_model._forward_tokens([10, 11, 12, 13], state)
    loss = sum(w.square().sum() for w in state.memory[0].adaptable().values())
    gradient, = torch.autograd.grad(loss, tiny_model.backbone.model.embed_tokens.weight)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    state._detach()
    assert all(t.grad_fn is None for m in state.memory.values() for t in m.tensors())


@pytest.mark.parametrize("compiled", [False, True])
def test_joint_fit_and_restore(tiny_model: TitansModel, tmp_path: Path, compiled: bool) -> None:
    """Fit across truncation boundaries, update theta/phi and restore learned weights.

    Args:
        tiny_model: Offline Llama/Qwen fixture; architecture lock is tested separately.
        tmp_path: Output directory for the library checkpoint.
        compiled: Exercise compiled functions and activation checkpointing together.
    """
    torch.compiler.reset()
    model = TitansModel(tiny_model.backbone, tiny_model.tokenizer,
                        replace(tiny_model.config, compile_kernels=compiled, chunk_size=1),
                        connection_factory=titans_mac_connection)
    model.backbone_kind = "scratch"
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = model.fit([TrainingEpisode("s", (Document("d", "Alice lives in Paris " * 8),))],
                       TrainingConfig(tmp_path / "joint", max_tokens=12, sequence_length=4,
                                      tokens_per_step=4, objective="language_modeling",
                                      activation_checkpointing=compiled))
    assert result.optimizer_steps == 3 and result.tokens_processed == 12
    for name in ("backbone.model.embed_tokens.weight", "backbone.model.layers.0.self_attn.q_proj.weight",
                 "backbone.model.layers.0.mlp.up_proj.weight", "backbone.model.norm.weight",
                 "memories.0.gates.weight"):
        assert not torch.equal(dict(model.named_parameters())[name], before[name]), name
    assert not any(p.requires_grad for p in model.backbone.parameters())
    restored = TitansModel.load_pretrained(result.checkpoint_path)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], weight, rtol=0, atol=0)
    state = restored.ingest(Document("d", "Alice lives in Rome"))
    restored.answer(Question("q", "Where Alice lives ?"), state)


def test_custom_joint_policy(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Custom checkpoint policies receive theta separately and work across multiple updates.

    Args:
        tiny_model: Offline frozen fixture to explicitly unfreeze for fitting.
        tmp_path: Checkpoint output root.
    """
    before = tiny_model.backbone.model.embed_tokens.weight.detach().clone()
    episode = TrainingEpisode("s", (Document("d", "Alice lives in Paris"),) * 2, (
        TrainingQuestion(1, Question("a", "Where Alice lives ?"), "Paris"),
        TrainingQuestion(2, Question("b", "Where Alice lives ?"), "Rome"),
    ))
    result = tiny_model.fit([episode], TrainingConfig(
        tmp_path / "custom", max_tokens=200, sequence_length=100, objective="document_qa",
        freeze_backbone=False,
    ), update=GradientUpdate(targets="joint"))
    assert result.optimizer_steps == 2
    assert not torch.equal(before, tiny_model.backbone.model.embed_tokens.weight)
    assert not any(p.requires_grad for p in tiny_model.backbone.parameters())


def test_joint_failure_refreezes_backbone(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Invalid supervision leaves the model safe for ordinary frozen-backbone inference.

    Args:
        tiny_model: Offline fixture.
        tmp_path: Unused checkpoint destination.
    """
    episode = TrainingEpisode("bad", (), (TrainingQuestion(1, Question("q", "Where?"), "Paris"),))
    with pytest.raises(ValueError, match="after_documents"):
        tiny_model.fit([episode], TrainingConfig(tmp_path / "bad", max_tokens=10, freeze_backbone=False))
    assert not any(p.requires_grad for p in tiny_model.backbone.parameters())
