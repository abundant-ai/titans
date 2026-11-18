"""Manually invoked H100 checks for the fixed scratch decoder and joint training."""

import gc
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from titans import Document, TitansConfig, TitansModel, TrainingConfig, TrainingEpisode


def run_pretraining_checks() -> dict[str, object]:
    """Compare full-size eager/optimized gradients, then run three joint AdamW steps.

    Returns:
        Measured parameter counts, gradient parity, training results and peak memory.
        Raises an assertion if a gradient, parameter update or checkpoint is wrong.
    """
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = TitansConfig(compile_kernels=False, use_liger=False, persistent_tokens=2,
                          inference_backend="torch", attention_backend="sdpa")
    print("Building fixed scratch decoder with the real Llama 2 tokenizer", flush=True)
    eager = TitansModel.from_scratch(config, device="cuda", seed=17)
    eager.backbone.requires_grad_(True)
    tokens = eager.tokenizer.encode("Alice lives in Paris. Bob lives in Rome.", add_special_tokens=False)[:8]
    eager.train()
    with sdpa_kernel(SDPBackend.MATH):
        state = eager._new_state()
        output = eager._forward_tokens(tokens, state)
        memory_loss = sum(t.square().sum() for t in state.memory[0].adaptable().values())
        inner_gradient, = torch.autograd.grad(memory_loss, eager.backbone.model.embed_tokens.weight,
                                             retain_graph=True)
        assert torch.isfinite(inner_gradient).all() and inner_gradient.norm() > 0
        loss = F.cross_entropy(output[:, :-1].reshape(-1, output.shape[-1]),
                               torch.tensor(tokens[1:], device="cuda"))
        loss.backward()
    expected = {name: p.grad.detach().cpu() for name, p in eager.named_parameters() if p.grad is not None}
    assert any(name.startswith("memories.") for name in expected)
    expected_loss = loss.item()
    parameter_count = sum(p.numel() for p in eager.backbone.parameters())
    total_count = sum(p.numel() for p in eager.parameters())
    del eager, state, output, loss, inner_gradient, memory_loss
    gc.collect()
    torch.cuda.empty_cache()
    print("Checking compiled memory, Liger kernels and checkpointed decoder gradients", flush=True)
    optimized = TitansModel.from_scratch(replace(config, compile_kernels=True, use_liger=True),
                                         device="cuda", seed=17)
    optimized.backbone.requires_grad_(True)
    optimized._checkpoint_backbone = True
    optimized.train()
    with sdpa_kernel(SDPBackend.MATH):
        state = optimized._new_state()
        output = optimized._forward_tokens(tokens, state)
        loss = F.cross_entropy(output[:, :-1].reshape(-1, output.shape[-1]),
                               torch.tensor(tokens[1:], device="cuda"))
        loss.backward()
    torch.testing.assert_close(loss.detach().cpu(), torch.tensor(expected_loss), rtol=2e-4, atol=2e-5)
    checked = 0
    max_error = 0.0
    for name, parameter in optimized.named_parameters():
        assert (parameter.grad is not None) == (name in expected), name
        if parameter.grad is None:
            continue
        actual = parameter.grad.detach().cpu()
        assert torch.isfinite(actual).all(), name
        max_error = max(max_error, (actual - expected[name]).abs().max().item())
        torch.testing.assert_close(actual, expected[name], rtol=3e-3, atol=3e-5, msg=name)
        checked += 1
    del expected, state, output, loss, actual
    optimized.zero_grad(set_to_none=True)
    watched_names = (
        "backbone.model.embed_tokens.weight", "backbone.model.layers.0.self_attn.q_proj.weight",
        "backbone.model.layers.0.mlp.up_proj.weight", "backbone.model.norm.weight",
        "backbone.lm_head.weight", "memories.0.gates.weight",
    )
    before = {name: p.detach().cpu().clone() for name, p in optimized.named_parameters()
              if name in watched_names}
    print("Running three joint steps with fused Liger loss and truncated backpropagation", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        result = optimized.fit(
            [TrainingEpisode("smoke", (Document("d", "Alice lives in Paris. Bob lives in Rome. " * 8),))],
            TrainingConfig(Path(directory) / "checkpoint", max_tokens=12, sequence_length=4,
                           tokens_per_step=4, objective="language_modeling"),
        )
        assert result.optimizer_steps == 3 and result.tokens_processed == 12
        for name, parameter in optimized.named_parameters():
            if name in before:
                assert not torch.equal(parameter.detach().cpu(), before[name]), name
        assert not any(p.requires_grad for p in optimized.backbone.parameters())
        with torch.no_grad():
            prediction = optimized._forward_tokens(tokens, optimized._new_state()).detach().cpu()
        # Restore the same learned weights with eager execution for independent parity.
        restored = TitansModel.load_pretrained(result.checkpoint_path, device="cuda")
        restored.config = replace(restored.config, compile_kernels=False, use_liger=False)
        for memory in restored.memories.values():
            memory.config = replace(memory.config, compile_kernels=False, use_liger=False)
        assert restored.backbone_kind == "scratch"
        with torch.no_grad():
            actual_prediction = restored._forward_tokens(tokens, restored._new_state()).cpu()
        torch.testing.assert_close(prediction, actual_prediction, rtol=3e-3, atol=3e-4)
        report = asdict(result)
        report["checkpoint_path"] = "temporary checkpoint (restoration verified, then removed)"
    return {
        "backbone_parameters": parameter_count, "total_parameters": total_count,
        "gradient_tensors_compared": checked, "maximum_gradient_absolute_error": max_error,
        "eager_loss": expected_loss, "joint_fit": report,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "checks": ["fixed full-size decoder", "Llama 2 tokenizer", "inner write to theta gradient",
                   "eager/compiled/Liger gradient parity", "three joint AdamW steps",
                   "truncation", "theta and phi changed", "checkpoint restore", "refrozen backbone"],
    }
