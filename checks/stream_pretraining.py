"""Manual H100 FP32 forward/backward checks for continuous packed pretraining."""

import time
from dataclasses import replace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import Tensor
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from titans import TitansConfig, TitansModel
from titans.runtime.backbones import scratch_config
from titans.training.pretraining import _block_loss, _prepare_next_block, fp32_execution


def gradient_pass(model: TitansModel, ids: Tensor) -> dict[str, float]:
    """Measure one fixed-input pass through the production block-loss operation.

    Args:
        model: Full-size FP32 scratch model with trainable backbone and method.
        ids: Consecutive inputs and one lookahead, shaped [1, tokens + 1].

    Returns:
        Loss, peak VRAM, forward and backward time, throughput, and gradient norm.
        Parameter gradients remain available for numerical comparison.

    Raises:
        AssertionError: Precision, memory writes or gradient finiteness is incorrect.
    """
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with fp32_execution(ids.device):
        state = model._new_state(prefix_hidden_only=True)
        loss = _block_loss(model, ids[:, :-1], ids[:, 1:], state) / (ids.shape[1] - 1)
        torch.cuda.synchronize()
        forwarded = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        finished = time.perf_counter()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert state.tokens_processed == ids.shape[1] - 1
    assert state.memory[0].pending.shape[1] == (ids.shape[1] - 1) % 16
    assert state.memory[0].momentum[0].abs().max() > 0
    assert all(c.keys.dtype == torch.float32 for c in state._attention.values())
    assert all(t.dtype == torch.float32 for m in state.memory.values()
               for t in m.tensors() if t.is_floating_point())
    norms = []
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32, name
        gradient = parameter.grad
        assert gradient is not None and gradient.isfinite().all(), name
        norms.append(gradient.double().norm())
    norm = torch.stack(norms).norm().item()
    return {
        "loss": loss.item(), "gradient_norm_fp64_reduction": norm,
        "forward_seconds": forwarded - started, "backward_seconds": finished - forwarded,
        "forward_tokens_per_second": state.tokens_processed / (forwarded - started),
        "forward_backward_tokens_per_second": state.tokens_processed / (finished - started),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
    }


def check_training_boundary(model: TitansModel, ids: Tensor) -> dict[str, object]:
    """Exercise memory continuation, ordinary KV reset and fresh prefix gradients.

    Args:
        model: Full-size FP32 model configured with the default MAC connection.
        ids: One 4096-token input block with its lookahead token.

    Returns:
        Cache sizes, continued token count and prefix-gradient norm. The probe
        performs two backward passes and zero optimizer steps.
    """
    model.zero_grad(set_to_none=True)
    with fp32_execution(ids.device):
        state = model._new_state(prefix_hidden_only=True)
        loss = _block_loss(model, ids[:, :-1], ids[:, 1:], state) / 4096
        loss.backward()
        del loss
        state._detach()
        memory = state.memory[0].copy(detach=True)
        position = state._position
        previous_cache = state._attention[1].keys.shape[-2]
        model.zero_grad(set_to_none=True)
        _prepare_next_block(model, state)
        current_cache = state._attention[1].keys.shape[-2]
        assert current_cache == model.config.persistent_tokens
        assert state._position == position and state.tokens_processed == 4096
        for actual, expected in zip(state.memory[0].tensors(), memory.tensors(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        continuation = torch.cat((ids[:, -1:], ids[:, 1:18]), dim=1)
        loss = _block_loss(model, continuation[:, :-1], continuation[:, 1:], state) / 17
        loss.backward()
        gradient = model.persistent.grad
        assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0
        assert state.tokens_processed == 4113 and state.memory[0].pending.shape[1] == 1
        return {
            "passed": True, "ordinary_kv_before": previous_cache,
            "ordinary_kv_after": current_cache, "tokens_processed": state.tokens_processed,
            "prefix_gradient_norm": gradient.double().norm().item(), "optimizer_steps": 0,
        }


def run_stream_pretraining_checks() -> dict[str, object]:
    """Check FP32 kernel parity and benchmark a full 4096-token connected graph.

    Returns:
        Fixed settings, short-pass gradient comparisons and full-block measurements.
        The check performs forward/backward passes without constructing an optimizer.
    """
    assert torch.cuda.get_device_capability()[0] == 9
    torch.manual_seed(17)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"x": 0})))
    config = TitansConfig(compile_kernels=False, use_liger=False)
    with torch.device("cuda"):
        model = TitansModel(LlamaForCausalLM(scratch_config()), tokenizer, config)
    model.requires_grad_(True)
    model.train()
    model._checkpoint_backbone = True
    generator = torch.Generator().manual_seed(29)
    ids = torch.randint(3, 32000, (1, 4097), generator=generator).to("cuda")
    watched = ("backbone.lm_head.weight", "backbone.model.layers.0.self_attn.q_proj.weight",
               "memories.0.qkv.weight", "memories.0.initial_weights.0")
    reference_gradients: dict[str, Tensor] = {}
    full_reference: dict[str, Tensor] = {}
    comparisons: dict[str, float] = {}
    full_comparisons: dict[str, float] = {}
    short = []
    measurements = {}
    for optimized in (False, True):
        model.config = replace(config, compile_kernels=optimized, use_liger=optimized)
        for memory in model.memories.values():
            memory.config = model.config
        print(f"Checking 33 tokens, optimized={optimized}", flush=True)
        short.append(gradient_pass(model, ids[:, :34]))
        for name in watched:
            gradient = model.get_parameter(name).grad.detach().cpu()
            if optimized:
                reference = reference_gradients[name]
                error = (gradient - reference).double().norm() / reference.double().norm().clamp_min(1e-12)
                comparisons[name] = error.item()
                assert error < 0.002, (name, error.item())
            else:
                reference_gradients[name] = gradient.clone()
        print(f"Warming up 4096 tokens, optimized={optimized}", flush=True)
        gradient_pass(model, ids)
        print(f"Measuring 4096 tokens, optimized={optimized}", flush=True)
        measurements["optimized" if optimized else "eager"] = gradient_pass(model, ids)
        for name in watched:
            gradient = model.get_parameter(name).grad.detach().cpu()
            if optimized:
                reference = full_reference[name]
                error = (gradient - reference).double().norm() / reference.double().norm().clamp_min(1e-12)
                full_comparisons[name] = error.item()
            else:
                full_reference[name] = gradient.clone()
    torch.testing.assert_close(torch.tensor(short[0]["loss"]), torch.tensor(short[1]["loss"]),
                               rtol=2e-5, atol=2e-5)
    print("Checking the training boundary and persistent-prefix gradients", flush=True)
    boundary = check_training_boundary(model, ids)
    return {
        "batch_size": 1, "input_tokens": 4096, "dtype": "float32", "tf32": False,
        "mac_segment_length": 512, "memory_write_chunk": 16,
        "activation_checkpointing": True, "optimizer_steps": 0,
        "short_gradient_relative_errors": comparisons, "short_checks": short,
        "full_gradient_relative_errors": full_comparisons,
        "full_gradient_parity_passed": all(error < 0.002 for error in full_comparisons.values()),
        "training_boundary": boundary,
        "full_block": measurements,
    }
