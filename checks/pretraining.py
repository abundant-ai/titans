"""Fixed-input H100 checks for streaming, mixed precision and fused loss."""

import statistics
import time
from dataclasses import replace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from titans import TitansConfig, TitansModel
from titans.optim.gpu_ops import flash_attention3, liger_loss
from titans.runtime.backbones import scratch_config


def run_pretraining_checks() -> dict[str, object]:
    """Compare fixed forwards and backwards on the locked FP32-stored scratch model.

    Returns:
        Numerical errors, peak allocated bytes, throughput and vocabulary-head calls.
        All runs use the same weights and token IDs. No optimizer is constructed.
    """
    assert torch.cuda.get_device_capability()[0] == 9
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"x": 0})))
    config = TitansConfig(compile_kernels=False, use_liger=False, inference_backend="torch")
    with torch.device("cuda"):
        model = TitansModel(LlamaForCausalLM(scratch_config()), tokenizer, config)
    ids = list(range(17, 50))
    targets = torch.tensor(ids[1:], device="cuda")
    model.backbone.requires_grad_(True)
    watched = ("backbone.lm_head.weight", "backbone.model.layers.0.self_attn.q_proj.weight",
               "memories.0.qkv.weight", "memories.0.initial_weights.0")
    gradients = {}
    losses = []
    gradient_errors = {}
    with sdpa_kernel(SDPBackend.MATH):
        for fused in (False, True):
            model.config = replace(config, use_liger=fused, compile_kernels=fused,
                                   attention_backend="auto" if fused else "sdpa")
            for memory in model.memories.values():
                memory.config = model.config
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                state = model._new_state()
                pred = model._forward_tokens(ids, state, return_hidden=fused)
                h = pred[0, :-1]
                loss = (liger_loss(model.backbone.lm_head.weight, h, targets) if fused else
                        F.cross_entropy(h.float(), targets, reduction="sum")) / len(targets)
            loss.backward()
            assert state.memory[0].pending.shape[1] == 1
            assert state.memory[0].weights[0].dtype == torch.float32
            assert state._attention[0].keys.dtype == torch.bfloat16
            assert state.memory[0].momentum[0].abs().max() > 0
            losses.append(loss.detach().item())
            for name in watched:
                gradient = model.get_parameter(name).grad
                assert gradient is not None and gradient.isfinite().all() and gradient.norm() > 0, name
                if fused:
                    reference = gradients[name]
                    actual = gradient.cpu()
                    relative = (actual - reference).norm() / reference.norm()
                    gradient_errors[name] = relative.item()
                    assert relative < 0.08, (name, relative.item())
                else:
                    gradients[name] = gradient.cpu().clone()
            del state, pred, h, loss
    torch.testing.assert_close(torch.tensor(losses[0]), torch.tensor(losses[1]), rtol=0.01, atol=0.01)
    model.zero_grad(set_to_none=True)
    model.eval()
    for memory in model.memories.values():
        memory.config = config
    with torch.no_grad():
        output_fp32 = []
        for backend in ("auto", "sdpa"):
            model.config = replace(config, attention_backend=backend)
            output_fp32.append(model._forward_tokens(ids, model._new_state(), last_only=True))
        torch.testing.assert_close(*output_fp32, rtol=0, atol=0)
        del output_fp32
    metrics = {}
    reference: Tensor | None = None
    sdpa: Tensor | None = None
    for label, stride, backend in (("tokenwise_sdpa", 1, "sdpa"), ("block_sdpa", 256, "sdpa"),
                                   ("block_fa3", config.prefill_chunk_size, config.attention_backend)):
        model.config = replace(config, prefill_chunk_size=stride, attention_backend=backend)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            state = model._new_state()
            with patch("titans.optim.gpu_ops.flash_attention3", wraps=flash_attention3) as kernel:
                outputs = model._forward_tokens(ids, state).float()
            fa3_calls = kernel.call_count
            del kernel  # Release retained Q/K/V arguments before measuring peak allocation.
            if label == "block_fa3":
                assert backend == "auto"
                assert fa3_calls == len(model.backbone.model.layers)
                assert sdpa is not None
                with patch("titans.optim.gpu_ops.flash_attention3", side_effect=NotImplementedError):
                    fallback = model._forward_tokens(ids, model._new_state()).float()
                torch.testing.assert_close(fallback.cpu(), sdpa, rtol=0, atol=0)
                del fallback
            elif label == "block_sdpa":
                assert fa3_calls == 0
                sdpa = outputs.cpu()
            if reference is None:
                reference = outputs.clone()
            torch.testing.assert_close(outputs, reference, rtol=0.06, atol=0.12)
            errors = (outputs - reference).abs().max().item()
            del state, outputs
            times = []
            for repeat in range(4):
                head_calls = []
                handle = model.backbone.lm_head.register_forward_hook(
                    lambda module, args, output: head_calls.append(output.shape[1]))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    state = model._new_state(prefix_hidden_only=True)
                    model._forward_tokens(ids, state, last_only=True)
                    torch.cuda.synchronize()
                finally:
                    handle.remove()
                if repeat:
                    times.append(time.perf_counter() - start)
                assert head_calls == [1]
                peak = torch.cuda.max_memory_allocated()
                del state
            metrics[label] = {"tokens_per_second": len(ids) / statistics.median(times),
                              "peak_allocated_bytes": peak, "head_calls": len(head_calls),
                              "max_logit_error": errors, "prefill_fa3_calls": fa3_calls}
    return {"real_tokens": len(ids), "segment": 512, "write_chunk": 16,
            "gradient_relative_l2": gradient_errors, "fp32_auto_sdpa_equal": True,
            "eager_loss": losses[0], "liger_loss": losses[1], "prefill": metrics}
