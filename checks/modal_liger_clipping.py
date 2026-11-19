"""Manual H100 check of selective Liger operations and actual global clipping."""

import json
from pathlib import Path

import modal

from checks.modal_pretraining import cache, image

app = modal.App("titans-selective-liger-clipping")


@app.function(image=image, gpu="H100!", cpu=4, memory=32768, timeout=1200,
              startup_timeout=300, max_containers=1, volumes={"/cache": cache})
def verify(previous_report: str) -> dict[str, object]:
    """Compare native, selective Liger, and default compiled paths on identical text.

    Args:
        previous_report: Saved real-text report containing the exact input IDs.

    Returns:
        Raw and genuinely clipped gradient comparisons, timing and source evidence.
        No optimizer is created or stepped; model weights remain fixed.
    """
    import hashlib
    import importlib.metadata
    import math
    import statistics
    import time
    import traceback
    from dataclasses import replace
    from typing import Never

    import torch
    from torch import Tensor
    from transformers import AutoTokenizer, LlamaForCausalLM

    import titans.optim.gpu_ops as ops
    from titans import TitansConfig, TitansModel
    from titans.runtime.backbones import scratch_config
    from titans.training.pretraining import _block_loss, fp32_execution

    def forbidden(*args: object, **kwargs: object) -> Never:
        """Fail if the runtime dispatches to an excluded Liger operation."""
        raise AssertionError("Liger RMSNorm and RoPE must not execute")

    def compare(reference: dict[str, Tensor], actual: dict[str, Tensor]) -> dict[str, object]:
        """Compare all parameter gradients using FP64 only for CPU diagnostics.

        Args:
            reference: Native gradients copied to CPU, before or after real clipping.
            actual: Corresponding candidate gradients with the same parameter names.

        Returns:
            Global and per-parameter relative L2 error, norms and cosine similarity.
        """
        rr = aa = dd = dot = 0.0
        details = {}
        for name, value in reference.items():
            left, right = value.double().flatten(), actual[name].double().flatten()
            l2, r2 = left.dot(left).item(), right.dot(right).item()
            delta = (left - right).square().sum().item()
            rr += l2
            aa += r2
            dd += delta
            dot += left.dot(right).item()
            details[name] = math.sqrt(delta) / max(math.sqrt(l2), 1e-30)
        return {"relative_l2": math.sqrt(dd) / max(math.sqrt(rr), 1e-30),
                "cosine": dot / max(math.sqrt(rr * aa), 1e-30),
                "reference_norm": math.sqrt(rr), "actual_norm": math.sqrt(aa),
                "parameters_compared": len(details), "per_parameter_relative_l2": details}

    previous = json.loads(previous_report)
    report: dict[str, object] = {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "liger_kernel": importlib.metadata.version('liger-kernel'),
        "source_sha256": {str(p.relative_to('/workspace')): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path('/workspace/titans').rglob('*.py'))},
        "input_token_ids_sha256": previous['data']['token_ids_sha256'],
        "settings": {"batch_size": 1, "input_tokens": 4096, "dtype": "float32", "tf32": False,
                     "mac_segment_length": 512, "memory_write_chunk": 16, "max_grad_norm": 1.0,
                     "optimizer_steps": 0, "seed": 17, "gradient_relative_tolerance": 0.002,
                     "liger_operations": ['swiglu', 'fused_cross_entropy'],
                     "native_operations": ['rms_norm', 'rope'], "activation_checkpointing": True},
    }
    original_norm, original_rope = ops.liger_norm, ops.liger_rotary
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(previous['data']['tokenizer'],
                                                   revision=previous['data']['tokenizer_revision'])
        ids = torch.tensor([previous['data']['input_ids_with_lookahead']], device='cuda')
        torch.manual_seed(17)
        config = TitansConfig(compile_kernels=False, use_liger=False)
        with torch.device('cuda'):
            model = TitansModel(LlamaForCausalLM(scratch_config()), tokenizer, config)
        model.requires_grad_(True)
        model.train()
        model._checkpoint_backbone = True
        parameters = model.outer_parameters(train_backbone=True)
        ops.liger_norm = forbidden
        ops.liger_rotary = forbidden

        def run_pass(liger: bool, compiled: bool, collect: bool
                     ) -> tuple[dict[str, float], dict[str, Tensor], dict[str, Tensor]]:
            """Run FP32 forward/backward and actual production clipping on fresh memory.

            Args:
                liger: Enable SwiGLU and fused loss.
                compiled: Enable the existing compiled tensor operations.
                collect: Copy every raw and clipped gradient to CPU after GPU timing.

            Returns:
                Timing and norms, raw gradients and actually clipped gradients.
            """
            model.config = replace(config, use_liger=liger, compile_kernels=compiled)
            for memory in model.memories.values():
                memory.config = model.config
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with fp32_execution(ids.device):
                state = model._new_state(prefix_hidden_only=True)
                loss = _block_loss(model, ids[:, :-1], ids[:, 1:], state) / 4096
                loss.backward()
                torch.cuda.synchronize()
                finished = time.perf_counter()
                raw = {name: p.grad.detach().cpu() for name, p in model.named_parameters()} if collect else {}
                clip_start = time.perf_counter()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                torch.cuda.synchronize()
                clip_seconds = time.perf_counter() - clip_start
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            post_norm = torch.stack([p.grad.double().norm() for p in parameters]).norm().item()
            assert 0.99999 <= post_norm <= 1.00001, post_norm
            assert state.tokens_processed == 4096 and state.memory[0].pending.shape[1] == 0
            assert all(p.dtype == torch.float32 for p in parameters)
            clipped = {name: p.grad.detach().cpu() for name, p in model.named_parameters()} if collect else {}
            return {"loss": loss.item(), "raw_norm_fp32": norm.item(), "clipped_norm": post_norm,
                    "forward_backward_seconds": finished - start, "clip_seconds": clip_seconds,
                    "tokens_per_second": 4096 / (finished - start + clip_seconds),
                    "peak_allocated_gib": peak}, raw, clipped

        results = {}
        reference_raw: dict[str, Tensor] = {}
        reference_clipped: dict[str, Tensor] = {}
        for name, liger, compiled in [('native', False, False), ('liger', True, False),
                                      ('default_compiled_liger', True, True)]:
            print('Checking', name, flush=True)
            try:
                run_pass(liger, compiled, False)
                passes = []
                comparisons = {}
                for repeat in range(3):
                    measured, raw, clipped = run_pass(liger, compiled, repeat == 0)
                    passes.append(measured)
                    if repeat == 0:
                        if name == 'native':
                            reference_raw, reference_clipped = raw, clipped
                        else:
                            comparisons = {'raw': compare(reference_raw, raw),
                                           'clipped': compare(reference_clipped, clipped)}
                    del raw, clipped
                passed = all(
                    x['relative_l2'] < 0.002
                    and max(x['per_parameter_relative_l2'].values()) < 0.002
                    for x in comparisons.values())
                results[name] = {'passed': passed, 'use_liger': liger, 'compile_kernels': compiled,
                                 'passes': passes, 'gradient_comparisons': comparisons,
                                 'median_tokens_per_second': statistics.median(x['tokens_per_second'] for x in passes),
                                 'peak_allocated_gib': max(x['peak_allocated_gib'] for x in passes)}
            except Exception:
                results[name] = {'passed': False, 'error': traceback.format_exc()}
        report['results'] = results
        report['passed'] = all(x['passed'] for x in results.values())
    except Exception:
        report['passed'] = False
        report['error'] = traceback.format_exc()
    finally:
        ops.liger_norm, ops.liger_rotary = original_norm, original_rope
    report['elapsed_seconds'] = time.perf_counter() - started
    return report


@app.local_entrypoint()
def main(source: str, output: str = '/private/tmp/titans-liger-clipping-report.json') -> None:
    """Run the manual GPU check and save evidence, including numerical failures.

    Args:
        source: Earlier real-text report providing the exact token IDs.
        output: Destination for this new validation report.
    """
    import time

    call = verify.spawn(Path(source).read_text())
    start = time.monotonic()
    while True:
        try:
            report = call.get(timeout=5)
            break
        except TimeoutError:
            elapsed = time.monotonic() - start
            allocated = any(x.task_id for x in call.get_call_graph())
            if (elapsed > 300 and not allocated) or elapsed > 1500:
                call.cancel(terminate_containers=True)
                report = {'passed': False, 'status': 'allocation_unavailable' if not allocated else 'execution_timeout'}
                break
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    summary = {k: v for k, v in report.items() if k not in ('source_sha256', 'results')}
    summary['results'] = {
        name: {**{k: v for k, v in result.items() if k != 'gradient_comparisons'},
               'gradient_comparisons': {phase: {k: v for k, v in values.items() if k != 'per_parameter_relative_l2'}
                                        for phase, values in result.get('gradient_comparisons', {}).items()}}
        for name, result in report.get('results', {}).items()}
    print(json.dumps(summary, indent=2))
