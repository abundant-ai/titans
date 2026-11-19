"""Ad hoc H100 Liger comparison on one real-text batch; no optimizer or CI job."""

import json
from pathlib import Path

import modal

from checks.modal_pretraining import cache, image

app = modal.App("titans-liger-real-text")


@app.function(image=image, gpu="H100!", cpu=4, memory=32768, timeout=1200,
              startup_timeout=300, max_containers=1, volumes={"/cache": cache})
def verify(source_json: str) -> dict[str, object]:
    """Compare native and Liger operations with fixed weights and real token IDs.

    Args:
        source_json: Unmodified FineWeb-Edu dataset-server response containing text.

    Returns:
        Reproducible input IDs, source hashes, timed passes and all-parameter gradient
        comparisons. The same batch is reused; no optimizer is constructed.
    """
    import hashlib
    import importlib.metadata
    import math
    import statistics
    import subprocess
    import time
    import traceback
    from dataclasses import replace

    import torch
    from huggingface_hub import model_info
    from torch import Tensor
    from transformers import AutoTokenizer, LlamaForCausalLM

    from titans import TitansConfig, TitansModel
    from titans.runtime.backbones import LLAMA_TOKENIZER, scratch_config
    from titans.training.pretraining import _block_loss, fp32_execution

    report: dict[str, object] = {
        "device": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True,
        ).strip(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "liger_kernel": importlib.metadata.version("liger-kernel"),
        "source_sha256": {
            str(p.relative_to("/workspace")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("/workspace/titans").rglob("*.py"))
        },
        "settings": {"batch_size": 1, "input_tokens": 4096, "dtype": "float32",
                     "tf32": False, "compile_kernels": False, "seed": 17,
                     "mac_segment_length": 512, "memory_write_chunk": 16,
                     "activation_checkpointing": True, "optimizer_steps": 0,
                     "gradient_relative_tolerance": 0.002},
    }
    started = time.perf_counter()
    try:
        revision = model_info(LLAMA_TOKENIZER).sha
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER, revision=revision)
        packed: list[int] = []
        documents: list[dict[str, object]] = []
        for item in json.loads(source_json)["rows"]:
            assert "text" not in item.get("truncated_cells", []), "Truncated source text"
            document = item["row"]
            text = document["text"]
            encoded = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            documents.append({"id": document["id"], "url": document["url"],
                              "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "start_token": len(packed), "tokens_with_eos": len(encoded)})
            packed.extend(encoded)
            if len(packed) >= 4097:
                break
        assert len(packed) >= 4097, "Source examples do not fill one batch"
        ids = torch.tensor([packed[:4097]], dtype=torch.long, device="cuda")
        report["data"] = {
            "dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train",
            "source_url": "https://datasets-server.huggingface.co/first-rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train",
            "response_sha256": hashlib.sha256(source_json.encode()).hexdigest(),
            "tokenizer": LLAMA_TOKENIZER, "tokenizer_revision": revision,
            "documents": documents, "input_ids_with_lookahead": packed[:4097],
            "token_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            "selection": "First 4097 tokens in source order, EOS between documents, no BOS or repetition",
        }
        torch.manual_seed(17)
        config = TitansConfig(compile_kernels=False, use_liger=False)
        with torch.device("cuda"):
            model = TitansModel(LlamaForCausalLM(scratch_config()), tokenizer, config)
        model.requires_grad_(True)
        model.train()
        model._checkpoint_backbone = True
        report["parameter_count"] = sum(p.numel() for p in model.parameters())

        def run_pass(liger: bool, *, collect: bool) -> tuple[dict[str, float], dict[str, Tensor]]:
            """Time one forward/backward pass with fresh active memory.

            Args:
                liger: Enable all existing Liger replacements, leaving compilation off.
                collect: Copy parameter gradients to CPU after timing for comparison.

            Returns:
                Loss, timing and allocated memory, plus optional FP32 CPU gradients.
            """
            model.config = replace(config, use_liger=liger)
            for memory in model.memories.values():
                memory.config = model.config
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            with fp32_execution(ids.device):
                state = model._new_state(prefix_hidden_only=True)
                loss = _block_loss(model, ids[:, :-1], ids[:, 1:], state) / 4096
                torch.cuda.synchronize()
                forwarded = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                finished = time.perf_counter()
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            assert loss.dtype == torch.float32
            assert state.tokens_processed == 4096
            assert state.memory[0].pending.shape[1] == 0
            assert state.memory[0].momentum[0].abs().max() > 0
            assert all(t.dtype == torch.float32 for m in state.memory.values()
                       for t in m.tensors() if t.is_floating_point())
            assert all(p.dtype == torch.float32 for p in model.parameters())
            gradients = {}
            if collect:
                for name, parameter in model.named_parameters():
                    assert parameter.grad is not None, name
                    gradients[name] = parameter.grad.detach().cpu()
            return {
                "loss": loss.item(), "forward_seconds": forwarded - begin,
                "backward_seconds": finished - forwarded,
                "forward_backward_seconds": finished - begin,
                "tokens_per_second": 4096 / (finished - begin), "peak_allocated_gib": peak,
            }, gradients

        def compare(reference: dict[str, Tensor], actual: dict[str, Tensor]) -> dict[str, object]:
            """Compare every shared-parameter gradient with stable FP64 diagnostics.

            Args:
                reference: Native PyTorch FP32 parameter gradients on CPU.
                actual: Liger or repeated-native gradients with identical parameter keys.

            Returns:
                Relative L2 errors, cosine similarities, and analytically clipped
                gradient error. FP64 is used only for these diagnostic reductions.
            """
            per_parameter = {}
            ref_sq = act_sq = diff_sq = dot = 0.0
            finite = True
            for name, original in reference.items():
                left, right = original.double().flatten(), actual[name].double().flatten()
                finite = finite and bool(torch.isfinite(right).all() and torch.isfinite(left).all())
                rr, aa = left.dot(left).item(), right.dot(right).item()
                dd, cross = (left - right).square().sum().item(), left.dot(right).item()
                ref_sq += rr
                act_sq += aa
                diff_sq += dd
                dot += cross
                per_parameter[name] = {
                    "native_norm": math.sqrt(rr), "actual_norm": math.sqrt(aa),
                    "relative_l2": math.sqrt(dd) / max(math.sqrt(rr), 1e-30),
                    "cosine": cross / max(math.sqrt(rr * aa), 1e-30),
                }
            rn, an = math.sqrt(ref_sq), math.sqrt(act_sq)
            rc, ac = min(1.0, 1.0 / (rn + 1e-6)), min(1.0, 1.0 / (an + 1e-6))
            clipped_diff = max(0.0, rc * rc * ref_sq + ac * ac * act_sq - 2 * rc * ac * dot)
            return {
                "all_finite": finite, "parameters_compared": len(per_parameter),
                "native_norm": rn, "actual_norm": an,
                "relative_l2": math.sqrt(diff_sq) / max(rn, 1e-30),
                "cosine": dot / max(rn * an, 1e-30),
                "after_global_clip_relative_l2": math.sqrt(clipped_diff) / max(rc * rn, 1e-30),
                "per_parameter": per_parameter,
            }

        print("Prepared one FineWeb-Edu batch; warming up native and Liger", flush=True)
        run_pass(False, collect=False)
        run_pass(True, collect=False)
        passes: dict[str, list[dict[str, float]]] = {"native": [], "liger": []}
        reference: dict[str, Tensor] = {}
        comparisons = {}
        for repeat in range(3):
            for name, liger in (("native", False), ("liger", True)):
                print(f"Measuring {name}, repetition {repeat + 1}", flush=True)
                metrics, gradients = run_pass(liger, collect=repeat < 2)
                passes[name].append(metrics)
                if repeat == 0 and name == "native":
                    reference = gradients
                elif gradients:
                    comparisons[f"{name}_{repeat + 1}_vs_native_1"] = compare(reference, gradients)
                del gradients
        report["passes"] = passes
        report["gradient_comparisons"] = comparisons
        report["medians"] = {
            name: {key: statistics.median(p[key] for p in measured)
                   for key in ("forward_backward_seconds", "tokens_per_second", "peak_allocated_gib")}
            for name, measured in passes.items()
        }
        report["passed"] = all(
            metrics["all_finite"] and metrics["relative_l2"] < 0.002
            and all(p["relative_l2"] < 0.002 for p in metrics["per_parameter"].values())
            for metrics in comparisons.values()
        ) and all(abs(p["loss"] - passes["native"][0]["loss"]) < 2e-5
                  for p in passes["liger"])
    except Exception:
        report["passed"] = False
        report["error"] = traceback.format_exc()
    report["elapsed_seconds"] = time.perf_counter() - started
    cache.commit()
    return report


@app.local_entrypoint()
def main(source: str, output: str = "/private/tmp/titans-liger-real-text-report.json") -> None:
    """Run the manually requested GPU comparison and save its complete evidence.

    Args:
        source: JSON response from the FineWeb-Edu dataset-server rows endpoint.
        output: Destination for the report, including numerical failures.
    """
    import time

    call = verify.spawn(Path(source).read_text())
    started = time.monotonic()
    while True:
        try:
            report = call.get(timeout=5)
            break
        except TimeoutError:
            graph = call.get_call_graph()
            elapsed = time.monotonic() - started
            allocated = any(item.task_id for item in graph)
            if (elapsed > 300 and not allocated) or elapsed > 1500:
                call.cancel(terminate_containers=True)
                report = {"passed": False, "status": "allocation_unavailable" if not allocated
                          else "execution_timeout", "elapsed_seconds": elapsed}
                break
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items()
               if key not in ("source_sha256", "data", "gradient_comparisons")}
    summary["gradient_comparisons"] = {
        name: {key: value for key, value in metrics.items() if key != "per_parameter"}
        for name, metrics in report.get("gradient_comparisons", {}).items()
    }
    print(json.dumps(summary, indent=2))
