"""Explicitly launched H100 fixed-input streaming and gradient checks; excluded from CI."""

import json
from pathlib import Path

import modal

from checks.modal_image import gpu_image

app = modal.App("titans-joint-pretraining-check")
image = (
    gpu_image
    .env({"PYTHONPATH": "/workspace", "TOKENIZERS_PARALLELISM": "false",
          "HF_HOME": "/cache/huggingface", "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor"})
    .add_local_dir("titans", "/workspace/titans")
    .add_local_dir("checks", "/workspace/checks")
)
cache = modal.Volume.from_name("titans-verification-cache", create_if_missing=True)


@app.function(image=image, gpu="H100!", cpu=4, memory=16384, timeout=1800, startup_timeout=300,
              max_containers=1, volumes={"/cache": cache})
def verify() -> dict[str, object]:
    """Run bounded full-size inference and gradient comparisons on one H100.

    Returns:
        Device/software/source provenance and pass/failure details, including failures.
    """
    import hashlib
    import importlib.metadata
    import subprocess
    import time
    import traceback

    import torch

    from checks.pretraining import run_pretraining_checks

    started = time.perf_counter()
    report: dict[str, object] = {
        "device": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True,
        ).strip(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "liger_kernel": importlib.metadata.version("liger-kernel"),
        "source_sha256": {
            str(p.relative_to("/workspace")): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("titans", "checks")
            for p in sorted((Path("/workspace") / folder).rglob("*.py"))
        },
    }
    try:
        report["results"] = run_pretraining_checks()
        report["passed"] = True
    except Exception:
        report["passed"] = False
        report["error"] = traceback.format_exc()
    report["elapsed_seconds"] = time.perf_counter() - started
    cache.commit()
    return report


@app.local_entrypoint()
def main(output: str = "validation/h100_pretraining.json") -> None:
    """Launch the manual H100 comparison and persist its evidence.

    Args:
        output: Local JSON report destination, including on test failure.

    Raises:
        RuntimeError: Validation did not complete successfully.
    """
    import time

    call = verify.spawn()
    started = time.monotonic()
    while True:
        try:
            report = call.get(timeout=5)
            break
        except TimeoutError:
            graph = call.get_call_graph()
            elapsed = time.monotonic() - started
            allocated = any(item.task_id for item in graph)
            if (elapsed >= 300 and not allocated) or elapsed >= 2100:
                call.cancel(terminate_containers=True)
                report = {
                    "passed": False,
                    "status": "allocation_unavailable" if not allocated else "execution_timeout",
                    "gpu_tests_executed": 0 if not allocated else "incomplete",
                    "elapsed_seconds": elapsed,
                    "call_graph": [{"function_call_id": item.function_call_id,
                                    "task_id": item.task_id, "status": item.status.name}
                                   for item in graph],
                }
                break
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"H100 pretraining check failed: {path}")
