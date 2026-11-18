"""Bounded H100 verification jobs, using the existing configured Modal account."""

import json
from pathlib import Path

import modal

from checks.modal_image import gpu_image

app = modal.App("titans-h100-verification")
image = (
    gpu_image
    .env({"HF_HOME": "/cache/huggingface", "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/workspace"})
    .add_local_dir("titans", "/workspace/titans")
    .add_local_dir("tests", "/workspace/tests")
    .add_local_dir("benchmarks", "/workspace/benchmarks")
    .add_local_dir("checks", "/workspace/checks")
    .add_local_dir("examples", "/workspace/examples")
)
cache = modal.Volume.from_name("titans-verification-cache", create_if_missing=True)


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=1800,
              max_containers=1, scaledown_window=30, volumes={"/cache": cache})
def inspect_runtime() -> dict[str, str]:
    """Verify the allocated GPU and installed kernel interfaces before benchmarking.

    Returns:
        Device identity and installed runtime versions.
    """
    import importlib.metadata

    import torch

    return {
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "liger_kernel": importlib.metadata.version("liger-kernel"),
    }


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=1800,
              max_containers=1, scaledown_window=30, volumes={"/cache": cache})
def benchmark() -> str:
    """Run CUDA equivalence tests before collecting native inference throughput.

    Returns:
        JSON containing measurements from the allocated H100.
    """
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pytest", "/workspace/checks/gpu_inference.py",
                    "-q", "--tb=short", "--show-capture=no"], check=True, cwd="/workspace")
    output = Path("/tmp/titans-h100-report.json")
    subprocess.run([sys.executable, "-m", "benchmarks.h100", "--output", str(output), "--profile"],
                   check=True, cwd="/workspace")
    return output.read_text()


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=3600,
              max_containers=1, scaledown_window=30, volumes={"/cache": cache})
def compare_backbones() -> str:
    """Benchmark the supported Qwen model on one physical H100.

    Each measurement has a fresh process to release device memory and isolate
    compiler specializations. Model downloads and compilation are outside steady timings.

    Returns:
        JSON with GPU identity, test output, inference and post-training results.
    """
    import subprocess
    import sys

    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True,
    ).strip()
    checked = subprocess.run(
        [sys.executable, "-m", "pytest", "/workspace/checks/gpu_inference.py", "-q",
         "--tb=short", "--show-capture=no"], check=False, cwd="/workspace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(checked.stdout, flush=True)
    checked.check_returncode()
    report: dict[str, object] = {"device": device, "gpu_tests": checked.stdout, "models": {}}
    models: dict[str, object] = {}
    for label, model in (("qwen2_5", "Qwen/Qwen2.5-1.5B-Instruct"),):
        measurements: dict[str, object] = {}
        for mode, module in (("inference", "benchmarks.h100"),
                             ("posttraining", "benchmarks.posttraining")):
            destination = Path(f"/cache/{label}-{mode}.json")
            subprocess.run([sys.executable, "-m", module, "--model", model,
                            "--output", str(destination)], check=True, cwd="/workspace")
            measurements[mode] = json.loads(destination.read_text())
            cache.commit()
        models[label] = measurements
    report["models"] = models
    return json.dumps(report, indent=2) + "\n"


@app.local_entrypoint()
def main(
    output: str = "validation/h100_runtime.json", run_benchmark: bool = False, compare: bool = False,
) -> None:
    """Run the hardware check and persist its actual measurements locally.

    Args:
        output: Local JSON destination for the remote result.
        compare: Benchmark Qwen inference and post-training.
        run_benchmark: Whether to run CUDA checks and inference timing after hardware inspection.
    """
    if compare:
        result = compare_backbones.remote()
        destination = Path("validation/h100_comparison.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result)
        print(result)
        return
    report = inspect_runtime.remote()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if run_benchmark:
        result = benchmark.remote()
        result_path = path.with_name("h100_qwen_inference.json")
        result_path.write_text(result)
        print(result)
