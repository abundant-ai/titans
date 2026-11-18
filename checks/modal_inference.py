"""Ad hoc, manually invoked Modal inference checks; never part of CI or training."""

import json
from pathlib import Path

import modal

from checks.modal_image import gpu_image

app = modal.App("titans-inference-check")
image = (
    gpu_image
    .env({"TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/workspace"})
    .add_local_dir("titans", "/workspace/titans")
    .add_local_dir("tests", "/workspace/tests")
    .add_local_dir("examples", "/workspace/examples")
    .add_local_dir("checks", "/workspace/checks")
)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800,
              max_containers=1, scaledown_window=30)
def verify_inference() -> dict[str, object]:
    """Run the native inference checks and collect their device and source metadata.

    Returns:
        Device identity, package versions, source hashes, test counts and full output.
        Failures remain in the report so the caller can save evidence before exiting.
    """
    import hashlib
    import importlib.metadata
    import subprocess
    import sys
    import time
    import xml.etree.ElementTree as ET

    import torch

    source_hashes = {
        str(path.relative_to("/workspace")): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("titans", "tests", "examples", "checks")
        for path in sorted((Path("/workspace") / folder).rglob("*.py"))
    }
    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True,
    ).strip()
    junit = Path("/tmp/inference-checks.xml")
    command = [sys.executable, "-m", "pytest", "/workspace/checks/gpu_inference.py",
               "-vv", "--tb=short", "--show-capture=no", f"--junitxml={junit}"]
    started = time.perf_counter()
    lines: list[str] = []
    with subprocess.Popen(command, cwd="/workspace", stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        returncode = process.wait()
    cases = list(ET.parse(junit).iter("testcase")) if junit.exists() else []
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {
        "mode": "inference_only", "device": device, "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "transformers": importlib.metadata.version("transformers"),
        "liger_kernel": importlib.metadata.version("liger-kernel"),
        "elapsed_seconds": time.perf_counter() - started, "returncode": returncode,
        "tests": len(cases), "passed": len(cases) - failures - errors - skipped,
        "failed": failures, "errors": errors, "skipped": skipped,
        "source_sha256": source_hashes, "output": "".join(lines),
    }


@app.local_entrypoint()
def main(output: str = "validation/gpu_inference.json", hopper: bool = False) -> None:
    """Invoke the manual GPU check and save its actual evidence locally.

    Args:
        output: Local report path; no model or trained checkpoint is produced.
        hopper: Run the same suite on a Hopper H100 instead of an L4.

    Raises:
        RuntimeError: A check failed or was skipped on the allocated GPU.
    """
    check = verify_inference.with_options(gpu="H100!") if hopper else verify_inference
    result = check.remote()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("output", "source_sha256")}, indent=2))
    if result["returncode"] != 0 or not result["tests"] or result["skipped"]:
        raise RuntimeError(f"GPU verification incomplete; report saved to {destination}")
