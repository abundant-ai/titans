# Manual GPU inference checks

Run from the repository root when GPU verification is needed:

```bash
uvx modal run --env testing checks/modal_inference.py --output validation/gpu_inference.json
```

This is an ad hoc local command that executes eight portable checks on one Modal
L4. It is not wired
into CI, installation, or the default `pytest` suite. The remote function has a
30-minute execution limit and permits one container. Add `--hopper` to run all 17
checks on one H200, including FA3 and its CUDA graph paths. Each invocation uses
one selected GPU function. Compiler caches are ephemeral; no shared volume is needed.

Together, the checks cover Llama and Qwen2 with Titans and linear memory, streaming past
the attention window, isolated answers, independent state forks, untrained
checkpoint restoration, FA3/Liger output parity, compiled memory transitions and
CUDA graph replay. The models are small, randomly initialized and constructed
offline; no pretrained weights are downloaded.

No training loop, backward pass, `fit`, `learn` or optimizer step is invoked.
Document ingestion still executes each method's test-time memory transition.
Shared model parameters remain unchanged. CPU tests exercise the training and
gradient protocols separately.

`gpu_inference.json` records the actual device, package versions, source hashes,
check counts and complete output. A failed or skipped GPU check makes the command
fail after saving the report. These checks do not measure QA quality or throughput.

## H100 streaming and gradient check

`uvx modal run --env testing checks/modal_pretraining.py` runs fixed inputs
through the full scratch decoder. It compares eager and compiled/Liger
gradients under BF16 autocast and measures FA3 and SDPA prefill. The job runs
outside CI. Optimizer-step tests run on CPU. See [pretraining](../docs/pretraining.md).
