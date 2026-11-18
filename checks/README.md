# Manual GPU inference checks

Run from the repository root when GPU verification is needed:

```bash
uvx modal run --env testing -m checks.modal_inference --output validation/gpu_inference.json
```

The suite runs on one Modal L4, or one H100 with `--hopper`. Both use the same
PyTorch and Liger environment. The checks cover Llama and Qwen2, native streaming,
answer isolation, checkpoint restoration, Liger kernel outputs, and compiled memory
updates. They use small offline models and run outside CI and default pytest discovery.

No training loop, backward pass, `fit`, `learn` or optimizer step is invoked.
Document ingestion still executes each method's test-time memory transition.
Shared model parameters remain unchanged. CPU tests exercise the training and
gradient protocols separately.

`gpu_inference.json` records the actual device, package versions, source hashes,
check counts and complete output. A failed or skipped GPU check makes the command
fail after saving the report. These checks do not measure QA quality or throughput.

## H100 streaming and gradient check

`uvx modal run --env testing -m checks.modal_pretraining` runs fixed inputs
through the full scratch decoder. It compares eager and compiled/Liger
gradients under BF16 autocast and measures tokenwise and block SDPA prefill. The job runs
outside CI. Optimizer-step tests run on CPU. See [pretraining](../docs/pretraining.md).
