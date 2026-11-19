# GPU runtime

Training, document ingestion and answering use one PyTorch model on one device.
Install the Linux GPU dependencies from the repository root with `uv sync --extra gpu`.
The GPU extra adds Liger Kernel 0.6.4 to the base PyTorch 2.9.1 and Transformers 4.57.1 installation.

```python
import torch
from titans import Document, Question, TitansModel

model = TitansModel.from_pretrained(device="cuda", dtype=torch.bfloat16)
state = model.ingest(Document("arrival", "The archive moved to room nine."))
answer = model.answer(Question("room", "Where is the archive?"), state)
```

This loads the fixed Qwen2.5-1.5B-Instruct backbone with newly initialized Titans memory.
Scratch Llama training uses the same runtime. Use the library's document and question
wrappers during post-training so the input format matches inference. See [pretraining](pretraining.md).

## Attention and state

All connections use PyTorch scaled dot-product attention. Each ordinary decoder query
can see the causal persistent prefix and up to `attention_window` real tokens, including
itself. MAC uses the interleaved retrieval and token positions of its current segment.
An explicit mask uses the stored key positions so a short query block can attend to the
correct part of a longer cache. Qwen retains its Q/K/V projection biases and grouped KV heads.

Each stream owns its KV cache, fast weights, momentum, convolution history and pending
writes. `answer` generates on a copy. The next document continues from the original
state. Historical checkpoints load into the local runtime and retain their saved
memory configuration, connection and segment behavior. Current checkpoints include the
rotary frequencies used by the backbone so a BF16 checkpoint round trip preserves them.

## Compilation and precision

`compile_kernels=True` compiles tensor operations with PyTorch Inductor, including the
memory projections, retrieval, write update and output gate. On CPU it also compiles
the backbone attention and MLP paths. On CUDA, PyTorch dispatches SDPA
directly. The CUDA training loop excludes cuDNN SDPA because its backward pass
produced nonfinite gradients on pretrained Qwen with the prefix and sliding-window mask.

`use_liger=True` enables CUDA SwiGLU and fused linear cross-entropy. Decoder RMSNorm
and rotary embeddings use the native PyTorch implementations, including when
compilation is enabled. The fused loss avoids allocating a dense vocabulary
projection for every supervised token. Set `use_liger=False` to use the native
loss and the existing native or compiled MLP path.
Set `compile_kernels=False` to run the tensor operations eagerly.
The fixed `pretrain` path uses FP32 and clips the joint parameter gradients to
global norm 1.0 immediately before each AdamW step. The ad hoc
[real-text clipping check](../checks/modal_liger_clipping.py) applies the actual
clipping operation without taking optimizer steps. With `compile_kernels=False`,
selective Liger passed raw and clipped gradient comparisons across all 232
parameter tensors on the recorded real-text batch. The existing compiled path
still failed gradient parity. Use `compile_kernels=False` for this validated
FP32 setup; see the [H100 report](../validation/h100_liger_clipping.json).

On supported CUDA devices, legacy `fit` training uses BF16 autocast and FP32 for
the memory-update arithmetic. Activation checkpointing can recompute the host MLP
during backward. See [PyTorch compilation](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html).

## Verification and measurement

The [manual GPU checks](../checks/README.md) cover inference and fixed-input gradients.
Run the native inference benchmark on an H100 with the pretrained weights available:

```bash
uv run python -m benchmarks.h100 --output validation/h100_qwen_inference.json
```

It warms up the model and synchronizes the GPU before recording latency. It measures
document ingestion, time to first token, and generation through the public API.
The report includes the actual number of tokens generated up to the 128-token cap.
Pass `--profile` for an additional fixed-token decode profile.

`benchmarks.posttraining` separately measures real optimizer steps when explicitly
invoked. Benchmarks in this checkout use MAC. Archived throughput measurements in
[validation](../validation/README.md) describe the runtime and connection recorded in each report.
