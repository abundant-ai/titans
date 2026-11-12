# GPU runtime

The selected backbone is
[`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).
Its [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/config.json)
uses `LlamaForCausalLM`, 24 layers and head dimension 64. The H100 benchmark checks
the actual parameter count is below two billion before measuring anything.

The CUDA environment pins PyTorch 2.9.1, Transformers 4.57.1, SGLang 0.5.7 and
Liger Kernel 0.6.4. Install on Linux with `uv sync --extra gpu`. The Modal runner
uses the official `lmsysorg/sglang:v0.5.7` image and adds Liger and pytest.

```python
import torch
from titans import TitansConfig, TitansModel

model = TitansModel.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    TitansConfig(
        memory_layers=(0, 6, 12, 18),
        memory_dim=64,
        memory_hidden_dim=128,
        chunk_size=32,
        prefill_chunk_size=256,
        inference_backend="sglang",
        attention_backend="fa3",
        compile_kernels=True,
        use_liger=True,
        cuda_graphs=True,
    ),
    device="cuda",
    dtype=torch.bfloat16,
)
# Call fit, ingest, answer and clone_state through the same public class.
# Release the local serving processes and their state when finished:
model.close()
```

## JIT and training kernels

`compile_kernels=True` is the default. Tensor-only kernels use
[`torch.compile(fullgraph=True)`](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html),
including the memory update, memory projections, retrieval and output gate.
The Torch path also compiles attention, RMSNorm and Llama's MLP. Full-graph capture
rejects unsupported graph breaks. CPU development can set `compile_kernels=False`
to avoid compilation startup costs.

On CUDA, Liger supplies RMSNorm, RoPE, SwiGLU and the fused linear cross-entropy loss.
The fused loss avoids materializing the full token-by-vocabulary logit matrix
during training. The hand-derived memory gradients remain in the compiled
Titans kernel. `activation_checkpointing=True` recomputes host MLP activations
during backward to reduce saved-activation memory.

`cuda_graphs=True` enables the Titans worker's graph runner for single-token
decoding once the attention window is full. It captures each pending-write phase
separately, including the chunk-commit update. Read-only answers use separate
graphs. RoPE positions remain GPU inputs, so replay advances absolute positions.
Prefill and shorter caches use the ordinary compiled-kernel path.

Graphs share a workspace and allocator pool. Copies back into persistent buffers
happen inside the graph. Before switching streams or returning to prefill, the
runner gives the old stream independent storage. This ownership rule preserves
clones and source states when a different request reuses the graph. Capture is
lazy; the first request to visit a phase pays its warmup and capture cost. See
[PyTorch's CUDA graph semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs).

## SGLang state handling

The external `TitansForCausalLM` model runs inside a local SGLang engine. SGLang
schedules requests and samples tokens. Its worker owns each stream's fast weights,
momentum, unfinished write chunk and bounded attention KV. Ingestion appends to
that state; answering creates a disposable branch; cloning copies the full state.
Document history is not replayed with each question.

This adapter currently allows one active request and one GPU. Native radix-prefix
sharing, native SGLang CUDA graphs, speculative decoding and overlapping scheduling are disabled
because their state handling has not been extended to include Titans memory.
The separate Titans graph runner includes the complete memory state.
Those limits matter for throughput comparisons with an ordinary SGLang Llama.
The client retains its training model while the worker loads an inference copy;
both copies occupy GPU memory. `fit` shuts down a running inference worker first,
then a subsequent inference call exports the updated weights and starts a new one.

The engine is an internal trusted client, not a public HTTP server. Custom request
metadata is transported through SGLang's custom-logit-processor interface. The
processor leaves logits unchanged. Requests carry state IDs and policies, while
tensor state remains in the worker. Deterministic SGLang inference enables the
request's sampling seed; repeated requests with the same state and seed are checked
on the H100. This flag also selects SGLang's batch-invariant kernels.

## FlashAttention 3

Inference calls SGLang's FA3 kernel with `ver=3`. Ordinary tokens use causal sliding
attention. Persistent prefix tokens remain globally visible: their attention
partition is merged with the FA3 local partition using log-sum-exp normalization.
This preserves the explicit local-plus-prefix mask; dropping the prefix or using
full-history attention would change the model.

The configured path requires Hopper hardware and BF16 or FP16. Training uses
autograd-capable SDPA; SGLang's cached FA3 entry point is used for inference.
[FA3 hardware requirements](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release)
and [SGLang's external-model registry](https://github.com/sgl-project/sglang/blob/v0.5.7/python/sglang/srt/models/registry.py)
are upstream dependencies, not additional Titans algorithm requirements.

## Reproduce the measurements

```bash
uv run pytest -q
uv run python -m benchmarks.memory_update --device cuda --output validation/memory_h100.json
uv run python -m benchmarks.h100 --output validation/h100_inference.json
```

For the configured Modal account, the task runner uploads only the package, tests
and benchmark scripts and uses one H100 per job:

```bash
uv run --with modal modal run --env testing benchmarks/modal_h100.py --run-benchmark
```

Compilation and engine startup are recorded separately. Timed requests run after
warmup. The report includes actual GPU identity, package versions, parameter
counts, document token count, raw latency samples, ingestion tokens/second,
time to first token, and generation speed. Decode-only throughput is an estimate
from matched 128-token and one-token requests; the report also includes the direct
end-to-end 128-token rate. The adapters are untrained for this speed test.
