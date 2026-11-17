# GPU runtime

This guide describes the existing Titans-specific accelerator path. Generic
methods and explicit QA feedback use the local PyTorch runtime. Historical H100
measurements predate the protocol refactor and have not been re-run for it.

`Qwen/Qwen2.5-1.5B-Instruct` is the default backbone for examples and benchmarks.
`HuggingFaceTB/SmolLM2-1.7B-Instruct` remains available as a comparison baseline.
Both use the same public API, memory kernels and SGLang worker. Select either model ID
in `TitansModel.from_pretrained`; checkpoints retain the backbone architecture.

| Backbone | HF architecture | Layers | Query/KV heads | Head dimension |
| --- | --- | ---: | ---: | ---: |
| SmolLM2-1.7B-Instruct | `LlamaForCausalLM` | 24 | 32/32 | 64 |
| Qwen2.5-1.5B-Instruct | `Qwen2ForCausalLM` | 28 | 12/2 | 128 |

Sources: [SmolLM2 configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/config.json)
and [Qwen2.5 configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json).
Qwen's Q/K/V projection biases and grouped KV heads are preserved in both eager
and compiled paths. Its smaller KV cache does not by itself establish a speedup;
the benchmark measures the complete Titans serving path.

Both backbones use the library's document/question wrappers in training and
inference. The library does not apply their native chat templates. Post-training
should use these same wrappers; this integration does not claim unchanged
out-of-the-box instruction-following quality.

The CUDA environment pins PyTorch 2.9.1, Transformers 4.57.1, SGLang 0.5.7 and
Liger Kernel 0.6.4. Install on Linux with `uv sync --extra gpu`. The Modal runner
uses the official `lmsysorg/sglang:v0.5.7` image and adds Liger and pytest.

```python
import torch
from titans import TitansConfig, TitansModel

model = TitansModel.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
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
The CPU path also compiles attention, RMSNorm and the host MLP.
CUDA attention uses native fused kernels and Liger RoPE: compiling the complete
SDPA wrapper produced nonfinite Qwen gradients in PyTorch 2.9.1, even with Liger
disabled. Memory kernels remain compiled on CUDA. Full-graph capture
rejects unsupported graph breaks. CPU development can set `compile_kernels=False`
to avoid compilation startup costs.

The two biased memory-projection kernels disable Inductor's shape-padding pass.
[SGLang 0.5.7's deterministic `addmm` replacement](https://github.com/sgl-project/sglang/blob/v0.5.7/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py)
does not accept the `alpha`/`beta` keywords used by that pass's benchmark. This
otherwise prevents Qwen-width memory from compiling in the serving worker.
Full-graph JIT remains enabled, and a CUDA regression test covers the affected
Qwen dimensions under SGLang's deterministic mode.

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
Those limits matter for throughput comparisons with an ordinary SGLang model.
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
autograd-capable SDPA with cuDNN excluded; SGLang's cached FA3 entry point is used
for inference. A pretrained Qwen training run produced NaNs in
`ScaledDotProductCudnnAttentionBackward0`, including with JIT and Liger disabled.
The reference run completed with finite gradients when cuDNN was excluded,
without changing the memory settings or 2,048-token backpropagation window.
The backend restriction is scoped to `fit` and applies to both backbones.
[FA3 hardware requirements](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release)
and [SGLang's external-model registry](https://github.com/sgl-project/sglang/blob/v0.5.7/python/sglang/srt/models/registry.py)
are upstream dependencies, not additional Titans algorithm requirements.

## Reproduce the measurements

```bash
uv run pytest -q
uv run python -m benchmarks.memory_update --device cuda --output validation/memory_h100.json
uv run python -m benchmarks.h100 --output validation/h100_qwen_inference.json
uv run python -m benchmarks.posttraining --output validation/h100_qwen_posttraining.json
```

For the configured Modal account, the task runner uploads only the package, tests
and benchmark scripts and uses one H100 per job:

```bash
uv run --with modal modal run --env testing benchmarks/modal_h100.py --run-benchmark
```

Both standalone benchmarks default to Qwen2.5-1.5B-Instruct. Pass
`--model HuggingFaceTB/SmolLM2-1.7B-Instruct` to select the comparison backbone.
The Modal `--run-benchmark` command writes `validation/h100_qwen_inference.json`.

Compilation and engine startup are recorded separately. Timed requests run after
warmup. The report includes actual GPU identity, package versions, parameter
counts, document token count, raw latency samples, ingestion tokens/second,
time to first token, and generation speed. Decode-only throughput is an estimate
from matched 128-token and one-token requests; the report also includes the direct
end-to-end 128-token rate. The adapters are untrained for this speed test.


To compare both backbones on the same physical H100, including real post-training:

```bash
uv run --with modal modal run --env testing benchmarks/modal_h100.py --compare
```

The comparison writes `validation/h100_comparison.json`. It runs each measurement
in a fresh process on the same GPU. Inference uses the same document text,
question and 128-token output limit, with five repetitions after warmup. Token
counts can differ between tokenizers; character throughput is also recorded.
Post-training uses identical text and exact token budgets: 8,192 warmup tokens,
then 32,768 measured tokens, with four AdamW steps. The frozen-backbone LM objective
uses 2,048-token backpropagation windows, 8,192 tokens per step, activation
checkpointing and Liger's fused loss. Warmup, training and checkpoint times are
recorded separately. This synthetic workload measures execution speed; it does
not evaluate the proposed mixed document/QA training recipe.


## Ad hoc inference checks on Modal

```bash
uvx modal run --env testing checks/modal_inference.py --output validation/gpu_inference.json
```

This manually invoked job requests one L4 with a 30-minute execution limit.
It runs eight portable checks from `checks/gpu_inference.py`: kernel output parity,
compiled memory transitions, both methods' streaming and answer isolation, and
untrained checkpoint restoration. Add `--hopper` to run all 17 checks on one H200,
including FA3 and CUDA graph state equivalence. No post-training, QA feedback update, backward
pass or optimizer step runs. Titans' analytic test-time memory transition remains
part of document ingestion; shared parameters remain unchanged.

The file is outside default pytest discovery and has no CI workflow. The report
records package versions, GPU identity, test output and SHA-256 hashes of the
uploaded sources. Small randomly initialized HF backbones establish execution
correctness only. The separate historical `--compare` benchmark above includes
post-training and is not invoked by this check.
