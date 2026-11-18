# Validation record

The reports below describe the revisions that produced them. The serving-engine and CUDA graph measurements are historical. Current training and inference run locally in PyTorch.

## Joint pretraining

The joint-training extension passes **131 CPU tests**, Ruff, and an isolated
wheel import/architecture smoke check. Tests cover joint theta/phi updates,
truncation, eager/compiled execution, custom joint policies, refreezing after
failure, architecture restrictions, checkpoint restoration and the previous
frozen-backbone behavior.

[`cpu_pretraining_smoke.json`](cpu_pretraining_smoke.json) records a separate
full-size CPU check with the real Llama 2 tokenizer: construction of the fixed
373,867,520-parameter decoder and finite, nonzero backward gradients in embeddings,
attention, FFN, output head and memory gates. It uses four input tokens and does
not establish pretraining convergence or long-context quality.

The new `checks/modal_pretraining.py` is an explicitly requested, manual H100
training check, separate from the unchanged inference-only check and outside CI.

[`h100_pretraining.json`](h100_pretraining.json) records the November 18 attempt:
Modal kept the H100 call pending for 301 seconds without a container ID. The
launcher cancelled it. Zero GPU tests and zero GPU optimizer steps executed;
H100 gradient parity is **unverified**, not passed. The report links the attempts
and gives the command for a later manual run. No GPU check was added to CI.

## Earlier architecture extension

The earlier architecture extension passed 111 CPU tests. GPU tests remain skipped. These cover the equation oracle,
outer gradients, finite differences, compiled/eager agreement, training, custom
methods/objectives/update rules, state ownership and checkpoint restoration.

GPU checks are manually invoked through `checks/modal_inference.py`, outside
default pytest discovery and CI. They cover inference only; they do not run
post-training, QA feedback learning or optimizer steps. The earlier measurements
below predate the protocol refactor and remain historical evidence.

That GPU verification was **skipped at the user's request** after Modal left
H100, H200 and L4 calls queued without allocating a container. The queued jobs
were stopped. [`modal_attempts.json`](modal_attempts.json) records the observed
status; zero GPU tests and zero GPU training steps ran for this refactor.

The historical [`optimization_preservation.json`](optimization_preservation.json),
recorded at commit `1cf5ad1` before the architecture extension, compares the
refactor with commit `2260f95668a4c3d40fd6050292a1d2f74d8f5a5c`. All 36 checked
optimization-related function bodies are unchanged, including memory equations,
compiler wrappers, attention execution, CUDA graph handling and SGLang serving.
`gpu_ops.py`, which supplies FA3 and Liger, is byte-for-byte unchanged. This is
source-preservation evidence; it does not substitute for a new GPU performance
measurement. GPU training was not run, as requested.


## Historical Qwen2.5 versus SmolLM2 on one H100

[`h100_comparison.json`](h100_comparison.json) records the completed paired run.
[`h100_comparison_verification.json`](h100_comparison_verification.json) links the
Modal app and verifies that all four measurements match the local package hash,
use identical Titans settings and share one physical H100 80GB. Both backbones
are below two billion parameters: 1,711,376,384 for SmolLM2 and 1,543,714,304 for Qwen.

| Measurement | SmolLM2-1.7B | Qwen2.5-1.5B |
| --- | ---: | ---: |
| Ingestion | 4,823 tokens/s | 4,942 tokens/s |
| Ingestion of identical text | 22,054 characters/s | 22,618 characters/s |
| Generation including query prefill | 184.4 tokens/s | 192.0 tokens/s |
| Estimated decode-only rate | 191.8 tokens/s | 201.0 tokens/s |
| One-token request latency | 32.0 ms | 34.6 ms |
| Training excluding checkpoint writing | 1,437 tokens/s | 1,272 tokens/s |
| Training including checkpoint writing | 1,276 tokens/s | 1,160 tokens/s |
| Measured training time, excluding checkpoint | 22.81 seconds | 25.77 seconds |
| Checkpoint writing | 2.88 seconds | 2.48 seconds |
| Peak allocated training memory | 5.88 GiB | 5.07 GiB |
| Trainable parameters | 18,984,460 | 11,109,900 |

The shared memory configuration has dimensions 64/128, two memory MLP layers,
placements `(0, 6, 12, 18)`, 32-token write chunks, a 512-token attention window,
four persistent tokens and 256-token prefill blocks. Different backbone widths
still produce different adapter parameter counts. Both use BF16, JIT memory
kernels and Liger. CUDA inference runs through SGLang with FA3 and Titans CUDA
graphs; training uses native non-cuDNN SDPA with autograd.

Inference measures medians of five requests after warmup. Each model receives
the same 4,600-character document and question; the document encodes to 1,006
SmolLM2 tokens or 1,005 Qwen tokens. Outputs are forced to 128 tokens. Timing
includes the generation request, state branching and question prefill, but
excludes the subsequent branch-release request. One-token latency is a proxy for
time to first token, measured with a complete one-token request rather than a
streaming response. Decode-only speed subtracts that latency from a matched
128-token request. Source preservation, cloning and seeded sampling passed.

Training calls the public `fit` entry point with a frozen backbone, a
2,048-token backpropagation window, 8,192 tokens per optimizer step, learning rate
`1e-4` and activation checkpointing. Each model processes 8,192 warmup tokens,
then 32,768 measured tokens and four AdamW steps on repeated synthetic document
text. Warmup updates are retained, but the measured `fit` starts a new optimizer
and document stream. Both runs update the memory gates, preserve the sampled
frozen backbone tensor and finish with finite memory parameters and loss.
Checkpoint writes are timed separately; peak memory is PyTorch's allocated
memory in the training process, not total device reservation.

The loss values are not a quality comparison. SmolLM2's warmup/timed mean losses
were 0.238/6.551; Qwen's were 0.0221/0.00587. The repeated text, different tokenizers,
stream positions and short optimization run do not establish useful document
memory. SmolLM2's rising loss needs recipe calibration before a larger training
run. Held-out QA quality remains unmeasured for both models.

### Runtime issues found and corrected

[`qwen_runtime_fixes.json`](qwen_runtime_fixes.json) records the diagnostics and
Modal runs. Qwen exposed three problems that tiny random-model checks missed:

- SGLang's deterministic `addmm` replacement rejects the `alpha`/`beta` arguments
  used by Inductor's shape-padding benchmark. The two biased memory-projection
  kernels now disable that optimization while retaining full-graph JIT.
- The first nonfinite gradient in an unfused Qwen run came from cuDNN attention
  backward. Excluding cuDNN allowed the reference run to complete without changing
  the memory settings or backpropagation window.
- The complete JIT attention wrapper still produced nonfinite gradients with
  Liger disabled. CUDA attention now uses native fused kernels outside that
  wrapper. Memory kernels remain compiled; Liger RoPE, RMSNorm, SwiGLU and fused
  loss stay enabled. CPU attention retains its compiled path.

The final H100 run passes 16 tests, including the SGLang deterministic-mode
regression, rotary output/gradient equivalence and both architectures' training
and CUDA graph checks. The full pretrained benchmarks then verify successful
optimizer updates, which the earlier failing runs could not provide.

## CPU memory-update benchmark

[`memory_cpu_jit.json`](memory_cpu_jit.json) records a float32, two-layer memory
with batch size 2, dimensions 64/128, and 32 tokens per update. Measurements use
one CPU thread on macOS arm64, five warmups and 30 timed repetitions.

| Implementation | Median update latency |
| --- | ---: |
| Literal per-token autograd reference | 2.525 ms |
| Weighted-matmul implementation, eager | 0.153 ms |
| Weighted-matmul implementation, JIT | 0.136 ms |

The JIT result is 18.5× faster than the literal reference and 1.12× faster than the
optimized eager implementation in this case. Its first call took 5.13 seconds,
including compilation/cache loading. The eager update's maximum absolute
difference from the reference was `8.94e-8`; compiled results also passed the
reference tolerance. These timings cover the memory update only. They do not
measure a complete LLM or compare against another optimized Titans library.

## Pretrained smoke check

[`pretrained_cpu.json`](pretrained_cpu.json) records the initial eager integration
check with SmolLM2-135M-Instruct at snapshot
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Disabling the memory output and persistent
prefix reproduced stock Llama logits exactly for the tested prompt. A 55-token,
one-step post-training run updated the memory write gates, preserved the sampled
frozen backbone tensor, and reproduced the generated answer after checkpoint
restoration. The model has 134,515,008 backbone parameters in that check.

The example answer was `Rome.`. The relevant facts fit inside the attention
window, so this is an execution and checkpoint check, not evidence of long-term
memory quality.

## Earlier SmolLM2-only H100 verification

The earlier SmolLM2-only run used an NVIDIA H100 80GB HBM3 in the abundant-ai `testing`
environment. All 10 GPU checks passed, including FA3/local-mask equivalence,
Liger loss/gradient comparisons, compiled memory gradients, a complete BF16
post-training step on a small test model, and CUDA graph state equivalence.
The measured SmolLM2-1.7B-Instruct has 1,711,376,384 backbone
parameters and 1,730,360,844 parameters including memory adapters.

| Measurement | Result |
| --- | ---: |
| Document ingestion | 4,269 tokens/s |
| SGLang generation, including query prefill | 182.7 tokens/s |
| Estimated decode-only rate | 191.4 tokens/s |
| Median time to first token | 37.0 ms |
| Direct numerical decode, without SGLang scheduling | 237.0 tokens/s |

[`h100_inference.json`](h100_inference.json) contains the raw five-repetition
samples, configuration, profiler output and uploaded-package SHA-256. That hash
identifies the earlier implementation before Qwen support. [`h100_runtime.json`](h100_runtime.json) records
PyTorch 2.9.1+cu129, SGLang 0.5.7, Liger 0.6.4 and the FA3 entry point.
[`h100_verification.json`](h100_verification.json) links the completed Modal run
and records the test results.

The benchmark used 1,006-token documents, forced 128-token outputs, BF16, one
active stream, memory at four layers, a 512-token local window and 32-token memory
chunks. Generation timing includes the SGLang call, branch creation and query
prefill; it excludes the subsequent branch-release control request. Decode-only
speed is estimated from matched 128-token and one-token request latencies.
All timed requests follow warmup. The report's `load_startup_and_warmup_seconds`
was 76.7 seconds; this field also includes the optional direct-path profiler.

Source-preservation, cloning and repeated sampling with the same seed passed.
The graph tests compare logits, KV, positions, fast weights, momentum, pending
writes and convolution history with the ordinary forward path. They exercise
all write phases, read-only steps, switching between streams and returning to
prefill. CUDA graph capture does not advance the source stream.

[`h100_before_graphs.json`](h100_before_graphs.json) measured 29.86 generated
tokens/s with fused RoPE and no Titans CUDA graphs. The final SGLang rate is
6.12× higher in this workload. [`h100_initial.json`](h100_initial.json) preserves
the earlier 31.14 tokens/s result before fused RoPE. These are successive versions
of this implementation, not a comparison with another Titans library or the
paper's reported throughput.

The speed benchmark uses newly initialized memory adapters. It does not establish
long-document QA accuracy or the cost of a full 10M-token post-training pilot.
