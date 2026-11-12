# Validation record

The CPU suite passes 55 tests, and all 10 CUDA tests pass on the H100. CUDA tests
are skipped on the local Mac. The CPU checks include the independent equation
oracle, outer gradients, finite differences, compiled/eager agreement, training,
state ownership and checkpoint restoration. Ruff and the explicit-type/docstring
audit pass.

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

## H100 verification

The final run used an NVIDIA H100 80GB HBM3 in the abundant-ai `testing`
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
samples, configuration, profiler output and uploaded-package SHA-256. The final
local package matches that hash. [`h100_runtime.json`](h100_runtime.json) records
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
