# Paper alignment

The reference is [Titans: Learning to Memorize at Test Time, v1](https://arxiv.org/html/2501.00663v1).
This library implements its associative neural-memory updates and attaches them
to a pretrained Llama or Qwen2 model. The retrofit has its own retrieval schedule and residual
gate. It does not reproduce the paper's complete from-scratch MAG architecture
or its published evaluation results.

| Paper component | Implementation | Evidence |
| --- | --- | --- |
| Associative squared-error objective, eq. 12 | Bias-free MLP, SiLU hidden layers, **summed** squared error | Independent autograd oracle and finite differences |
| Forgetting and momentum, eqs. 13–14 | Input-dependent alpha, eta and theta; independent fast weights and momentum per stream | Final matrices and outer derivatives compared with literal recurrence |
| Chunk-start gradients, §3.2 | All gradients in a write chunk evaluated at its initial weights | Oracle evaluates each token's gradient separately at those weights |
| Retrieval, eq. 15 | Forward pass through committed fast weights | Read-only policy and source-preservation tests |
| Persistent inputs, §3.3 | Learned prefix embeddings with retained attention KV | Bounded-cache, checkpoint and causality tests |
| Gated memory, §4.2 | Add a normalized, projected memory branch to the host's attention residual | Explicit retrofit choice; stock Llama/Qwen2 equivalence when disabled |

## Exact recurrence and optimized chunks

For each token, the core rule is

```text
loss_i = ||MLP(M_start, key_i) - value_i||²
S_i    = eta_i * S_(i-1) - theta_i * gradient(loss_i)
M_i    = (1 - alpha_i) * M_(i-1) + S_i
```

With `chunk_size=1`, `M_start` is the preceding token's updated memory.
With larger chunks, `M_start` stays fixed through that chunk, following §3.2.
The two settings are different learning trajectories. Increasing chunk size
is an algorithmic choice as well as a performance setting.

`chunk_update` reduces the gradients with two scalar coefficient sequences.
Writing `b_i = 1 - alpha_i` and `g_i = gradient(loss_i)`, the final state is

```text
S_n = product(eta) * S_0 - sum_i theta_i * product_(j>i)(eta_j) * g_i
M_n = product(b) * M_0 + eta_0 * h_0 * S_0 - sum_i theta_i * h_i * g_i
h_(n-1) = 1
h_i = product_(j>i)(b_j) + eta_(i+1) * h_(i+1)
```

The implementation computes suffix products without division, so gates equal to
zero remain valid. A parallel affine scan computes `h`. Analytic MLP derivatives
then reduce the two weighted gradient sums with batched matrix multiplications.
It never allocates a `[batch, tokens, number_of_memory_parameters]` gradient tensor.
Ordinary differentiation through these expressions preserves outer gradients.
The squared-error gradient includes the factor of two from eq. 12; the learning
rate is consequently calibrated to that convention.

## Retrieval and integration choices

Reads use the last **completed** write chunk. A partial chunk retains keys, values
and gates across calls and document boundaries. Questions can therefore arrive
while a chunk is incomplete. `read_only` reads the committed memory; `temporary`
can complete the pending chunk on its disposable branch. Neither policy modifies
the source stream. The local attention window covers recent tokens independently.

The host keeps its pretrained MLP, normalization and rotary embedding
parameters. A sigmoid gate controls the added memory residual. Memory QKV uses
causal depthwise convolution, SiLU, and L2-normalized queries and keys. Persistent
prefix tokens do not write into fast memory. These choices define this retrofit;
they should be held fixed when comparing memory-update algorithms.

`prefill_chunk_size` controls host-model work blocks independently of memory
`chunk_size`. Changing prefill splitting preserves the computation within floating
point tolerance; changing the memory chunk size changes the update rule.

## What verification establishes

The CPU suite compares depths 1–3, ordinary and extreme gates, nonzero momentum,
outer derivatives, causality, arbitrary input splits, deep cloning, training
budgets and checkpoint round trips. Compiled kernels are checked separately
against the same autograd oracle. The GPU suite compares FA3 against an explicit
SDPA mask and compares Liger losses and gradients with PyTorch. CUDA graph checks
compare logits, caches, fast weights, momentum, pending writes and convolution
history with the ordinary forward path across commits and stream switches.

These checks establish numerical and API contracts for the tested configurations.
They do not establish learned long-document recall. That requires post-training
and held-out streams containing new facts, conflicting updates, distractors and
unanswerable questions. Score each question against the documents available at
its checkpoint. Keep answers out of the committed stream and report ingestion
cost, generation cost and memory use alongside accuracy. Local attention across
multiple layers can carry information beyond one layer's window, so a single
window-length distractor gap does not by itself isolate neural memory.
