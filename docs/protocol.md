# Continual-learning protocol

The decoder architecture is fixed to one of two supported configurations. Its
parameters theta are frozen during ordinary streaming and trainable during joint
pretraining. Phi is the shared method initialization, projections and other
registered memory/connection parameters. W is algorithm-owned active memory within a stream.
Different methods may use different W representations and auxiliary buffers.

## Fixed ingestion

`ContinualModel.ingest(document, state)` encodes the document with the runtime's
fixed boundary format, processes every token in arrival order, and carries the
same stream through bounded blocks. It neither shuffles documents nor resets W
at a document boundary. Only an omitted state, `start_stream`, or a new training
episode creates a new stream. `ingest` is marked final in the type contract.

Each selected layer calls its method's `transition(hidden, state, write=...)`.
The runtime retains ownership of attention, position accounting, token budgets,
branching and the fixed decoder architecture. The connection protocol controls attention inputs and output fusion, including
attention-driven memory writes. See [architecture and connection contracts](architecture.md)
for the MAC default, available hooks, and segmentation limits.

## Memory method

`MemoryArchitecture(hidden_size) -> MemoryModule` (also exported as `MemoryFactory`) constructs a new independent module
per selected layer. Factory closures may capture arbitrary algorithm settings.
The runtime moves registered parameters to the backbone's device and dtype.

The module provides:

- `initial_state(batch_size)`: independent tensors, optionally differentiable
  copies of a learned initialization.
- `transition(hidden, state, write)`: a causal memory read and optional state
  update. It returns the same shape/device/dtype as `hidden`.

A transition must preserve meaning when input blocks are split across calls.
It may keep incomplete update chunks in its state. With `write=False`, adaptable
memory stays unchanged, although read-side causal buffers can advance on the
isolated branch. A transition must not update shared parameters. Explicit phi
updates belong to the update policy, where revision invalidation can occur.

During training, enabled autograd must retain the differentiable transitions.
At inference, the runtime ordinarily uses `torch.no_grad`. Methods requiring
inner autograd must enable it locally and release the graph afterward. Titans
uses analytic differentiable updates and needs no inner autograd invocation.

Algorithm state provides `copy`, `detach`, `tensors`, `adaptable`, and `replace`.
Copy must clone every mutable tensor. Detach must sever every graph. Adaptable
returns named floating-point W tensors; replace accepts shape-compatible updates
to those names. Momentum, replay buffers or other history belong to that same
state, not global variables. `StreamState.memory` exposes it by layer.

All Titans method code is in `titans/methods/titans.py`: settings, state, associative MLP,
update equations, projections, and the `titans(config)` factory. `kernels.py`
contains compiler wrappers and shared backbone execution kernels.

## Objective

`Objective(LossContext) -> LossValue` is independent of update target selection.
The context includes aligned logits `[targets, vocabulary]`, token IDs `[targets]`,
stream state, phi parameters and a document/answer phase marker. In joint training,
`backbone_parameters` separately supplies theta. Return a scalar
loss sum and a finite positive normalization weight. The default cross-entropy
uses the number of target tokens as the weight. An arbitrary state regularizer
can instead use unit weight.

On explicit QA learning, context.state is the pre-question stream. Prompt and
teacher tokens exist only on a disposable branch with memory writes disabled.
In accumulated LM/QA training, the context is the currently processed branch.
Custom objectives should use the appropriate state semantics for their chosen
training path. They must not mutate state or parameters.

## Update policy

`UpdateRule(UpdateContext) -> UpdateResult` receives normalized scalar loss,
shared phi parameters and named active W tensors. Joint training also supplies
theta in `backbone_parameters`. The policy can
select a subset, use autograd or another solver, and own optimizer state.

It changes selected phi/theta tensors in place and returns W replacements keyed by
`(layer_index, parameter_name)`. It must accurately report `shared_changed`.
The runtime validates replacement names, shapes, dtypes, devices and finite values,
commits replacements to the continuing stream, and detaches its old graph.
Momentum is not automatically reset when W is replaced; a method defines how
its auxiliary state should respond in `replace`.

The built-in `GradientUpdate` performs clipped SGD on `phi`, `memory`, `both`
(phi and W), or `joint` (phi and theta).
If targets include phi, the gradient can pass through the current document unroll.
Selecting both computes both derivatives on the same loss graph before changes.
Learning W directly does not change the shared initial W0 unless the selected
phi objective also provides a gradient to W0.

After a shared update, other streams are stale and rejected. The supplied stream
keeps its existing history; it is not reconstructed under the new parameters.
If a custom updater fails after it may have changed phi, the runtime invalidates
existing streams. This is a correctness boundary, not a sandbox for untrusted code.

## Data and checkpoint scheduling

`DataGenerator(seed)` returns a lazy iterable of `TrainingEpisode`. An episode
contains ordered documents and questions with `after_documents` checkpoints.
Zero means before the first document; N means after the Nth document. Questions
at the same checkpoint are handled in their supplied order.

The generator controls source material, facts, updates, distractors, question
content, labels and curriculum. The runtime fixes chronology and when labels
become available. The generator does not receive hidden evaluation data.

The same `ingest`, `answer` and `learn` operations support an ongoing application
stream. Prediction never triggers learning from a hidden label. A caller may
score a prediction, explicitly release feedback, then call learn. Generated
supervision is also possible: generating it is the caller's policy, and its cost
must be counted by the experiment.

## Training paths and gradient boundaries

The accumulated AdamW path updates phi and, in joint training, theta. It supports document LM, answer QA or
mixed supervision, custom objectives and token-based gradient accumulation.
The existing Liger optimization is used only for its built-in cross-entropy.

Supplying `update=` selects checkpoint training. Each QA item is an explicit
learning boundary with the same semantics as `learn`. Its complete prompt and
answer must fit inside sequence_length and the remaining token budget. If
recent document history plus feedback would exceed the unroll, that history is
detached before feedback. Acquisition examples therefore need enough unroll
space for both their supporting documents and answer supervision.

The graph is detached after each update and at document unroll boundaries.
Values survive truncation; gradients cannot reach writes before that boundary.
W-only checkpoint training produces a continuing state but does not post-train
shared initialization. The final state is returned separately from the saved
model artifact. Feedback losses are averaged per checkpoint in this path;
accumulated training normalizes by the objective's supplied weights.

Test-time document updates run the chosen transition. Test-time QA adaptation
requires an explicit learn call with released supervision. Both keep theta frozen
by default; `learn(..., train_backbone=True)` explicitly enables joint feedback updates. For exact experiment replay, record runtime settings, factory settings,
objective/update implementation, generator seed and checkpoint revision together.

## Scope

The default Titans path retains compiled memory kernels, Liger backbone kernels
and fused cross-entropy, activation checkpointing, truncated backpropagation and
token-based gradient accumulation. Titans inference retains SGLang, FA3 and CUDA
graphs. The generic runtime can enable its shared compile/Liger settings;
specialized Titans serving still requires `TitansModel`.

An arbitrary custom objective uses ordinary logits/loss evaluation rather than
the fused cross-entropy shortcut. Custom checkpoint updates use their own policy
instead of the accumulated AdamW schedule. These are explicit extension choices;
the original optimized default training path remains available.

The fixed scratch Llama and pretrained Qwen2.5-1.5B-Instruct are supported on one device.
See [pretraining](pretraining.md) for joint theta/phi updates. Generic methods and explicit feedback
run locally in PyTorch. Remote SGLang state cannot be passed to learn; the existing
Titans serving specialization is retained for ingestion and answering. Shared
checkpoint save/load is supported; live-state and optimizer-resume serialization
are not yet provided. No held-out continual-learning quality claim follows from
the protocol tests alone.
