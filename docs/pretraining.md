# Pretraining and backbone selection

There are two supported starts. `TitansModel.from_scratch()` initializes a fixed
Llama-style decoder, and `TitansModel.from_pretrained()` loads only
`Qwen/Qwen2.5-1.5B-Instruct`. `ContinualModel` provides the same choices for custom
memory methods. Backbone configuration overrides and arbitrary pretrained model
IDs or directories are rejected. `load_pretrained(path)` restores a complete
library checkpoint with one of the supported decoder configurations.

## Fixed scratch decoder

| Setting | Value |
| --- | --- |
| Decoder blocks | 24 |
| Hidden width | 1,024 |
| Query / KV heads | 16 / 16 |
| SwiGLU intermediate width | 2,816 |
| Vocabulary | 32,000 |
| Position limit / RoPE base | 4,096 / 10,000 |
| RMSNorm epsilon | 1e-5 |
| Attention / MLP biases | Disabled |
| Embedding and output weights | Untied |
| Attention dropout | 0 |
| Initializer standard deviation | 0.02 |

The block count, width and head count follow the paper's 340M configuration
([Table 5](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf#page=32)).
The remaining settings are this library's fixed Llama-style choices. The decoder
has **373,867,520 parameters** before memory and connections, so this is not an
exact reproduction of the paper's 340M parameter budget. The Llama 2 tokenizer
is loaded from the public `NousResearch/Llama-2-7b-hf` mirror; its model weights
are never downloaded. Initialization uses an isolated seed.

## Joint training

```python
from pathlib import Path
from titans import Document, TitansModel, TrainingConfig, TrainingEpisode

model = TitansModel.from_scratch(device="cuda", seed=17)
episodes = [TrainingEpisode("stream-1", (
    Document("doc-1", "Alice lives in Paris."),
    Document("doc-2", "Bob lives in Rome."),
))]
result = model.fit(episodes, TrainingConfig(
    output_dir=Path("checkpoints/pretraining"),
    max_tokens=100_000,
    sequence_length=512,
    tokens_per_step=8192,
    objective="language_modeling",
))
```

Replace the toy episodes with a data generator for a useful run. Training counts
actual forwarded tokens, including the library's document wrappers. This is a
single-device trainer, without distributed sharding or optimizer-resume support.

The inner transition updates active memory W while processing the stream. The
outer AdamW optimizer updates backbone parameters theta and shared memory,
connection and persistent-prefix parameters phi. Active W is never an outer
optimizer parameter. Its learned initialization is part of phi.

Backpropagation includes memory writes within `sequence_length`. The trainer
scores the next known supervised token before detaching a window's prediction.
The target is used only as a label at that point; its embedding enters the stream
when forwarding resumes. Eager and Liger use the same labels and normalization.
This applies to language-model targets and explicit QA targets, including an
answer beginning exactly at a window boundary.

At a cut, memory values and real-token caches survive while their gradient history
is detached. Persistent-prefix KV is recomputed using the current parameters so
later windows can train the persistent embeddings. Prefix refresh does not
write to fast memory, advance real-token positions, or consume token budget.
Real-token caches carried across an optimizer step still reflect the parameters
used when those tokens arrived. The model returns to frozen-backbone execution
after fitting, including on a training failure.

`TrainingConfig.freeze_backbone=None` selects joint training for scratch Llama
and frozen-backbone training for Qwen. Set it explicitly to `True` for memory-only
training or `False` for joint training on either supported decoder.

Custom `Objective` receives phi in `shared_parameters` and, in joint mode, theta
in `backbone_parameters`. The same separation applies to `UpdateContext`.
`GradientUpdate(targets="joint")` updates theta and phi; existing `phi`, `memory`
and `both` selections keep their meaning. Custom update rules in `fit` retain
checkpointed `document_qa` routing. `learn(..., train_backbone=True)` exposes theta
for explicitly released QA feedback. Prediction never receives hidden targets.

Memory architecture, connection factory, inner transition, data generator and
objective remain configurable. The default MAC segment holds 512 real tokens and the default
memory write chunk holds 16. Standard training on BF16-capable CUDA uses BF16
autocast for backbone operations. With default storage, parameters, AdamW state,
fast weights, momentum and loss reductions use FP32. Explicit storage dtypes
and custom update rules retain their original ownership.

## Manual H100 validation

```bash
uvx modal run --env testing -m checks.modal_pretraining \
  --output validation/h100_memory_gradients.json
```

The job compares eager and Liger forward/backward passes on the fixed scratch
decoder with 33 deterministic token IDs. It measures peak allocated GPU memory,
prefill throughput and vocabulary-head calls for tokenwise and block SDPA on the
same weights. Optimizer-step tests run on CPU. GPU checks are
manual and outside CI.

The [H100 report](../validation/h100_memory_gradients.json) includes the fixed-input
losses, gradient differences, throughput, peak allocation and source hashes.


## Choose whether episodes share memory

**Set `TrainingConfig.episode_memory` explicitly when preparing a training run.**

| Value | Behavior at the start of each episode | Use |
| --- | --- | --- |
| `"reset"` (default) | Fresh fast weights from learned initialization, zero momentum, empty pending writes and document caches | Independent examples, users, or worker-local streams |
| `"carry"` | Continue the previous episode's document state inside this `fit` call | Consecutive parts of one ordered continual-learning stream |

```python
# Independent episodes, including datasets partitioned between workers.
config = TrainingConfig(
    output_dir=Path("checkpoints/independent"),
    max_tokens=10_000_000,
    objective="language_modeling",
    episode_memory="reset",
)

# Consecutive episodes from the same document stream.
config = TrainingConfig(
    output_dir=Path("checkpoints/continuous"),
    max_tokens=10_000_000,
    objective="language_modeling",
    episode_memory="carry",
)
result = model.fit(episodes, config)
continuing_state = result.final_state
```

Carry mode preserves fast weights, momentum, incomplete write chunks, convolution
history, segment retrieval state, real-token KV, and stream counters. Gradient
history is detached between episodes. Persistent-prefix representations are
refreshed from the current learned embeddings.

Episode boundaries still delimit supervised sequences. The first document token
of a carried episode is not scored against the preceding episode's last prediction.
Within each episode, next-token targets cross execution blocks, training windows,
and document boundaries. For a fully continuous language-model objective, place
the related documents in one episode.

`TrainingQuestion(after_documents=0, ...)` runs before any documents in the
current episode. In carry mode, it can already access documents from preceding
episodes. All `after_documents` counts remain local to their episode, while
`state.documents_seen` is cumulative. Standard QA branches discard teacher tokens.
Custom updates through `learn` can deliberately change the continuing memory.

Both modes retain learned shared parameters and any trained backbone weights.
A new `fit` call starts a new stream. Carry mode returns its final in-memory
state in `TrainingResult.final_state`; `save_pretrained` saves model parameters,
and does not serialize that live stream or resume optimizer state.

**This flag does not enable parallel or distributed training.** Each worker must
own its model execution and stream state. Use `"reset"` for independent episodes.
Use `"carry"` only where each worker receives consecutive parts of its own stream.
The built-in trainer remains sequential and single-device.

For custom objectives, `LossContext.state` describes the forwarded prefix. A
final lookahead target may appear in `targets` before that token is ingested.
Callbacks can occur for document or answer supervision, including a lookahead
answer at the end of its question prompt.
