<p align="center">
  <img src="assets/titans.svg" width="112" height="112" alt="Titans logo" />
</p>

<h1 align="center">Titans</h1>

<p align="center">Long-term neural memory for pretrained language models.</p>

Abundant's research implementation for attaching memory to a pretrained Llama,
post-training it on document streams, and asking questions as new documents arrive.
One `TitansModel` owns the trained parameters; each `StreamState` holds an independent
stream's memory.

**Status: under active development.** This is a gated-memory retrofit inspired by
[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663).
Its memory core implements the paper's update rule; the pretrained integration is
a separate design choice. Published benchmark results and GPU speedups have not
been reproduced here. Newly attached memory modules need post-training before
their document recall can be meaningfully evaluated.

## Install

Python 3.12 or later is required. From a checkout of this repository:

```bash
uv sync
```

This installs the package and development dependencies, including the pinned
PyTorch and Transformers versions. With pip, use `pip install -e .` for the library.

## Four entry points

| Method | Input | Output | What changes |
| --- | --- | --- | --- |
| `fit` | Training episodes and configuration | `TrainingResult` | Trained parameters; existing stream states become invalid |
| `ingest` | A document and optional stream state | `StreamState` | The supplied stream, in place; omitting it starts a new stream |
| `answer` | A question, stream state and decoding configuration | `Answer` | A disposable branch; the supplied stream stays unchanged |
| `clone_state` | A stream state | `StreamState` | Creates an independent copy, including caches and unfinished writes |

The public records are defined in [`titans/types.py`](titans/types.py).
Training supervision uses `TrainingQuestion`: a question, its target answer, and
the number of documents that should have arrived before asking it.

## Attach and post-train

`from_pretrained` loads a backbone and attaches **new, untrained** memory modules.
Pass a local Hugging Face Llama checkpoint directory or an accessible model ID.
The example uses a local directory so it does not assume model access or download
weights automatically.

```python
from pathlib import Path

from titans import (
    Document,
    Question,
    TitansConfig,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
)

model = TitansModel.from_pretrained(
    "./backbone",  # An existing Llama checkpoint and tokenizer.
    config=TitansConfig(memory_layers=(0,), chunk_size=32),
    device="cpu",  # Use "cuda:0" for a CUDA device.
)

episode = TrainingEpisode(
    episode_id="office-update",
    documents=(
        Document("notice-1", "The team meets in room Cedar."),
        Document("notice-2", "The meeting has moved to room Birch."),
    ),
    questions=(
        TrainingQuestion(1, Question("q1", "Where does the team meet?"), "Cedar"),
        TrainingQuestion(2, Question("q2", "Where does the team meet now?"), "Birch"),
    ),
)

result = model.fit(
    episodes=[episode],
    config=TrainingConfig(
        output_dir=Path("checkpoints/pilot"),
        max_tokens=10_000,
        sequence_length=2048,
        tokens_per_step=8192,
        learning_rate=1e-4,
        objective="mixed",
        freeze_backbone=True,
    ),
)
print(result)
```

This small episode illustrates the data format; it is not enough to train useful
memory. `fit` makes one pass over the supplied episodes and stops at the token
budget or the end of the data. The budget counts forwarded document, question and
target tokens, excluding persistent prefix tokens. The output directory must be new.

The `mixed` objective combines document next-token prediction with answer
supervision. Use `language_modeling` or `document_qa` to select one objective.
The backbone is frozen by default, while the memory components are trained.
`sequence_length` bounds backpropagation through the stream; dependencies beyond
that boundary do not receive training gradients through earlier writes.

## Ingest and answer

After training, create a fresh stream and ask questions at any document boundary:

```python
from titans import AnswerConfig, Document, Question

state = model.ingest(Document("notice-1", "The team meets in room Cedar."))
before_update = model.clone_state(state)

state = model.ingest(
    Document("notice-2", "The meeting has moved to room Birch."),
    state=state,
)
answer = model.answer(
    Question("current-room", "Where does the team meet now?"),
    state=state,
    config=AnswerConfig(max_new_tokens=32, memory_policy="read_only"),
)
print(answer.text)
```

`before_update` keeps the earlier state for comparison. Repeated document IDs are
processed as new arrivals; they do not delete or replace old records automatically.
Correctly learning an updated fact is a model behavior to evaluate.

`answer` supports two policies:

- `read_only`: retrieve from the accumulated fast weights without writing to them.
- `temporary` (default): allow question and generation tokens to update memory on
  a private branch, then discard that branch.

Both preserve the original stream and trained parameters. `Answer.abstained` is
true when the generated text is `[unknown]`; it is not a factuality score.

## Save and restore

`fit` saves a complete checkpoint at `result.checkpoint_path`. Restore its trained
memory with `load_pretrained`:

```python
restored = TitansModel.load_pretrained("checkpoints/pilot", device="cpu")
```

Use `model.save_pretrained("checkpoints/snapshot")` to save another snapshot to a
new directory. Checkpoints include the backbone, tokenizer, memory parameters and
configuration. They exclude live document states and optimizer state. Start a new
stream after loading; states belong to the model instance that created them.

## Memory and efficiency

The memory core uses associative squared-error learning with input-dependent
forgetting, momentum and learning-rate gates. Its chunk update follows equations
12–14 and the chunk-start gradient evaluation described in section 3.2 of the
[paper](https://arxiv.org/html/2501.00663v1).

- Weighted matrix multiplications compute memory updates without allocating a
  parameter-gradient tensor for every token.
- The update remains differentiable for post-training. Inference uses the same
  analytical update without invoking autograd inside the memory loop.
- Attention uses a sliding window in every backbone layer, plus the persistent
  prefix. Old documents do not accumulate in an unrestricted attention cache.
- Fast weights and momentum use float32, or float64 when the model is float64.

`chunk_size=1` uses the tokenwise recurrence. Larger chunks evaluate gradients at
the chunk's initial weights and commit a write when the chunk is complete. Reads
use the last committed weights. Incomplete chunks carry across document boundaries;
increasing chunk size changes update and retrieval timing as well as throughput.

The public model currently supports one stream per call and one device. It accepts
unquantized `LlamaForCausalLM` backbones with static default, linear or Llama 3 RoPE.
Other architectures, sharded device maps and quantized backbones are unsupported.

## Development

```bash
uv run pytest -q
uv run ruff check .
```

The tests include an independent autograd reference for the memory recurrence,
outer-gradient comparisons, causal reads and chunk-boundary behavior. These checks
address implementation correctness; document recall and speed require separate
experiments on held-out streams and the intended hardware.

| File | Responsibility |
| --- | --- |
| [`titans/model.py`](titans/model.py) | Public entry points, Llama integration and checkpoints |
| [`titans/memory.py`](titans/memory.py) | Neural memory and chunk update |
| [`titans/state.py`](titans/state.py) | Stream ownership, caches and independent copies |
| [`titans/training.py`](titans/training.py) | Post-training and token-budget accounting |
| [`titans/types.py`](titans/types.py) | Typed inputs, outputs and configuration |

Reference: Ali Behrouz, Peilin Zhong and Vahab Mirrokni,
[*Titans: Learning to Memorize at Test Time*](https://arxiv.org/abs/2501.00663),
arXiv:2501.00663. This repository is an independent implementation.
