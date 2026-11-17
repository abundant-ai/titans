<p align="center">
  <img src="assets/titans.svg" width="112" height="112" alt="Titans logo" />
</p>

<h1 align="center">Continual learning with a frozen LLM</h1>

Stream documents through a pretrained Llama or Qwen2 backbone. Supply the memory
method, training data, objective and update policy. Titans is the reference method,
implemented in [titans/titans.py](titans/titans.py).

This is a research library. Its Titans integration is a gated pretrained retrofit,
not a reproduction of the paper's complete architecture or published QA results.
Numerical tests and throughput measurements do not establish long-range recall.

## Install

Python 3.12 or later is required:

```bash
uv sync
# Or install the library into an existing environment:
pip install -e .
```

## Fixed runtime and replaceable policies

| Boundary | Contract |
| --- | --- |
| Fixed | Backbone weights stay frozen; causal ingestion follows arrival order; attention state stays bounded; answers run on isolated branches. |
| Memory method | `MemoryFactory` constructs a `MemoryModule` with `initial_state` and `transition`. The module owns its shared parameters and state representation. |
| Objective | `Objective(LossContext) -> LossValue` defines the loss and its normalization. It can use logits, labels, shared parameters, memory, or combinations. |
| Update policy | `UpdateRule(UpdateContext) -> UpdateResult` chooses targets and optimizer behavior. Built-in SGD selects `phi`, `memory`, or `both`. |
| Data | `DataGenerator(seed) -> Iterable[TrainingEpisode]` defines documents, question timing and training targets. An ordinary iterable also works. |

The contracts are in [protocols.py](titans/protocols.py), with lifecycle and
training semantics in [the protocol guide](docs/protocol.md). The runtime calls
these policies; it does not accept a replacement ingestion loop.

## Instantiate a method

```python
from titans import ContinualModel, Document, Question, RuntimeConfig, TitansConfig, titans

model = ContinualModel.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    config=RuntimeConfig(memory_layers=(0,), attention_window=512),
    memory_factory=titans(TitansConfig(memory_dim=64, memory_hidden_dim=128)),
    device="cpu",
)

state = model.start_stream()
model.ingest(Document("notice-1", "The team meets in room Cedar."), state)
prediction = model.answer(Question("q1", "Where does the team meet?"), state)
model.ingest(Document("notice-2", "The meeting has moved to room Birch."), state)
```

Pass the same state to continue learning. Document boundaries and questions do
not reset it. `answer` makes an isolated branch; its generated text never enters
the continuing stream. Use `clone_state` for an independent fork. The constructor
`TitansModel.from_pretrained(...)` remains a convenience for the same Titans method.

A different representation and transition can use the same runtime. See
[examples/linear_memory.py](examples/linear_memory.py), which has a single linear
associative matrix and no Titans momentum, convolution or pending-write state.

## Learn from released feedback

```python
from titans import GradientUpdate

feedback = model.learn(
    Question("q2", "Where does the team meet now?"),
    "Birch",
    state,
    update=GradientUpdate(targets="memory", learning_rate=1e-3),
)
```

`learn` accepts explicit supervision. Never pass hidden evaluation targets before
scoring a prediction. Its default loss is answer cross-entropy, and its default
update changes active memory W only. Select `phi` to update shared method parameters,
or `both`. Custom losses and optimizers implement the independent protocols.
Teacher-forced question/answer tokens stay on a disposable branch. Their token cost
is reported even though they are not committed as document history.

A phi update invalidates other live streams; the supplied stream continues with
its current state under the updated parameters. These cached activations were
computed before the update. This is an online-learning approximation, not replay
of earlier documents. W-only updates do not affect other streams.

## Post-train with a generator

```python
from collections.abc import Iterable
from pathlib import Path
from titans import TrainingConfig, TrainingEpisode, TrainingQuestion


def generate(seed: int) -> Iterable[TrainingEpisode]:
    """Yield a small example stream using an explicit generation seed.

    Args:
        seed: Run seed supplied by fit; used here only in the episode identifier.

    Yields:
        A document and training question at its arrival checkpoint.
    """
    yield TrainingEpisode(
        f"example-{seed}",
        (Document("d1", "The team meets in room Cedar."),),
        (TrainingQuestion(1, Question("q1", "Where does the team meet?"), "Cedar"),
    )


result = model.fit(
    generate,
    TrainingConfig(Path("checkpoints/run"), max_tokens=10_000,
                   sequence_length=2048, objective="document_qa"),
    update=GradientUpdate(targets="phi", learning_rate=1e-4),
)
```

This tiny episode demonstrates the interface, not a useful training corpus.
Each episode starts a new stream; all documents within it share state. The
custom-update path takes one update per complete QA item and truncates document
history at the configured unroll limit. Feedback must fit in one unroll. If the
remaining token budget cannot fit it, training stops before that item.

Without an explicit `update`, `fit` retains the accumulated AdamW path for phi,
with `document_qa`, `language_modeling`, or `mixed` supervision routing. `loss=` can
replace its cross-entropy as well. Explicit update rules currently use
`document_qa` routing; `tokens_per_step` applies only to accumulated AdamW.

## Save and restore

`save_pretrained` stores the backbone and shared method parameters. It does not
store live W, attention history, or an optimizer-resume checkpoint. A custom-update
fit returns `result.final_state` for continuing its final stream; W-only learning
is retained there, not in the saved shared initialization.

Generic restoration requires the same explicitly supplied method factory and
configuration. The library does not import executable code named by a checkpoint:

```python
model.save_pretrained("checkpoints/saved")
restored = ContinualModel.load_pretrained(
    "checkpoints/saved",
    memory_factory=titans(TitansConfig(memory_dim=64, memory_hidden_dim=128)),
)
```

`TitansModel.load_pretrained` reconstructs the Titans factory and also accepts
checkpoints from the previous library version. Original backbone weights remain
fixed in every supported training mode; `freeze_backbone=False` is rejected.

## Validation and accelerators

```bash
uv run ruff check titans tests benchmarks examples checks
uv run pytest -q
```

Tests cover recurrence and outer gradients, causal split equivalence, bounded
attention, alternative state representations, phi/W/both target selection,
feedback isolation, custom generators/objectives, truncation and checkpoints.
They use tiny locally constructed Llama and Qwen2 models and do not download weights.

GPU checks are ad hoc and excluded from default pytest discovery and CI. Launch
the inference-only check manually from this checkout:

```bash
uvx modal run --env testing checks/modal_inference.py --output validation/gpu_inference.json
```

This uses one L4 for portable kernel parity, streaming, isolated answers and
checkpoint restoration. Add `--hopper` to include FA3 and CUDA graph checks on
one H200. It runs no post-training, QA feedback updates or optimizer steps.
See [the manual checks](checks/README.md) for the exact scope.

The generic plugin and explicit feedback paths use the local PyTorch runtime.
The existing SGLang/CUDA-graph adapter is specific to Titans. Use
`inference_backend="torch"` for explicit streams and feedback. Custom methods do
not automatically inherit the specialized serving adapter.

See [GPU runtime](docs/gpu_runtime.md), [paper alignment](docs/paper_alignment.md),
and [earlier pilot notes](docs/pilot_notes.md). Historical H100 measurements in
[validation](validation/README.md) predate this refactor; GPU performance and
held-out QA quality must be measured for the new protocols.
