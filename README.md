<p align="center">
  <img src="assets/titans.svg" width="112" height="112" alt="Titans logo" />
</p>

<h1 align="center">Continual learning and joint pretraining</h1>

Start from a fixed random Llama-style decoder or pretrained Qwen2.5-1.5B-Instruct. Supply the memory
method, training data, objective and update policy. Titans is the reference method,
implemented in [titans/methods/titans.py](titans/methods/titans.py). The default MAC connection uses
512 real tokens per segment and commits a memory write every 16 tokens.
MAG and the residual connection are also selectable.

This is a research library. Its Titans integration supports joint pretraining and
frozen-backbone adaptation; it is not a reproduction of the paper's complete architecture or published QA results.
Numerical tests and throughput measurements do not establish long-range recall.

## Install

Python 3.12 or later is required:

```bash
uv sync
# Or install the library into an existing environment:
pip install -e .
```

On Linux with CUDA, run `uv sync --extra gpu` to install Liger. Training and inference use local PyTorch SDPA and `torch.compile`.

## Fixed runtime and replaceable policies

| Boundary | Contract |
| --- | --- |
| Fixed | Two fixed decoder choices; causal ingestion follows arrival order; attention state stays bounded; answers run on isolated branches. |
| Architecture and connection | `MemoryArchitecture` builds the neural memory; `ConnectionFactory` controls its backbone wiring. Defaults: Titans memory and causal MAC. See [architecture contracts](docs/architecture.md). |
| Memory method | `MemoryFactory` constructs a `MemoryModule` with `initial_state` and `transition`. The module owns its shared parameters and state representation. |
| Objective | `Objective(LossContext) -> LossValue` defines the loss and its normalization. It can use logits, labels, shared parameters, memory, or combinations. |
| Update policy | `UpdateRule(UpdateContext) -> UpdateResult` chooses targets and optimizer behavior. Built-in SGD selects `phi`, `memory`, `both`, or `joint` (phi + backbone). |
| Data | `DataGenerator(seed) -> Iterable[TrainingEpisode]` defines documents, question timing and training targets. An ordinary iterable also works. |

The contracts are in [protocols.py](titans/core/protocols.py), with lifecycle and
training semantics in [the protocol guide](docs/protocol.md). The runtime calls
these policies; it does not accept a replacement ingestion loop.

## Repository layout

```text
titans/
  core/       # Protocols, configuration and input/output records
  runtime/    # Fixed backbone choices, stream lifecycle and connections
  methods/
    titans.py # Titans architecture, transition, MAC/MAG and convenience model
  training/   # Post-training loop, objectives and feedback update policies
  optim/      # PyTorch compiled kernels and optional Liger CUDA operations
tests/
  support/    # Shared offline fixtures and independent numerical oracle
checks/       # Manual inference-only GPU checks, outside CI
benchmarks/   # Throughput and post-training measurements
examples/     # Alternative memory implementations
docs/         # Contracts, architecture and runtime guides
```

Public imports such as `from titans import ContinualModel, TitansModel` stay the
same. Internal modules use the package paths above. Checkpoint loading accepts
the previous built-in connection paths as well as the current ones.

## Start from scratch

```python
from titans import TitansModel

model = TitansModel.from_scratch(device="cuda", seed=17)
```

The decoder has 24 blocks, hidden width 1,024 and 16 attention heads. Its remaining
architecture settings are also fixed. `fit` defaults to joint backbone/memory
training for this model. See [pretraining](docs/pretraining.md) for the exact
configuration, training example and manual H100 check.

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
checkpoints using the supported decoder configurations. Qwen defaults to a frozen
backbone; scratch Llama defaults to joint training. `freeze_backbone` can explicitly
select either training mode. Arbitrary backbone IDs and local backbone directories
are no longer accepted by `from_pretrained`.

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
uvx modal run --env testing -m checks.modal_inference --output validation/gpu_inference.json
```

This uses one L4 for portable kernel parity, streaming, isolated answers and
checkpoint restoration. Add `--hopper` to run the same suite on one H100. It runs no post-training, QA feedback updates or optimizer steps.
See [the manual checks](checks/README.md) for the exact scope.

See [GPU runtime](docs/gpu_runtime.md), [paper alignment](docs/paper_alignment.md),
and [earlier pilot notes](docs/pilot_notes.md). Historical H100 measurements in
[validation](validation/README.md) predate this refactor; GPU performance and
held-out QA quality must be measured for the new protocols.


### Memory across training episodes

`TrainingConfig(…, episode_memory="reset")` starts each episode with fresh
memory and is the default. Use `episode_memory="carry"` to continue document
memory sequentially across episodes within one `fit` call. Question checkpoint
counts stay local to each episode. This option does not enable parallel execution
or share state between workers. See [episode memory policy](docs/pretraining.md#choose-whether-episodes-share-memory)
for examples, gradient boundaries and checkpoint behavior.
