<p align="center">
  <img src="assets/titans.svg" width="112" height="112" alt="Titans logo" />
</p>

<h1 align="center">Titans</h1>

<p align="center">Long-term neural memory for pretrained language models.</p>

Abundant's research implementation for attaching memory to a pretrained Llama or Qwen2,
post-training it on document streams, and asking questions as new documents arrive.
One `TitansModel` owns the trained parameters; each `StreamState` holds an independent
stream's memory.

**Status: under active development.** This is a gated-memory retrofit inspired by
[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663).
Its memory core implements the paper's update rule; the pretrained integration is
a separate design choice. The paper's published quality results have not
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
Pass a local Hugging Face Llama or Qwen2 checkpoint directory or an accessible model ID.
The default backbone for examples and benchmarks is `Qwen/Qwen2.5-1.5B-Instruct`.
The example downloads its weights on first use; a local checkpoint path also works.

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
    "Qwen/Qwen2.5-1.5B-Instruct",
    config=TitansConfig(memory_layers=(0,), chunk_size=32),
    device="cpu",  # For CUDA/BF16, follow the GPU setup below.
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

## Post-training recipe

Start with a 10M-token pilot on Qwen2.5-1.5B-Instruct, keeping the backbone frozen
and training the memory components and persistent prefix. The goal is to learn
how to store, retrieve and revise facts as documents arrive. This is a proposed
recipe: the numerical implementation has been tested, but this corpus mix and
its long-document QA quality have not been evaluated yet.

### Data sources and preparation

The following mix allocates 10M processed tokens, including document wrappers,
questions and target answers. Count tokens with the backbone's tokenizer; repeated
passes count again. Reserve validation and test data before filling these budgets.

| Pilot allocation | Source | Role in training |
| --- | --- | --- |
| 6M tokens | Abundant documents available for training, or a streamed sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Natural document language and realistic distractors. FineWeb-Edu contains educational web text; it does not provide the checkpointed question/answer labels this recipe needs. |
| 3M tokens | Generated fact-update streams with a programmatically maintained answer key | Introduce fictional entities and random identifiers, revise their facts, and ask questions before and after each revision. Mix in distractors, delayed questions and unanswerable questions. |
| 1M tokens | The training split of [QASPER](https://huggingface.co/datasets/allenai/qasper/blob/main/README.md), including its paper text | Questions with annotated answers and supporting evidence from research papers. Keep examples answerable from text; omit questions requiring unavailable figures or tables. |

Deduplicate and split by original document or paper before chunking. Keep related
versions in the same split. For generated streams, separate generator seeds and
template families across training and evaluation. Record source IDs, dataset
revisions, split assignments and attribution alongside the prepared episodes.
FineWeb-Edu lists ODC-By and Common Crawl terms; QASPER lists CC BY 4.0 in their
linked dataset cards.

Convert each stream into a `TrainingEpisode` with ordered `Document` records.
Attach each `TrainingQuestion` to the appropriate `after_documents` checkpoint.
Use only evidence available at that checkpoint, and use `[unknown]` when the
answer is absent. For QASPER, ask after the complete paper has arrived unless the
evidence annotations establish an earlier valid checkpoint. Training targets stay
on disposable branches and never become committed document history.

Natural documents can contribute language-modeling loss without QA labels.
Additional QA pairs can be written by people or generated by a teacher model,
then checked against their supporting passages. Teacher generation and review
are separate costs. Corpus converters and a complete quality-evaluation harness
are not bundled with this library; the current training entry point consumes
prepared `TrainingEpisode` objects.

### Training and evaluation

1. Load the BF16 backbone on one H100 using the configuration in
   [GPU setup](docs/gpu_runtime.md): memory layers `(0, 6, 12, 18)`, memory
   dimensions 64/128, `chunk_size=32`, `attention_window=512` and
   `prefill_chunk_size=256`. Enable compiled kernels and Liger.
2. Run a separate calibration job on roughly 50k to 100k processed tokens.
   Record compilation time, warmed training tokens/s, peak GPU memory, loss and
   whether the memory parameters change. Use this measured training rate to
   update the cost estimate below. Reload the original initialization for the
   pilot so calibration examples do not become an uncounted extra training pass.
3. Start the pilot with `max_tokens=10_000_000`, `objective="mixed"`,
   `freeze_backbone=True`, `learning_rate=1e-4`, `sequence_length=2048`,
   `tokens_per_step=8192`, `max_grad_norm=1.0` and
   `activation_checkpointing=True`. Supply enough prepared episodes to consume
   the budget; `fit` stops when the input ends and does not repeat it automatically.
   Each output directory must be new. Training uses PyTorch autograd and Liger;
   SGLang, cached FA3 attention and decode CUDA graphs serve inference.
4. Begin with short fact-to-question gaps that fit inside an autograd unroll,
   then include longer streams and distractor gaps. Stream state carries across
   unrolls, but gradients stop at each `sequence_length` boundary. A longer input
   stream does not by itself train credit assignment across that boundary.
   Increase the unroll length only after measuring its memory and speed costs.
5. Evaluate a separate document split and unseen generated streams at increasing
   lengths, for example 2k, 8k and 32k tokens. Ask at intermediate arrivals as well
   as at the end. Report answer accuracy or token F1, accuracy after fact updates,
   abstention precision/recall, and performance by evidence-to-question distance.
   Keep gold answers out of the document stream. Compare with an otherwise
   identical model with memory reads disabled, and with a control that resets
   the entire neural-memory state before each question. Keep decoding settings
   and the query-time memory policy fixed across comparisons. Multiple attention
   layers can carry facts beyond one layer's window, so exceeding 512 tokens
   alone does not isolate memory.

QASPER's validation split supplies a separate real-document check.
[BABILong](https://github.com/booydar/babilong) provides an additional evaluation
of facts distributed among distractors; use its evaluation sets without mixing
them into the training corpus. A streaming adaptation must document its chunking
and question schedule, since those choices affect comparability with published
scores. Scale to 100M training tokens only after the pilot improves held-out QA
over the controls. Report ingestion speed, answer latency and peak memory beside
quality scores.

`fit` saves the trained checkpoint at completion. It does not currently save
periodic optimizer-resume checkpoints. Once the memory module is trained, load
that checkpoint and use `ingest` for new documents; another `fit` is not required
for every document.

### Compute cost estimate

Budget $20 to $50 for an initial 10M-token pilot, including room for startup,
checkpointing, evaluation and small retries. This is a planning allowance, not a
measured training bill or a guarantee of useful memory quality.

[Modal's listed rates](https://modal.com/pricing), checked on November 12, 2026,
are $0.001097 per H100 GPU-second, $0.0000131 per CPU-core-second and $0.00000222
per GiB-second of host RAM. One H100 with 8 physical CPU cores and 32 GiB RAM is
about $4.58/hour at those rates, before credits or pricing premiums.

The [matched H100 benchmark](validation/h100_comparison.json) measured frozen-backbone
post-training at 1,437 tokens/s for SmolLM2 and 1,272 tokens/s for Qwen2.5, excluding
compilation and checkpoint writing. Each model completed 32,768 measured tokens
and four optimizer steps after 8,192 warmup tokens, with a 2,048-token
backpropagation window and Liger enabled.

| Backbone | Measured training tokens/s | Extrapolated time for 10M tokens | Processing compute at $4.58/hour |
| --- | ---: | ---: | ---: |
| SmolLM2-1.7B-Instruct | 1,437 | 1.93 hours | $8.86 |
| Qwen2.5-1.5B-Instruct | 1,272 | 2.18 hours | $10.01 |

These are extrapolations from repeated synthetic document text with an LM
objective. They are not measurements of the proposed mixed document/QA corpus
or a full 10M-token run. SmolLM2's mean loss rose from 0.24 during warmup to 6.55
in the timed run; calibrate the training recipe before committing to the pilot.
Finite optimizer updates establish that training executes, not that it improves
memory quality. Both models still need held-out document QA evaluation.

Calculate processing cost as
`processed_tokens / training_tokens_per_second / 3600 * hourly_rate`.
The table excludes startup, compilation, checkpointing, evaluation, data
preparation and teacher-model calls. In this cached environment, each 8,192-token
warmup took about 21 seconds including checkpointing; first-time setup can take
longer. The measured checkpoint writes took 2.5 to 2.9 seconds. Three passes
through 10M tokens count as 30M processed tokens. Recalibrate when changing the
objective, backpropagation horizon, memory dimensions or backbone training policy.

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
unquantized `LlamaForCausalLM` and `Qwen2ForCausalLM` backbones with static default,
linear or Llama 3 RoPE.
Other architectures, sharded device maps and quantized backbones are unsupported.

## CUDA inference and post-training

CUDA inference defaults to a local SGLang engine with a custom Titans model.
The default H100 backbone is Qwen2.5-1.5B-Instruct. SmolLM2-1.7B-Instruct remains
supported as a comparison baseline; both are below two billion parameters.
Use BF16 and install the Linux GPU dependencies with `uv sync --extra gpu`.

Memory kernels use `torch.compile(fullgraph=True)` by default. CUDA operations use
Liger RMSNorm, RoPE and SwiGLU; training uses its fused linear cross-entropy loss.
Inference uses FlashAttention 3 with local attention and a correctly normalized
persistent-prefix contribution. `prefill_chunk_size` controls host-model work
blocks independently of the memory update's `chunk_size`.

The initial SGLang adapter supports one active request on one GPU. Its worker
owns the full Titans stream state; ordinary token-prefix caching cannot substitute
for fast weights and momentum. Native prefix sharing and CUDA graphs are disabled
in SGLang. The Titans-specific CUDA graph runner captures the full memory update
and bounded cache once the attention window is full. Call `model.close()`
to release a serving worker when finished.

See [GPU setup and benchmark commands](docs/gpu_runtime.md),
[equation alignment and verification scope](docs/paper_alignment.md), and
[recorded validation results](validation/README.md).


## Matched H100 results

Both models ran sequentially on the same H100 80GB in BF16, using SGLang 0.5.7,
FA3 inference, Liger and compiled memory kernels. Memory settings were identical:
layers `(0, 6, 12, 18)`, dimensions 64/128, a 512-token attention window, 32-token
write chunks and four persistent tokens.

| Measurement | SmolLM2-1.7B | Qwen2.5-1.5B |
| --- | ---: | ---: |
| Ingestion | 4,823 tokens/s | 4,942 tokens/s |
| Generation, including query prefill | 184.4 tokens/s | 192.0 tokens/s |
| One-token request latency | 32.0 ms | 34.6 ms |
| Post-training, excluding checkpoint writing | 1,437 tokens/s | 1,272 tokens/s |
| Peak allocated training GPU memory | 5.88 GiB | 5.07 GiB |

Qwen generated about 4% faster in this run and used about 14% less training memory.
SmolLM2 trained about 13% faster. The inference differences are small and should
be remeasured for the intended workload. Inference uses five warmed repetitions
and forced 128-token outputs; training uses four measured optimizer steps on
synthetic LM data. Training attention uses native non-cuDNN SDPA, while the
Titans memory kernels remain JIT-compiled. These measurements do not establish
long-term recall quality.

The [raw comparison](validation/h100_comparison.json) records samples, parameter
counts, configuration and source hashes. See the [validation record](validation/README.md)
for verification and the CUDA compatibility fixes needed by Qwen.

## Development

```bash
uv run pytest -q
uv run ruff check .
```

The tests include an independent autograd reference for the memory recurrence,
outer-gradient comparisons, causal reads and chunk-boundary behavior. These checks
address implementation correctness; document recall and speed require separate
experiments. H100 throughput measurements are in the validation record;
document recall still requires post-training and held-out streams.

| File | Responsibility |
| --- | --- |
| [`titans/model.py`](titans/model.py) | Public entry points, Llama/Qwen2 integration and checkpoints |
| [`titans/memory.py`](titans/memory.py) | Neural memory and chunk update |
| [`titans/state.py`](titans/state.py) | Stream ownership, caches and independent copies |
| [`titans/training.py`](titans/training.py) | Post-training and token-budget accounting |
| [`titans/types.py`](titans/types.py) | Typed inputs, outputs and configuration |
| [`titans/kernels.py`](titans/kernels.py) | JIT-compiled tensor paths |
| [`titans/gpu_ops.py`](titans/gpu_ops.py) | FlashAttention 3 and Liger operations |
| [`titans/cuda_graphs.py`](titans/cuda_graphs.py) | Stateful CUDA graph capture and replay |
| [`titans/sglang_backend.py`](titans/sglang_backend.py) | SGLang client and remote state handles |

Reference: Ali Behrouz, Peilin Zhong and Vahab Mirrokni,
[*Titans: Learning to Memorize at Test Time*](https://arxiv.org/abs/2501.00663),
arXiv:2501.00663. This repository is an independent implementation.
