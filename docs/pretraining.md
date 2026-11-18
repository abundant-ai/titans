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

Backpropagation includes memory writes within `sequence_length`. At each unroll
boundary the trainer preserves state values but detaches their history. It
accumulates gradients until `tokens_per_step` or the end of the input/budget.
Cached activations carried across an optimizer step were computed using older
parameters; this is truncated online training, without replay. The model returns
to frozen-backbone execution after fitting, including on a training failure.

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
objective remain configurable. MAC remains the default with one real token per
segment. This version does not reproduce the paper's larger MAC segments.

## Manual H100 validation

```bash
uvx modal run --env testing checks/modal_pretraining.py \
  --output validation/h100_pretraining.json
```

This separate, ad hoc job uses the full fixed scratch decoder and the real Llama
2 tokenizer. It checks gradients through inner writes, eager versus compiled/Liger
backbone and memory gradients in FP32, three joint AdamW steps with truncation and fused
loss, and checkpoint restoration. Its JSON report includes the GPU, software
versions and source hashes. It is outside default pytest discovery and CI.
The existing `checks/modal_inference.py` remains inference-only.

These checks establish execution and gradient correctness on a short synthetic
stream. They do not establish convergence, long-range retention, throughput for
large unrolls, or BABILong quality.
