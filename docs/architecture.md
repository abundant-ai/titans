# Architecture and backbone connections

`ContinualModel` and `TitansModel` default to Titans neural memory with MAC
(memory as context). `MemoryArchitecture` constructs the memory network;
`ConnectionFactory` constructs its connection at each selected backbone layer.
Both are callable protocols. `MemoryFactory` remains an alias for compatibility.

```python
from titans import (
    ContinualModel, RuntimeConfig, TitansArchitecture, TitansConfig,
    titans_mac_connection,
)

model = ContinualModel.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    config=RuntimeConfig(memory_layers=(0, 6, 12), attention_window=512),
    memory_architecture=TitansArchitecture(TitansConfig(
        memory_dim=64, memory_hidden_dim=128, memory_depth=2, chunk_size=16,
    )),
    connection_factory=titans_mac_connection,
)
```

The agent can replace the entire memory network, including initialization,
representation, depth, projections and transition. The built-in architecture is
a deep associative MLP with the existing Titans update. All Titans-specific
architectures and connections are implemented in `titans/methods/titans.py`.

## Connection protocol

- `prepare(context, memory, state, write=...) -> ConnectionPlan` chooses what
  memory reads/writes, what attention receives, and any retrieved context tokens.
- `finish(context, attention, plan, memory, state, write=...) -> Tensor` can write
  attention outputs back to memory and choose how to produce the residual stream.
- `combine(context, attention, plan) -> Tensor` is the simpler fusion hook used
  by the default `finish`. A policy can override either or both.

`context` contains the layer index, residual input and normalized input. A plan
contains attention input and a memory read. Connection parameters are registered
PyTorch parameters: they join phi, receive gradients, and are saved with the model.
Mutable temporal buffers belong in the stream's `MemoryState`, not in a connection.

## Choosing decoder layers

`RuntimeConfig.memory_layers` lists the zero-based decoder-layer indices where
the runtime creates a memory module and its connection. In a 24-layer decoder,
the valid indices are 0 through 23. The default `memory_layers=(0,)` selects
only the first decoder layer.

| Configuration for a 24-layer decoder | Selected layers |
| --- | --- |
| `memory_layers=(0,)` | First layer only |
| `memory_layers=(0, 6, 12, 18)` | Four layers spread through the decoder |
| `memory_layers=tuple(range(24))` | All 24 layers |

With the default MAC connection, the first configuration produces this stack.

```text
Token embeddings
    |
Layer 0     MAC attention and neural memory -> feed-forward network
    |
Layer 1     ordinary attention              -> feed-forward network
    |
   ...
    |
Layer 23    ordinary attention              -> feed-forward network
    |
Vocabulary prediction
```

At a selected layer, MAC uses that decoder layer's existing attention projection
weights. It adds memory retrievals to the attention input, writes from the
attention output, and gates the output with a memory read. The layer's
feed-forward network still follows this operation. Unselected layers execute
the runtime's ordinary sliding-window attention with the persistent prefix.

`connection_factory` determines the connection used at the selected indices.
Its default is MAC. Supplying `titans_mag_connection` chooses MAG at those
indices, and a custom factory can choose a different connection for each
`layer_index`. The factory receives `(layer_index, hidden_size)`.

Each selected layer owns an independent memory module, connection instance,
and memory state in each stream. Adding a selected layer adds another set of
memory parameters and stream state. A stream's memory at layer 0 is separate
from its memory at layer 6.

The examples above illustrate placement options. Their accuracy and speed
depend on the backbone, training setup, and workload.

## MAC default and its limits

The default follows the operation order in the [Titans paper, section 4.1,
equations 21–25](https://arxiv.org/html/2501.00663v1#S4.SS1): retrieve old memory,
place the retrieval in the attention context, update memory from attention output,
then read the updated memory to gate the output.

A new MAC model starts each **512-real-token segment** with a differentiable
copy of the incoming memory. Each token queries that copy using its own input
and the retrieval convolution's causal history. The MAC layer attends to
`prefix, r1, x1, r2, x2, …` using segment-local rotary positions. Token 100
can attend to the first 100 retrievals and real tokens. Token 513 starts with
fresh segment KV and keeps the persistent prefix. Other layers use their usual
sliding attention window.

Each real-token attention output supplies one query, key, value and set of
gates. The built-in memory commits after 16 writes. Token 15 reads the old
weights. Token 16 reads the newly committed weights with the same query it
computed before the write. Each token uses that read to gate its attention
output. Incomplete writes carry across documents and prefill calls.

A question can arrive halfway through either interval. Its branch copies the
segment cache, retrieval cursor and live memory, including pending writes.
`temporary` lets the branch commit writes. `read_only` uses the committed
weights. The next document continues from the source state.

`mac_segment_length` counts real tokens, `chunk_size` counts writes, and
`prefill_chunk_size` bounds execution blocks. Only the first two change the
model's predictions. Host blocks of 256 and training detaches at 2,048 tokens
preserve the segment offset. Custom connections retain their declared context
bound and execution order.

## Fixed versus overridable

| Boundary | Fixed runtime contract | Agent controls |
| --- | --- | --- |
| Backbone | Fixed scratch Llama or Qwen2.5-1.5B-Instruct | Choose scratch/pretrained start and outer training mode |
| Architecture | Registered modules and independent state ownership | `MemoryArchitecture`, network size/structure and initialization |
| Transition | Causality and stream continuity | `MemoryModule.transition` and state representation |
| Connection | Shape/device checks, bounded attention, real-token order | `ConnectionFactory`, `prepare`, `finish`, `combine`, layer placement |
| Data | Ordered episodes and explicit feedback availability | `DataGenerator`, documents, questions, answers, checkpoint timing |
| Objective | Finite scalar and normalization contract | `Objective` functional form |
| Updates | State/revision ownership; theta separate from phi and W | `UpdateRule`, phi/W/both/joint, optimizer and rates |
| Execution | `ingest`, branching, token accounting, checkpoint lifecycle | Supported window, unroll and optimization settings |

## Compatibility and optimized execution

`titans_mag_connection` selects the parallel normalized MAG-style gate.
`residual_connection` selects the prior pretrained residual retrofit. Checkpoints
from formats 1 and 2 restore residual connections automatically; format 3 saves
connection types and weights. Formats 1–3 restore historical one-token MAC segments
and the saved write chunk (one when absent). Format 4 records the segment length,
shared causal retrieval policy and each built-in memory configuration. The corrected
convolution read can change historical MAC predictions. Custom connections require an explicit factory when
loading. A checkpoint never imports arbitrary code from its metadata.

Compiled memory kernels, Liger, activation checkpointing, gradient accumulation
and truncation remain available. MAC, MAG, residual and custom connections all
use the same PyTorch runtime for local training and inference.

The original optimization audit describes commit `1cf5ad1`; its source hashes do
not describe this architecture extension. GPU checks run manually on Modal.


## Memory queries and training parameters

At each selected decoder layer, the memory's own QKV projection receives the
normalized hidden state. A causal depthwise convolution and SiLU follow that
projection. Splitting its channels gives query, key and value, and L2
normalization produces the final query. These projection parameters belong to
the outer-trained shared parameters. Each arriving token gets its own query.

For the pre-attention MAC read, that query reads the segment-start snapshot.
The attention output supplies a second projected query for the post-write gate.
A 16-token write chunk shares its gradient evaluation weights; its tokens still
have individual keys, values, queries and retrieved outputs.

Fresh models use a residual GELU memory network with fourfold expansion.
Its implementation choices and historical checkpoint behavior are described in
[paper alignment](paper_alignment.md#memory-network-used-by-fresh-models).

Persistent embeddings train with the outer parameters, including when the
backbone is frozen. Standard training refreshes their prefix representations
after truncation so later losses can reach them. Runtime inference keeps those
learned embeddings fixed while fast memory adapts.

## Layer-placement experiments

The following are candidate configurations for comparing placement. The current
default remains `memory_layers=(0,)`.

| Backbone | Four distributed layers | All layers |
| --- | --- | --- |
| Scratch Llama, 24 layers | `(5, 11, 17, 23)` | `tuple(range(24))` |
| Qwen2.5-1.5B, 28 layers | `(6, 13, 20, 27)` | `tuple(range(28))` |

Use the same document/QA data and training token budget across candidates.
Record delayed-fact accuracy, overwrite accuracy, parameter counts, peak memory,
and tokens per second. Adding layers changes both placement and capacity.
Select a default using measurements from the intended workload.
