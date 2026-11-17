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
        memory_dim=64, memory_hidden_dim=128, memory_depth=2, chunk_size=1,
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

The factory receives `(layer_index, hidden_size)`, so the agent can vary wiring
by layer. `RuntimeConfig.memory_layers` chooses placement. A connection can put
memory before attention, run it alongside attention, prepend retrieved context,
or update it from attention outputs. Each layer gets independent module instances.

## MAC default and its limits

The default follows the operation order in the [Titans paper, section 4.1,
equations 21–25](https://arxiv.org/html/2501.00663v1#S4.SS1): retrieve old memory,
place the retrieval in the attention context, update memory from attention output,
then read the updated memory to gate the output.

This streaming implementation uses **one real token per MAC segment**, with one
retrieved context token. Only persistent-prefix KV survives between segments.
The memory state carries the history. This ensures causal execution and equal
results when the caller splits a document differently. Virtual context tokens
use segment-local rotary positions and do not count as real input/label tokens.
The backbone remains a pretrained, frozen decoder, with its original FFN.

The default memory write chunk is one token. Choosing a larger memory write
chunk delays commits; that is an algorithmic choice distinct from MAC segmentation.
Reads for context run on independent state copies; only attention-driven writes
advance the live state. Read-only answers never commit their temporary state.

This is a MAC retrofit with a specific causal segmentation policy. It does not
reproduce the paper's larger-segment experiments, establish their accuracy, or
promise that MAC outperforms every other connection. Larger MAC segments are not
yet implemented in the fixed runtime. Context-expanding policies currently run
one real token at a time, and declare a finite `max_context_tokens` bound.

## Fixed versus overridable

| Boundary | Fixed runtime contract | Agent controls |
| --- | --- | --- |
| Backbone | Frozen weights; supported Llama/Qwen2 execution | Pretrained checkpoint |
| Architecture | Registered modules and independent state ownership | `MemoryArchitecture`, network size/structure and initialization |
| Transition | Causality and stream continuity | `MemoryModule.transition` and state representation |
| Connection | Shape/device checks, bounded attention, real-token order | `ConnectionFactory`, `prepare`, `finish`, `combine`, layer placement |
| Data | Ordered episodes and explicit feedback availability | `DataGenerator`, documents, questions, answers, checkpoint timing |
| Objective | Finite scalar and normalization contract | `Objective` functional form |
| Updates | Backbone excluded; state/revision ownership | `UpdateRule`, phi/W/both, optimizer and rates |
| Execution | `ingest`, branching, token accounting, checkpoint lifecycle | Supported window, unroll and optimization settings |

## Compatibility and optimized execution

`titans_mag_connection` selects the parallel normalized MAG-style gate.
`residual_connection` selects the prior pretrained residual retrofit. Checkpoints
from formats 1 and 2 restore residual connections automatically; format 3 saves
connection types and weights. Custom connections require an explicit factory when
loading. A checkpoint never imports arbitrary code from its metadata.

Compiled memory kernels, Liger, activation checkpointing, gradient accumulation
and truncation remain available. SGLang and the existing optimized CUDA graph
serving adapter support the residual Titans connection. Select it explicitly for
that path. MAC and custom connections use the local PyTorch runtime; they do not
silently dispatch through an incompatible serving adapter. The historical
throughput benchmarks explicitly select the residual connection.

The original optimization audit describes commit `1cf5ad1`; its source hashes do
not describe this architecture extension. GPU checks remain manual and were not
run for this change.
