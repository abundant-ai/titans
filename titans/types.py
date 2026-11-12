"""Public data contracts for document streams and post-training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Document:
    """One document; arrival order, including repeated IDs, defines stream order."""

    document_id: str
    text: str


@dataclass(frozen=True)
class Question:
    """A question about the documents seen so far."""

    question_id: str
    text: str


@dataclass(frozen=True)
class TrainingQuestion:
    """Supervision after exactly ``after_documents`` documents in an episode."""

    after_documents: int
    question: Question
    target_answer: str


@dataclass(frozen=True)
class TrainingEpisode:
    """An independent stream; answers never become part of its document memory."""

    episode_id: str
    documents: tuple[Document, ...]
    questions: tuple[TrainingQuestion, ...] = ()


@dataclass(frozen=True)
class TitansConfig:
    """Memory and local-attention settings for a Llama retrofit.

    Chunk size one implements tokenwise updates. Larger chunks use gradients
    evaluated at the chunk's initial weights and commit only complete chunks.
    Memory reads use the last committed weights. ``memory_layers`` is zero-based.
    Fast weights and their updates use float32 unless the whole model is float64.
    """

    memory_dim: int = 128
    memory_hidden_dim: int = 256
    memory_depth: int = 2
    memory_layers: tuple[int, ...] = (0,)
    chunk_size: int = 32
    attention_window: int = 512
    persistent_tokens: int = 4
    conv_kernel: int = 3
    max_inner_learning_rate: float = 0.01
    gate_bias: float = -4.0
    compile_kernels: bool = True
    prefill_chunk_size: int = 256
    inference_backend: Literal["auto", "torch", "sglang"] = "auto"
    attention_backend: Literal["auto", "sdpa", "fa3"] = "auto"
    use_liger: bool = True
    cuda_graphs: bool = True

    def __post_init__(self) -> None:
        """Reject invalid dimensions and update rates before allocating a model."""
        dimensions = (
            self.memory_dim, self.memory_hidden_dim, self.memory_depth,
            self.chunk_size, self.attention_window, self.conv_kernel, self.prefill_chunk_size,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("Memory dimensions, chunk size, window and kernel must be positive")
        if self.persistent_tokens < 0:
            raise ValueError("persistent_tokens must be nonnegative")
        if len(set(self.memory_layers)) != len(self.memory_layers):
            raise ValueError("memory_layers must contain unique indices")
        if not self.memory_layers or min(self.memory_layers) < 0:
            raise ValueError("At least one nonnegative memory layer is required")
        if not 0 < self.max_inner_learning_rate <= 1:
            raise ValueError("max_inner_learning_rate must be in (0, 1]")
        if self.inference_backend not in ("auto", "torch", "sglang"):
            raise ValueError("Unknown inference backend")
        if self.attention_backend not in ("auto", "sdpa", "fa3"):
            raise ValueError("Unknown attention backend")


@dataclass(frozen=True)
class TrainingConfig:
    """Post-training budget and optimization settings.

    ``max_tokens`` counts document, question and target tokens actually forwarded;
    persistent prefix tokens are excluded. ``sequence_length`` bounds each
    autograd unroll. Optimizer steps occur at unroll boundaries after at least
    ``tokens_per_step`` forwarded tokens, or at the end of the data/budget.
    """

    output_dir: Path
    max_tokens: int
    sequence_length: int = 2048
    tokens_per_step: int = 8192
    learning_rate: float = 1e-4
    objective: Literal["language_modeling", "document_qa", "mixed"] = "mixed"
    freeze_backbone: bool = True
    seed: int = 0
    max_grad_norm: float = 1.0
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        """Validate the finite token budget and optimizer controls."""
        if min(self.max_tokens, self.sequence_length, self.tokens_per_step) < 1:
            raise ValueError("Token budgets and sequence_length must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.objective not in ("language_modeling", "document_qa", "mixed"):
            raise ValueError("Unknown training objective")


@dataclass(frozen=True)
class TrainingResult:
    """Measurements for one fit call; loss is averaged over supervised tokens."""

    checkpoint_path: Path
    tokens_processed: int
    optimizer_steps: int
    mean_loss: float | None
    elapsed_seconds: float
    stop_reason: Literal["token_budget", "data_exhausted"]


@dataclass(frozen=True)
class AnswerConfig:
    """Decoding controls; temporary memory writes are always discarded."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    memory_policy: Literal["read_only", "temporary"] = "temporary"
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject invalid decoding lengths, temperatures and memory policies."""
        if self.max_new_tokens < 1 or self.temperature < 0:
            raise ValueError("max_new_tokens must be positive and temperature nonnegative")
        if self.memory_policy not in ("read_only", "temporary"):
            raise ValueError("Unknown memory policy")


@dataclass(frozen=True)
class Answer:
    """Generated response; abstained means the model emitted ``[unknown]``."""

    question_id: str
    text: str
    abstained: bool
    generated_tokens: int
    stopped_by_limit: bool
