"""Public data contracts for document streams and post-training."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from titans.runtime.state import StreamState


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
class RuntimeConfig:
    """Fixed backbone execution settings, independent of the memory algorithm."""

    memory_layers: tuple[int, ...] = (0,)
    attention_window: int = 512
    persistent_tokens: int = 4
    compile_kernels: bool = True
    prefill_chunk_size: int = 256
    inference_backend: Literal["auto", "torch", "sglang"] = "torch"
    attention_backend: Literal["sdpa", "auto"] = "sdpa"
    use_liger: bool = False
    cuda_graphs: bool = True
    mac_segment_length: int = field(default=512, kw_only=True)

    def __post_init__(self) -> None:
        """Reject invalid runtime settings before allocating the backbone adapter."""
        if min(self.mac_segment_length, self.attention_window, self.prefill_chunk_size) < 1:
            raise ValueError("MAC segment length, attention window and prefill size must be positive")
        if self.persistent_tokens < 0:
            raise ValueError("persistent_tokens must be nonnegative")
        if not self.memory_layers or min(self.memory_layers) < 0:
            raise ValueError("At least one nonnegative memory layer is required")
        if len(set(self.memory_layers)) != len(self.memory_layers):
            raise ValueError("memory_layers must contain unique indices")
        if self.inference_backend not in ("auto", "torch", "sglang"):
            raise ValueError("Unknown inference backend")
        if self.attention_backend not in ("sdpa", "auto"):
            raise ValueError("Titans uses PyTorch SDPA. Set attention_backend to 'sdpa' or 'auto'.")


@dataclass(frozen=True)
class TrainingConfig:
    """Training budget and optimization settings.

    freeze_backbone=None selects joint training for scratch Llama and memory-only
    training for Qwen. An explicit boolean overrides that training selection.

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
    freeze_backbone: bool | None = None
    seed: int = 0
    max_grad_norm: float = 1.0
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        """Validate the finite token budget and optimizer controls."""
        if min(self.max_tokens, self.sequence_length, self.tokens_per_step) < 1:
            raise ValueError("Token budgets and sequence_length must be positive")
        if any(not math.isfinite(value) or value <= 0
               for value in (self.learning_rate, self.max_grad_norm)):
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.objective not in ("language_modeling", "document_qa", "mixed"):
            raise ValueError("Unknown training objective")


@dataclass(frozen=True)
class TrainingResult:
    """Measurements for one fit call and an optional continuing stream.

    Default AdamW training averages loss using Objective weights. Custom-update
    training reports the mean normalized checkpoint loss and returns final_state;
    W-only changes live there and are not part of the shared-parameter checkpoint.
    """

    checkpoint_path: Path
    tokens_processed: int
    optimizer_steps: int
    mean_loss: float | None
    elapsed_seconds: float
    stop_reason: Literal["token_budget", "data_exhausted"]
    final_state: "StreamState | None" = None


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


@dataclass(frozen=True)
class LearningResult:
    """One explicit feedback update; teacher tokens are accounted but not committed."""

    loss: float
    tokens_processed: int
    shared_changed: bool
    memory_tensors_changed: int
