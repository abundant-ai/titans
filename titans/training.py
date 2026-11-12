"""Single-device truncated backpropagation through document-memory updates."""

import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .state import StreamState
from .types import TrainingConfig, TrainingEpisode, TrainingResult

if TYPE_CHECKING:
    from .model import TitansModel


class _TrainingRun:
    """Private accounting for bounded autograd windows and token-weighted updates."""

    def __init__(self, model: "TitansModel", config: TrainingConfig) -> None:
        """Create an optimizer for the parameters enabled by the caller.

        Args:
            model: Model whose parameters will be updated.
            config: Training budget and optimization controls.
        """
        self.model = model
        self.config = config
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate)
        self.tokens = 0
        self.steps = 0
        self.window_tokens = 0
        self.step_tokens = 0
        self.step_targets = 0
        self.total_targets = 0
        self.total_loss = 0.0
        self.losses: list[Tensor] = []

    def _step(self) -> None:
        """Normalize accumulated summed losses by their actual target-token count."""
        if self.step_targets:
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(self.step_targets)
            torch.nn.utils.clip_grad_norm_(
                self.parameters, self.config.max_grad_norm, error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.step_tokens = 0
        self.step_targets = 0

    def flush(self, states: tuple[StreamState, ...], *, final: bool = False) -> None:
        """Backpropagate one unroll and detach every live branch before a weight update.

        Args:
            states: Every stream whose graph belongs to this unroll.
            final: Whether to apply a final partial optimizer step.
        """
        if self.losses:
            torch.stack(self.losses).sum().backward()
            self.losses.clear()
        for state in states:
            state._detach()
        self.window_tokens = 0
        if final or self.step_tokens >= self.config.tokens_per_step:
            self._step()

    def consume(
        self,
        tokens: list[int],
        state: StreamState,
        *,
        supervise: bool,
        live_states: tuple[StreamState, ...],
    ) -> bool:
        """Process tokens up to the exact remaining budget, with optional causal loss.

        Args:
            tokens: Document, prompt or answer IDs in processing order.
            state: Stream receiving these tokens.
            supervise: Whether each token is a next-token prediction target.
            live_states: All active states to detach at an unroll boundary.

        Returns:
            True if the entire input was consumed, False if the budget stopped it.
        """
        offset = 0
        use_liger = self.model.config.use_liger and self.model.backbone.device.type == "cuda"
        while offset < len(tokens) and self.tokens < self.config.max_tokens:
            count = min(
                len(tokens) - offset, self.model.config.prefill_chunk_size,
                self.config.sequence_length - self.window_tokens,
                self.config.max_tokens - self.tokens,
            )
            block = tokens[offset:offset + count]
            previous_logits = state._next_hidden if use_liger else state._next_logits
            logits = self.model._forward_tokens(block, state, last_only=not supervise,
                                                return_hidden=use_liger)
            if supervise:
                if previous_logits is None:
                    predictions, labels = logits[:, :-1], block[1:]
                else:
                    predictions = torch.cat((previous_logits, logits[:, :-1]), 1)
                    labels = block
                if labels:
                    targets = torch.tensor(labels, device=logits.device)
                    if use_liger:
                        from .gpu_ops import liger_loss

                        loss = liger_loss(self.model.backbone.lm_head.weight,
                                          predictions.reshape(-1, predictions.shape[-1]), targets)
                    else:
                        loss = F.cross_entropy(
                            predictions.reshape(-1, predictions.shape[-1]).float(),
                            targets, reduction="sum",
                        )
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("Nonfinite training loss; decrease the learning rate")
                    self.losses.append(loss)
                    self.step_targets += len(labels)
                    self.total_targets += len(labels)
                    self.total_loss += float(loss.detach().item())
            offset += count
            self.tokens += count
            self.window_tokens += count
            self.step_tokens += count
            if self.window_tokens == self.config.sequence_length:
                self.flush(live_states)
        return offset == len(tokens)

    def episode(self, episode: TrainingEpisode) -> None:
        """Train one independent stream with questions at specified document counts.

        Args:
            episode: Ordered documents and checkpointed supervision.

        Raises:
            ValueError: A question references a nonexistent document checkpoint.
        """
        for item in episode.questions:
            if not 0 <= item.after_documents <= len(episode.documents):
                raise ValueError("after_documents must be between zero and the document count")
            if not item.target_answer.strip():
                raise ValueError("Training answers must be nonempty; use [unknown] for abstention")
        state = self.model._new_state()
        use_qa = self.config.objective in ("document_qa", "mixed")
        use_lm = self.config.objective in ("language_modeling", "mixed")
        for count in range(len(episode.documents) + 1):
            if use_qa:
                for item in episode.questions:
                    if item.after_documents != count or self.tokens >= self.config.max_tokens:
                        continue
                    branch = state._copy()
                    active = (state, branch)
                    self.consume(self.model._question_tokens(item.question), branch,
                                 supervise=False, live_states=active)
                    targets = self.model.tokenizer.encode(item.target_answer, add_special_tokens=False)
                    if self.model.tokenizer.eos_token_id is not None:
                        targets.append(self.model.tokenizer.eos_token_id)
                    self.consume(targets, branch, supervise=True, live_states=active)
                    # The branch's graph may still contribute to pending losses;
                    # the loss tensors own it until the next flush.
            if count == len(episode.documents) or self.tokens >= self.config.max_tokens:
                break
            complete = self.consume(
                self.model._document_tokens(episode.documents[count]), state,
                supervise=use_lm, live_states=(state,),
            )
            if complete:
                state._documents += 1
            else:
                break
        self.flush((state,))


def fit_model(
    model: "TitansModel", episodes: Iterable[TrainingEpisode], config: TrainingConfig,
) -> TrainingResult:
    """Execute a single post-training pass and write its complete checkpoint.

    Args:
        model: Host model and memory components.
        episodes: Independent streams; no replay or implicit epoch repetition.
        config: Validated training settings and new output directory.

    Returns:
        Actual budget usage, supervised loss and checkpoint location.

    Raises:
        FileExistsError: Output directory exists, before any model mutation.
        FloatingPointError: A loss or gradient becomes nonfinite.
    """
    if config.output_dir.exists():
        raise FileExistsError(config.output_dir)
    model.close()
    started = time.perf_counter()
    was_training = model.training
    was_checkpointing = model._checkpoint_backbone
    model._checkpoint_backbone = config.activation_checkpointing
    model._revision += 1
    model.backbone.requires_grad_(not config.freeze_backbone)
    model.memories.requires_grad_(True)
    model.persistent.requires_grad_(True)
    model.train()
    run = _TrainingRun(model, config)
    device = model.backbone.model.embed_tokens.weight.device
    devices = [device.index or 0] if device.type == "cuda" else []
    # cuDNN SDPA backward produced NaNs on pretrained Qwen2.5 with the local/prefix
    # mask. Keep the same attention equation using the other autograd backends.
    attention_context = sdpa_kernel([
        SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
    ]) if device.type == "cuda" else nullcontext()
    try:
        with torch.random.fork_rng(devices=devices), attention_context:
            torch.manual_seed(config.seed)
            for episode in episodes:
                if run.tokens >= config.max_tokens:
                    break
                run.episode(episode)
            run.flush((), final=True)
        checkpoint = model.save_pretrained(config.output_dir)
    finally:
        run.optimizer.zero_grad(set_to_none=True)
        model.train(was_training)
        model._checkpoint_backbone = was_checkpointing
    return TrainingResult(
        checkpoint, run.tokens, run.steps,
        run.total_loss / run.total_targets if run.total_targets else None,
        time.perf_counter() - started,
        "token_budget" if run.tokens >= config.max_tokens else "data_exhausted",
    )
