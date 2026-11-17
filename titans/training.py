"""Single-device truncated backpropagation through document-memory updates."""

import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .learning import cross_entropy, feedback_tokens, validate_loss
from .protocols import DataGenerator, LossContext, Objective, UpdateRule
from .state import StreamState
from .types import TrainingConfig, TrainingEpisode, TrainingResult

if TYPE_CHECKING:
    from .model import ContinualModel


class _TrainingRun:
    """Private accounting for bounded autograd windows and token-weighted updates."""

    def __init__(
        self, model: "ContinualModel", config: TrainingConfig, loss: Objective | None,
    ) -> None:
        """Create an optimizer for the parameters enabled by the caller.

        Args:
            model: Model whose parameters will be updated.
            config: Training budget and optimization controls.
            loss: Optional custom objective over aligned supervised tokens.
        """
        self.model = model
        self.config = config
        self.objective = loss
        self.parameters = list(model.shared_parameters())
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
        phase: Literal["document", "answer"] = "document",
    ) -> bool:
        """Process tokens up to the exact remaining budget, with optional causal loss.

        Args:
            tokens: Document, prompt or answer IDs in processing order.
            state: Stream receiving these tokens.
            supervise: Whether each token is a next-token prediction target.
            live_states: All active states to detach at an unroll boundary.
            phase: Whether supervision came from a document or answer.

        Returns:
            True if the entire input was consumed, False if the budget stopped it.
        """
        offset = 0
        use_liger = (self.objective is None and self.model.config.use_liger
                     and self.model.backbone.device.type == "cuda")
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
                    weight = float(len(labels))
                    if self.objective is not None:
                        result = self.objective(LossContext(
                            predictions.reshape(-1, predictions.shape[-1]), targets, state,
                            self.model.shared_parameters(), phase,
                        ))
                        validate_loss(result)
                        loss, weight = result.value, result.weight
                    elif use_liger:
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
                    self.step_targets += weight
                    self.total_targets += weight
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
                    self.consume(targets, branch, supervise=True, live_states=active, phase="answer")
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
    model: "ContinualModel", episodes: Iterable[TrainingEpisode] | DataGenerator,
    config: TrainingConfig, *, loss: Objective | None = None, update: UpdateRule | None = None,
) -> TrainingResult:
    """Execute a single post-training pass and write its complete checkpoint.

    Args:
        model: Host model and memory components.
        episodes: Independent streams; no replay or implicit epoch repetition.
        config: Validated training settings and new output directory.
        loss: Optional custom scalar objective and normalization.
        update: Optional checkpoint updater targeting phi, W, or both.

    Returns:
        Actual budget usage, supervised loss and checkpoint location.

    Raises:
        FileExistsError: Output directory exists, before any model mutation.
        FloatingPointError: A loss or gradient becomes nonfinite.
    """
    if config.output_dir.exists():
        raise FileExistsError(config.output_dir)
    if update is not None and config.objective != "document_qa":
        raise ValueError("Explicit update rules require document_qa supervision routing")
    if callable(episodes):
        episodes = episodes(config.seed)
    model.close()
    started = time.perf_counter()
    was_training = model.training
    was_checkpointing = model._checkpoint_backbone
    model._checkpoint_backbone = config.activation_checkpointing
    model._revision += 1
    model.backbone.requires_grad_(False)
    model.memories.requires_grad_(True)
    model.persistent.requires_grad_(True)
    model.train()
    run = _TrainingRun(model, config, loss)
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
            if update is not None:
                return fit_checkpoints(model, episodes, config, loss or cross_entropy, update,
                                       started)
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


def fit_checkpoints(
    model: "ContinualModel", episodes: Iterable[TrainingEpisode], config: TrainingConfig,
    loss: Objective, update: UpdateRule, started: float,
) -> TrainingResult:
    """Run custom QA updates while keeping document ingestion order fixed.

    Each complete feedback item is one optimizer boundary. Its prompt and target
    must fit in one unroll. Oversized feedback is rejected; an insufficient token
    budget stops before feedback rather than learning from a partial answer.
    Document state persists across unrolls and questions, and resets per episode.

    Args:
        model: Prepared runtime with a frozen backbone.
        episodes: Lazy ordered streams and questions at document checkpoints.
        config: Token budget, unroll limit and destination checkpoint.
        loss: Arbitrary scalar QA/state objective.
        update: Caller-owned target and optimizer policy.
        started: Monotonic timestamp covering this fit call.

    Returns:
        Trained artifact, accounting and the final live stream, including adapted W.
        W-only updates change that stream, not the saved shared initialization.
    """
    tokens = steps = window_tokens = 0
    total_loss = 0.0
    exhausted_budget = False
    state: StreamState | None = None
    for episode in episodes:
        for item in episode.questions:
            if not 0 <= item.after_documents <= len(episode.documents):
                raise ValueError("after_documents must reference a document checkpoint")
            prompt, targets = feedback_tokens(model, item.question, item.target_answer)
            if len(prompt) + len(targets) > config.sequence_length:
                raise ValueError("Feedback must fit in sequence_length; increase the unroll")
        if tokens >= config.max_tokens:
            exhausted_budget = True
            break
        state = model._new_state()
        window_tokens = 0
        for count in range(len(episode.documents) + 1):
            for item in episode.questions:
                if item.after_documents != count:
                    continue
                prompt, targets = feedback_tokens(model, item.question, item.target_answer)
                size = len(prompt) + len(targets)
                if tokens + size > config.max_tokens:
                    exhausted_budget = True
                    break
                if window_tokens + size > config.sequence_length:
                    state._detach()
                result = model.learn(item.question, item.target_answer, state,
                                     loss=loss, update=update, retain_history=True)
                tokens += result.tokens_processed
                total_loss += result.loss
                steps += 1
                window_tokens = 0
            if exhausted_budget or count == len(episode.documents):
                break
            document = episode.documents[count]
            ids = model._document_tokens(document)
            offset = 0
            while offset < len(ids) and tokens < config.max_tokens:
                if window_tokens == config.sequence_length:
                    state._detach()
                    window_tokens = 0
                size = min(len(ids) - offset, model.config.prefill_chunk_size,
                           config.sequence_length - window_tokens, config.max_tokens - tokens)
                model._forward_tokens(ids[offset:offset + size], state, last_only=True)
                offset += size
                tokens += size
                window_tokens += size
            if offset == len(ids):
                state._documents += 1
            if tokens >= config.max_tokens:
                exhausted_budget = True
                break
        state._detach()
        if exhausted_budget:
            break
    checkpoint_path = model.save_pretrained(config.output_dir)
    return TrainingResult(checkpoint_path, tokens, steps, total_loss / steps if steps else None,
                          time.perf_counter() - started,
                          "token_budget" if exhausted_budget else "data_exhausted", state)
