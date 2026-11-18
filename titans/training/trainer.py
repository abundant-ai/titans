"""Single-device truncated backpropagation through document-memory updates."""

import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from titans.core.protocols import DataGenerator, LossContext, Objective, UpdateRule
from titans.core.types import TrainingConfig, TrainingEpisode, TrainingResult
from titans.runtime.state import StreamState
from titans.training.learning import cross_entropy, feedback_tokens, validate_loss

if TYPE_CHECKING:
    from titans.runtime.model import ContinualModel


def refresh_prefix(model: "ContinualModel", state: StreamState) -> None:
    """Recompute persistent-prefix KV on a fresh graph without ingesting real tokens.

    Args:
        model: Runtime using the current shared and backbone parameters.
        state: Continuing detached stream whose document caches and memory survive.

    Prefix positions are input-independent and can be rebuilt using an empty
    stream. Only prefix KV is replaced, preserving real-token cache history.
    """
    count = model.config.persistent_tokens
    if not count:
        return
    prefix = model._new_state(prefix_hidden_only=True)
    for layer, cache in state._attention.items():
        fresh = prefix._attention[layer]
        cache.keys = torch.cat((fresh.keys, cache.keys[:, :, count:]), dim=2)
        cache.values = torch.cat((fresh.values, cache.values[:, :, count:]), dim=2)


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
        self.train_backbone = not (config.freeze_backbone if config.freeze_backbone is not None
                                   else model.backbone_kind != "scratch")
        self.parameters = list(model.outer_parameters(train_backbone=self.train_backbone))
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate)
        self.use_liger = (loss is None and model.config.use_liger and model.backbone.device.type == "cuda")
        self.mixed_precision = False
        device = model.backbone.device
        if device.type == "cuda" and model.backbone.dtype in (torch.bfloat16, torch.float32):
            with torch.cuda.device(device):
                self.mixed_precision = torch.cuda.is_bf16_supported(including_emulation=False)
        self.tokens = 0
        self.steps = 0
        self.window_tokens = 0
        self.step_tokens = 0
        self.step_targets = 0
        self.total_targets = 0
        self.total_loss = 0.0
        self.losses: list[Tensor] = []
        self.state: StreamState | None = None

    def _autocast(self) -> torch.autocast:
        """Use BF16 on supported CUDA devices for the standard training forward pass."""
        return torch.autocast(self.model.backbone.device.type, dtype=torch.bfloat16,
                              enabled=self.mixed_precision)

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
        if not final:
            with self._autocast():
                for state in states:
                    refresh_prefix(self.model, state)

    def score(self, predictions: Tensor, labels: list[int], state: StreamState,
              phase: Literal["document", "answer"]) -> None:
        """Accumulate a summed loss for known causal next-token labels.

        Args:
            predictions: Logits or normalized hidden states on the live graph.
            labels: Target IDs; a final lookahead label has not been forwarded.
            state: State after the forwarded input block, before lookahead ingestion.
            phase: Source of supervision for custom objectives.
        """
        if not labels:
            return
        targets = torch.tensor(labels, device=predictions.device)
        if self.objective is not None:
            result = self.objective(LossContext(
                predictions.reshape(-1, predictions.shape[-1]), targets, state,
                self.model.shared_parameters(), phase,
                tuple(self.model.backbone.parameters()) if self.train_backbone else (),
            ))
            validate_loss(result)
            loss, weight = result.value, result.weight
        elif self.use_liger:
            from titans.optim.gpu_ops import liger_loss

            with self._autocast():
                loss = liger_loss(self.model.backbone.lm_head.weight,
                                  predictions.flatten(0, 1), targets).float()
            weight = float(len(labels))
        else:
            loss = F.cross_entropy(predictions.reshape(-1, predictions.shape[-1]).float(),
                                   targets, reduction="sum")
            weight = float(len(labels))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Nonfinite training loss; decrease the learning rate")
        self.losses.append(loss)
        self.step_targets += weight
        self.total_targets += weight
        self.total_loss += float(loss.detach().item())

    def consume(
        self,
        tokens: list[int],
        state: StreamState,
        *,
        supervise: bool,
        live_states: tuple[StreamState, ...],
        phase: Literal["document", "answer"] = "document",
        next_target: int | None = None,
        next_phase: Literal["document", "answer"] = "document",
    ) -> bool:
        """Process tokens up to the exact remaining budget, with optional causal loss.

        Args:
            tokens: Document, prompt or answer IDs in processing order.
            state: Stream receiving these tokens.
            supervise: Whether each token is a next-token prediction target.
            live_states: All active states to detach at an unroll boundary.
            phase: Whether supervision came from a document or answer.
            next_target: Optional next supervised token after the entire input.
                It is scored on this branch before truncation and never forwarded early.
            next_phase: Supervision phase of that final lookahead label.

        Returns:
            True if the entire input was consumed, False if the budget stopped it.
        """
        offset = 0
        use_liger = self.use_liger
        while offset < len(tokens) and self.tokens < self.config.max_tokens:
            count = min(
                len(tokens) - offset, self.model.config.prefill_chunk_size,
                self.config.sequence_length - self.window_tokens,
                self.config.max_tokens - self.tokens,
            )
            block = tokens[offset:offset + count]
            previous = state._next_hidden if use_liger else state._next_logits
            already_scored = state._next_target_scored
            intermediate_prompt = not supervise and offset + count < len(tokens)
            with self._autocast():
                logits = self.model._forward_tokens(block, state, last_only=not supervise,
                                                    return_hidden=use_liger or intermediate_prompt)
            predictions = logits[:, :0]
            labels: list[int] = []
            if supervise:
                predictions, labels = logits[:, :-1], block[1:]
                if previous is not None and not already_scored:
                    predictions = torch.cat((previous, predictions), 1)
                    labels = block
            following = None
            following_phase = phase
            if self.tokens + count < self.config.max_tokens:
                if offset + count < len(tokens):
                    if supervise:
                        following = tokens[offset + count]
                else:
                    following, following_phase = next_target, next_phase
            if following is not None and following_phase == phase:
                predictions = torch.cat((predictions, logits[:, -1:]), 1)
                labels = labels + [following]
            self.score(predictions, labels, state, phase)
            if following is not None and following_phase != phase:
                self.score(logits[:, -1:], [following], state, following_phase)
            state._next_target_scored = following is not None
            offset += count
            self.tokens += count
            self.window_tokens += count
            self.step_tokens += count
            if self.window_tokens == self.config.sequence_length:
                self.flush(live_states)
        return offset == len(tokens)

    def episode(self, episode: TrainingEpisode) -> None:
        """Train an episode with reset or carried state and local document counts.

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
        with self._autocast():
            if self.config.episode_memory == "reset" or self.state is None:
                self.state = self.model._new_state(prefix_hidden_only=self.use_liger)
            else:
                # Episodes delimit supervised sequences, even when memory carries.
                self.state._next_logits = None
                self.state._next_hidden = None
                self.state._next_target_scored = False
                refresh_prefix(self.model, self.state)
        state = self.state
        use_qa = self.config.objective in ("document_qa", "mixed")
        use_lm = self.config.objective in ("language_modeling", "mixed")
        for count in range(len(episode.documents) + 1):
            if use_qa:
                for item in episode.questions:
                    if item.after_documents != count or self.tokens >= self.config.max_tokens:
                        continue
                    # Every question due here branches from documents[:count].
                    # The next question or document resumes from the original state.
                    branch = state._copy()
                    active = (state, branch)
                    targets = self.model.tokenizer.encode(item.target_answer, add_special_tokens=False)
                    if self.model.tokenizer.eos_token_id is not None:
                        targets.append(self.model.tokenizer.eos_token_id)
                    self.consume(self.model._question_tokens(item.question), branch,
                                 supervise=False, live_states=active, next_target=targets[0],
                                 next_phase="answer")
                    self.consume(targets, branch, supervise=True, live_states=active, phase="answer")
                    # The branch's graph may still contribute to pending losses;
                    # the loss tensors own it until the next flush.
            if count == len(episode.documents) or self.tokens >= self.config.max_tokens:
                break
            document_tokens = self.model._document_tokens(episode.documents[count])
            next_document_target = None
            if use_lm and count + 1 < len(episode.documents):
                question_budget = 0
                if use_qa:
                    for item in episode.questions:
                        if item.after_documents == count + 1:
                            prompt, answer = feedback_tokens(
                                self.model, item.question, item.target_answer)
                            question_budget += len(prompt) + len(answer)
                if self.tokens + len(document_tokens) + question_budget < self.config.max_tokens:
                    next_ids = self.model._document_tokens(episode.documents[count + 1])
                    next_document_target = next_ids[0] if next_ids else None
            complete = self.consume(
                document_tokens, state, supervise=use_lm, live_states=(state,),
                next_target=next_document_target,
            )
            if complete:
                state._documents += 1
            else:
                break
        self.flush((state,))
        state._detach()


def fit_model(
    model: "ContinualModel", episodes: Iterable[TrainingEpisode] | DataGenerator,
    config: TrainingConfig, *, loss: Objective | None = None, update: UpdateRule | None = None,
) -> TrainingResult:
    """Execute a memory-only or joint training pass and save its learned weights.

    Args:
        model: Host model and memory components.
        episodes: Ordered episodes; episode_memory selects isolation or continuity.
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
    started = time.perf_counter()
    was_training = model.training
    was_checkpointing = model._checkpoint_backbone
    model._checkpoint_backbone = config.activation_checkpointing
    model._revision += 1
    train_backbone = not (config.freeze_backbone if config.freeze_backbone is not None
                          else model.backbone_kind != "scratch")
    model.backbone.requires_grad_(train_backbone)
    model.memories.requires_grad_(True)
    model.connections.requires_grad_(True)
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
        model.backbone.requires_grad_(False)
        model.train(was_training)
        model._checkpoint_backbone = was_checkpointing
    return TrainingResult(
        checkpoint, run.tokens, run.steps,
        run.total_loss / run.total_targets if run.total_targets else None,
        time.perf_counter() - started,
        "token_budget" if run.tokens >= config.max_tokens else "data_exhausted",
        run.state if config.episode_memory == "carry" else None,
    )


def fit_checkpoints(
    model: "ContinualModel", episodes: Iterable[TrainingEpisode], config: TrainingConfig,
    loss: Objective, update: UpdateRule, started: float,
) -> TrainingResult:
    """Run custom QA updates while keeping document ingestion order fixed.

    Each complete feedback item is one optimizer boundary. Its prompt and target
    must fit in one unroll. Oversized feedback is rejected; an insufficient token
    budget stops before feedback rather than learning from a partial answer.
    Document state persists across unrolls and questions. The episode_memory
    policy selects reset or carry at episode boundaries.

    Args:
        model: Prepared runtime with the selected outer-training targets.
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
        if config.episode_memory == "reset" or state is None:
            state = model._new_state()
        else:
            refresh_prefix(model, state)
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
                    refresh_prefix(model, state)
                result = model.learn(item.question, item.target_answer, state,
                                     loss=loss, update=update, retain_history=True,
                                     train_backbone=not (config.freeze_backbone if config.freeze_backbone is not None
                                                         else model.backbone_kind != "scratch"))
                tokens += result.tokens_processed
                total_loss += result.loss
                steps += 1
                window_tokens = 0
                refresh_prefix(model, state)
            if exhausted_budget or count == len(episode.documents):
                break
            document = episode.documents[count]
            ids = model._document_tokens(document)
            offset = 0
            while offset < len(ids) and tokens < config.max_tokens:
                if window_tokens == config.sequence_length:
                    state._detach()
                    refresh_prefix(model, state)
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
