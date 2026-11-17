"""Explicit feedback learning with independent objectives and update policies."""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from titans.core.protocols import (
    LossContext,
    LossValue,
    Objective,
    UpdateContext,
    UpdateResult,
    UpdateRule,
)
from titans.core.types import LearningResult, Question
from titans.runtime.state import StreamState

if TYPE_CHECKING:
    from titans.runtime.model import ContinualModel


def cross_entropy(context: LossContext) -> LossValue:
    """Compute summed token cross-entropy with explicit target-count normalization.

    Args:
        context: Aligned predictions and targets, plus optional state for custom losses.

    Returns:
        Summed loss and number of target tokens. No stream state is changed.
    """
    return LossValue(F.cross_entropy(context.predictions.float(), context.targets,
                                    reduction="sum"), float(context.targets.numel()))


def validate_loss(result: LossValue) -> None:
    """Check a user-supplied scalar objective before any parameter update.

    Args:
        result: Differentiable loss and its normalization weight.

    Raises:
        ValueError: The loss is nonscalar or its weight is not finite and positive.
        FloatingPointError: The objective is not finite.
    """
    if result.value.ndim != 0 or not math.isfinite(result.weight) or result.weight <= 0:
        raise ValueError("Objective must return a scalar loss and a finite positive weight")
    if not bool(torch.isfinite(result.value).item()):
        raise FloatingPointError("Nonfinite objective; no update was applied")


@dataclass(frozen=True)
class GradientUpdate:
    """SGD policy selecting phi, active memory W, or both; B is never a target.

    Phi is updated in place. W replacements are returned functionally so they
    can be committed to the continuing stream without committing teacher tokens.
    Custom UpdateRule implementations can instead own Adam state or other solvers.
    """

    targets: Literal["phi", "memory", "both"] = "memory"
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid target groups, nonfinite rates and clipping bounds."""
        if self.targets not in ("phi", "memory", "both"):
            raise ValueError("targets must be phi, memory or both")
        if any(not math.isfinite(x) or x <= 0
               for x in (self.learning_rate, self.max_grad_norm)):
            raise ValueError("Learning rate and gradient bound must be finite and positive")

    def __call__(self, context: UpdateContext) -> UpdateResult:
        """Differentiate the loss only with respect to the selected target groups.

        Args:
            context: Normalized scalar loss and permitted phi/W tensors.

        Returns:
            New active weights and whether shared parameters received an update.

        Raises:
            ValueError: No selected tensors are differentiable.
            FloatingPointError: Any selected gradient is nonfinite.
        """
        shared = context.shared_parameters if self.targets != "memory" else ()
        memory = dict(context.memory_parameters) if self.targets != "phi" else {}
        parameters = shared + tuple(memory.values())
        if not parameters or not context.loss.requires_grad:
            raise ValueError("The loss has no differentiable targets")
        gradients = torch.autograd.grad(context.loss, parameters, allow_unused=True)
        present = [gradient for gradient in gradients if gradient is not None]
        if not present:
            raise ValueError("The objective is disconnected from the selected targets")
        norm = torch.stack([g.float().square().sum() for g in present]).sum().sqrt()
        if not bool(torch.isfinite(norm).item()):
            raise FloatingPointError("Nonfinite gradient; no update was applied")
        scale = (self.max_grad_norm / (norm + 1e-12)).clamp(max=1).detach()
        changed = False
        with torch.no_grad():
            for parameter, gradient in zip(shared, gradients[:len(shared)], strict=True):
                if gradient is not None:
                    parameter.add_(-self.learning_rate * scale * gradient)
                    changed = True
            replacements = {
                key: value - self.learning_rate * scale * gradient
                for (key, value), gradient in zip(memory.items(), gradients[len(shared):], strict=True)
                if gradient is not None
            }
        return UpdateResult(replacements, changed)


def feedback_tokens(
    model: "ContinualModel", question: Question, target_answer: str,
) -> tuple[list[int], list[int]]:
    """Encode feedback identically for budget checks and learning.

    Args:
        model: Runtime owning tokenization and prompting.
        question: Question whose target was explicitly released.
        target_answer: Nonempty target text.

    Returns:
        Prompt IDs and answer IDs including the tokenizer's EOS, when available.

    Raises:
        ValueError: The target is empty or tokenization yields no answer tokens.
    """
    if not target_answer.strip():
        raise ValueError("Feedback requires a nonempty target answer")
    prompt = model._question_tokens(question)
    targets = model.tokenizer.encode(target_answer, add_special_tokens=False)
    if model.tokenizer.eos_token_id is not None:
        targets.append(model.tokenizer.eos_token_id)
    if not targets:
        raise ValueError("Feedback target produced no tokens")
    return prompt, targets


def learn(
    model: "ContinualModel", question: Question, target_answer: str, state: StreamState,
    *, loss: Objective | None = None, update: UpdateRule | None = None,
    retain_history: bool = False,
) -> LearningResult:
    """Apply released QA supervision to phi, W, or both, on an isolated branch.

    Args:
        model: Frozen-backbone runtime. Phi updates invalidate other live streams.
        question: Question being learned from, not an evaluator's hidden query.
        target_answer: Explicitly available answer or correction.
        state: Continuing local stream; teacher prompts and answers are never appended.
        loss: Objective over predictions, phi and pre-question memory.
        update: Target selection and optimizer; defaults to W-only SGD.
        retain_history: Keep recent document graphs for outer meta-gradients. A completed
            update always detaches the continuing state before parameters are reused.

    Returns:
        Normalized loss, forwarded feedback tokens and update accounting.

    Raises:
        ValueError: The state is invalid, feedback is empty or remote state was supplied.
    """
    model._validate_state(state)
    if state._remote_id is not None:
        raise ValueError("Explicit learning requires a local torch stream")
    prompt, targets = feedback_tokens(model, question, target_answer)
    model.backbone.requires_grad_(False)
    objective = loss or cross_entropy
    rule = update or GradientUpdate()
    shared = model.shared_parameters()
    was_training = model.training
    model.eval()
    attention_context = sdpa_kernel([
        SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
    ]) if model.backbone.model.embed_tokens.weight.device.type == "cuda" else nullcontext()
    try:
        with torch.enable_grad(), attention_context:
            working = state._copy(detach=not retain_history)
            memory: dict[tuple[int, str], Tensor] = {}
            for layer, algorithm_state in working._memory.items():
                for name, tensor in algorithm_state.adaptable().items():
                    if not tensor.is_floating_point():
                        raise ValueError("Adaptable memory tensors must be floating point")
                    tensor.requires_grad_(True)
                    memory[layer, name] = tensor
            branch = working._copy()
            model._forward_tokens(prompt, branch, write=False, last_only=True)
            previous = branch._next_logits
            if previous is None:
                raise ValueError("Question prompt produced no prediction")
            logits = model._forward_tokens(targets, branch, write=False)
            predictions = torch.cat((previous, logits[:, :-1]), 1).reshape(-1, logits.shape[-1])
            result = objective(LossContext(
                predictions, torch.tensor(targets, device=logits.device), working, shared, "answer",
            ))
            validate_loss(result)
            normalized = result.value / result.weight
            value = float(normalized.detach().item())
            # Any exception after invoking an external updater can leave phi changed.
            # Invalidate the supplied state on failure rather than reusing a partial update.
            try:
                changed = rule(UpdateContext(normalized, shared, memory))
                for (layer, name), tensor in changed.memory.items():
                    if (layer, name) not in memory:
                        raise ValueError("Updater returned an undeclared memory target")
                    original = memory[layer, name]
                    if (tensor.shape != original.shape or tensor.device != original.device
                            or tensor.dtype != original.dtype):
                        raise ValueError("Memory replacement changed shape, device or dtype")
                    if not bool(torch.isfinite(tensor).all().item()):
                        raise FloatingPointError("Updater returned nonfinite memory")
                    working._memory[layer].replace({name: tensor})
            except Exception:
                model._revision += 1
                state._valid = False
                raise
            if changed.shared_changed:
                model.close()
                model._revision += 1
            state._memory = working._memory
            state._revision = model._revision
            state._detach()
    finally:
        model.train(was_training)
    return LearningResult(value, len(prompt) + len(targets), changed.shared_changed,
                          len(changed.memory))
