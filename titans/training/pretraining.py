"""One continuous FP32 stream with one joint optimizer step per input block."""

import math
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedTokenizerBase

from titans.core.types import TrainingResult
from titans.runtime.state import StreamState
from titans.training.trainer import refresh_prefix

if TYPE_CHECKING:
    from titans.runtime.model import ContinualModel


PRETRAINING_BLOCK_SIZE = 4096


def pretraining_batches(documents: Iterable[str], tokenizer: PreTrainedTokenizerBase
                       ) -> Iterator[tuple[Tensor, Tensor]]:
    """Pack ordered documents into consecutive single-instance next-token batches.

    Append EOS after each document. Retain one lookahead token between batches,
    so every emitted label is scored before entering instance memory. The final
    EOS is a target only; an unfinished final block contains no padding.

    Args:
        documents: Lazy document texts in stream order.
        tokenizer: Backbone tokenizer with an EOS separator ID.

    Yields:
        CPU int64 input/label pairs [1, 4096], with a possibly shorter final pair.

    Raises:
        ValueError: The tokenizer has no EOS separator.
    """
    separator = tokenizer.eos_token_id
    if separator is None:
        raise ValueError("Packing documents requires a tokenizer EOS separator")
    pending: list[int] = []
    for document in documents:
        pending.extend(tokenizer.encode(document, add_special_tokens=False))
        pending.append(separator)
        start = 0
        while len(pending) - start > PRETRAINING_BLOCK_SIZE:
            tokens = torch.tensor([pending[start:start + PRETRAINING_BLOCK_SIZE + 1]],
                                  dtype=torch.long)
            yield tokens[:, :-1], tokens[:, 1:]
            start += PRETRAINING_BLOCK_SIZE
        pending = pending[start:]
    if len(pending) > 1:
        tokens = torch.tensor([pending], dtype=torch.long)
        yield tokens[:, :-1], tokens[:, 1:]


@contextmanager
def fp32_execution(device: torch.device) -> Iterator[None]:
    """Scope full-precision operations and restore the caller's precision policy.

    Args:
        device: Single model device. CUDA also disables TF32 and cuDNN SDPA,
            whose backward is excluded by the existing training runtime.

    Yields:
        An execution scope with autocast disabled, including under outer autocast.
    """
    if device.type != "cuda":
        with torch.autocast(device.type, enabled=False):
            yield
        return
    previous_matmul = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast("cuda", enabled=False), sdpa_kernel([
            SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
        ]):
            yield
    finally:
        torch.set_float32_matmul_precision(previous_matmul)
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def _validate_batch(inputs: Tensor, labels: Tensor, vocabulary: int) -> None:
    """Validate one dense, causally aligned single-instance batch before forwarding.

    Args:
        inputs: Input IDs with one instance and at most 4096 tokens.
        labels: Corresponding next-token IDs, including the final lookahead.
        vocabulary: Number of supported token IDs.

    Raises:
        ValueError: Shape, dtype, token range or next-token alignment is invalid.
    """
    if (inputs.ndim != 2 or inputs.shape[0] != 1 or labels.shape != inputs.shape
            or not 1 <= inputs.shape[1] <= PRETRAINING_BLOCK_SIZE):
        raise ValueError("Pretraining requires batch size 1 and matching [1, T] tensors, 1 <= T <= 4096")
    if inputs.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("Pretraining input_ids and labels must have dtype torch.int64")
    for ids in (inputs, labels):
        if ids.min().item() < 0 or ids.max().item() >= vocabulary:
            raise ValueError("Pretraining token IDs must be inside the backbone vocabulary")
    if not torch.equal(inputs[:, 1:], labels[:, :-1]):
        raise ValueError("Pretraining labels must be shifted next-token targets")


def _block_loss(model: "ContinualModel", inputs: Tensor, labels: Tensor,
                state: StreamState) -> Tensor:
    """Forward a complete block with inner writes and return summed causal loss.

    Args:
        model: FP32 model with joint gradients enabled.
        inputs: Validated device-local input IDs [1, T].
        labels: Aligned device-local labels [1, T].
        state: Continuing instance memory and cached context, mutated in place.

    Returns:
        FP32 summed next-token cross-entropy retaining the entire block's graph.

    Raises:
        ValueError: A custom memory produced floating-point buffers outside FP32.
    """
    fused = model.config.use_liger and inputs.device.type == "cuda"
    predictions = model._forward_embeddings(
        model.backbone.model.embed_tokens(inputs), state, return_hidden=fused,
    )
    for memory in (*state.memory.values(), *state._retrieval.values()):
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in memory.tensors()):
            raise ValueError("Pretraining memory buffers must use FP32")
    if fused:
        from titans.optim.gpu_ops import liger_loss

        loss = liger_loss(model.backbone.lm_head.weight, predictions.flatten(0, 1), labels.flatten())
    else:
        loss = F.cross_entropy(predictions.flatten(0, 1), labels.flatten(), reduction="sum")
    state._tokens += inputs.numel()
    state._next_target_scored = True
    return loss.float()


def _prepare_next_block(model: "ContinualModel", state: StreamState) -> None:
    """Reset ordinary attention context and rebuild a differentiable persistent prefix.

    Args:
        model: FP32 model after the preceding outer optimizer step.
        state: Detached instance memory and cached context. Active memory and
            retrieval cursors survive. Connections can retain their own KV using
            retain_training_cache, including a MAC segment spanning the boundary.
    """
    executions = model._connection_executions(state._position, 1)
    count = model.config.persistent_tokens
    for layer, cache in state._attention.items():
        execution = executions.get(layer)
        if execution is None or not execution.retain_training_cache:
            cache.keys = cache.keys[:, :, :count].clone()
            cache.values = cache.values[:, :, :count].clone()
            cache.positions = cache.positions[:count].clone()
    refresh_prefix(model, state)


def pretrain_model(model: "ContinualModel", batches: Iterable[tuple[Tensor, Tensor]], *,
                   output_dir: str | Path, learning_rate: float = 1e-4) -> TrainingResult:
    """Train a scratch decoder and its method on one continuous FP32 token stream.

    Each full block has 4096 inputs. One shorter final block is accepted. Model
    parameters are converted to FP32 before AdamW is constructed. Active memory,
    pending writes and connection-owned context survive with detached history.
    Ordinary decoder KV is reset before each subsequent block, and persistent
    prefix representations are rebuilt using the updated shared parameters.

    Args:
        model: Scratch-backbone model with protocol-supplied memory/connections.
        batches: Consecutive input/next-token-label tensor pairs, batch size one.
        output_dir: New directory for trained model parameters.
        learning_rate: Positive finite AdamW learning rate.

    Returns:
        Checkpoint, token-weighted mean loss, step count and detached final stream.
        Model checkpoints store shared parameters; the live stream is returned
        separately and optimizer state is local to this call.

    Raises:
        ValueError: Backbone, batch alignment, sequence lengths or rate is invalid.
        FileExistsError: The checkpoint destination already exists.
        FloatingPointError: A loss is nonfinite.
        RuntimeError: Gradient clipping detects a nonfinite norm.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    if model.backbone_kind != "scratch":
        raise ValueError("pretrain requires the scratch backbone from from_scratch()")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    started = time.perf_counter()
    was_training = model.training
    was_checkpointing = model._checkpoint_backbone
    parameters = model.outer_parameters(train_backbone=True)
    previous_grad = tuple(p.requires_grad for p in parameters)
    device = model.backbone.model.embed_tokens.weight.device
    state: StreamState | None = None
    steps = tokens = 0
    total_loss = 0.0
    previous_target: Tensor | None = None
    previous_size = PRETRAINING_BLOCK_SIZE
    optimizer: torch.optim.AdamW | None = None
    model._revision += 1
    try:
        model.float()
        model.train()
        model._checkpoint_backbone = True
        for parameter in parameters:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        with torch.enable_grad(), fp32_execution(device):
            for inputs, labels in batches:
                _validate_batch(inputs, labels, model.backbone.config.vocab_size)
                if previous_size < PRETRAINING_BLOCK_SIZE:
                    raise ValueError("Only the final pretraining block may be shorter than 4096")
                inputs, labels = inputs.to(device), labels.to(device)
                if previous_target is not None and not torch.equal(previous_target, inputs[:, :1]):
                    raise ValueError("Pretraining batches must continue the preceding lookahead token")
                if state is None:
                    state = model._new_state(prefix_hidden_only=True)
                else:
                    _prepare_next_block(model, state)
                optimizer.zero_grad(set_to_none=True)
                summed_loss = _block_loss(model, inputs, labels, state)
                loss = summed_loss / labels.numel()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite pretraining loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                state._detach()
                total_loss += summed_loss.detach().item()
                tokens += labels.numel()
                steps += 1
                previous_size = inputs.shape[1]
                previous_target = labels[:, -1:].detach().clone()
                del loss, summed_loss
            if state is None:
                raise ValueError("Pretraining requires at least one batch")
        checkpoint_path = model.save_pretrained(destination)
    except Exception:
        if state is not None:
            state._valid = False
        raise
    finally:
        if state is not None:
            state._detach()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for parameter, enabled in zip(parameters, previous_grad, strict=True):
            parameter.requires_grad_(enabled)
        model.train(was_training)
        model._checkpoint_backbone = was_checkpointing
    return TrainingResult(checkpoint_path, tokens, steps, total_loss / tokens,
                          time.perf_counter() - started, "data_exhausted", state)
