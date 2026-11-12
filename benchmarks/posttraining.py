"""Measure complete frozen-backbone Titans post-training steps on an H100."""

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.h100 import BASE_MODEL, source_digest
from titans import Document, TitansConfig, TitansModel, TrainingConfig, TrainingEpisode


class TimedTitansModel(TitansModel):
    """Measure checkpoint serialization separately without bypassing public fit."""

    checkpoint_seconds: float = 0.0

    def save_pretrained(self, path: str | Path) -> Path:
        """Write the complete checkpoint and record synchronized serialization time.

        Args:
            path: New checkpoint directory owned by this benchmark.

        Returns:
            The successfully saved checkpoint path.
        """
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = super().save_pretrained(path)
        torch.cuda.synchronize()
        self.checkpoint_seconds = time.perf_counter() - started
        return result


def main() -> None:
    """Warm compilation, then time four real AdamW steps through the public fit API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(31)
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("This benchmark explicitly requires an H100")
    config = TitansConfig(memory_dim=64, memory_hidden_dim=128, memory_layers=(0, 6, 12, 18),
                          chunk_size=32, prefill_chunk_size=256, attention_window=512,
                          persistent_tokens=4, inference_backend="sglang", attention_backend="fa3",
                          compile_kernels=True, use_liger=True)
    model = TimedTitansModel.from_pretrained(args.model, config, device="cuda", dtype=torch.bfloat16)
    assert isinstance(model, TimedTitansModel)
    # Identical source text; exact token budgets compensate for tokenizer differences.
    episode = TrainingEpisode("archive", (Document("batch", (
        "The archive records a blue box in room seven. "
        "An updated inventory moves the blue box to room nine. "
        "The previous location is retained in the historical register. "
    ) * 4096),))
    results: list[dict[str, object]] = []
    for name, budget in (("warmup", 8192), ("measured", args.tokens)):
        output = Path(f"/tmp/titans-training-{name}-{time.time_ns()}")
        training = TrainingConfig(output, max_tokens=budget, sequence_length=2048,
                                  tokens_per_step=8192, learning_rate=1e-4,
                                  objective="language_modeling", freeze_backbone=True,
                                  activation_checkpointing=True, seed=31)
        before = model.memories["0"].gates.weight.detach().clone()
        frozen_before = model.backbone.model.layers[0].self_attn.q_proj.weight.detach().clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = model.fit([episode], training)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        compute_seconds = elapsed - model.checkpoint_seconds
        assert result.tokens_processed == budget
        assert result.optimizer_steps == math.ceil(budget / training.tokens_per_step)
        assert result.mean_loss is not None and math.isfinite(result.mean_loss)
        assert not torch.equal(before, model.memories["0"].gates.weight)
        assert torch.equal(frozen_before, model.backbone.model.layers[0].self_attn.q_proj.weight)
        assert all(bool(torch.isfinite(p).all()) for p in model.memories.parameters())
        measured = {
            "phase": name, "tokens": result.tokens_processed,
            "optimizer_steps": result.optimizer_steps, "mean_loss": result.mean_loss,
            "wall_seconds_including_checkpoint": elapsed,
            "checkpoint_seconds": model.checkpoint_seconds,
            "training_seconds_excluding_checkpoint": compute_seconds,
            "tokens_per_second_excluding_checkpoint": budget / compute_seconds,
            "tokens_per_second_including_checkpoint": budget / elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "training_config": {**asdict(training), "output_dir": str(output)},
            "finite_parameters_changed_memory_frozen_backbone_check": "passed",
        }
        results.append(measured)
        print(json.dumps(measured), flush=True)
        shutil.rmtree(output)
    report = {
        "base_model": args.model, "gpu": torch.cuda.get_device_name(),
        "package_source_sha256": source_digest(), "config": asdict(config),
        "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "scope": "synthetic repeated document LM; frozen backbone; throughput, not learning quality",
        "attention": "autograd SDPA with cuDNN excluded during training; FA3 during inference only",
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    model.close()


if __name__ == "__main__":
    main()
