"""Verify attachment and a real post-training step on a cached Qwen checkpoint."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from titans import (
    AnswerConfig,
    Document,
    Question,
    TitansConfig,
    TitansModel,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
)
from titans.runtime.connections import residual_connection


def main() -> None:
    """Compare stock Qwen logits, train frozen-backbone memory, and test a reload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B-Instruct",
                        choices=["Qwen/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    started = time.perf_counter()
    config = TitansConfig(memory_dim=16, memory_hidden_dim=32, memory_layers=(0, 1),
                          attention_window=128, persistent_tokens=0, chunk_size=16)
    model = TitansModel.from_pretrained(args.backbone, config, local_files_only=True,
                                         connection_factory=residual_connection)
    tokens = model.tokenizer.encode("Alice lives in Paris. Later she moves to Rome. " * 3)
    with torch.no_grad():
        for memory in model.memories.values():
            memory.output.weight.zero_()
        stock = model.backbone(torch.tensor([tokens]), use_cache=False).logits
        attached = model._forward_tokens(tokens, model._new_state())
    max_error = float((stock - attached).abs().max().item())
    torch.testing.assert_close(attached, stock, rtol=1e-3, atol=2e-4)
    # Reinitialize nonzero adapters before testing real post-training.
    model = TitansModel(model.backbone, model.tokenizer, replace(config, persistent_tokens=2),
                        connection_factory=residual_connection)
    memory_before = model.memories["0"].gates.weight.detach().clone()
    backbone_sample = model.backbone.model.layers[0].self_attn.q_proj.weight.detach().clone()
    episodes = [TrainingEpisode("cities", (
        Document("one", "Alice lives in Paris."),
        Document("two", "Alice moved to Rome. Her current city is Rome."),
    ), (
        TrainingQuestion(1, Question("first", "Where does Alice live?"), "Paris"),
        TrainingQuestion(2, Question("second", "Where does Alice live now?"), "Rome"),
    ))]
    result = model.fit(episodes, TrainingConfig(
        args.checkpoint, max_tokens=128, sequence_length=128, tokens_per_step=128,
        learning_rate=1e-4, objective="mixed",
    ))
    assert result.optimizer_steps > 0
    assert not torch.equal(memory_before, model.memories["0"].gates.weight)
    torch.testing.assert_close(backbone_sample,
                               model.backbone.model.layers[0].self_attn.q_proj.weight,
                               rtol=0, atol=0)
    state = None
    for document in episodes[0].documents:
        state = model.ingest(document, state)
    assert state is not None
    question = Question("check", "Where does Alice live now?")
    answer = model.answer(question, state, AnswerConfig(max_new_tokens=8))
    loaded = TitansModel.load_pretrained(args.checkpoint)
    restored_state = None
    for document in episodes[0].documents:
        restored_state = loaded.ingest(document, restored_state)
    assert restored_state is not None
    restored_answer = loaded.answer(question, restored_state, AnswerConfig(max_new_tokens=8))
    assert restored_answer == answer
    report = {
        "backbone_snapshot": str(args.backbone), "device": "cpu", "threads": args.threads,
        "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
        "adapter_parameters": sum(p.numel() for p in model.memories.parameters())
                              + model.persistent.numel(),
        "stock_llama_max_logit_error_with_memory_disabled": max_error,
        "training": {**asdict(result), "checkpoint_path": str(result.checkpoint_path)},
        "memory_write_gates_updated": True, "backbone_sample_unchanged": True,
        "checkpoint_roundtrip_answer_identical": True, "example_answer": asdict(answer),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "execution and gradient smoke check; one optimizer step is not a QA evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
