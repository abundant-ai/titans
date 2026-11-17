"""Measure actual SGLang Titans ingestion and decoding on a sub-2B model."""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from titans import AnswerConfig, Document, Question, TitansConfig, TitansModel
from titans.optim.cuda_graphs import GraphDecoder
from titans.runtime.connections import residual_connection
from titans.serving.registry import RequestControl

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def source_digest() -> str:
    """Hash the actual uploaded package code so measurements identify their implementation.

    Returns:
        SHA-256 over relative paths and contents of all package Python files.
    """
    import titans

    root = Path(titans.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def profile_decode(model: TitansModel, tokens: list[int]) -> dict[str, object]:
    """Measure direct decoding and profile the numerical path independently of scheduling.

    Args:
        model: CUDA Titans model whose private numerical path is under test.
        tokens: Document tokens used to initialize a realistic full attention window.

    Returns:
        Direct-call throughput and CPU/CUDA operator summaries. Profiling overhead
        is excluded from the direct throughput measurement.
    """
    with torch.no_grad():
        decoder = GraphDecoder(model)
        state = model._new_state()
        model._forward_tokens(tokens, state, last_only=True)
        ids = torch.tensor([[7]], device="cuda")
        for _ in range(40):
            decoder.forward(ids, state)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(64):
            decoder.forward(ids, state)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(32):
                decoder.forward(ids, state)
            torch.cuda.synchronize()
    events = profile.key_averages()
    return {
        "direct_decode_tokens_per_second": 64 / seconds,
        "captured_cuda_graphs": len(decoder.graphs),
        "cuda_graph_replays": decoder.replays,
        "cpu_operator_profile": events.table(sort_by="self_cpu_time_total", row_limit=18),
        "cuda_operator_profile": events.table(sort_by="self_cuda_time_total", row_limit=18),
    }


def main() -> None:
    """Run warmup, source-preservation checks, and timed SGLang/FA3 requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--model", default=BASE_MODEL)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(31)
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("This benchmark explicitly requires an H100")
    config = TitansConfig(memory_dim=64, memory_hidden_dim=128, memory_layers=(0, 6, 12, 18),
                          chunk_size=32, prefill_chunk_size=256, attention_window=512,
                          persistent_tokens=4, inference_backend="sglang", attention_backend="fa3",
                          compile_kernels=True, use_liger=True)
    started = time.perf_counter()
    model = TitansModel.from_pretrained(args.model, config, device="cuda", dtype=torch.bfloat16,
                                              connection_factory=residual_connection)
    backbone_parameters = sum(p.numel() for p in model.backbone.parameters())
    assert backbone_parameters < 2_000_000_000
    report: dict[str, object] = {
        "base_model": args.model, "backbone_parameters": backbone_parameters,
        "package_source_sha256": source_digest(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "sglang": importlib.metadata.version("sglang"),
        "liger_kernel": importlib.metadata.version("liger-kernel"),
        "config": asdict(config), "repeats": args.repeats,
        "scope": "single active stream; untrained memory adapters; throughput, not QA accuracy",
    }
    document = Document("batch", "The archive records a blue box in room seven. " * 100)
    tokens_per_document = len(model._document_tokens(document))
    question = Question("speed", "Write a detailed description of the archive.")
    ingest_samples: list[float] = []
    generation_samples: list[float] = []
    first_token_samples: list[float] = []
    if args.profile:
        report["profile"] = profile_decode(model, model._document_tokens(document))
    try:
        # Startup and compilation are measured separately and excluded below.
        warm_state = model.ingest(document)
        backend = model._serving_backend()
        for limit in (1, 128):
            control = RequestControl("answer", uuid.uuid4().hex, source_id=warm_state._remote_id or "")
            backend.engine.generate(
                input_ids=model._question_tokens(question),
                sampling_params={"temperature": 0.0, "max_new_tokens": limit, "ignore_eos": True,
                                 "sampling_seed": 0,
                                 "custom_params": asdict(control)},
                custom_logit_processor=backend.processor,
            )
            backend._request([0], RequestControl("release", control.state_id))
        report["load_startup_and_warmup_seconds"] = time.perf_counter() - started
        first = model.answer(question, warm_state, AnswerConfig(max_new_tokens=8))
        second = model.answer(question, warm_state, AnswerConfig(max_new_tokens=8))
        assert first == second, "Answering changed the document source state"
        sampled = model.answer(question, warm_state, AnswerConfig(max_new_tokens=16,
                                                                 temperature=0.7, seed=31))
        assert sampled == model.answer(question, warm_state, AnswerConfig(max_new_tokens=16,
                                                                          temperature=0.7, seed=31))
        fork = model.clone_state(warm_state)
        assert model.answer(question, fork, AnswerConfig(max_new_tokens=8)) == first
        backend._request([0], RequestControl("release", fork._remote_id or ""))
        for iteration in range(args.repeats):
            start = time.perf_counter()
            state = model.ingest(document)
            ingest_samples.append(time.perf_counter() - start)
            for limit, samples in ((1, first_token_samples), (128, generation_samples)):
                control = RequestControl("answer", f"bench-{iteration}-{limit}",
                                         source_id=state._remote_id or "")
                start = time.perf_counter()
                result = backend.engine.generate(
                    input_ids=model._question_tokens(question),
                    sampling_params={"temperature": 0.0, "max_new_tokens": limit, "ignore_eos": True,
                                     "sampling_seed": 0,
                                     "custom_params": asdict(control)},
                    custom_logit_processor=backend.processor,
                )
                samples.append(time.perf_counter() - start)
                assert result["meta_info"]["completion_tokens"] == limit
                backend._request([0], RequestControl("release", control.state_id))
            backend._request([0], RequestControl("release", state._remote_id or ""))
        ingest_seconds = statistics.median(ingest_samples)
        generation_seconds = statistics.median(generation_samples)
        first_seconds = statistics.median(first_token_samples)
        report.update({
            "document_tokens": tokens_per_document,
            "document_characters": len(document.text),
            "ingest_characters_per_second": len(document.text) / ingest_seconds,
            "ingest_seconds_samples": ingest_samples,
            "ingest_tokens_per_second": tokens_per_document / ingest_seconds,
            "generation_128_tokens_seconds_samples": generation_samples,
            "time_to_first_token_seconds_samples": first_token_samples,
            "generation_including_prefill_tokens_per_second": 128 / generation_seconds,
            "estimated_decode_tokens_per_second": 127 / (generation_seconds - first_seconds),
            "median_time_to_first_token_seconds": first_seconds,
            "source_preservation_and_clone_check": "passed",
            "seeded_sampling_check": "passed",
        })
    finally:
        model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
