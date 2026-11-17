"""Compare equal-math chunk updates; run with python -m benchmarks.memory_update."""

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from tests.support.reference import reference_chunk
from titans.methods.titans import chunk_update
from titans.optim.kernels import compiled_chunk_update


def synchronize(device: torch.device) -> None:
    """Wait for accelerator work so timing includes completed execution.

    Args:
        device: CPU, CUDA or MPS device under measurement.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_call(call: Callable[[], object], device: torch.device, repeats: int) -> float:
    """Measure the median latency after five warmup calls.

    Args:
        call: One complete update to measure.
        device: Device whose work must complete before reading the clock.
        repeats: Number of measured calls.

    Returns:
        Median wall-clock milliseconds per update.
    """
    for _ in range(5):
        call()
    samples: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        call()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    """Measure inference updates and write machine-readable results when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1)
    device = torch.device(args.device)
    dimensions = [args.dim] + [2 * args.dim] * (args.depth - 1) + [args.dim]
    weights = tuple((torch.randn(args.batch_size, out_dim, in_dim, device=device) * 0.05)
                    .requires_grad_()
                    for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:], strict=True))
    momentum = tuple(torch.randn_like(w) * 0.001 for w in weights)
    keys = torch.randn(args.batch_size, args.chunk_size, args.dim, device=device)
    keys = torch.nn.functional.normalize(keys, dim=-1)
    values = torch.randn_like(keys)
    alpha = torch.rand(args.batch_size, args.chunk_size, device=device) * 0.01
    eta = torch.rand_like(alpha) * 0.1 + 0.85
    theta = torch.rand_like(alpha) * 0.01

    def optimized() -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Execute the production chunk update with inference graph settings.

        Returns:
            Final fast weights and momentum.
        """
        with torch.no_grad():
            return chunk_update(weights, momentum, keys, values, alpha, eta, theta)

    def reference() -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Execute the literal autograd reference without outer training graphs.

        Returns:
            Final fast weights and momentum for the identical chunk semantics.
        """
        return reference_chunk(weights, momentum, keys, values, alpha, eta, theta,
                               create_graph=False)

    def compiled() -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Run the JIT-compiled update with the same inference inputs.

        Returns:
            Updated fast weights and momentum.
        """
        with torch.no_grad():
            return compiled_chunk_update(weights, momentum, keys, values, alpha, eta, theta)

    actual, expected = optimized(), reference()
    maximum_error = 0.0
    for got, want in zip(actual[0] + actual[1], expected[0] + expected[1], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        maximum_error = max(maximum_error, float((got - want).abs().max().item()))
    fast_ms = time_call(optimized, device, args.repeats)
    reference_ms = time_call(reference, device, args.repeats)
    report = {
        "device": str(device), "platform": platform.platform(), "torch": str(torch.__version__),
        "dtype": "float32", "threads": args.threads, "batch_size": args.batch_size,
        "memory_dim": args.dim, "hidden_dim": 2 * args.dim, "depth": args.depth,
        "chunk_size": args.chunk_size, "repeats": args.repeats,
        "optimized_median_ms": fast_ms, "autograd_reference_median_ms": reference_ms,
        "speedup_over_literal_reference": reference_ms / fast_ms,
        "max_absolute_error": maximum_error,
        "scope": "memory update only; excludes LLM, projections, attention and decoding",
    }
    if not args.skip_compile:
        synchronize(device)
        first_call = time.perf_counter()
        compiled_result = compiled()
        synchronize(device)
        first_call_seconds = time.perf_counter() - first_call
        for got, want in zip(compiled_result[0] + compiled_result[1],
                             expected[0] + expected[1], strict=True):
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        compiled_ms = time_call(compiled, device, args.repeats)
        report.update({
            "compiled_first_call_seconds": first_call_seconds,
            "compiled_median_ms": compiled_ms,
            "compiled_speedup_over_eager": fast_ms / compiled_ms,
            "compiled_speedup_over_literal_reference": reference_ms / compiled_ms,
        })
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
