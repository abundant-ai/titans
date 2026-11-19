"""Manual audit of Liger integration on the previously recorded real-text batch."""

import json
from pathlib import Path

import modal

from checks.modal_pretraining import cache, image

app = modal.App("titans-liger-integration-audit")


@app.function(image=image, gpu="H100!", cpu=4, memory=32768, timeout=1200,
              startup_timeout=300, max_containers=1, volumes={"/cache": cache})
def audit(batch_report: str) -> dict[str, object]:
    """Isolate wrappers, checkpointing and full-stream kernel replacements.

    Args:
        batch_report: Earlier real-text report containing exact token IDs and seed.

    Returns:
        Component errors and full-stream ablations, with no optimizer steps.
    """
    import hashlib
    import inspect
    import time
    import traceback
    from collections.abc import Callable
    from dataclasses import replace

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from torch import Tensor
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    import titans.optim.gpu_ops as ops
    from titans import TitansConfig, TitansModel
    from titans.runtime.backbones import scratch_config
    from titans.training.pretraining import _block_loss, fp32_execution

    def native_norm(hidden: Tensor, weight: Tensor, epsilon: float) -> Tensor:
        """Apply native Llama RMSNorm to supplied activations and scale."""
        value = hidden.float()
        return weight * (value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + epsilon)).to(hidden.dtype)

    def native_rope(query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the backbone's native rotary encoding to fresh Q and K."""
        return apply_rotary_pos_emb(query, key, cosine, sine)

    def native_mlp(hidden: Tensor, gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor,
                   gate_bias: Tensor | None, up_bias: Tensor | None, down_bias: Tensor | None) -> Tensor:
        """Apply native SwiGLU with exactly the wrapper's linear projections."""
        return F.linear(F.silu(F.linear(hidden, gate_weight, gate_bias))
                        * F.linear(hidden, up_weight, up_bias), down_weight, down_bias)

    def native_loss(weight: Tensor, hidden: Tensor, targets: Tensor) -> Tensor:
        """Apply native summed cross-entropy without shifting already aligned labels."""
        return F.cross_entropy(F.linear(hidden, weight), targets, reduction="sum")

    def error(actual: Tensor, expected: Tensor) -> dict[str, float]:
        """Compare FP32 values using FP64 diagnostic reductions on CPU."""
        a, b = actual.detach().double().cpu().flatten(), expected.detach().double().cpu().flatten()
        return {"relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
                "max_absolute": (a - b).abs().max().item(),
                "norm": a.norm().item(), "reference_norm": b.norm().item(),
                "cosine": F.cosine_similarity(a, b, dim=0, eps=1e-30).item()}

    def component(native: Callable[..., Tensor], liger: Callable[..., Tensor],
                  values: tuple[Tensor, ...], *, scale: float = 1.0) -> dict[str, object]:
        """Compare one forward and vector-Jacobian product with fresh input storage.

        Args:
            native: Independent native operation.
            liger: Production Liger wrapper.
            values: Differentiable floating-point inputs shared between both paths.
            scale: Multiplier applied to the identical upstream gradient.

        Returns:
            Output error and gradient errors for every supplied input.
        """
        references = None
        upstream = None
        result = {}
        for index, operation in enumerate((native, liger)):
            inputs = tuple(v.detach().clone().requires_grad_(True) for v in values)
            output = operation(*inputs)
            if upstream is None:
                upstream = torch.randn_like(output) * scale
            gradients = torch.autograd.grad(output, inputs, grad_outputs=upstream.clone())
            if index == 0:
                references = (output.detach().clone(), tuple(g.detach().clone() for g in gradients))
            else:
                assert references is not None
                result = {"output": error(output, references[0]),
                          "gradients": [error(a, b) for a, b in zip(gradients, references[1], strict=True)]}
        return result

    previous = json.loads(batch_report)
    report: dict[str, object] = {"torch": str(torch.__version__), "optimizer_steps": 0,
        "input_token_ids_sha256": previous["data"]["token_ids_sha256"],
        "source_sha256": {str(p.relative_to('/workspace')): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path('/workspace/titans').rglob('*.py'))}}
    actual = {"norm": ops.liger_norm, "rope": ops.liger_rotary,
              "mlp": ops.liger_mlp, "loss": ops.liger_loss}
    native = {"norm": native_norm, "rope": native_rope, "mlp": native_mlp, "loss": native_loss}
    attributes = {"norm": "liger_norm", "rope": "liger_rotary", "mlp": "liger_mlp", "loss": "liger_loss"}
    start = time.perf_counter()
    try:
        if report["source_sha256"] != previous["source_sha256"]:
            raise ValueError("Historical all-Liger audit requires the source hashes from its input report; "
                             "use modal_liger_clipping for the current selective-Liger runtime")
        import liger_kernel.ops.fused_linear_cross_entropy as ce_source
        import liger_kernel.ops.rms_norm as norm_source
        import liger_kernel.ops.rope as rope_source
        import liger_kernel.ops.swiglu as mlp_source

        report["installed_liger_source_sha256"] = {
            name: hashlib.sha256(inspect.getsource(module).encode()).hexdigest()
            for name, module in (("loss", ce_source), ("norm", norm_source),
                                 ("rope", rope_source), ("mlp", mlp_source))}
        torch.manual_seed(17)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"x": 0})))
        config = TitansConfig(compile_kernels=False, use_liger=False)
        with torch.device("cuda"):
            model = TitansModel(LlamaForCausalLM(scratch_config()), tokenizer, config)
        model.requires_grad_(True)
        model.train()
        ids = torch.tensor([previous["data"]["input_ids_with_lookahead"]], device="cuda")
        captured: list[Tensor] = []

        def capture_head(module: torch.nn.Module, arguments: tuple[Tensor, ...]) -> None:
            """Capture real final hidden activations for an isolated loss comparison."""
            captured.append(arguments[0].detach().clone())

        watched = ("backbone.lm_head.weight", "backbone.model.layers.0.self_attn.q_proj.weight",
                   "memories.0.qkv.weight", "memories.0.initial_weights.0")
        gradients: dict[str, dict[str, Tensor]] = {}
        outputs = {}
        modes = (("native", set(), True), ("native_wrappers", set(), True),
                 ("loss_only", {"loss"}, True), ("norm_only", {"norm"}, True),
                 ("rope_only", {"rope"}, True), ("mlp_only", {"mlp"}, True),
                 ("all", set(actual), True), ("all_no_checkpoint", set(actual), False),
                 ("native_no_checkpoint", set(), False))
        with fp32_execution(ids.device):
            for name, enabled, use_checkpoint in modes:
                print("Ablation:", name, flush=True)
                for key, attribute in attributes.items():
                    setattr(ops, attribute, actual[key] if key in enabled else native[key])
                model.config = replace(config, use_liger=not name.startswith("native_no") and name != "native")
                for memory in model.memories.values():
                    memory.config = model.config
                model._checkpoint_backbone = use_checkpoint
                model.zero_grad(set_to_none=True)
                handle = model.backbone.lm_head.register_forward_pre_hook(capture_head) if name == "native" else None
                state = model._new_state(prefix_hidden_only=True)
                loss = _block_loss(model, ids[:, :-1], ids[:, 1:], state) / 4096
                loss.backward()
                if handle is not None:
                    handle.remove()
                gradients[name] = {n: model.get_parameter(n).grad.detach().cpu() for n in watched}
                outputs[name] = {"loss": loss.item(),
                                 "gradients_vs_native": {n: error(gradients[name][n], gradients["native"][n])
                                                         for n in watched}}
                del loss, state
            report["ablations"] = outputs
            report["checkpoint_effect"] = {
                mode: {n: error(gradients[mode + "_no_checkpoint"][n], gradients[mode][n]) for n in watched}
                for mode in ("all", "native")}
            model.zero_grad(set_to_none=True)
            for key, attribute in attributes.items():
                setattr(ops, attribute, actual[key])
            hidden = torch.cat(captured, dim=1).flatten(0, 1)
            assert hidden.shape == (4096, 1024)
            labels = ids[:, 1:].flatten()

            def loss_native(weight: Tensor, h: Tensor) -> Tensor:
                """Apply native mean loss to fixed real-text activations."""
                return native_loss(weight, h, labels) / 4096

            def loss_liger(weight: Tensor, h: Tensor) -> Tensor:
                """Apply production summed Liger loss and its outer normalization."""
                return actual["loss"](weight, h, labels) / 4096

            def norm_native(h: Tensor, weight: Tensor) -> Tensor:
                """Apply native RMSNorm with the fixed decoder epsilon."""
                return native_norm(h, weight, 1e-5)

            def norm_liger(h: Tensor, weight: Tensor) -> Tensor:
                """Apply the production RMSNorm wrapper with the same epsilon."""
                return actual["norm"](h, weight, 1e-5)

            def mlp_native(h: Tensor, gate: Tensor, up: Tensor, down: Tensor) -> Tensor:
                """Apply the decoder's bias-free native feed-forward operation."""
                return native_mlp(h, gate, up, down, None, None, None)

            def mlp_liger(h: Tensor, gate: Tensor, up: Tensor, down: Tensor) -> Tensor:
                """Apply production Liger SwiGLU with fresh projected activations."""
                return actual["mlp"](h, gate, up, down, None, None, None)

            def mlp_checkpoint(h: Tensor, gate: Tensor, up: Tensor, down: Tensor) -> Tensor:
                """Exercise the exact non-reentrant checkpoint wrapper used in training."""
                return checkpoint(mlp_liger, h, gate, up, down, use_reentrant=False)

            cosine, sine = model.backbone.model.rotary_emb(hidden[:512][None], torch.arange(512, device='cuda')[None])

            def rope_native(q: Tensor, k: Tensor) -> Tensor:
                """Concatenate native Q/K outputs under fixed rotary embeddings."""
                return torch.cat(native_rope(q, k, cosine, sine), dim=0)

            def rope_liger(q: Tensor, k: Tensor) -> Tensor:
                """Concatenate Liger outputs for the same fresh Q/K tensors."""
                return torch.cat(actual["rope"](q, k, cosine, sine), dim=0)

            print("Checking isolated loss, RMSNorm, SwiGLU, checkpoint, and RoPE", flush=True)
            layer = model.backbone.model.layers[0]
            values = (hidden[:512][None], layer.mlp.gate_proj.weight,
                      layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)
            report["components"] = {
                "loss": component(loss_native, loss_liger, (model.backbone.lm_head.weight, hidden), scale=0.25),
                "rms_norm": component(norm_native, norm_liger, (hidden[:512][None], layer.input_layernorm.weight)),
                "swiglu": component(mlp_native, mlp_liger, values),
                "swiglu_checkpoint": component(mlp_liger, mlp_checkpoint, values),
                "rope": component(rope_native, rope_liger,
                                  (torch.randn(1,512,16,64,device='cuda').transpose(1,2),
                                   torch.randn(1,512,16,64,device='cuda').transpose(1,2))),
            }
        report["component_checks_passed"] = all(
            c['output']['relative_l2'] < 2e-5 and all(g['relative_l2'] < 2e-5 for g in c['gradients'])
            for c in report['components'].values())
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        for key, attribute in attributes.items():
            setattr(ops, attribute, actual[key])
    report['elapsed_seconds'] = time.perf_counter() - start
    return report


@app.local_entrypoint()
def main(source: str, output: str = '/private/tmp/titans-liger-audit-report.json') -> None:
    """Launch bounded H100 diagnostics and retain evidence without modifying models.

    Args:
        source: Existing report containing the exact previously evaluated batch.
        output: New JSON audit report path.
    """
    import time

    call = audit.spawn(Path(source).read_text())
    start = time.monotonic()
    while True:
        try:
            report = call.get(timeout=5)
            break
        except TimeoutError:
            elapsed = time.monotonic() - start
            allocated = any(c.task_id for c in call.get_call_graph())
            if (elapsed > 300 and not allocated) or elapsed > 1500:
                call.cancel(terminate_containers=True)
                report = {'status': 'allocation_unavailable' if not allocated else 'execution_timeout'}
                break
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))
