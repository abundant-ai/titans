"""SGLang model adapter with bounded, explicit Titans state ownership.

This adapter uses SGLang scheduling and sampling, while owning its memory and KV
state separately. Native radix-prefix sharing and native SGLang CUDA graphs
are disabled. The Titans graph runner includes fast weights and pending writes.
"""

from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, Qwen2Config

from titans.cuda_graphs import GraphDecoder
from titans.model import TitansModel
from titans.serving import RequestControl, StateRegistry
from titans.types import TitansConfig


class TitansMetadataProcessor(CustomLogitProcessor):
    """Carry typed state metadata through SGLang without changing sampling logits."""

    def __call__(
        self, logits: Tensor, custom_param_list: list[dict[str, object]] | None = None,
    ) -> Tensor:
        """Preserve logits; metadata has already been consumed by the model adapter.

        Args:
            logits: SGLang's next-token logits.
            custom_param_list: Request metadata supplied by the local client.

        Returns:
            The input logits unchanged.
        """
        return logits


class TitansForCausalLM(nn.Module):
    """Single-device Titans implementation registered as an external SGLang model."""

    def __init__(
        self, config: LlamaConfig | Qwen2Config, quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Construct the worker architecture; SGLang's loader supplies backbone weights.

        Args:
            config: Exported Llama or Qwen2 config containing Titans settings and adapter path.
            quant_config: Must be None; quantized fast-memory inference is unsupported.
            prefix: SGLang module prefix, unused for this single-device implementation.
        """
        super().__init__()
        if quant_config is not None:
            raise ValueError("Quantized Titans serving is not supported")
        settings = dict(config.titans_settings)
        settings["memory_layers"] = tuple(settings["memory_layers"])
        settings["inference_backend"] = "torch"
        settings["attention_backend"] = "fa3"
        self.adapter_path = Path(config.titans_adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path, local_files_only=True)
        self.titans = TitansModel(
            AutoModelForCausalLM.from_config(config), tokenizer, TitansConfig(**settings),
        )
        self.registry = StateRegistry(self.titans)
        self.decoder = GraphDecoder(self.titans)
        self.config = config

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None:
        """Load original backbone parameter names and restore the trained memory adapter.

        Args:
            weights: SGLang's checkpoint tensor iterator.

        Raises:
            ValueError: An unexpected backbone tensor is present.
        """
        parameters = dict(self.titans.backbone.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.endswith("rotary_emb.inv_freq"):
                continue
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                continue
            if name not in parameters:
                raise ValueError(f"Unexpected backbone tensor: {name}")
            parameters[name].copy_(tensor)
            loaded.add(name)
        missing = set(parameters) - loaded
        if missing:
            raise ValueError(f"Missing backbone tensors: {sorted(missing)}")
        adapter = load_file(self.adapter_path)
        self.titans.memories.load_state_dict({
            key.removeprefix("memories."): value for key, value in adapter.items()
            if key.startswith("memories.")
        }, strict=True)
        self.titans.persistent.copy_(adapter["persistent"])

    @torch.no_grad()
    def forward(
        self, input_ids: Tensor, positions: Tensor, forward_batch: ForwardBatch,
        input_embeds: Tensor | None = None,
    ) -> LogitsProcessorOutput:
        """Execute a SGLang request against its explicit stream or disposable branch.

        Args:
            input_ids: Flattened scheduled token IDs, already on the GPU.
            positions: SGLang request positions; Titans owns absolute stream positions.
            forward_batch: Scheduling and sampling metadata for the request.
            input_embeds: Unsupported; ordinary token IDs are required.

        Returns:
            Next-token logits for SGLang's sampler.

        Raises:
            ValueError: Batching or input embeddings violate this adapter's contract.
        """
        if forward_batch.batch_size != 1 or input_embeds is not None:
            raise ValueError("Titans serving currently supports one active request and token IDs")
        sampling = forward_batch.sampling_info
        metadata = sampling.custom_params if sampling is not None else None
        control = RequestControl.parse(metadata[0]) if metadata and metadata[0] else None
        state = self.registry.prepare(control) if control else self.titans._new_state()
        if state is None:
            self.decoder.deactivate()
            logits = torch.zeros(1, self.config.vocab_size, device=input_ids.device)
            return LogitsProcessorOutput(next_token_logits=logits)
        try:
            self.decoder.forward(input_ids[None, :], state, write=control.write if control else True)
            if control and control.document_complete:
                state._documents += 1
            if state._next_logits is None:
                raise ValueError("SGLang sent an empty forward block")
            return LogitsProcessorOutput(next_token_logits=state._next_logits[:, -1].float())
        except Exception:
            state._valid = False
            raise

    def get_embed_and_head(self) -> tuple[Tensor, Tensor]:
        """Expose the host embedding/head tensors expected by SGLang utilities.

        Returns:
            Input embedding and output projection weights.
        """
        return self.titans.backbone.model.embed_tokens.weight, self.titans.backbone.lm_head.weight


EntryClass = TitansForCausalLM
