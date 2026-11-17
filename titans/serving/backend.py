"""Local SGLang engine client for the public TitansModel stream operations."""

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from titans.core.types import Answer, AnswerConfig, Document, Question
from titans.runtime.state import StreamState
from titans.serving.registry import RequestControl

if TYPE_CHECKING:
    from titans.runtime.model import ContinualModel


class SGLangBackend:
    """One local engine; states stay on its GPU worker between API calls."""

    def __init__(self, model: "ContinualModel") -> None:
        """Export current trained weights and start the custom Titans engine.

        Args:
            model: CUDA model whose parameters define this inference revision.
        """
        import sglang as sgl
        from sglang.srt.models.registry import ModelRegistry

        from titans.serving.models.titans_llama import TitansForCausalLM, TitansMetadataProcessor

        self.model = model
        dtype = model.backbone.model.embed_tokens.weight.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Load the CUDA backbone with dtype=torch.bfloat16 or torch.float16 for FA3")
        if model.config.attention_backend == "sdpa":
            raise ValueError("SGLang Titans requires FA3; select the torch backend for SDPA")
        ModelRegistry.models["TitansForCausalLM"] = TitansForCausalLM
        self.directory = Path(tempfile.mkdtemp(prefix="titans-sglang-"))
        self.processor = TitansMetadataProcessor.to_str()
        try:
            checkpoint = model.save_pretrained(self.directory / "checkpoint")
            config_path = checkpoint / "backbone" / "config.json"
            config = json.loads(config_path.read_text())
            config["architectures"] = ["TitansForCausalLM"]
            config["titans_settings"] = asdict(model.config)
            config["titans_adapter_path"] = str(checkpoint / "memory.safetensors")
            config_path.write_text(json.dumps(config) + "\n")
            previous = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
            os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "titans.serving.models"
            try:
                self.engine = sgl.Engine(
                    model_path=str(checkpoint / "backbone"), dtype=str(dtype).removeprefix("torch."),
                    tp_size=1,
                    attention_backend="fa3", disable_radix_cache=True, disable_cuda_graph=True,
                    disable_overlap_schedule=True, enable_custom_logit_processor=True,
                    enable_deterministic_inference=True,
                    max_running_requests=1, chunked_prefill_size=-1, context_length=8192,
                    max_total_tokens=8192, mem_fraction_static=0.35, log_level="warning",
                )
            finally:
                if previous is None:
                    os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
                else:
                    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = previous
        except Exception:
            shutil.rmtree(self.directory, ignore_errors=True)
            raise

    def _request(
        self, tokens: list[int], control: RequestControl, config: AnswerConfig | None = None,
    ) -> dict[str, object]:
        """Execute one bounded request using SGLang scheduling and sampling.

        Args:
            tokens: Prompt IDs; control operations use a single dummy token.
            control: Worker state operation.
            config: Generation settings, or None for a one-token control/ingest result.

        Returns:
            Validated outer response mapping from the engine.
        """
        parameters: dict[str, object] = {
            "temperature": config.temperature if config else 0.0,
            "max_new_tokens": config.max_new_tokens if config else 1,
            "custom_params": asdict(control),
            "sampling_seed": config.seed if config else 0,
        }
        result: object = self.engine.generate(input_ids=tokens, sampling_params=parameters,
                                             custom_logit_processor=self.processor)
        if not isinstance(result, dict) or not all(isinstance(key, str) for key in result):
            raise RuntimeError("Unexpected SGLang response")
        return {str(key): value for key, value in result.items()}

    def ingest(self, document: Document, state: StreamState | None) -> StreamState:
        """Append all document tokens to a worker-owned state without replaying history.

        Args:
            document: Next document in arrival order.
            state: Compatible remote state, or None to initialize one.

        Returns:
            The updated state handle; a failure invalidates it.
        """
        if state is None:
            state = StreamState(self.model._owner, self.model._revision, _remote_id=uuid.uuid4().hex)
        self.model._validate_state(state)
        if state._remote_id is None:
            raise ValueError("A local Torch state cannot be passed to SGLang")
        tokens = self.model._document_tokens(document)
        try:
            for start in range(0, len(tokens), 2048):
                block = tokens[start:start + 2048]
                complete = start + len(block) == len(tokens)
                self._request(block, RequestControl("ingest", state._remote_id,
                                                   document_complete=complete))
                state._tokens += len(block)
            state._documents += 1
        except Exception:
            state._valid = False
            raise
        return state

    def clone_state(self, state: StreamState) -> StreamState:
        """Clone fast weights, momentum, pending writes and KV inside the worker.

        Args:
            state: Source remote state owned by this model revision.

        Returns:
            An independently mutable worker state handle.
        """
        self.model._validate_state(state)
        if state._remote_id is None:
            raise ValueError("Expected a SGLang state")
        identifier = uuid.uuid4().hex
        self._request([0], RequestControl("clone", identifier, source_id=state._remote_id))
        return StreamState(state._owner, state._revision, _tokens=state._tokens,
                           _documents=state._documents, _remote_id=identifier)

    def answer(self, question: Question, state: StreamState, config: AnswerConfig) -> Answer:
        """Generate on a worker-side branch and release it even after a failed request.

        Args:
            question: Query text and caller identifier.
            state: Source document stream, preserved throughout generation.
            config: Sampling settings and temporary/read-only memory policy.

        Returns:
            SGLang's generated text and stop metadata.
        """
        self.model._validate_state(state)
        if state._remote_id is None:
            raise ValueError("Expected a SGLang state")
        identifier = uuid.uuid4().hex
        control = RequestControl("answer", identifier, source_id=state._remote_id,
                                 write=config.memory_policy == "temporary")
        try:
            result = self._request(self.model._question_tokens(question), control, config)
        finally:
            self._request([0], RequestControl("release", identifier))
        text = result.get("text")
        metadata = result.get("meta_info")
        if not isinstance(text, str) or not isinstance(metadata, dict):
            raise RuntimeError("SGLang response is missing text or generation metadata")
        count = metadata.get("completion_tokens")
        reason = metadata.get("finish_reason")
        if not isinstance(count, int):
            raise RuntimeError("SGLang response is missing its generated token count")
        limited = isinstance(reason, dict) and reason.get("type") == "length"
        text = text.strip()
        return Answer(question.question_id, text, text.casefold() == "[unknown]", count, limited)

    def close(self) -> None:
        """Stop the engine, release its GPU states and remove the temporary weight export."""
        try:
            self.engine.shutdown()
        finally:
            shutil.rmtree(self.directory, ignore_errors=True)
