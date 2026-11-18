"""Architecture and backbone-wiring contracts on offline Llama and Qwen models."""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from examples.linear_memory import LinearMemory
from titans import (
    BackboneConnection,
    ConnectionContext,
    ConnectionFactory,
    ConnectionPlan,
    ContinualModel,
    Document,
    GradientUpdate,
    MemoryModule,
    MemoryState,
    Question,
    RuntimeConfig,
    TitansArchitecture,
    TitansMACConnection,
    TitansModel,
)
from titans.runtime.connections import ResidualConnection


class ProjectedLayerConnection(BackboneConnection):
    """Feed a projected memory read into attention, with no parallel residual branch."""

    def __init__(self, hidden_size: int) -> None:
        """Register a learned read/write projection.

        Args:
            hidden_size: Backbone width.
        """
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def prepare(self, context: ConnectionContext, memory: MemoryModule,
                state: MemoryState, *, write: bool) -> ConnectionPlan:
        """Use memory as the input to the bounded attention operation.

        Args:
            context: Layer input.
            memory: Chosen memory network.
            state: Owned state to advance causally.
            write: Whether learning from these tokens is permitted.

        Returns:
            The read as the attention input, demonstrating overridable topology.
        """
        read = memory(self.projection(context.normalized), state, write=write)
        return ConnectionPlan(read, read)

    def combine(self, context: ConnectionContext, attention: Tensor,
                plan: ConnectionPlan) -> Tensor:
        """Retain the frozen decoder residual around sequential memory and attention.

        Args:
            context: Original residual input.
            attention: Attention applied to the memory read.
            plan: Prepared intermediate values.

        Returns:
            The residual stream before the FFN.
        """
        return context.hidden + attention


def projected_connection(layer_index: int, hidden_size: int) -> BackboneConnection:
    """Build a new connection per selected layer.

    Args:
        layer_index: Backbone placement.
        hidden_size: Residual width.

    Returns:
        Independent trainable sequential connection.
    """
    return ProjectedLayerConnection(hidden_size)


def test_defaults_and_legacy_connection(tiny_model: TitansModel) -> None:
    """New defaults use Titans/MAC; legacy fixtures explicitly retain residual wiring.

    Args:
        tiny_model: Frozen offline backbone fixture.
    """
    model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer,
                          RuntimeConfig(persistent_tokens=0, compile_kernels=False))
    fresh = TitansModel(tiny_model.backbone, tiny_model.tokenizer, tiny_model.config)
    assert isinstance(fresh.connections["0"], TitansMACConnection)
    assert isinstance(model.connections['0'], TitansMACConnection)
    assert type(model.memories['0']) is type(TitansArchitecture()(16))
    assert all(type(connection) is ResidualConnection for connection in tiny_model.connections.values())


def test_custom_architecture_wiring_gradients_and_checkpoint(
    tiny_model: TitansModel, tmp_path: Path,
) -> None:
    """Custom topology participates in phi learning, checkpointing and frozen-B execution.

    Args:
        tiny_model: Offline Llama/Qwen fixture.
        tmp_path: Isolated checkpoint location.
    """
    config = RuntimeConfig(memory_layers=(0, 1), persistent_tokens=0,
                           attention_window=8, compile_kernels=False)
    model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer, config,
                          memory_architecture=LinearMemory, connection_factory=projected_connection)
    assert model.connections['0'] is not model.connections['1']
    before = model.connections['0'].projection.weight.detach().clone()
    backbone = [p.detach().clone() for p in model.backbone.parameters()]
    state = model.ingest(Document('d', 'Alice lives in Paris'))
    model.learn(Question('q', 'Where Alice lives ?'), 'Rome', state,
                update=GradientUpdate('phi', learning_rate=0.1))
    assert not torch.equal(before, model.connections['0'].projection.weight)
    for old, new in zip(backbone, model.backbone.parameters(), strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
        assert not new.requires_grad
    path = model.save_pretrained(tmp_path / 'custom')
    with pytest.raises(ValueError, match='connection_factory'):
        ContinualModel.load_pretrained(path, memory_architecture=LinearMemory)
    restored = ContinualModel.load_pretrained(path, memory_architecture=LinearMemory,
                                               connection_factory=projected_connection)
    with torch.no_grad():
        expected = model._forward_tokens([7, 10, 11, 12, 13], model.start_stream())
        actual = restored._forward_tokens([7, 10, 11, 12, 13], restored.start_stream())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('connection', [None, projected_connection])
def test_connection_preserves_split_causality(
    tiny_model: TitansModel, connection: ConnectionFactory | None,
) -> None:
    """MAC and sequential connections preserve arbitrary ingest split boundaries.

    Args:
        tiny_model: Offline architecture fixture.
        connection: Selected factory; None selects MAC.
    """
    model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer,
                          RuntimeConfig(persistent_tokens=0, compile_kernels=False),
                          memory_architecture=LinearMemory, connection_factory=connection)
    with torch.no_grad():
        whole, split = model.start_stream(), model.start_stream()
        tokens = [7, 10, 11, 12, 13, 8, 10, 11, 12, 14]
        expected = model._forward_tokens(tokens, whole)
        actual = torch.cat((model._forward_tokens(tokens[:3], split),
                            model._forward_tokens(tokens[3:], split)), 1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('prefix', [0, 2])
def test_mac_retains_only_prefix_and_differentiates_writes(
    tiny_model: TitansModel, prefix: int,
) -> None:
    """MAC discards past segment KV while preserving differentiable neural memory.

    Args:
        tiny_model: Offline backbone fixture.
        prefix: Number of persistent context tokens.
    """
    model = ContinualModel(tiny_model.backbone.double(), tiny_model.tokenizer,
                          RuntimeConfig(persistent_tokens=prefix, mac_segment_length=4, compile_kernels=False),
                          memory_architecture=LinearMemory).double()
    state = model._new_state()
    model._forward_tokens([10, 11, 12, 13], state)
    assert state._attention[0].keys.shape[2] == prefix
    assert state.tokens_processed == 4
    matrix = state.memory[0].adaptable()['matrix']
    derivative, = torch.autograd.grad(matrix.square().sum(), model.memories['0'].rate)
    assert torch.isfinite(derivative) and derivative.abs() > 0
    source = model.clone_state(state)
    model.answer(Question('q', 'Where Alice lives ?'), source)
    for actual, expected in zip(source.memory[0].tensors(), state.memory[0].tensors(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mac_checkpoint_and_legacy_migration(tiny_model: TitansModel, tmp_path: Path) -> None:
    """Reload MAC automatically, and retain residual semantics for old checkpoints.

    Args:
        tiny_model: Legacy residual fixture.
        tmp_path: Isolated checkpoint directories.
    """
    import json

    model = ContinualModel(tiny_model.backbone, tiny_model.tokenizer,
                          RuntimeConfig(persistent_tokens=2, compile_kernels=False),
                          memory_architecture=LinearMemory)
    path = model.save_pretrained(tmp_path / 'mac')
    restored = ContinualModel.load_pretrained(path, memory_architecture=LinearMemory)
    assert isinstance(restored.connections['0'], TitansMACConnection)
    with torch.no_grad():
        expected = model._forward_tokens([10, 11, 12, 13], model.start_stream())
        actual = restored._forward_tokens([10, 11, 12, 13], restored.start_stream())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    legacy = tiny_model.save_pretrained(tmp_path / 'legacy')
    metadata_path = legacy / 'titans.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['format_version'] = 2
    metadata.pop('connection_types')
    metadata_path.write_text(json.dumps(metadata))
    old = TitansModel.load_pretrained(legacy)
    assert all(type(c) is ResidualConnection for c in old.connections.values())


@pytest.mark.parametrize("connection_name", ["ResidualConnection", "TitansMACConnection", "TitansMAGConnection"])
def test_flat_package_checkpoint_names(
    tiny_model: TitansModel, tmp_path: Path, connection_name: str,
) -> None:
    """Restore built-in format-3 checkpoints saved before the package reorganization.

    Args:
        tiny_model: Offline backbone and tokenizer fixture.
        tmp_path: Directory for a complete checkpoint.
        connection_name: Built-in connection whose previous module path is encoded.
    """
    import json

    from titans import residual_connection, titans_mac_connection, titans_mag_connection

    factories = {
        "ResidualConnection": residual_connection,
        "TitansMACConnection": titans_mac_connection,
        "TitansMAGConnection": titans_mag_connection,
    }
    model = TitansModel(tiny_model.backbone, tiny_model.tokenizer, tiny_model.config,
                        connection_factory=factories[connection_name])
    path = model.save_pretrained(tmp_path / connection_name)
    metadata_path = path / "titans.json"
    metadata = json.loads(metadata_path.read_text())
    old_module = "titans.connections" if connection_name == "ResidualConnection" else "titans.titans"
    metadata["connection_types"] = {
        key: f"{old_module}.{connection_name}" for key in metadata["connection_types"]
    }
    metadata_path.write_text(json.dumps(metadata))
    restored = TitansModel.load_pretrained(path)
    assert all(type(c).__name__ == connection_name for c in restored.connections.values())
    with torch.no_grad():
        expected = model._forward_tokens([10, 11, 12, 13], model.start_stream())
        actual = restored._forward_tokens([10, 11, 12, 13], restored.start_stream())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
