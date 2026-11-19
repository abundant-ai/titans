"""Joint scratch pretraining and Qwen continual learning with pluggable memory."""

from titans.core.protocols import (
    BackboneConnection,
    ConnectionContext,
    ConnectionExecution,
    ConnectionFactory,
    ConnectionPlan,
    DataGenerator,
    LossContext,
    LossValue,
    MemoryArchitecture,
    MemoryFactory,
    MemoryModule,
    MemoryState,
    Objective,
    UpdateContext,
    UpdateResult,
    UpdateRule,
)
from titans.core.types import (
    Answer,
    AnswerConfig,
    Document,
    LearningResult,
    Question,
    RuntimeConfig,
    TrainingConfig,
    TrainingEpisode,
    TrainingQuestion,
    TrainingResult,
)
from titans.methods.titans import (
    TitansArchitecture,
    TitansConfig,
    TitansMACConnection,
    TitansMAGConnection,
    TitansModel,
    titans,
    titans_mac_connection,
    titans_mag_connection,
)
from titans.runtime.connections import ResidualConnection, residual_connection
from titans.runtime.model import ContinualModel
from titans.runtime.state import StreamState
from titans.training.learning import GradientUpdate, cross_entropy
from titans.training.pretraining import pretraining_batches

__all__ = [
    "pretraining_batches",
    "TitansMACConnection", "titans_mac_connection",
    "BackboneConnection", "ConnectionContext", "ConnectionExecution", "ConnectionFactory", "ConnectionPlan",
    "MemoryArchitecture", "TitansArchitecture", "TitansMAGConnection", "ResidualConnection",
    "residual_connection", "titans_mag_connection",
    "GradientUpdate", "cross_entropy", "DataGenerator", "LossContext", "LossValue",
    "MemoryFactory", "MemoryModule", "MemoryState", "Objective", "UpdateContext",
    "UpdateResult", "UpdateRule",
    "Answer", "AnswerConfig", "Document", "Question", "StreamState", "TitansConfig",
    "ContinualModel", "RuntimeConfig", "LearningResult", "titans", "TitansModel", "TrainingConfig", "TrainingEpisode", "TrainingQuestion", "TrainingResult",
]
