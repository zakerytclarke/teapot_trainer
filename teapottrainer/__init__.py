from .config import TrainingConfig, ModelConfig, DatasetSpec
from .trainer import TeapotTrainer
from .datasets import (
    DatasetSourceConfig,
    DatasetRegistry,
    FINEWEB_EDU,
    SMOLTALK,
    GSM8K
)

__version__ = "0.1.0"
__all__ = [
    "TeapotTrainer",
    "TrainingConfig",
    "ModelConfig",
    "DatasetSpec",
    "DatasetSourceConfig",
    "DatasetRegistry",
    "FINEWEB_EDU",
    "SMOLTALK",
    "GSM8K"
]