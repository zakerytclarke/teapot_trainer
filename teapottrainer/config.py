from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any, Union
from .datasets import DatasetSourceConfig, FINEWEB_EDU

class ModelConfig(BaseModel):
    model_type: str = "llama"
    hidden_size: int = 768
    layers: int = Field(12, alias="num_hidden_layers")
    heads: int = Field(12, alias="num_attention_heads")
    kv_heads: int = Field(4, alias="num_key_value_heads")
    intermediate: int = Field(3072, alias="intermediate_size")
    seq_len: int = Field(2048, alias="max_position_embeddings")
    
    custom_config_cls: Optional[Any] = None
    custom_model_cls: Optional[Any] = None

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class DatasetSpec(BaseModel):
    source: Union[DatasetSourceConfig, str] = FINEWEB_EDU
    target_tokens: int = 1_000_000_000
    weight: float = 1.0

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class TrainingConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig, alias="model_config")
    
    pretrain_datasets: List[DatasetSpec] = Field(
        default_factory=lambda: [DatasetSpec(source=FINEWEB_EDU, target_tokens=2_000_000_000)]
    )
    sft_datasets: List[DatasetSpec] = Field(default_factory=list)
    eval_datasets: List[DatasetSpec] = Field(default_factory=list)
    
    tokenizer: str = "Qwen/Qwen2.5-0.5B"
    batch_size: int = 4
    grad_accum: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.1
    compile: bool = True
    logging_dir: Optional[str] = None
    
    eval_every_percent: float = 5.0
    holdout_ratio: float = 0.05
    seed: int = 42

    model_config = ConfigDict(populate_by_name=True)