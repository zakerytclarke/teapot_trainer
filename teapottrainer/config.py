from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator
from typing import Optional, List, Any, Union
from .chat_formats import get_chat_format
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
    # Applied to supervised examples only. Pretrain text stays raw.
    # Names: plain, chatml, smollm, llama2, llama3, gemma, mistral, phi, zephyr, alpaca, vicuna.
    chat_format: str = "plain"
    # Hugging Face id or local directory. None builds a new model from the config.
    init_from: Optional[str] = None

    @field_validator("chat_format")
    @classmethod
    def _known_chat_format(cls, value: str) -> str:
        return get_chat_format(value).name

    batch_size: int = 4
    grad_accum: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.1
    compile: bool = True
    logging_dir: Optional[str] = None
    
    eval_every_percent: float = 5.0
    # At each pretrain eval checkpoint, train a copy on the full SFT mix and
    # save it beside the base weights. The pretrain model keeps training.
    sft_at_checkpoints: bool = False
    holdout_ratio: float = 0.05
    seed: int = 42
    # Called with the phase name before each training step. None leaves the
    # model's own reasoning_effort alone. Return an int of at least 1.
    sample_reasoning_effort: Optional[Any] = None

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @field_serializer("sample_reasoning_effort")
    def _dump_reasoning_hook(self, _value):
        # The callable stays on the object. The run hash only needs a stable dump.
        return None