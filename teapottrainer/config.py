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
    # Raw text trained after pretrain and before supervised training. Empty skips it.
    # The loop count can change per batch, so this phase stays eager.
    bridge_datasets: List[DatasetSpec] = Field(default_factory=list)
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
    # None uses 0.2x lr after a pretrain phase, and lr when there is no pretrain.
    sft_lr: Optional[float] = None
    weight_decay: float = 0.1
    compile: bool = True
    logging_dir: Optional[str] = None
    
    eval_every_percent: float = 5.0
    # When positive, checkpoint every this many tokens instead of by percent.
    # Percent mode names folders by the integer percent, so a step under 1%
    # makes two checkpoints share a folder and the second supervised copy is skipped.
    eval_every_tokens: int = 0
    # Stop pretrain when the FineWeb holdout improves by less than this many
    # nats across plateau_window_tokens. Zero window disables the check.
    plateau_window_tokens: int = 0
    plateau_min_loss_drop: float = 0.02
    # At each pretrain or bridge eval checkpoint, train a copy on the full SFT
    # mix and save it beside the base weights. The base model keeps training.
    sft_at_checkpoints: bool = False
    holdout_ratio: float = 0.05
    seed: int = 42
    # Called as hook(model, phase_name) before each training batch. A hook may
    # assign any field on model.config. The trainer restores the config after
    # the batch. Pass one function or a list of them. None runs nothing.
    # The same hook works for Llama, Qwen, GPT-2, or a custom model.
    before_batch: Optional[Any] = None

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @field_serializer("before_batch")
    def _dump_before_batch(self, value):
        # Keep the callable on the object. The run hash stores a stable name.
        if value is None:
            return None
        hooks = [value] if callable(value) else list(value)
        labels = []
        for hook in hooks:
            labels.append(getattr(hook, "__qualname__", type(hook).__name__))
        return labels