from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator, model_validator
from typing import Optional, List, Any, Union
from .chat_formats import get_chat_format
from .datasets import DatasetSourceConfig, FINEWEB_EDU


def _hook_labels(value: Any) -> list[str]:
    """Stable names for callables. The run hash and the config dump both use them."""
    if value is None:
        return []
    hooks = [value] if callable(value) else list(value)
    return [getattr(hook, "__qualname__", type(hook).__name__) for hook in hooks]

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


class CheckpointRule(BaseModel):
    """When a phase saves a mid-run checkpoint and scores its eval sets.

    ``every_tokens`` wins when it is positive. Percent mode names folders by
    the integer percent. A step that lands on the phase budget is the phase
    end, not an interior checkpoint.
    """

    every_tokens: int = 0
    every_percent: float = 100.0


class EarlyStop(BaseModel):
    """Stop the phase when a holdout loss flattens.

    ``metric`` is a substring of the eval loss key (``fineweb`` matches
    ``fineweb-edu/train/loss``). The check looks back ``window_tokens``.
    The phase stops after ``patience`` checks in a row each improve by less
    than ``min_delta`` nats. Patience 1 is the old pretrain plateau.
    """

    metric: str = "fineweb"
    window_tokens: int = 0
    min_delta: float = 0.02
    patience: int = 1


class TrainingPhase(BaseModel):
    """One stretch of training in a recipe.

    Phases run in list order. ``datasets`` is the train mix. ``token_budget``
    is the total for the phase; when it differs from the sum of the mix, the
    mix is scaled to that total and the weights stay. ``eval_datasets`` is
    what the checkpoints score. ``None`` uses the config-level eval list.

    ``on_checkpoint`` runs after each interior checkpoint, and ``on_complete``
    runs when the phase's budget is finished. A hook is
    ``hook(trainer, model, tokenizer, checkpoint_name, step)``. That is how a
    supervised copy is trained beside a pretrain checkpoint without a special
    case in the loop. Hooks should skip work they have already saved.

    ``fork`` trains a fresh copy from a saved checkpoint instead of the live
    weights. ``fork_from="latest"`` uses the bridge checkpoint when it exists,
    otherwise pretrain. ``load_fork`` makes that copy the live model afterward.
    """

    name: str
    datasets: List[DatasetSpec] = Field(default_factory=list)
    eval_datasets: Optional[List[DatasetSpec]] = None
    token_budget: Optional[int] = None
    lr: Optional[float] = None
    # Multiplies TrainingConfig.lr when ``lr`` is omitted. The bridge uses 0.1.
    lr_scale: Optional[float] = None
    compile: Optional[bool] = None
    checkpoint: CheckpointRule = Field(default_factory=CheckpointRule)
    early_stop: Optional[EarlyStop] = None
    # None uses TrainingConfig.before_batch.
    before_batch: Optional[Any] = None
    on_checkpoint: List[Any] = Field(default_factory=list)
    on_complete: List[Any] = Field(default_factory=list)
    task_eval: bool = False
    eval_during: bool = True
    fork: bool = False
    fork_required: bool = False
    fork_from: Optional[str] = None
    load_fork: bool = False
    save: bool = True
    # Fresh start of this phase begins at step 0. Resume still continues its step.
    reset_step: bool = False
    restore_boundary_rng: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("before_batch", "on_checkpoint", "on_complete")
    def _dump_hooks(self, value: Any):
        if value is None:
            return None
        return _hook_labels(value)


class TrainingConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig, alias="model_config")

    # The recipe. Empty means "build phases from the dataset lists below".
    phases: List[TrainingPhase] = Field(default_factory=list)
    
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
        return _hook_labels(value)

    @model_validator(mode="after")
    def _mirror_phases(self):
        """Keep the older fields aligned with a phase recipe.

        The trainer runs ``phases`` when it is set. These fields stay filled
        so a resume hash, a checkpoint copy, and older readers still see the
        pretrain, bridge, and supervised mixes.
        """
        if not self.phases:
            return self
        by_name = {}
        for phase in self.phases:
            by_name.setdefault(phase.name, phase)
        pretrain = by_name.get("pretrain")
        bridge = by_name.get("bridge")
        supervised = by_name.get("sft")
        if pretrain is not None and not pretrain.fork:
            self.pretrain_datasets = list(pretrain.datasets)
            if pretrain.lr is not None:
                self.lr = float(pretrain.lr)
            if pretrain.compile is not None:
                self.compile = bool(pretrain.compile)
            self.eval_every_tokens = int(pretrain.checkpoint.every_tokens or 0)
            self.eval_every_percent = float(pretrain.checkpoint.every_percent)
            if pretrain.early_stop is not None:
                self.plateau_window_tokens = int(pretrain.early_stop.window_tokens)
                self.plateau_min_loss_drop = float(pretrain.early_stop.min_delta)
        else:
            self.pretrain_datasets = []
        if bridge is not None and not bridge.fork:
            self.bridge_datasets = list(bridge.datasets)
        else:
            self.bridge_datasets = []
        if supervised is not None:
            self.sft_datasets = list(supervised.datasets)
            if supervised.lr is not None:
                self.sft_lr = float(supervised.lr)
        else:
            self.sft_datasets = []
        for phase in self.phases:
            if phase.eval_datasets:
                self.eval_datasets = list(phase.eval_datasets)
                break
        for phase in self.phases:
            if phase.before_batch is not None:
                self.before_batch = phase.before_batch
                break
        if any(phase.on_checkpoint or phase.on_complete for phase in self.phases):
            self.sft_at_checkpoints = True
        return self