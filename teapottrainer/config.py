"""Serializable training and model specs.

Sources and evals are ordinary Python objects on ``Stage`` so a run can mix
memmap corpora, Hugging Face datasets, and in-memory chat rows.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelSpec(BaseModel):
    """Architecture knobs for a from-scratch causal LM.

    ``architecture`` is ``llama``, ``gpt2``, ``mistral``, or ``custom``.
    For a custom model set ``custom_builder(spec, vocab_size) -> PreTrainedModel``
    or pass a Transformers config class and model class.
    """

    architecture: str = "llama"
    hidden_size: int = 512
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 1408
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = True
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    custom_builder: Optional[Any] = None
    custom_config_cls: Optional[Any] = None
    custom_model_cls: Optional[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def public_dict(self) -> dict:
        return self.model_dump(
            exclude={"custom_builder", "custom_config_cls", "custom_model_cls"}
        )


PRESETS: dict[str, ModelSpec] = {
    # Fits a CPU/GPU smoke test. Vocab still comes from the tokenizer.
    "tiny": ModelSpec(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=384,
        max_position_embeddings=256,
    ),
    # Matches teapot_foundation's 100.3M tied-embedding Llama.
    "100m": ModelSpec(
        hidden_size=512,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1408,
        max_position_embeddings=4096,
        rope_theta=500_000.0,
    ),
    # Deep-thin agent-sized model, about 350M with a 32k vocab.
    "350m": ModelSpec(
        hidden_size=960,
        num_hidden_layers=32,
        num_attention_heads=15,
        num_key_value_heads=5,
        intermediate_size=2560,
        max_position_embeddings=4096,
        rope_theta=500_000.0,
    ),
}


def preset(name: str) -> ModelSpec:
    key = name.lower()
    if key not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown model preset {name!r}; choose from {known}")
    return PRESETS[key].model_copy(deep=True)


class Stage(BaseModel):
    """One training phase (pretrain, SFT, or anything in between)."""

    name: str
    sources: list[Any]
    target_tokens: int
    evals: list[Any] = Field(default_factory=list)
    lr: Optional[float] = None
    seq_len: Optional[int] = None
    warmup_steps: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainConfig(BaseModel):
    """Everything the trainer needs, except the objects hanging off each stage."""

    tokenizer: str
    stages: list[Stage]
    model: Optional[ModelSpec] = None
    init_from: Optional[str] = None
    micro_batch: int = 8
    grad_accum: int = 1
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    seq_len: Optional[int] = None
    seed: int = 0
    output_dir: str = "runs/teapot"
    gradient_checkpointing: bool = True
    eval_every_tokens: Optional[int] = None
    eval_at_start: bool = True
    log_every: int = 1
    local_files_only: bool = False
    trust_remote_code: bool = False
    patience_evals: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _require_model(self):
        if self.model is None and not self.init_from:
            raise ValueError("TrainConfig needs model=... or init_from=...")
        if not self.stages:
            raise ValueError("TrainConfig needs at least one stage")
        for stage in self.stages:
            if stage.target_tokens < 1:
                raise ValueError(f"stage {stage.name!r} target_tokens must be positive")
            if not stage.sources:
                raise ValueError(f"stage {stage.name!r} needs at least one source")
        return self

    def public_dict(self) -> dict:
        stages = []
        for stage in self.stages:
            stages.append(
                {
                    "name": stage.name,
                    "target_tokens": stage.target_tokens,
                    "lr": stage.lr,
                    "seq_len": stage.seq_len,
                    "sources": [
                        getattr(src, "name", type(src).__name__) for src in stage.sources
                    ],
                    "weights": [float(getattr(src, "weight", 1.0)) for src in stage.sources],
                    "evals": [getattr(ev, "name", type(ev).__name__) for ev in stage.evals],
                }
            )
        return {
            "tokenizer": self.tokenizer,
            "init_from": self.init_from,
            "model": None if self.model is None else self.model.public_dict(),
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "min_lr_frac": self.min_lr_frac,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "max_grad_norm": self.max_grad_norm,
            "seq_len": self.seq_len,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_every_tokens": self.eval_every_tokens,
            "eval_at_start": self.eval_at_start,
            "patience_evals": self.patience_evals,
            "stages": stages,
        }
