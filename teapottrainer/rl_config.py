"""Config surface for reinforcement learning runs.

Parallel to ``TrainingConfig``. The unit of work is an episode sampled from an
``RlDataset`` (init / sample / grade), not a supervised (prompt, answer) pair.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from .rl_datasets import RlDataset, SYNTH_MATH
from .tool_formats import get_tool_format


class RlConfig(BaseModel):
    """Config-driven RL training, peer to ``TrainingConfig`` for CE/SFT."""

    # Hugging Face id or local checkpoint directory. Required: RL assumes SFT done.
    init_from: str
    # Defaults to ``init_from`` when omitted.
    tokenizer: Optional[str] = None
    # How tools are injected into prompts and how calls/results are parsed.
    # ``tokenizer`` defers to Hugging Face ``apply_chat_template(..., tools=)``
    # and ``parse_response`` when the tokenizer supports them.
    tool_format: str = "tokenizer"
    # Optional override of chat role markers. None keeps the tokenizer template.
    chat_format: Optional[str] = None

    # Mixture of RL datasets. Each has init_fn / sample_fn / grade_fn / tools.
    datasets: List[RlDataset] = Field(default_factory=lambda: [SYNTH_MATH])

    algorithm: str = "grpo"
    group_size: int = 4
    batch_prompts: int = 1
    max_turns: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95

    lr: float = 5e-7
    weight_decay: float = 0.0
    kl_coef: float = 0.05
    max_grad_norm: float = 1.0

    seed: int = 42
    logging_dir: Optional[str] = None
    max_train_episodes: int = 200
    eval_every_episodes: int = 50
    eval_episodes: int = 32
    resume_every_episodes: int = 50

    before_batch: Optional[Any] = None

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @field_validator("tool_format")
    @classmethod
    def _known_tool_format(cls, value: str) -> str:
        return get_tool_format(value).name

    @field_validator("algorithm")
    @classmethod
    def _known_algorithm(cls, value: str) -> str:
        key = str(value or "").strip().lower()
        if key not in {"grpo"}:
            raise ValueError(f"Unsupported algorithm {value!r}. Supported: grpo")
        return key

    @field_validator("group_size")
    @classmethod
    def _group_size_ok(cls, value: int) -> int:
        if int(value) < 2:
            raise ValueError("group_size must be >= 2 for GRPO")
        return int(value)

    @field_validator("datasets")
    @classmethod
    def _datasets_nonempty(cls, value: List[RlDataset]) -> List[RlDataset]:
        if not value:
            raise ValueError("RlConfig.datasets must contain at least one RlDataset")
        return value

    @field_serializer("before_batch")
    def _dump_before_batch(self, value):
        if value is None:
            return None
        hooks = [value] if callable(value) else list(value)
        return [getattr(hook, "__qualname__", type(hook).__name__) for hook in hooks]

    def tokenizer_name(self) -> str:
        return self.tokenizer or self.init_from
