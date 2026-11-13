"""RlDataset type and episode helpers."""

from __future__ import annotations

import random
from typing import Any, Callable, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer

InitFn = Callable[[], Any]
SampleFn = Callable[[Any, random.Random], dict[str, Any]]
GradeFn = Callable[..., Union[float, dict[str, Any]]]
ToolFn = Callable[..., Any]


class RlDataset(BaseModel):
    """One RL data source in the mixture."""

    name: str
    init_fn: InitFn = Field(exclude=True)
    sample_fn: SampleFn = Field(exclude=True)
    grade_fn: GradeFn = Field(exclude=True)
    # Python functions exposed as tools for episodes from this dataset.
    tools: list[ToolFn] = Field(default_factory=list, exclude=True)
    weight: float = 1.0
    target_episodes: int = 1_000

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("init_fn", "sample_fn", "grade_fn")
    def _dump_fn(self, value):
        return getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))

    @field_serializer("tools")
    def _dump_tools(self, value):
        return [getattr(fn, "__name__", type(fn).__name__) for fn in (value or [])]


def normalize_grade(raw: Union[float, int, dict[str, Any]]) -> tuple[float, dict[str, float]]:
    """Accept a bare float or a dict with ``score`` / ``reward`` plus metrics."""
    if isinstance(raw, (int, float)):
        return float(raw), {"score": float(raw)}
    if not isinstance(raw, dict):
        raise TypeError(f"grade_fn must return float or dict, got {type(raw)!r}")
    score = raw.get("score", raw.get("reward", raw.get("grade")))
    if score is None:
        raise ValueError("grade dict must include score/reward/grade")
    metrics = {
        str(k): float(v)
        for k, v in raw.items()
        if k not in {"score", "reward", "grade"} and isinstance(v, (int, float))
    }
    metrics.setdefault("score", float(score))
    return float(score), metrics


def prepare_dataset(dataset: RlDataset) -> Any:
    """Run ``init_fn`` once and return the opaque dataset state."""
    return dataset.init_fn()


def sample_episode(dataset: RlDataset, state: Any, rng: random.Random) -> dict[str, Any]:
    """Sample one episode seed.

    Required keys:
      * ``messages``: initial HF-style chat messages for the policy

    Optional keys:
      * ``meta``: passed through to ``grade_fn``
      * ``sample_id``: logging id
      * ``tools``: override the dataset tool list for this episode
    """
    sample = dataset.sample_fn(state, rng)
    if not isinstance(sample, dict) or "messages" not in sample:
        raise ValueError(f"{dataset.name}.sample_fn must return a dict with a messages list")
    messages = sample["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{dataset.name}.sample_fn messages must be a non-empty list")
    out = dict(sample)
    out.setdefault("meta", {})
    out.setdefault("sample_id", dataset.name)
    out.setdefault("tools", list(dataset.tools or []))
    return out


def grade_episode(
    dataset: RlDataset,
    conversation: list[dict],
    meta: dict | None = None,
) -> tuple[float, dict[str, float]]:
    """Grade a finished conversation JSON."""
    raw = dataset.grade_fn(conversation, meta or {})
    return normalize_grade(raw)
