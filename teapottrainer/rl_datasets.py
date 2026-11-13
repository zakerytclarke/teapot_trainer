"""Extensible RL datasets.

An RL dataset is three callables plus optional Python tool functions:

* ``init_fn()`` — download or build whatever the dataset needs once
* ``sample_fn(state, rng)`` — pick one episode seed (messages + meta)
* ``grade_fn(conversation, meta)`` — score a finished conversation JSON

``conversation`` is the Hugging Face-style message list after the rollout
(including assistant ``tool_calls`` and ``role=tool`` results). ``meta`` is
whatever ``sample_fn`` attached (gold answers, task ids, …). The trainer runs
GRPO on the numeric grade.

Ship presets like ``SYNTH_MATH`` for convenience; users can also build their
own ``RlDataset`` instances the same way SFT presets wrap ``DatasetSourceConfig``.
"""

from __future__ import annotations

import operator
import random
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .rl.rewards import score_numeric_answer
from .rl.tools import calculator, conversation_tool_calls, final_assistant_text


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


# ---------------------------------------------------------------------------
# Preset: synthetic math (generated, no download)
# ---------------------------------------------------------------------------

_OPS = [
    ("+", operator.add, "plus"),
    ("-", operator.sub, "minus"),
    ("*", operator.mul, "times"),
]


def init_synth_math() -> dict[str, Any]:
    """No download — just marks the generator ready."""
    return {"name": "synth_math", "ops": list(_OPS)}


def sample_synth_math(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Draw a random arithmetic / word problem with a known numeric gold."""
    ops = state.get("ops") or _OPS
    mode = rng.choice(["easy", "word"])
    a = int(rng.randint(1, 20))
    b = int(rng.randint(1, 20))
    op_sym, op_fn, op_word = ops[int(rng.randrange(len(ops)))]
    gold = float(op_fn(a, b))
    expression = f"{a} {op_sym} {b}"

    if mode == "word":
        question = (
            f"What is {a} {op_word} {b}? "
            "You may use the calculator tool. "
            "Reply with a short assistant answer that includes the final number."
        )
    else:
        question = (
            f"Compute {expression}. "
            "You may use the calculator tool if helpful. "
            "Give a brief answer that includes the result."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. "
                "When arithmetic is needed, you may call the calculator tool. "
                "Then answer the user in natural language and include the final number."
            ),
        },
        {"role": "user", "content": question},
    ]
    return {
        "sample_id": f"math:{mode}:{expression}={gold}",
        "messages": messages,
        "meta": {
            "value": gold,
            "expression": expression,
            "mode": mode,
        },
        "tools": [calculator],
    }


def grade_synth_math(conversation: list[dict], meta: dict[str, Any] | None = None) -> dict[str, float]:
    """Score a finished math conversation against ``meta['value']``."""
    meta = meta or {}
    text = final_assistant_text(conversation)
    calls = conversation_tool_calls(conversation)
    used = any(call.get("name") == "calculator" for call in calls)
    reward, metrics = score_numeric_answer(
        text,
        float(meta["value"]),
        used_calculator=used,
    )
    metrics["num_tool_calls"] = float(len(calls))
    metrics["score"] = float(reward)
    return metrics


SYNTH_MATH = RlDataset(
    name="synth_math",
    init_fn=init_synth_math,
    sample_fn=sample_synth_math,
    grade_fn=grade_synth_math,
    tools=[calculator],
    weight=1.0,
    target_episodes=1_000,
)
