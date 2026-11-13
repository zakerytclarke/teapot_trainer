"""Generated arithmetic RL dataset (no download)."""

from __future__ import annotations

import operator
import random
from typing import Any

from ..rewards import score_numeric_answer
from ..tools import calculator, conversation_tool_calls, final_assistant_text
from .base import RlDataset

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
