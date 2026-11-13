"""Generated arithmetic RL dataset (no download).

Problems are solvable without tools, so the grade requires a calculator call.
Otherwise GRPO has no reason to prefer tool use over answering from weights.
"""

from __future__ import annotations

import operator
import random
from typing import Any

from ..rewards import extract_number, numbers_equal, score_numeric_answer
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
    # Larger operands so the tool is more useful, but still exact ints.
    a = int(rng.randint(12, 97))
    b = int(rng.randint(12, 97))
    op_sym, op_fn, op_word = ops[int(rng.randrange(len(ops)))]
    gold = float(op_fn(a, b))
    expression = f"{a} {op_sym} {b}"

    if mode == "word":
        question = (
            f"What is {a} {op_word} {b}? "
            "You must use the calculator tool before answering. "
            "After the tool result, reply with a short assistant answer that includes the final number."
        )
    else:
        question = (
            f"Compute {expression}. "
            "You must call the calculator tool with that expression before answering. "
            "Then give a brief natural-language answer that includes the result."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. "
                "For every arithmetic question you must call the calculator tool first. "
                "Do not answer from memory. After the tool returns, reply in natural language "
                "and include the final number."
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


def _calculator_returned_gold(conversation: list[dict], gold: float) -> bool:
    """True if any calculator tool message contains the gold value."""
    for message in conversation:
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        if name and name != "calculator":
            continue
        content = str(message.get("content") or "")
        if content.startswith("ERROR"):
            continue
        got = extract_number(content)
        if numbers_equal(got, gold):
            return True
    return False


def grade_synth_math(conversation: list[dict], meta: dict[str, Any] | None = None) -> dict[str, float]:
    """Score a finished math conversation; calculator use is required."""
    meta = meta or {}
    gold = float(meta["value"])
    text = final_assistant_text(conversation)
    calls = conversation_tool_calls(conversation)
    used = any(call.get("name") == "calculator" for call in calls)
    calc_ok = _calculator_returned_gold(conversation, gold)
    reward, metrics = score_numeric_answer(
        text,
        gold,
        used_calculator=used,
        require_calculator=True,
        calculator_result_ok=calc_ok,
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
