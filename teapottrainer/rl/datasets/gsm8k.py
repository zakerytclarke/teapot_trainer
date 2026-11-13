"""GSM8K as an RL dataset (optional calculator tool)."""

from __future__ import annotations

import random
from typing import Any

from ..rewards import score_gsm8k_answer
from ..tools import calculator, conversation_tool_calls, final_assistant_text
from ._hf import load_hf_rows, pick_row
from .base import RlDataset

_SYSTEM = (
    "You are a careful math tutor. Solve the grade-school word problem. "
    "You may call the calculator tool for arithmetic. "
    "End with the final numeric answer."
)


def init_gsm8k() -> dict[str, Any]:
    rows = load_hf_rows("openai/gsm8k", subset="main", split="train")
    return {"name": "gsm8k", "rows": rows}


def sample_gsm8k(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = pick_row(state.get("rows") or [], rng)
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question},
    ]
    return {
        "sample_id": f"gsm8k:{hash(question) & 0xFFFFFFFF:x}",
        "messages": messages,
        "meta": {"answer": answer, "question": question},
        "tools": [calculator],
    }


def grade_gsm8k(conversation: list[dict], meta: dict[str, Any] | None = None) -> dict[str, float]:
    meta = meta or {}
    text = final_assistant_text(conversation)
    calls = conversation_tool_calls(conversation)
    used = any(call.get("name") == "calculator" for call in calls)
    # GSM8K does not require the calculator: many steps are reasoning, not arithmetic.
    reward, metrics = score_gsm8k_answer(
        text,
        str(meta.get("answer") or ""),
        used_calculator=used,
        require_calculator=False,
    )
    metrics["num_tool_calls"] = float(len(calls))
    metrics["score"] = float(reward)
    return metrics


GSM8K_RL = RlDataset(
    name="gsm8k",
    init_fn=init_gsm8k,
    sample_fn=sample_gsm8k,
    grade_fn=grade_gsm8k,
    tools=[calculator],
    weight=1.0,
    target_episodes=2_000,
)
