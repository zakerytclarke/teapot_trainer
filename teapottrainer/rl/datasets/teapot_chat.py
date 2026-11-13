"""TeapotChat as an RL keepalive dataset — exact match, no tools.

Keeps assistant / identity style from drifting while GRPO pushes math and
open-web search. Episodes always expose ``tools=[]``.
"""

from __future__ import annotations

import random
from typing import Any

from ..rewards import score_exact_answer
from ..tools import conversation_tool_calls, final_assistant_text
from ._hf import load_hf_rows, pick_row
from .base import RlDataset

_DEFAULT_SYSTEM = 'You are Teapot. If a user asks who you are, reply "I am Teapot".'


def init_teapot_chat() -> dict[str, Any]:
    rows = load_hf_rows("teapotai/teapot-chat", split="train")
    return {"name": "teapot_chat", "rows": rows}


def sample_teapot_chat(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = pick_row(state.get("rows") or [], rng)
    system = str(row.get("system_prompt") or "").strip() or _DEFAULT_SYSTEM
    query = str(row.get("query") or "").strip()
    response = str(row.get("response") or "").strip()
    context = str(row.get("context") or "").strip()
    user = f"Context:\n{context}\n\nQuestion: {query}" if context else query
    return {
        "sample_id": f"teapot_chat:{hash(query) & 0xFFFFFFFF:x}",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "meta": {"answer": response, "query": query},
        "tools": [],
    }


def grade_teapot_chat(conversation: list[dict], meta: dict[str, Any] | None = None) -> dict[str, float]:
    meta = meta or {}
    text = final_assistant_text(conversation)
    calls = conversation_tool_calls(conversation)
    reward, metrics = score_exact_answer(text, str(meta.get("answer") or ""))
    if calls:
        reward = 0.0
        metrics["tool_violation"] = 1.0
    else:
        metrics["tool_violation"] = 0.0
    metrics["num_tool_calls"] = float(len(calls))
    metrics["score"] = float(reward)
    return metrics


TEAPOT_CHAT_RL = RlDataset(
    name="teapot_chat",
    init_fn=init_teapot_chat,
    sample_fn=sample_teapot_chat,
    grade_fn=grade_teapot_chat,
    tools=[],
    weight=1.0,
    target_episodes=2_000,
)
