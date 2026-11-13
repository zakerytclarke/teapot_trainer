"""Teapot SynthQA as an RL dataset — context-only, no tools.

Uses the same system rules as supervised SynthQA:

* answer from the provided context only
* if the answer is missing / type is unanswerable, reply exactly
  ``The context does not contain the answer.``

Episodes always expose ``tools=[]`` so search/calculator cannot be bound in.
"""

from __future__ import annotations

import random
from typing import Any

from ...formatting import ABSTAIN_TEXT, RAG_SYSTEM
from ..rewards import score_synthqa_answer
from ..tools import conversation_tool_calls, final_assistant_text
from ._hf import load_hf_rows, pick_row
from .base import RlDataset


def init_synthqa() -> dict[str, Any]:
    rows = load_hf_rows("teapotai/synthqa", split="train")
    return {"name": "synthqa", "rows": rows}


def sample_synthqa(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = pick_row(state.get("rows") or [], rng)
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    row_type = str(row.get("type") or "qa")
    unanswerable = row_type == "unanswerable" or not answer
    gold = ABSTAIN_TEXT if unanswerable else answer
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return {
        "sample_id": f"synthqa:{row_type}:{hash(question) & 0xFFFFFFFF:x}",
        "messages": [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": user},
        ],
        "meta": {
            "answer": gold,
            "answers": [gold],
            "unanswerable": unanswerable,
            "type": row_type,
            "question": question,
        },
        # Hard-empty: never inherit search/calc tools from a mistaken bind_tools.
        "tools": [],
    }


def grade_synthqa(conversation: list[dict], meta: dict[str, Any] | None = None) -> dict[str, float]:
    meta = meta or {}
    text = final_assistant_text(conversation)
    calls = conversation_tool_calls(conversation)
    reward, metrics = score_synthqa_answer(
        text,
        gold_answer=str(meta.get("answer") or ABSTAIN_TEXT),
        unanswerable=bool(meta.get("unanswerable")),
    )
    # Any tool call on this task is a protocol violation.
    if calls:
        reward = 0.0
        metrics["tool_violation"] = 1.0
    else:
        metrics["tool_violation"] = 0.0
    metrics["num_tool_calls"] = float(len(calls))
    metrics["score"] = float(reward)
    return metrics


SYNTHQA_RL = RlDataset(
    name="synthqa",
    init_fn=init_synthqa,
    sample_fn=sample_synthqa,
    grade_fn=grade_synthqa,
    tools=[],
    weight=1.0,
    target_episodes=2_000,
)
