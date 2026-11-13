"""Open-domain / multi-hop QA datasets meant to be paired with search tools.

Presets ship with an empty tool list. Bind Brave / Exa / LLM-search tools at
config time::

    from teapottrainer import NQ_OPEN_RL, SearchSettings, bind_tools, build_search_tools

    tools = build_search_tools(SearchSettings.from_env())
    dataset = bind_tools(NQ_OPEN_RL, tools)
"""

from __future__ import annotations

import random
from typing import Any

from ..rewards import score_text_answer, used_search_tools
from ..tools import conversation_tool_calls, final_assistant_text
from ._hf import load_hf_rows, pick_row
from .base import RlDataset

_SEARCH_SYSTEM = (
    "You are a helpful research assistant with web search tools. "
    "For factual questions, call a search tool before answering. "
    "After tool results arrive, reply in natural language with a concise answer."
)


def _as_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("answer") or "").strip()
            else:
                text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, dict):
        return _as_answers(value.get("text") or value.get("answer") or list(value.values()))
    text = str(value).strip()
    return [text] if text else []


def _sample_search_episode(
    *,
    sample_id: str,
    question: str,
    answers: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {"answers": answers, "question": question, "require_search": True}
    if extra_meta:
        meta.update(extra_meta)
    return {
        "sample_id": sample_id,
        "messages": [
            {"role": "system", "content": _SEARCH_SYSTEM},
            {"role": "user", "content": question},
        ],
        "meta": meta,
        # Tools are attached via bind_tools / dataset.tools at config time.
        "tools": [],
    }


def grade_web_qa(conversation: list[dict], meta: dict[str, Any] | None = None) -> dict[str, float]:
    """Shared grader for open web / multi-hop search datasets."""
    meta = meta or {}
    text = final_assistant_text(conversation)
    calls = conversation_tool_calls(conversation)
    used = used_search_tools(calls)
    require = bool(meta.get("require_search", True))
    reward, metrics = score_text_answer(
        text,
        meta.get("answers") or meta.get("answer") or [],
        used_search=used,
        require_search=require,
    )
    metrics["num_tool_calls"] = float(len(calls))
    metrics["score"] = float(reward)
    return metrics


# --- Natural Questions open (short answers, good search practice) -------------


def init_nq_open() -> dict[str, Any]:
    rows = load_hf_rows("google-research-datasets/nq_open", split="train")
    return {"name": "nq_open", "rows": rows}


def sample_nq_open(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = pick_row(state.get("rows") or [], rng)
    question = str(row.get("question") or "").strip()
    answers = _as_answers(row.get("answer") or row.get("answers"))
    return _sample_search_episode(
        sample_id=f"nq_open:{hash(question) & 0xFFFFFFFF:x}",
        question=question,
        answers=answers,
        extra_meta={"source": "nq_open"},
    )


NQ_OPEN_RL = RlDataset(
    name="nq_open",
    init_fn=init_nq_open,
    sample_fn=sample_nq_open,
    grade_fn=grade_web_qa,
    tools=[],
    weight=1.0,
    target_episodes=2_000,
)


# --- HotpotQA distractor (multi-hop; model should search, not read context) ---


def init_hotpot_rl() -> dict[str, Any]:
    rows = load_hf_rows("hotpotqa/hotpot_qa", subset="distractor", split="train")
    return {"name": "hotpot", "rows": rows}


def sample_hotpot_rl(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = pick_row(state.get("rows") or [], rng)
    question = str(row.get("question") or "").strip()
    answers = _as_answers(row.get("answer"))
    # Intentionally omit Wikipedia context so the policy must use search tools.
    return _sample_search_episode(
        sample_id=f"hotpot:{hash(question) & 0xFFFFFFFF:x}",
        question=question,
        answers=answers,
        extra_meta={"source": "hotpot", "level": row.get("level"), "type": row.get("type")},
    )


HOTPOT_RL = RlDataset(
    name="hotpot",
    init_fn=init_hotpot_rl,
    sample_fn=sample_hotpot_rl,
    grade_fn=grade_web_qa,
    tools=[],
    weight=1.0,
    target_episodes=2_000,
)


# --- WebQuestions -------------------------------------------------------------


def init_web_questions() -> dict[str, Any]:
    # Compact Freebase-backed open QA set commonly used with web retrieval.
    rows = load_hf_rows("web_questions", split="train")
    return {"name": "web_questions", "rows": rows}


def sample_web_questions(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = pick_row(state.get("rows") or [], rng)
    question = str(row.get("question") or "").strip()
    answers = _as_answers(row.get("answers") or row.get("answer"))
    return _sample_search_episode(
        sample_id=f"webq:{hash(question) & 0xFFFFFFFF:x}",
        question=question,
        answers=answers,
        extra_meta={"source": "web_questions"},
    )


WEB_QUESTIONS_RL = RlDataset(
    name="web_questions",
    init_fn=init_web_questions,
    sample_fn=sample_web_questions,
    grade_fn=grade_web_qa,
    tools=[],
    weight=1.0,
    target_episodes=1_000,
)


# --- TriviaQA (rc.nocontext: question + answer aliases, no passage) ------------


def init_trivia_qa() -> dict[str, Any]:
    rows = load_hf_rows("trivia_qa", subset="rc.nocontext", split="train")
    return {"name": "trivia_qa", "rows": rows}


def sample_trivia_qa(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = pick_row(state.get("rows") or [], rng)
    question = str(row.get("question") or "").strip()
    answer_blob = row.get("answer") or {}
    aliases: list[str] = []
    if isinstance(answer_blob, dict):
        aliases.extend(_as_answers(answer_blob.get("aliases")))
        aliases.extend(_as_answers(answer_blob.get("normalized_aliases")))
        value = answer_blob.get("value") or answer_blob.get("normalized_value")
        aliases.extend(_as_answers(value))
    else:
        aliases.extend(_as_answers(answer_blob))
    # De-dupe while preserving order.
    seen = set()
    answers = []
    for item in aliases:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        answers.append(item)
    return _sample_search_episode(
        sample_id=f"trivia:{hash(question) & 0xFFFFFFFF:x}",
        question=question,
        answers=answers,
        extra_meta={"source": "trivia_qa"},
    )


TRIVIA_QA_RL = RlDataset(
    name="trivia_qa",
    init_fn=init_trivia_qa,
    sample_fn=sample_trivia_qa,
    grade_fn=grade_web_qa,
    tools=[],
    weight=1.0,
    target_episodes=2_000,
)
