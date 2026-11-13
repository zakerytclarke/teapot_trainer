"""Verifiable reward helpers for RL environments."""

from __future__ import annotations

import re
from typing import Iterable

from ..metrics import best_span_scores, gold_gsm_number, is_abstention, pred_gsm_number, score_gsm


_NUM = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")

_SEARCH_TOOL_NAMES = frozenset({"brave_search", "exa_search", "llm_search", "web_search"})


def extract_number(text: str) -> float | None:
    """Best-effort last number in a reply."""
    matches = _NUM.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def numbers_equal(pred: float | None, gold: float | None, tol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    if abs(gold) < 1.0:
        return abs(pred - gold) <= tol
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


def used_named_tools(calls: Iterable[dict], names: Iterable[str]) -> bool:
    wanted = {str(n).strip() for n in names if str(n).strip()}
    for call in calls or []:
        if str(call.get("name") or "").strip() in wanted:
            return True
    return False


def used_search_tools(calls: Iterable[dict]) -> bool:
    return used_named_tools(calls, _SEARCH_TOOL_NAMES)


def score_numeric_answer(
    prediction: str,
    gold_value: float,
    *,
    used_calculator: bool = False,
    require_calculator: bool = False,
    calculator_result_ok: bool = False,
) -> tuple[float, dict[str, float]]:
    """Reward for synthetic / word-math answers with a known numeric gold.

    When ``require_calculator`` is True, a correct final answer with no calculator
    call scores 0. That is intentional: otherwise easy arithmetic never teaches
    tool use, because the model can already solve it from weights alone.
    """
    pred = extract_number(prediction)
    correct = 1.0 if numbers_equal(pred, float(gold_value)) else 0.0
    prose = 1.0 if (prediction or "").strip() and pred is not None else 0.0
    calc = 1.0 if used_calculator else 0.0
    calc_ok = 1.0 if calculator_result_ok else 0.0

    if require_calculator:
        # Must call the tool and land the right final answer.
        if not used_calculator:
            reward = 0.0
        elif correct and calculator_result_ok:
            reward = 1.0
        elif correct:
            # Called something, final answer right, but tool output didn't match gold.
            reward = 0.4
        elif calculator_result_ok:
            # Tool got the value; final prose missed it.
            reward = 0.5
        else:
            reward = 0.0
    else:
        reward = correct
        if correct and used_calculator and calculator_result_ok:
            reward += 0.25

    if reward > 0 and prose and len((prediction or "").split()) > 1:
        reward += 0.1

    metrics = {
        "accuracy": correct,
        "parsed": 1.0 if pred is not None else 0.0,
        "used_calculator": calc,
        "calculator_result_ok": calc_ok,
        "prose": prose,
    }
    return float(reward), metrics


def score_gsm8k_answer(
    prediction: str,
    gold_answer: str,
    *,
    used_calculator: bool = False,
    require_calculator: bool = False,
) -> tuple[float, dict[str, float]]:
    """GSM8K reward using the same number extraction as supervised evals."""
    correct = float(score_gsm(prediction, gold_answer))
    gold = gold_gsm_number(gold_answer)
    pred = pred_gsm_number(prediction)
    prose = 1.0 if (prediction or "").strip() else 0.0
    calc = 1.0 if used_calculator else 0.0

    if require_calculator and not used_calculator:
        reward = 0.0
    else:
        reward = correct
        if correct and used_calculator:
            reward += 0.1

    if reward > 0 and prose and len((prediction or "").split()) > 3:
        reward += 0.05

    metrics = {
        "accuracy": correct,
        "parsed": 1.0 if pred is not None else 0.0,
        "used_calculator": calc,
        "prose": prose,
        "gold_parsed": 1.0 if gold is not None else 0.0,
    }
    return float(reward), metrics


def score_text_answer(
    prediction: str,
    gold_answers: str | list[str],
    *,
    used_search: bool = False,
    require_search: bool = False,
) -> tuple[float, dict[str, float]]:
    """Open-domain / web QA reward.

    Success is **containment-first**: if any gold answer string appears in the
    reply (after normalization), the model gets full credit even when it adds
    surrounding prose. Token F1 is only a fallback for paraphrases that do not
    literally contain the gold span — long answers are not punished for length.
    """
    if isinstance(gold_answers, str):
        golds = [gold_answers]
    else:
        golds = [str(g) for g in (gold_answers or []) if str(g).strip()]
    em, f1, contained = best_span_scores(prediction or "", golds)
    prose = 1.0 if (prediction or "").strip() else 0.0
    search = 1.0 if used_search else 0.0

    if require_search and not used_search:
        reward = 0.0
    elif contained or em:
        # Has the answer somewhere in the reply — surrounding text is fine.
        reward = 1.0
        if used_search:
            reward += 0.15
    elif f1 >= 0.5:
        # Paraphrase without literal span; keep soft credit, no length penalty.
        reward = float(f1)
        if used_search:
            reward += 0.1
    else:
        reward = 0.0

    metrics = {
        "accuracy": float(em),
        "f1": float(f1),
        "contained": float(contained),
        "used_search": search,
        "prose": prose,
    }
    return float(reward), metrics


def score_exact_answer(
    prediction: str,
    gold_answer: str,
) -> tuple[float, dict[str, float]]:
    """Binary reward: normalized prediction must equal the gold string exactly."""
    em, f1, contained = best_span_scores(prediction or "", [gold_answer])
    reward = float(em)
    metrics = {
        "accuracy": float(em),
        "f1": float(f1),
        "contained": float(contained),
        "prose": 1.0 if (prediction or "").strip() else 0.0,
    }
    return reward, metrics


def score_synthqa_answer(
    prediction: str,
    *,
    gold_answer: str,
    unanswerable: bool,
) -> tuple[float, dict[str, float]]:
    """SynthQA keepalive reward: exact match only (no tools, no soft F1).

    Unanswerable rows must reply with the exact abstain sentence. Answerable
    rows must match the gold answer exactly after normalization.
    """
    text = prediction or ""
    abstained = 1.0 if is_abstention(text) else 0.0
    if unanswerable:
        # Prefer the exact abstain string; is_abstention is a soft fallback check.
        reward, metrics = score_exact_answer(text, gold_answer)
        if reward < 1.0 and abstained:
            # Still accept known abstain paraphrases only if gold is the canonical line
            # and the model used a recognized abstain — keep strict: require EM.
            reward = 0.0
        metrics["abstained"] = abstained
        metrics["unanswerable"] = 1.0
        return float(reward), metrics

    reward, metrics = score_exact_answer(text, gold_answer)
    if abstained:
        reward = 0.0
    metrics["abstained"] = abstained
    metrics["unanswerable"] = 0.0
    return float(reward), metrics
