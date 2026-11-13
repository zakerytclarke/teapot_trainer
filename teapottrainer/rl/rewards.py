"""Verifiable reward helpers for RL environments."""

from __future__ import annotations

import re


_NUM = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


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
