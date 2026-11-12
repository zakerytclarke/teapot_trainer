"""Verifiable reward helpers for RL environments."""

from __future__ import annotations

import re
from typing import Any


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
) -> tuple[float, dict[str, float]]:
    """Reward for synthetic / word-math answers with a known numeric gold."""
    pred = extract_number(prediction)
    correct = 1.0 if numbers_equal(pred, float(gold_value)) else 0.0
    # Light shape bonus: non-empty assistant prose that still contains the number.
    prose = 1.0 if (prediction or "").strip() and pred is not None else 0.0
    calc = 1.0 if used_calculator else 0.0
    reward = correct
    if require_calculator and correct and not used_calculator:
        reward *= 0.5
    # Prefer answers that are not a bare number-only dump once correct.
    if correct and prose and len((prediction or "").split()) > 1:
        reward += 0.1
    metrics = {
        "accuracy": correct,
        "parsed": 1.0 if pred is not None else 0.0,
        "used_calculator": calc,
        "prose": prose,
    }
    return float(reward), metrics
