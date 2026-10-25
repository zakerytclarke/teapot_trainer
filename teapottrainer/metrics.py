"""Scorers for assistant and agent behavior.

Each function is pure so the metric can be checked against gold rows without
running a model. Higher is better unless the name says otherwise.
"""

from __future__ import annotations

import ast
import json
import re
import string
from collections import Counter
from typing import Any

from .formatting import ABSTAIN_TEXT

_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _parse_jsonish(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return None


def _coerce_call(obj: Any) -> dict | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        parsed = _parse_jsonish(args)
        args = parsed if isinstance(parsed, dict) else {"_raw": args}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"name": name.strip(), "arguments": _canonicalize(args)}


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"-?\d+\.\d+", stripped):
            number = float(stripped)
            return int(number) if number.is_integer() else number
        return stripped
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def extract_tool_calls(text: str) -> list[dict]:
    """Parse ``<tool_call>`` blocks or a bare JSON call. Empty means no call."""
    if not text:
        return []
    calls: list[dict] = []
    blocks = _TOOL_BLOCK.findall(text)
    payloads = blocks or []
    if not payloads:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            payloads = [stripped]
    for payload in payloads:
        parsed = _parse_jsonish(payload)
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            call = _coerce_call(item)
            if call is not None:
                calls.append(call)
    return calls


def score_tool_prediction(prediction: str, gold: str) -> dict[str, float]:
    gold_calls = extract_tool_calls(gold)
    pred_calls = extract_tool_calls(prediction)
    gold_is_call = bool(gold_calls)
    pred_is_call = bool(pred_calls)
    if not gold_is_call:
        return {
            "parse": 0.0,
            "name": 0.0,
            "args": 0.0,
            "restraint": 1.0 if not pred_is_call else 0.0,
            "gold_is_call": 0.0,
        }
    names_ok = [g["name"] for g in gold_calls] == [p["name"] for p in pred_calls]
    args_ok = names_ok and [
        g["arguments"] for g in gold_calls
    ] == [
        p["arguments"] for p in pred_calls
    ]
    return {
        "parse": 1.0 if pred_is_call else 0.0,
        "name": 1.0 if names_ok else 0.0,
        "args": 1.0 if args_ok else 0.0,
        "restraint": 0.0,
        "gold_is_call": 1.0,
    }


def classify_decision(text: str) -> str:
    """Map a reply to When2Call's decision: call, ask, refuse, or answer."""
    if extract_tool_calls(text):
        return "tool_call"
    lowered = (text or "").lower()
    asks = (
        "?" in (text or "")
        or "could you" in lowered
        or "can you provide" in lowered
        or "please provide" in lowered
        or "i'll need" in lowered
        or "i will need" in lowered
        or "i need" in lowered
        or "need to know" in lowered
        or "which specific" in lowered
    )
    refuses = any(
        phrase in lowered
        for phrase in (
            "unable",
            "cannot",
            "can't",
            "can not",
            "sorry",
            "apolog",
            "not able",
            "no tool",
            "do not have",
            "don't have",
            "cannot answer",
        )
    )
    if asks and not refuses:
        return "request_for_info"
    if refuses:
        return "cannot_answer"
    if asks:
        return "request_for_info"
    return "direct"


def _normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch if ch not in set(string.punctuation) else " " for ch in text)
    tokens = [tok for tok in text.split() if tok not in {"a", "an", "the"}]
    return " ".join(tokens)


def is_abstention(text: str) -> bool:
    normalized = _normalize_answer(text)
    if not normalized:
        return True
    phrases = (
        _normalize_answer(ABSTAIN_TEXT),
        "does not contain the answer",
        "context does not contain",
        "not enough information",
        "cannot be answered",
        "no answer in the context",
        "unable to answer",
    )
    return any(phrase in normalized for phrase in phrases)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_span_scores(prediction: str, golds: list[str]) -> tuple[float, float]:
    if not golds:
        return 0.0, 0.0
    em = 0.0
    f1 = 0.0
    pred_norm = _normalize_answer(prediction)
    for gold in golds:
        if pred_norm == _normalize_answer(gold):
            em = 1.0
        f1 = max(f1, token_f1(prediction, gold))
    return em, f1


def _canon_number(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip().replace(",", "").rstrip(".")
    if not text:
        return None
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def gold_gsm_number(answer: str) -> str | None:
    if "####" not in (answer or ""):
        return _canon_number(answer)
    tail = answer.split("####")[-1]
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", tail)
    return _canon_number(match.group(0) if match else None)


def pred_gsm_number(text: str) -> str | None:
    if not text:
        return None
    if "####" in text:
        return gold_gsm_number(text)
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return _canon_number(matches[-1])


def score_gsm(prediction: str, gold_answer: str) -> float:
    gold = gold_gsm_number(gold_answer)
    pred = pred_gsm_number(prediction)
    if gold is None or pred is None:
        return 0.0
    return 1.0 if gold == pred else 0.0


def _bullet_count(response: str) -> int:
    return len(re.findall(r"(?m)^\s*[\*\-]\s+\S", response or ""))


def _keyword_list(blob: str) -> list[str]:
    quoted = re.findall(r"[\"“”']([^\"“”']+)[\"“”']", blob)
    starred = re.findall(r"\*([^*]+)\*", blob)
    bracketed = re.findall(r"\[([^\]]+)\]", blob)
    words: list[str] = []
    for chunk in quoted + starred + bracketed:
        words.extend(part.strip() for part in chunk.split(",") if part.strip())
    if not words:
        words = [part.strip(" *.,") for part in re.split(r",| and ", blob) if part.strip(" *.,")]
    cleaned = []
    for word in words:
        word = word.strip(" *.,;:")
        if word and word.lower() not in {"and", "or"}:
            cleaned.append(word)
    return cleaned


def constraint_checks(instruction: str, response: str) -> list[tuple[str, bool]]:
    """Check the constraint patterns that show up in SmolTalk smol-constraints."""
    text = instruction or ""
    response = response or ""
    checks: list[tuple[str, bool]] = []

    match = re.search(r"at least (\d+) sentences", text, re.IGNORECASE)
    if match:
        sentences = [part for part in re.split(r"[.!?]+", response) if part.strip()]
        checks.append(("min_sentences", len(sentences) >= int(match.group(1))))

    match = re.search(r"exactly (\d+) bullet", text, re.IGNORECASE)
    if match:
        checks.append(("exact_bullets", _bullet_count(response) == int(match.group(1))))
    else:
        match = re.search(r"at least (\d+) bullet", text, re.IGNORECASE)
        if match:
            checks.append(("min_bullets", _bullet_count(response) >= int(match.group(1))))

    match = re.search(r"less than (\d+) words", text, re.IGNORECASE)
    if match:
        checks.append(("max_words", len(_WORD.findall(response)) < int(match.group(1))))

    match = re.search(
        r"the word [\"“”'](.+?)[\"“”'] should appear at least (\d+)",
        text,
        re.IGNORECASE,
    )
    if match:
        found = len(re.findall(rf"\b{re.escape(match.group(1))}\b", response, re.IGNORECASE))
        checks.append(("word_count", found >= int(match.group(2))))

    if re.search(r"title", text, re.IGNORECASE) and "<<" in text:
        checks.append(("title", bool(re.search(r"<<[^<>\n]+>>", response))))

    if re.search(r"postscript starting with\s+[\"“”']?P\.?S", text, re.IGNORECASE):
        checks.append(("postscript", bool(re.search(r"P\.S\.", response))))

    if re.search(r"all lowercase|no capital letters", text, re.IGNORECASE):
        letters = [ch for ch in response if ch.isalpha()]
        checks.append(("lowercase", bool(letters) and all(ch.islower() for ch in letters)))

    match = re.search(r"at least (\d+) placeholders", text, re.IGNORECASE)
    if match:
        found = re.findall(r"\[[^\[\]\n]{1,60}\]", response)
        checks.append(("placeholders", len(found) >= int(match.group(1))))

    match = re.search(r"exactly (\d+) paragraphs", text, re.IGNORECASE)
    at_least = re.search(r"at least (\d+) paragraphs", text, re.IGNORECASE)
    if match or at_least:
        parts = [part.strip() for part in re.split(r"\*\*\*", response) if part.strip()]
        if match:
            checks.append(("paragraphs", len(parts) == int(match.group(1))))
        else:
            checks.append(("paragraphs", len(parts) >= int(at_least.group(1))))

    match = re.search(r"include keywords\s*\[([^\]]+)\]", text, re.IGNORECASE)
    if match:
        keywords = [part.strip() for part in match.group(1).split(",") if part.strip()]
        checks.append(("keywords", all(word.lower() in response.lower() for word in keywords)))
    else:
        match = re.search(r"include keywords such as ([^.\n]+)", text, re.IGNORECASE)
        if match:
            keywords = _keyword_list(match.group(1))
            checks.append((
                "keywords",
                bool(keywords) and all(word.lower() in response.lower() for word in keywords),
            ))

    if re.search(r"highlight at least (\d+) section", text, re.IGNORECASE):
        highlights = re.findall(r"(?<![\*\n])\*[^*\n]+\*(?!\*)", response)
        checks.append(("highlight", len(highlights) >= 1))

    return checks


def score_constraints(instruction: str, response: str) -> dict[str, float] | None:
    checks = constraint_checks(instruction, response)
    if not checks:
        return None
    passed = sum(1 for _, ok in checks if ok)
    return {
        "check_pass": passed / len(checks),
        "example_pass": 1.0 if passed == len(checks) else 0.0,
        "n_checks": float(len(checks)),
    }


HIGHER_IS_BETTER = (
    "parse",
    "name",
    "args",
    "restraint",
    "decision_accuracy",
    "answerable_em",
    "answerable_f1",
    "unanswerable_abstain",
    "em",
    "f1",
    "numeric_em",
    "example_pass",
    "check_pass",
)
