"""Grade previously generated tool traces (after the fact).

Reads ``tool_traces.jsonl`` (or parquet with ``*_json`` columns) produced by
``generate_tool_traces.py`` and writes a scored copy.

Example::

    python grade_tool_traces.py --in tool_traces.jsonl --out tool_traces_scored.parquet \\
      --keep-min-score 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from teapottrainer.formatting import ABSTAIN_TEXT
from teapottrainer.metrics import (
    _normalize_answer,
    best_span_scores,
    gold_gsm_number,
    is_abstention,
    score_gsm,
)
from teapottrainer.rl.rewards import used_named_tools, used_search_tools
from teapottrainer.rl.tools import conversation_tool_calls, final_assistant_text

BROWSER_TOOLS = frozenset({"browser_goto", "browser_read", "browser_act"})
CONTEXT_TAG_OPEN = "[INJECTED_CONTEXT]"
CONTEXT_TAG_CLOSE = "[/INJECTED_CONTEXT]"


def extract_injected_context(messages: list[dict[str, Any]], fallback: str = "") -> str:
    chunks = []
    for message in messages:
        content = str(message.get("content") or "")
        if CONTEXT_TAG_OPEN in content and CONTEXT_TAG_CLOSE in content:
            start = content.find(CONTEXT_TAG_OPEN) + len(CONTEXT_TAG_OPEN)
            end = content.find(CONTEXT_TAG_CLOSE)
            if end > start:
                chunks.append(content[start:end].strip())
    if chunks:
        return "\n".join(chunks)
    return fallback or ""


def extract_tool_evidence(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    )


def evidence_corpus(messages: list[dict[str, Any]], injected: str = "") -> str:
    return "\n".join(
        p
        for p in (injected, extract_injected_context(messages, injected), extract_tool_evidence(messages))
        if p and str(p).strip()
    )


def gold_supported_by_evidence(golds: list[str], evidence: str) -> bool:
    ev = _normalize_answer(evidence or "")
    if not ev:
        return False
    for gold in golds:
        num = gold_gsm_number(gold)
        if num and _normalize_answer(num) in ev:
            return True
        g = _normalize_answer(gold)
        if g and g in ev:
            return True
    return False


def answer_supported_by_evidence(prediction: str, evidence: str, golds: list[str]) -> bool:
    if gold_supported_by_evidence(golds, evidence):
        _, _, contained = best_span_scores(prediction, golds)
        if contained:
            return True
        if golds and score_gsm(prediction, golds[0]) >= 1.0:
            return True
    pred_norm = _normalize_answer(prediction)
    ev = _normalize_answer(evidence)
    if not pred_norm or not ev:
        return False
    for gold in golds:
        g = _normalize_answer(gold_gsm_number(gold) or gold)
        if g and g in pred_norm and g in ev:
            return True
    return False


def effective_required_tools(required: list[str], available: list[str]) -> list[str]:
    avail = set(available)
    out = []
    for name in required:
        if name == "web_search" and any(t in avail for t in ("web_search", "brave_search", "exa_search")):
            out.append("web_search")
        elif name == "calculator" and "calculator" in avail:
            out.append("calculator")
        elif name == "browser" and (BROWSER_TOOLS & avail):
            out.append("browser")
        elif name in avail:
            out.append(name)
    return out


def required_tools_used(required: list[str], calls: list[dict]) -> bool:
    if not required:
        return True
    for name in required:
        if name == "web_search":
            if not used_search_tools(calls):
                return False
        elif name == "browser":
            if not used_named_tools(calls, BROWSER_TOOLS):
                return False
        elif not used_named_tools(calls, [name]):
            return False
    return True


def grade_row(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    trace = row.get("messages_trace") or []
    if isinstance(trace, str):
        trace = json.loads(trace)
    available = row.get("tools_available") or []
    if isinstance(available, str):
        available = json.loads(available)
    required_raw = row.get("tools_required") or []
    if isinstance(required_raw, str):
        required_raw = json.loads(required_raw)
    golds = row.get("gold_answers") or []
    if isinstance(golds, str):
        golds = json.loads(golds)
    task_type = str(row.get("task_type") or "search")
    unanswerable = bool(row.get("unanswerable"))
    injected = str(row.get("injected_context") or "")

    final = final_assistant_text(trace) or str(row.get("final_assistant") or "")
    calls = conversation_tool_calls(trace)
    required = effective_required_tools(list(required_raw), list(available))
    used_ok = required_tools_used(required, calls)
    abstained = is_abstention(final)
    evidence = evidence_corpus(trace, injected=injected)
    has_evidence = bool(_normalize_answer(evidence))

    metrics: dict[str, Any] = {
        "used_required_tools": 1.0 if used_ok else 0.0,
        "abstained": 1.0 if abstained else 0.0,
        "has_evidence": 1.0 if has_evidence else 0.0,
        "required_tools": required,
        "task_type": task_type,
    }

    if unanswerable:
        if abstained and used_ok:
            score = 1.0
        elif abstained:
            score = 0.5
        else:
            score = 0.0
        metrics["grounded"] = 1.0 if abstained else 0.0
        metrics["answer_contained"] = 0.0
        metrics["hallucinated"] = 0.0 if abstained else 1.0
        return float(score), metrics

    if not used_ok:
        metrics["grounded"] = 0.0
        metrics["answer_contained"] = 0.0
        metrics["hallucinated"] = 1.0
        return 0.0, metrics

    if task_type == "calc":
        gold = golds[0] if golds else ""
        answer_ok = score_gsm(final, gold) >= 1.0
        grounded = answer_ok and (
            gold_supported_by_evidence(golds, evidence) or used_named_tools(calls, ["calculator"])
        )
        if used_named_tools(calls, ["calculator"]) and evidence.strip():
            grounded = answer_ok and gold_supported_by_evidence(golds, evidence)
        metrics["answer_contained"] = 1.0 if answer_ok else 0.0
        metrics["grounded"] = 1.0 if grounded else 0.0
        metrics["hallucinated"] = (
            1.0 if ((answer_ok and not grounded) or (not answer_ok and not abstained)) else 0.0
        )
        if answer_ok and grounded:
            return 1.0, metrics
        if abstained and not has_evidence:
            return 0.4, metrics
        return 0.0, metrics

    _, _, contained = best_span_scores(final, golds)
    answer_ok = bool(contained)
    grounded = answer_supported_by_evidence(final, evidence, golds)
    metrics["answer_contained"] = 1.0 if answer_ok else 0.0
    metrics["grounded"] = 1.0 if grounded else 0.0
    metrics["hallucinated"] = 1.0 if ((not abstained) and (not grounded)) else 0.0

    if answer_ok and grounded and used_ok:
        return 1.0, metrics
    if abstained and not gold_supported_by_evidence(golds, evidence):
        return 0.5, metrics
    return 0.0, metrics


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        rows = []
        for batch in table.to_pylist():
            row = dict(batch)
            # Unflatten JSON columns if present.
            for key, raw_key in (
                ("gold_answers", "gold_answers_json"),
                ("tools_available", "tools_available_json"),
                ("tools_required", "tools_required_json"),
                ("messages_input", "messages_input_json"),
                ("messages_trace", "messages_trace_json"),
            ):
                if raw_key in row and key not in row:
                    row[key] = json.loads(row[raw_key])
            rows.append(row)
        return rows
    raise SystemExit(f"unsupported input type: {path}")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        flat = []
        for row in rows:
            flat.append(
                {
                    "sample_id": row.get("sample_id"),
                    "dataset": row.get("dataset"),
                    "task_type": row.get("task_type"),
                    "question": row.get("question"),
                    "gold_answers_json": json.dumps(row.get("gold_answers") or [], ensure_ascii=False),
                    "tools_available_json": json.dumps(row.get("tools_available") or [], ensure_ascii=False),
                    "tools_required_json": json.dumps(row.get("tools_required") or [], ensure_ascii=False),
                    "messages_input_json": json.dumps(row.get("messages_input") or [], ensure_ascii=False),
                    "messages_trace_json": json.dumps(row.get("messages_trace") or [], ensure_ascii=False),
                    "final_assistant": row.get("final_assistant") or "",
                    "injected_context": row.get("injected_context") or "",
                    "model": row.get("model") or "",
                    "tool_format": row.get("tool_format") or "",
                    "unanswerable": bool(row.get("unanswerable")),
                    "score": float(row.get("score") or 0.0),
                    "metrics_json": json.dumps(row.get("metrics") or {}, ensure_ascii=False),
                }
            )
        pq.write_table(pa.Table.from_pylist(flat), path)
        return
    raise SystemExit(f"unsupported output type: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", required=True, help="Input .jsonl or .parquet from generate_tool_traces.py")
    p.add_argument("--out", required=True, help="Scored output .jsonl or .parquet")
    p.add_argument("--keep-min-score", type=float, default=-1.0, help="If >=0, drop rows below this score")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.inp))
    scored: list[dict[str, Any]] = []
    for row in rows:
        score, metrics = grade_row(row)
        row = dict(row)
        row["score"] = score
        row["metrics"] = metrics
        if not row.get("final_assistant"):
            row["final_assistant"] = final_assistant_text(row.get("messages_trace") or [])
        if args.keep_min_score >= 0 and score < args.keep_min_score:
            continue
        scored.append(row)
    _write_rows(Path(args.out), scored)
    print(f"graded {len(rows)} -> kept {len(scored)} -> {args.out}")


if __name__ == "__main__":
    main()
