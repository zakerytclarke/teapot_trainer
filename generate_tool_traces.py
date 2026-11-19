"""Generate teacher tool-use traces for SFT distillation.

Runs a strong instruct model (default Qwen2.5-14B) with a **random subset** of
tools on general assistant / QnA tasks (no coding). Each row stores:

* ``messages_input`` — conversation before generation
* ``messages_trace`` — full agent loop (input + tool calls + final answer)
* ``score`` / ``metrics`` — grounded outcome

Grading rules (high level):

* **GSM8K / calc tasks:** if ``calculator`` was available, it must be used; final
  answer must match gold. Hallucinated arithmetic without the tool scores 0.
* **Search / browser tasks:** if the matching tool was available, it must be used;
  the gold answer must appear in the final reply **and** be supported by tool
  evidence (or injected context). Answering from parametric memory alone scores 0.
* **Context QA (e.g. SQuAD):** evidence is the ``[INJECTED_CONTEXT]`` block. The
  model must answer from that context, or abstain when the context is insufficient.
  Making something up that is not in context scores 0.

Example::

    python generate_tool_traces.py \\
      --model Qwen/Qwen2.5-14B-Instruct \\
      --load-in-4bit \\
      --max-samples 200 \\
      --out traces.parquet
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from teapottrainer import (
    BrowserSettings,
    SearchSettings,
    build_browser_toolkit,
    build_search_tools,
    calculator,
    ensure_playwright_browsers,
    run_agent,
)
from teapottrainer.formatting import ABSTAIN_TEXT
from teapottrainer.metrics import (
    _normalize_answer,
    best_span_scores,
    gold_gsm_number,
    is_abstention,
    score_gsm,
)
from teapottrainer.rl.rewards import used_named_tools, used_search_tools
from teapottrainer.rl.tools import ToolFn, conversation_tool_calls, final_assistant_text

BROWSER_TOOLS = frozenset({"browser_goto", "browser_read", "browser_act"})
CONTEXT_TAG_OPEN = "[INJECTED_CONTEXT]"
CONTEXT_TAG_CLOSE = "[/INJECTED_CONTEXT]"


# ---------------------------------------------------------------------------
# Prompt / tool lottery
# ---------------------------------------------------------------------------


def build_system_prompt(*, today: str | None = None) -> str:
    day = today or date.today().isoformat()
    return f"""You are a careful tool-using assistant. Today is {day} (ISO date).

Evidence rules (strict):
- Only state facts that appear in tool results or in {CONTEXT_TAG_OPEN}...{CONTEXT_TAG_CLOSE} blocks.
- {CONTEXT_TAG_OPEN} blocks are machine-injected evidence, not the user speaking.
- If the available evidence is insufficient, refuse briefly (e.g. "{ABSTAIN_TEXT}") instead of guessing.
- Never invent biographies, numbers, dates, or citations.

Tool rules:
- calculator — required for arithmetic when available.
- web_search — look up open-domain facts when available.
- browser_goto / browser_read / browser_act — use for live pages when available.
- Prefer a short final answer after gathering evidence.
"""


TOOL_LOTTERY: list[tuple[str, ...]] = [
    ("calculator",),
    ("web_search",),
    ("browser",),
    ("calculator", "web_search"),
    ("web_search", "browser"),
    ("calculator", "browser"),
    ("calculator", "web_search", "browser"),
]


def sample_tool_bundle(
    rng: random.Random,
    *,
    search_tools: list[ToolFn],
    browser_tools: list[ToolFn],
    names: tuple[str, ...] | None = None,
) -> list[ToolFn]:
    pick = names or rng.choice(TOOL_LOTTERY)
    out: list[ToolFn] = []
    if "calculator" in pick:
        out.append(calculator)
    if "web_search" in pick:
        out.extend(search_tools)
    if "browser" in pick:
        out.extend(browser_tools)
    return out


def tool_names(tools: Iterable[ToolFn]) -> list[str]:
    return [str(getattr(fn, "__name__", "tool")) for fn in tools]


# ---------------------------------------------------------------------------
# Episode specs
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSpec:
    sample_id: str
    dataset: str
    task_type: str  # calc | search | context_qa | browser
    question: str
    gold_answers: list[str]
    messages_input: list[dict[str, Any]]
    required_tools: list[str] = field(default_factory=list)
    # If True, gold is "must abstain" (unanswerable / insufficient evidence).
    unanswerable: bool = False
    # Extra evidence already in the prompt (injected context).
    injected_context: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


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


def _messages_for_question(system: str, question: str, *, injected_context: str = "") -> list[dict[str, Any]]:
    msgs = [{"role": "system", "content": system}]
    if injected_context.strip():
        msgs.append(
            {
                "role": "user",
                "content": (
                    f"{CONTEXT_TAG_OPEN}\n{injected_context.strip()}\n{CONTEXT_TAG_CLOSE}\n\n"
                    f"Question: {question}"
                ),
            }
        )
    else:
        msgs.append({"role": "user", "content": question})
    return msgs


# --- Dataset loaders --------------------------------------------------------


def load_gsm8k(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit else rows


def load_nq_open(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("google-research-datasets/nq_open", split=split)
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit else rows


def load_web_questions(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("stanfordnlp/web_questions", split=split)
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit else rows


def load_trivia_qa(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit else rows


def load_hotpot(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit else rows


def load_squad_v2(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit else rows


def make_gsm8k_episode(row: dict[str, Any], system: str, idx: int) -> EpisodeSpec:
    q = str(row.get("question") or "").strip()
    gold = str(row.get("answer") or "")
    return EpisodeSpec(
        sample_id=f"gsm8k:{idx}",
        dataset="gsm8k",
        task_type="calc",
        question=q,
        gold_answers=[gold],
        messages_input=_messages_for_question(system, q),
        required_tools=["calculator"],  # enforced only if calculator is in the bundle
        meta={"raw_answer": gold},
    )


def make_search_episode(
    *,
    dataset: str,
    sample_id: str,
    question: str,
    answers: list[str],
    system: str,
) -> EpisodeSpec:
    return EpisodeSpec(
        sample_id=sample_id,
        dataset=dataset,
        task_type="search",
        question=question,
        gold_answers=answers,
        messages_input=_messages_for_question(system, question),
        required_tools=["web_search"],
    )


def make_squad_episode(row: dict[str, Any], system: str, idx: int) -> EpisodeSpec:
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    answers = _as_answers((row.get("answers") or {}).get("text") if isinstance(row.get("answers"), dict) else row.get("answers"))
    # SQuAD v2: empty answers => unanswerable
    unanswerable = len(answers) == 0 or bool(row.get("is_impossible"))
    gold = [ABSTAIN_TEXT] if unanswerable else answers
    return EpisodeSpec(
        sample_id=f"squad_v2:{idx}",
        dataset="squad_v2",
        task_type="context_qa",
        question=question,
        gold_answers=gold,
        messages_input=_messages_for_question(system, question, injected_context=context),
        required_tools=[],  # answer from injected context; tools optional noise
        unanswerable=unanswerable,
        injected_context=context,
    )


def build_episode_pool(
    system: str,
    *,
    per_dataset: int,
    seed: int,
) -> list[EpisodeSpec]:
    rng = random.Random(seed)
    episodes: list[EpisodeSpec] = []

    for i, row in enumerate(load_gsm8k(limit=per_dataset)):
        episodes.append(make_gsm8k_episode(row, system, i))

    for i, row in enumerate(load_nq_open(limit=per_dataset)):
        q = str(row.get("question") or "").strip()
        ans = _as_answers(row.get("answer") or row.get("answers"))
        episodes.append(
            make_search_episode(
                dataset="nq_open",
                sample_id=f"nq_open:{i}",
                question=q,
                answers=ans,
                system=system,
            )
        )

    for i, row in enumerate(load_web_questions(limit=per_dataset)):
        q = str(row.get("question") or "").strip()
        ans = _as_answers(row.get("answers") or row.get("answer"))
        episodes.append(
            make_search_episode(
                dataset="web_questions",
                sample_id=f"web_questions:{i}",
                question=q,
                answers=ans,
                system=system,
            )
        )

    for i, row in enumerate(load_trivia_qa(limit=per_dataset)):
        q = str(row.get("question") or "").strip()
        blob = row.get("answer") or {}
        ans: list[str] = []
        if isinstance(blob, dict):
            ans.extend(_as_answers(blob.get("aliases")))
            ans.extend(_as_answers(blob.get("normalized_aliases")))
            ans.extend(_as_answers(blob.get("value") or blob.get("normalized_value")))
        else:
            ans.extend(_as_answers(blob))
        # de-dupe
        seen = set()
        uniq = []
        for a in ans:
            key = a.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(a)
        episodes.append(
            make_search_episode(
                dataset="trivia_qa",
                sample_id=f"trivia_qa:{i}",
                question=q,
                answers=uniq,
                system=system,
            )
        )

    for i, row in enumerate(load_hotpot(limit=per_dataset)):
        q = str(row.get("question") or "").strip()
        ans = _as_answers(row.get("answer"))
        episodes.append(
            make_search_episode(
                dataset="hotpot",
                sample_id=f"hotpot:{i}",
                question=q,
                answers=ans,
                system=system,
            )
        )

    for i, row in enumerate(load_squad_v2(limit=per_dataset)):
        episodes.append(make_squad_episode(row, system, i))

    rng.shuffle(episodes)
    return episodes


# ---------------------------------------------------------------------------
# Evidence + grading
# ---------------------------------------------------------------------------


def extract_injected_context(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for message in messages:
        content = str(message.get("content") or "")
        if CONTEXT_TAG_OPEN in content and CONTEXT_TAG_CLOSE in content:
            start = content.find(CONTEXT_TAG_OPEN) + len(CONTEXT_TAG_OPEN)
            end = content.find(CONTEXT_TAG_CLOSE)
            if end > start:
                chunks.append(content[start:end].strip())
    return "\n".join(chunks)


def extract_tool_evidence(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


def evidence_corpus(messages: list[dict[str, Any]], injected: str = "") -> str:
    return "\n".join(
        p
        for p in (injected, extract_injected_context(messages), extract_tool_evidence(messages))
        if p and p.strip()
    )


def gold_supported_by_evidence(golds: list[str], evidence: str) -> bool:
    """True if any gold span (normalized) appears in the evidence text."""
    ev = _normalize_answer(evidence or "")
    if not ev:
        return False
    for gold in golds:
        g = _normalize_answer(gold)
        # GSM8K gold may be a long rationale with #### N — prefer the number.
        num = gold_gsm_number(gold)
        if num and _normalize_answer(num) in ev:
            return True
        if g and g in ev:
            return True
        # soft: any reasonably long gold token overlap
        if g and len(g) >= 3 and g in ev:
            return True
    return False


def answer_supported_by_evidence(prediction: str, evidence: str, golds: list[str]) -> bool:
    """Final answer must be grounded: gold in evidence, or prediction span in evidence."""
    if gold_supported_by_evidence(golds, evidence):
        # Also require the reply itself to contain the gold (answered correctly).
        _, _, contained = best_span_scores(prediction, golds)
        if contained:
            return True
        if score_gsm(prediction, golds[0]) >= 1.0:
            return True
    # Fallback: predicted number / short answer literal in evidence
    pred_norm = _normalize_answer(prediction)
    ev = _normalize_answer(evidence)
    if not pred_norm or not ev:
        return False
    # Avoid counting huge replies as "in evidence"
    for gold in golds:
        g = _normalize_answer(gold_gsm_number(gold) or gold)
        if g and g in pred_norm and g in ev:
            return True
    return False


def effective_required_tools(spec: EpisodeSpec, available: list[str]) -> list[str]:
    """Only require tools that were actually offered this episode."""
    avail = set(available)
    out = []
    for name in spec.required_tools:
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


def grade_trace(
    spec: EpisodeSpec,
    messages_trace: list[dict[str, Any]],
    *,
    available_tools: list[str],
) -> tuple[float, dict[str, Any]]:
    final = final_assistant_text(messages_trace)
    calls = conversation_tool_calls(messages_trace)
    required = effective_required_tools(spec, available_tools)
    used_ok = required_tools_used(required, calls)
    abstained = is_abstention(final)
    evidence = evidence_corpus(messages_trace, injected=spec.injected_context)
    has_evidence = bool(_normalize_answer(evidence))

    metrics: dict[str, Any] = {
        "used_required_tools": 1.0 if used_ok else 0.0,
        "abstained": 1.0 if abstained else 0.0,
        "has_evidence": 1.0 if has_evidence else 0.0,
        "required_tools": required,
        "task_type": spec.task_type,
    }

    # --- Unanswerable / must abstain ---
    if spec.unanswerable:
        # Good: abstain. Bad: confident answer.
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

    # --- Must use required tools when offered ---
    if not used_ok:
        metrics["grounded"] = 0.0
        metrics["answer_contained"] = 0.0
        metrics["hallucinated"] = 1.0
        return 0.0, metrics

    # --- Answer match ---
    if spec.task_type == "calc":
        gold = spec.gold_answers[0] if spec.gold_answers else ""
        answer_ok = score_gsm(final, gold) >= 1.0
        # Prefer evidence that literally contains the number; else require a calc call.
        grounded = answer_ok and (
            gold_supported_by_evidence(spec.gold_answers, evidence)
            or used_named_tools(calls, ["calculator"])
        )
        if used_named_tools(calls, ["calculator"]) and evidence.strip():
            grounded = answer_ok and gold_supported_by_evidence(spec.gold_answers, evidence)
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

    # search / context_qa / browser
    _, _, contained = best_span_scores(final, spec.gold_answers)
    answer_ok = bool(contained)
    grounded = answer_supported_by_evidence(final, evidence, spec.gold_answers)

    metrics["answer_contained"] = 1.0 if answer_ok else 0.0
    metrics["grounded"] = 1.0 if grounded else 0.0

    # Hallucination: answered something not supported by evidence.
    hallucinated = (not abstained) and (not grounded)
    metrics["hallucinated"] = 1.0 if hallucinated else 0.0

    if answer_ok and grounded and used_ok:
        return 1.0, metrics
    # Correct abstain when evidence truly lacks gold (model refused to hallucinate).
    if abstained and not gold_supported_by_evidence(spec.gold_answers, evidence):
        return 0.5, metrics
    # Right final string but not supported by tool/context evidence => hallucination.
    return 0.0, metrics


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not serializable: {type(obj)!r}")


def rows_to_table(rows: list[dict[str, Any]]):
    import pyarrow as pa

    # Nested structures stored as JSON strings for a flat, portable schema.
    flat = []
    for row in rows:
        flat.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "question": row["question"],
                "gold_answers_json": json.dumps(row["gold_answers"], ensure_ascii=False),
                "tools_available_json": json.dumps(row["tools_available"], ensure_ascii=False),
                "tools_required_json": json.dumps(row["tools_required"], ensure_ascii=False),
                "messages_input_json": json.dumps(row["messages_input"], ensure_ascii=False),
                "messages_trace_json": json.dumps(row["messages_trace"], ensure_ascii=False),
                "final_assistant": row["final_assistant"],
                "score": float(row["score"]),
                "metrics_json": json.dumps(row["metrics"], ensure_ascii=False, default=_json_default),
                "model": row["model"],
                "tool_format": row["tool_format"],
                "unanswerable": bool(row["unanswerable"]),
            }
        )
    return pa.Table.from_pylist(flat)


def write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        pq.write_table(rows_to_table(rows), path)
    elif suffix == ".csv":
        import csv

        table = rows_to_table(rows)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table.column_names)
            writer.writeheader()
            for i in range(table.num_rows):
                writer.writerow({name: table[name][i].as_py() for name in table.column_names})
    else:
        # jsonl fallback
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--tool-format", default="hermes")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--per-dataset", type=int, default=32, help="Rows to pull from each source")
    p.add_argument("--max-samples", type=int, default=64, help="Episodes to generate this run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--mock-search", action="store_true")
    p.add_argument("--mock-browser", action="store_true")
    p.add_argument("--out", default="tool_traces.parquet")
    p.add_argument("--keep-min-score", type=float, default=-1.0, help="If >=0, only write rows with score >= this")
    return p.parse_args()


def load_model(model_id: str, *, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    system = build_system_prompt()

    print("building episode pool…")
    pool = build_episode_pool(system, per_dataset=args.per_dataset, seed=args.seed)
    if args.max_samples > 0:
        pool = pool[: int(args.max_samples)]
    print(f"episodes: {len(pool)}")

    if not args.mock_browser:
        ensure_playwright_browsers()

    search = SearchSettings(mock=bool(args.mock_search), default_providers=["llmsearchindex"])
    browser = BrowserSettings(mock=bool(args.mock_browser), eager=True, headless=True)
    search_tools = build_search_tools(search)
    toolkit = build_browser_toolkit(browser)

    print(f"loading teacher {args.model} …")
    model, tokenizer = load_model(args.model, load_in_4bit=bool(args.load_in_4bit))

    rows: list[dict[str, Any]] = []
    try:
        for i, spec in enumerate(pool):
            tools = sample_tool_bundle(
                rng,
                search_tools=search_tools,
                browser_tools=toolkit.tools,
            )
            available = tool_names(tools)
            # Context-QA episodes: allow empty tool set sometimes so it must use injected context.
            if spec.task_type == "context_qa" and rng.random() < 0.35:
                tools = []
                available = []

            print(f"[{i+1}/{len(pool)}] {spec.sample_id} tools={available}")
            try:
                traj = run_agent(
                    model,
                    tokenizer,
                    spec.messages_input,
                    tools,
                    tool_format=args.tool_format,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    sample_id=spec.sample_id,
                    meta={"dataset": spec.dataset, "task_type": spec.task_type},
                )
                trace = traj.messages
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")
                continue

            score, metrics = grade_trace(spec, trace, available_tools=available)
            final = final_assistant_text(trace)
            row = {
                "sample_id": spec.sample_id,
                "dataset": spec.dataset,
                "task_type": spec.task_type,
                "question": spec.question,
                "gold_answers": spec.gold_answers,
                "tools_available": available,
                "tools_required": effective_required_tools(spec, available),
                "messages_input": spec.messages_input,
                "messages_trace": trace,
                "final_assistant": final,
                "score": score,
                "metrics": metrics,
                "model": args.model,
                "tool_format": args.tool_format,
                "unanswerable": spec.unanswerable,
            }
            if args.keep_min_score >= 0 and score < args.keep_min_score:
                print(f"  drop score={score:.2f} metrics={metrics}")
                continue
            rows.append(row)
            print(f"  score={score:.2f} grounded={metrics.get('grounded')} contained={metrics.get('answer_contained')}")
    finally:
        toolkit.close()

    out = Path(args.out)
    write_output(out, rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
