"""Generate teacher tool-use traces (no grading).

Runs a strong instruct model (default ``Qwen/Qwen3.8-27B``) with a **random
subset** of tools on general assistant / QnA tasks (no coding). Each row stores
the prompt, full agent trace, golds, and tool metadata — filter/score later with
``grade_tool_traces.py``.

Sources: GSM8K, NQ-Open, WebQuestions, TriviaQA, Hotpot, SQuAD v2.

Example (full multi-day run)::

    python generate_tool_traces.py \\
      --model Qwen/Qwen3.8-27B \\
      --load-in-4bit \\
      --per-dataset 0 \\
      --max-samples 0 \\
      --out tool_traces.jsonl \\
      --flush-every 10
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
from teapottrainer.rl.tools import ToolFn, final_assistant_text

CONTEXT_TAG_OPEN = "[INJECTED_CONTEXT]"
CONTEXT_TAG_CLOSE = "[/INJECTED_CONTEXT]"


def build_system_prompt(*, today: str | None = None) -> str:
    day = today or date.today().isoformat()
    return f"""You are a careful tool-using assistant. Today is {day} (ISO date).

Evidence rules (strict):
- Only state facts that appear in tool results or in {CONTEXT_TAG_OPEN}...{CONTEXT_TAG_CLOSE} blocks.
- {CONTEXT_TAG_OPEN} blocks are machine-injected evidence, not the user speaking.
- If the available evidence is insufficient, refuse briefly (e.g. "{ABSTAIN_TEXT}") instead of guessing.
- Never invent biographies, numbers, dates, or citations.

Tool rules:
- calculator — use for arithmetic when available.
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


@dataclass
class EpisodeSpec:
    sample_id: str
    dataset: str
    task_type: str  # calc | search | context_qa | browser
    question: str
    gold_answers: list[str]
    messages_input: list[dict[str, Any]]
    # Intended tools for *later* grading (only enforced if present in tools_available).
    required_tools: list[str] = field(default_factory=list)
    unanswerable: bool = False
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


def _load_rows(name: str, *, subset: str | None = None, split: str = "train", limit: int | None = None) -> list[dict]:
    if subset:
        ds = load_dataset(name, subset, split=split)
    else:
        ds = load_dataset(name, split=split)
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


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
        required_tools=["calculator"],
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
    answers = _as_answers(
        (row.get("answers") or {}).get("text") if isinstance(row.get("answers"), dict) else row.get("answers")
    )
    unanswerable = len(answers) == 0 or bool(row.get("is_impossible"))
    gold = [ABSTAIN_TEXT] if unanswerable else answers
    return EpisodeSpec(
        sample_id=f"squad_v2:{idx}",
        dataset="squad_v2",
        task_type="context_qa",
        question=question,
        gold_answers=gold,
        messages_input=_messages_for_question(system, question, injected_context=context),
        required_tools=[],
        unanswerable=unanswerable,
        injected_context=context,
    )


def build_episode_pool(system: str, *, per_dataset: int, seed: int) -> list[EpisodeSpec]:
    """``per_dataset <= 0`` means use the full train split for each source."""
    rng = random.Random(seed)
    limit = None if per_dataset <= 0 else int(per_dataset)
    episodes: list[EpisodeSpec] = []

    for i, row in enumerate(_load_rows("openai/gsm8k", subset="main", limit=limit)):
        episodes.append(make_gsm8k_episode(row, system, i))

    for i, row in enumerate(_load_rows("google-research-datasets/nq_open", limit=limit)):
        episodes.append(
            make_search_episode(
                dataset="nq_open",
                sample_id=f"nq_open:{i}",
                question=str(row.get("question") or "").strip(),
                answers=_as_answers(row.get("answer") or row.get("answers")),
                system=system,
            )
        )

    for i, row in enumerate(_load_rows("stanfordnlp/web_questions", limit=limit)):
        episodes.append(
            make_search_episode(
                dataset="web_questions",
                sample_id=f"web_questions:{i}",
                question=str(row.get("question") or "").strip(),
                answers=_as_answers(row.get("answers") or row.get("answer")),
                system=system,
            )
        )

    for i, row in enumerate(_load_rows("mandarjoshi/trivia_qa", subset="rc.nocontext", limit=limit)):
        blob = row.get("answer") or {}
        ans: list[str] = []
        if isinstance(blob, dict):
            ans.extend(_as_answers(blob.get("aliases")))
            ans.extend(_as_answers(blob.get("normalized_aliases")))
            ans.extend(_as_answers(blob.get("value") or blob.get("normalized_value")))
        else:
            ans.extend(_as_answers(blob))
        seen: set[str] = set()
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
                question=str(row.get("question") or "").strip(),
                answers=uniq,
                system=system,
            )
        )

    for i, row in enumerate(_load_rows("hotpotqa/hotpot_qa", subset="distractor", limit=limit)):
        episodes.append(
            make_search_episode(
                dataset="hotpot",
                sample_id=f"hotpot:{i}",
                question=str(row.get("question") or "").strip(),
                answers=_as_answers(row.get("answer")),
                system=system,
            )
        )

    for i, row in enumerate(_load_rows("rajpurkar/squad_v2", limit=limit)):
        episodes.append(make_squad_episode(row, system, i))

    rng.shuffle(episodes)
    return episodes


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not serializable: {type(obj)!r}")


def row_to_flat(row: dict[str, Any]) -> dict[str, Any]:
    return {
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
        "injected_context": row.get("injected_context") or "",
        "model": row["model"],
        "tool_format": row["tool_format"],
        "unanswerable": bool(row["unanswerable"]),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row_to_flat(r) for r in rows]), path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--tool-format", default="hermes")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument(
        "--per-dataset",
        type=int,
        default=0,
        help="Rows per source (0 = full train split)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap total episodes after shuffle (0 = no cap)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--mock-search", action="store_true")
    p.add_argument("--mock-browser", action="store_true")
    p.add_argument("--out", default="tool_traces.jsonl", help="Prefer .jsonl for long runs (append-safe)")
    p.add_argument("--flush-every", type=int, default=10, help="Also rewrite parquet snapshot every N rows when --out ends with .parquet")
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
    out = Path(args.out)

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

    # For long runs, always also append jsonl next to parquet.
    jsonl_path = out if out.suffix.lower() == ".jsonl" else out.with_suffix(".jsonl")
    if jsonl_path.exists():
        print(f"appending to existing {jsonl_path}")

    rows: list[dict[str, Any]] = []
    written = 0
    try:
        for i, spec in enumerate(pool):
            tools = sample_tool_bundle(
                rng,
                search_tools=search_tools,
                browser_tools=toolkit.tools,
            )
            available = tool_names(tools)
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

            row = {
                "sample_id": spec.sample_id,
                "dataset": spec.dataset,
                "task_type": spec.task_type,
                "question": spec.question,
                "gold_answers": spec.gold_answers,
                "tools_available": available,
                "tools_required": list(spec.required_tools),
                "messages_input": spec.messages_input,
                "messages_trace": trace,
                "final_assistant": final_assistant_text(trace),
                "injected_context": spec.injected_context,
                "model": args.model,
                "tool_format": args.tool_format,
                "unanswerable": spec.unanswerable,
            }
            append_jsonl(jsonl_path, row)
            rows.append(row)
            written += 1
            print(f"  wrote ({written}) final={row['final_assistant'][:120]!r}")

            if out.suffix.lower() == ".parquet" and args.flush_every > 0 and written % args.flush_every == 0:
                write_parquet(out, rows)
                print(f"  parquet snapshot -> {out}")
    finally:
        toolkit.close()

    if out.suffix.lower() == ".parquet":
        write_parquet(out, rows)
    print(f"done: {written} traces -> {jsonl_path}" + (f" and {out}" if out.suffix.lower() == ".parquet" else ""))


if __name__ == "__main__":
    main()
