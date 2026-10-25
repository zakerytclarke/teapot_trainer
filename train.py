#!/usr/bin/env python3
"""Train a small local agent, or pretrain the base model it starts from.

The library stays generic. This script is the use case: a deep-thin Llama
pretrained on the local Ultra-FineWeb token stream, then supervised on chat,
tool calls, tool restraint, and grounded answers that know when to abstain.

Short runs:

    python train.py --smoke
    python train.py --confidence-pretrain
    python train.py --confidence-sft

Full runs, once those metrics look right:

    python train.py --pretrain --model 100m --tokens 2000000000
    python train.py --pretrain --model 350m --tokens 20000000000
    python train.py --sft --init runs/pretrain-100m/checkpoints/best --tokens 50000000

Pretrain reads ../teapot_foundation/tokenized, or TEAPOT_TOKEN_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from teapottrainer import (
    GenerateEval,
    ListMessages,
    MemmapTokens,
    Stage,
    TaskEval,
    TeapotTrainer,
    TokenPerplexity,
    TrainConfig,
    preset,
    rag_messages,
    tool_messages,
)

SYSTEM = (
    "You are a local assistant. When a tool is required, reply with exactly one "
    '<tool_call>{"name":"...","arguments":{...}}</tool_call> and no other text. '
    "If no tool is required, answer in plain text and do not call a tool. "
    "If context is provided and it does not contain the answer, reply exactly: "
    "The context does not contain the answer."
)

ABSTAIN = "The context does not contain the answer."
NO_TOOL = "I can answer without using a tool."

TOOLS = [
    {"name": "get_weather", "description": "Current weather for a city."},
    {"name": "calculator", "description": "Arithmetic on two numbers. op is + or - or *."},
    {"name": "create_event", "description": "Add a calendar event with a title and a day."},
    {"name": "search_docs", "description": "Search the local knowledge base."},
]

TRAIN_CITIES = [
    "Lisbon", "Osaka", "Naples", "Quebec", "Bergen", "Tunis", "Malmo", "Pune",
    "Accra", "Basel", "Dakar", "Haifa", "Kyoto", "Lima", "Minsk", "Nice",
    "Omaha", "Perm", "Quito", "Riga", "Sofia", "Tulsa", "Umea", "Valencia",
    "Windhoek", "Xian", "Yokohama", "Zagreb",
]
EVAL_CITIES = ["Reykjavik", "Hobart", "Salvador", "Inverness", "Gaborone", "Antwerp", "Busan", "Odessa"]

TRAIN_NAMES = ["Amira", "Boris", "Chioma", "Dmitri", "Elena", "Farid", "Greta", "Hiro"]
EVAL_NAMES = ["Quinlan", "Saskia", "Taavi", "Ulrika"]
TRAIN_ITEMS = ["brass compass", "cedar flute", "indigo notebook", "copper kettle"]
EVAL_ITEMS = ["silver thimble", "maple astrolabe"]
TRAIN_PLACES = ["kitchen drawer", "library attic", "boat locker", "garden shed"]
EVAL_PLACES = ["observatory desk", "lighthouse cupboard"]


def token_dir() -> Path:
    env = os.environ.get("TEAPOT_TOKEN_DIR")
    if env:
        path = Path(env)
    else:
        path = Path(__file__).resolve().parent.parent / "teapot_foundation" / "tokenized"
    needed = ["train.bin", "val.bin", "test.bin", "tokenizer"]
    missing = [name for name in needed if not (path / name).exists()]
    if missing:
        raise SystemExit(f"{path} is missing {missing}. Set TEAPOT_TOKEN_DIR.")
    return path


def _tool_row(user: str, call: dict | None) -> list[dict]:
    return tool_messages(user, TOOLS, call, system=SYSTEM, abstain=NO_TOOL)


def _rag_row(passages: list[str], question: str, answer: str | None) -> list[dict]:
    return rag_messages(passages, question, answer, system=SYSTEM, abstain=ABSTAIN)


def _prompt_only(messages: list[dict]) -> list[dict]:
    return [msg for msg in messages if msg["role"] != "assistant"]


def build_agent_rows(seed: int = 0) -> tuple[list, list]:
    """Train chats and held-out scored prompts. Eval entities never appear in train."""
    rng = random.Random(seed)
    train: list[list[dict]] = []
    evals: list[dict] = []

    for city in TRAIN_CITIES:
        train.append(_tool_row(f"What is the weather in {city}?", {"name": "get_weather", "arguments": {"city": city}}))
        train.append(_tool_row(
            f"The weather in {city} is already 11C and rainy. Should I wear a coat? Answer in one sentence.",
            None,
        ))
        train.append(_tool_row(f"Write a two-line haiku about {city}. Do not use a tool.", None))

    for city in EVAL_CITIES:
        called = _tool_row(f"What is the weather in {city}?", {"name": "get_weather", "arguments": {"city": city}})
        evals.append({
            "messages": _prompt_only(called),
            "expect": {"kind": "tool", "name": "get_weather", "arguments": {"city": city}, "metric_prefix": "tools"},
        })
        restrained = _tool_row(
            f"Don't check the weather in {city}. Just say good morning.",
            None,
        )
        evals.append({
            "messages": _prompt_only(restrained),
            "expect": {"kind": "no_tool", "metric": "tools/restraint"},
        })

    for a, b in ((i, i + 3) for i in range(3, 28)):
        train.append(_tool_row(
            f"What is {a} * {b}? Use the calculator.",
            {"name": "calculator", "arguments": {"a": a, "b": b, "op": "*"}},
        ))
        train.append(_tool_row(f"What is {a} + {b}? Use the calculator.", {"name": "calculator", "arguments": {"a": a, "b": b, "op": "+"}}))
    for a, b in ((60, 7), (64, 9), (71, 4), (78, 6)):
        called = _tool_row(f"What is {a} * {b}? Use the calculator.", {"name": "calculator", "arguments": {"a": a, "b": b, "op": "*"}})
        evals.append({
            "messages": _prompt_only(called),
            "expect": {"kind": "tool", "name": "calculator", "arguments": {"a": a, "b": b, "op": "*"}, "metric_prefix": "tools"},
        })

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for i, title in enumerate(["dentist", "standup", "grocery run", "piano lesson", "code review"]):
        day = days[i % len(days)]
        train.append(_tool_row(
            f"Schedule {title} on {day}.",
            {"name": "create_event", "arguments": {"title": title, "day": day}},
        ))
    for title, day in (("patent review", "Saturday"), ("greenhouse visit", "Sunday")):
        called = _tool_row(
            f"Schedule {title} on {day}.",
            {"name": "create_event", "arguments": {"title": title, "day": day}},
        )
        evals.append({
            "messages": _prompt_only(called),
            "expect": {"kind": "tool", "name": "create_event", "arguments": {"title": title, "day": day}, "metric_prefix": "tools"},
        })

    for query in ["rope scaling notes", "kv cache layout", "bf16 overflow", "tokenizer training"]:
        train.append(_tool_row(f"Search the docs for {query}.", {"name": "search_docs", "arguments": {"query": query}}))
    called = _tool_row("Search the docs for sliding window attention.", {"name": "search_docs", "arguments": {"query": "sliding window attention"}})
    evals.append({
        "messages": _prompt_only(called),
        "expect": {"kind": "tool", "name": "search_docs", "arguments": {"query": "sliding window attention"}, "metric_prefix": "tools"},
    })
    plain = _tool_row("What tools do you have? Answer in one sentence and do not call one.", None)
    evals.append({"messages": _prompt_only(plain), "expect": {"kind": "no_tool", "metric": "tools/restraint"}})

    for name in TRAIN_NAMES:
        for item in TRAIN_ITEMS:
            place = rng.choice(TRAIN_PLACES)
            fact = f"{name} keeps a {item} in the {place}."
            other = f"{rng.choice(TRAIN_NAMES)} paints maps on Sundays."
            row = _rag_row([fact, other], f"What does {name} keep in the {place}?", fact)
            train.append(row)
    for name in EVAL_NAMES:
        for item, place in zip(EVAL_ITEMS, EVAL_PLACES):
            fact = f"{name} keeps a {item} in the {place}."
            distractor = "The reading room closes at dusk."
            row = _rag_row([fact, distractor], f"What does {name} keep in the {place}?", fact)
            evals.append({
                "messages": _prompt_only(row),
                "expect": {"kind": "contains", "text": fact, "metric": "rag/answer"},
            })
            abstain = _rag_row(
                [fact, distractor],
                "How many moons does Mars have?",
                None,
            )
            evals.append({
                "messages": _prompt_only(abstain),
                "expect": {"kind": "phrase", "text": ABSTAIN, "metric": "rag/abstain"},
            })

    for user, answer in (
        ("Say hello in one short sentence.", "Hello. How can I help?"),
        ("What is two plus two? Answer with the digit only and do not call a tool.", "4"),
        ("Name the color of a clear daytime sky in one word.", "blue"),
    ):
        train.append([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ])
    evals.append({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Reply with the single word pomegranate."},
        ],
        "expect": {"kind": "contains", "text": "pomegranate", "metric": "chat/contains"},
    })

    rng.shuffle(train)
    return train, evals


def pretrain_config(model_name: str, tokens: int, output: str, smoke: bool = False) -> TrainConfig:
    data = token_dir()
    if smoke:
        micro, accum, seq, lr, warmup = 16, 1, 128, 1e-3, 5
        eval_tokens, val_tokens, every = 32_768, 32_768, 32_768
        spec = preset("tiny")
        spec.max_position_embeddings = max(spec.max_position_embeddings, seq)
    elif model_name == "350m":
        micro, accum, seq, lr, warmup = 8, 8, 4096, 8e-4, 300
        eval_tokens, val_tokens, every = None, 1_048_576, None
        spec = preset("350m")
    else:
        micro, accum, seq, lr, warmup = 8, 4, 4096, 1e-3, 300
        eval_tokens, val_tokens, every = None, 1_048_576, None
        spec = preset("100m" if model_name == "100m" else model_name)
        if model_name == "100m":
            spec = preset("100m")
    if smoke:
        pass
    return TrainConfig(
        model=spec,
        tokenizer=str(data / "tokenizer"),
        output_dir=output,
        micro_batch=micro,
        grad_accum=accum,
        seq_len=seq,
        lr=lr,
        warmup_steps=5 if tokens < 50_000_000 else warmup,
        weight_decay=0.1,
        eval_every_tokens=every if smoke else max(tokens // 5, seq),
        eval_at_start=True,
        log_every=1 if smoke or tokens <= 20_000_000 else 10,
        local_files_only=True,
        stages=[
            Stage(
                name="pretrain",
                target_tokens=tokens,
                sources=[MemmapTokens(str(data / "train.bin"), name="ultrafineweb")],
                evals=[
                    TokenPerplexity(str(data / "val.bin"), name="val", max_tokens=val_tokens if smoke else 524_288, batch_size=4),
                    TokenPerplexity(str(data / "test.bin"), name="test", max_tokens=val_tokens // 2 if smoke else 262_144, batch_size=4),
                    GenerateEval(["The capital city of France is", "Once upon a time"], max_new_tokens=16 if smoke else 32),
                ],
            )
        ],
    )


def sft_config(init: str, tokens: int, output: str, rows: list, evals: list, seq_len: int, micro: int, accum: int) -> TrainConfig:
    return TrainConfig(
        init_from=init,
        tokenizer=init,
        output_dir=output,
        micro_batch=micro,
        grad_accum=accum,
        seq_len=seq_len,
        lr=2e-5,
        min_lr_frac=0.1,
        warmup_steps=20,
        weight_decay=0.0,
        eval_every_tokens=max(tokens // 4, 1),
        eval_at_start=True,
        log_every=5,
        local_files_only=True,
        gradient_checkpointing=True,
        stages=[
            Stage(
                name="sft",
                target_tokens=tokens,
                lr=2e-5,
                sources=[ListMessages(rows, name="agent-mix")],
                evals=[
                    TaskEval(evals, name="agent", max_new_tokens=40, batch_size=8),
                ],
            )
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Pretrain or agent-SFT a small language model.")
    parser.add_argument("--smoke", action="store_true", help="Tiny model, a minute or less, checks the loop.")
    parser.add_argument("--confidence-pretrain", action="store_true", help="100M model, a few million tokens.")
    parser.add_argument("--confidence-sft", action="store_true", help="Short Qwen2.5-0.5B agent SFT with before/after scores.")
    parser.add_argument("--pretrain", action="store_true", help="From-scratch pretrain. This is the long run.")
    parser.add_argument("--sft", action="store_true", help="Agent SFT. Defaults to a multi-epoch run, not the short one.")
    parser.add_argument("--model", default="100m", choices=["tiny", "100m", "350m"])
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--init", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    modes = [args.smoke, args.confidence_pretrain, args.confidence_sft, args.pretrain, args.sft]
    if sum(bool(m) for m in modes) != 1:
        parser.error("choose exactly one of --smoke, --confidence-pretrain, --confidence-sft, --pretrain, --sft")

    if args.smoke:
        cfg = pretrain_config("tiny", args.tokens or 65_536, args.output or "runs/smoke", smoke=True)
    elif args.confidence_pretrain:
        cfg = pretrain_config("100m", args.tokens or 4_000_000, args.output or "runs/confidence-pretrain", smoke=False)
        cfg.warmup_steps = 4
        cfg.micro_batch = 8
        cfg.grad_accum = 4
        cfg.log_every = 1
        cfg.eval_every_tokens = 2_000_000
    elif args.pretrain:
        tokens = args.tokens or (20_000_000_000 if args.model == "350m" else 2_000_000_000)
        cfg = pretrain_config(args.model, tokens, args.output or f"runs/pretrain-{args.model}", smoke=False)
    else:
        rows, evals = build_agent_rows()
        short = args.confidence_sft
        tokens = args.tokens or (80_000 if short else 5_000_000)
        output = args.output or ("runs/confidence-sft" if short else "runs/sft-agent")
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "system_prompt.txt").write_text(SYSTEM + "\n", encoding="utf-8")
        (out / "eval_tasks.json").write_text(json.dumps(evals, indent=2) + "\n", encoding="utf-8")
        print(f"agent mix: {len(rows)} train rows, {len(evals)} eval prompts", flush=True)
        cfg = sft_config(
            init=args.init,
            tokens=tokens,
            output=output,
            rows=rows,
            evals=evals,
            seq_len=256,
            micro=8,
            accum=2 if short else 4,
        )
        if short:
            cfg.warmup_steps = 10
            cfg.log_every = 5

    TeapotTrainer(cfg).train(resume=args.resume)


if __name__ == "__main__":
    main()
