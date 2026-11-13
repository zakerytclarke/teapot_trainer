"""Cold-start SFT on Hermes calculator traces, then GRPO on the saved checkpoint.

teapot1alpha (and similar small models) often open ``<tool_call>`` but never emit
a valid JSON body, so GRPO stays at score 0. This recipe:

1. synthesizes multi-turn calculator conversations in the same Hermes wire format
   RL uses (``tool_format=\"hermes\"``)
2. CE-trains with ``TeapotTrainer`` from ``--init-from``
3. runs ``RlTrainer`` GRPO from the SFT run directory

Example (Colab / local)::

    python sft_then_rl_math.py \\
      --init-from teapotai/teapotllm \\
      --phase all \\
      --num-examples 4000 \\
      --sft-tokens 800000 \\
      --rl-episodes 200

``--phase sft`` stops after the supervised checkpoint. ``--phase rl`` skips SFT
and loads ``--sft-dir`` (or the path written by a prior run).
"""

from __future__ import annotations

import argparse
import json
import operator
import random
from pathlib import Path

from transformers import AutoConfig

from teapottrainer import (
    DatasetSourceConfig,
    DatasetSpec,
    ModelConfig,
    RlConfig,
    RlTrainer,
    SYNTH_MATH,
    TeapotTrainer,
    TrainingConfig,
    get_tool_format,
)
from teapottrainer.datasets import STOP_TOKEN_LOSS_WEIGHT, eos_loss_weight
from teapottrainer.formatting import messages_format
from teapottrainer.rl.tools import calculator, call_tool, tool_schemas


_OPS = [
    ("+", operator.add, "plus"),
    ("-", operator.sub, "minus"),
    ("*", operator.mul, "times"),
]

_SYSTEM = (
    "You are a helpful math assistant. "
    "For every arithmetic question you must call the calculator tool first. "
    "Do not answer from memory. After the tool returns, reply in natural language "
    "and include the final number."
)


def _model_config_from_checkpoint(path: str, seq_len: int) -> ModelConfig:
    """Match ``ModelConfig`` numbers to a Hub id or local checkpoint."""
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    hidden = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd"))
    layers = int(getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer"))
    heads = int(getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head"))
    kv = int(getattr(cfg, "num_key_value_heads", None) or heads)
    intermediate = int(getattr(cfg, "intermediate_size", None) or hidden * 4)
    positions = int(getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", seq_len))
    return ModelConfig(
        model_type="llama",
        hidden_size=hidden,
        layers=layers,
        heads=heads,
        kv_heads=kv,
        intermediate=intermediate,
        seq_len=min(int(seq_len), positions),
    )


def _question(mode: str, a: int, b: int, op_sym: str, op_word: str, expression: str) -> str:
    if mode == "word":
        return (
            f"What is {a} {op_word} {b}? "
            "You must use the calculator tool before answering. "
            "After the tool result, reply with a short assistant answer that includes the final number."
        )
    return (
        f"Compute {expression}. "
        "You must call the calculator tool with that expression before answering. "
        "Then give a brief natural-language answer that includes the result."
    )


def build_hermes_math_row(rng: random.Random, fmt) -> dict:
    """One multi-turn row: tool call, then prose answer after the tool result."""
    mode = rng.choice(["easy", "word"])
    a = int(rng.randint(12, 97))
    b = int(rng.randint(12, 97))
    op_sym, op_fn, op_word = _OPS[int(rng.randrange(len(_OPS)))]
    gold = float(op_fn(a, b))
    expression = f"{a} {op_sym} {b}"
    schemas = tool_schemas([calculator])
    system_msgs = fmt.inject_tools([{"role": "system", "content": _SYSTEM}], schemas)
    system = str(system_msgs[0]["content"])
    user = _question(mode, a, b, op_sym, op_word, expression)
    call_text = fmt.format_assistant_tool_calls(
        [{"name": "calculator", "arguments": {"expression": expression}}]
    )
    tool_value = call_tool(calculator, {"expression": expression})
    # ``messages_format`` only trains system/user/assistant turns, so the tool
    # observation is written as a user turn with the Hermes ``<tool_response>``
    # body — the same text RL injects for named formats.
    tool_as_user = fmt.format_tool_result("calculator", tool_value)
    if float(gold).is_integer():
        answer = f"The result is {int(gold)}."
    else:
        answer = f"The result is {gold}."
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": call_text},
            {"role": "user", "content": tool_as_user},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"expression": expression, "value": gold, "mode": mode},
    }


def write_sft_jsonl(path: Path, num_examples: int, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fmt = get_tool_format("hermes")
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(int(num_examples)):
            handle.write(json.dumps(build_hermes_math_row(rng, fmt), ensure_ascii=False) + "\n")
    print(f"Wrote {num_examples} Hermes calculator rows to {path}")
    return path


def run_sft(
    *,
    init_from: str,
    tokenizer: str,
    data_path: Path,
    sft_tokens: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seq_len: int,
    seed: int,
    chat_format: str,
) -> str:
    """CE-train on the synthetic Hermes traces. Returns the run directory."""
    model = _model_config_from_checkpoint(init_from, seq_len)
    source = DatasetSourceConfig(
        name=str(data_path.resolve()),
        split="train",
        is_sft=True,
        chat_template_fn=messages_format,
        token_loss_weight_fn=eos_loss_weight(STOP_TOKEN_LOSS_WEIGHT),
    )
    config = TrainingConfig(
        model=model,
        init_from=init_from,
        tokenizer=tokenizer or init_from,
        chat_format=chat_format,
        pretrain_datasets=[],
        bridge_datasets=[],
        sft_datasets=[DatasetSpec(source=source, target_tokens=int(sft_tokens))],
        eval_datasets=[],
        batch_size=int(batch_size),
        grad_accum=int(grad_accum),
        lr=float(lr),
        sft_lr=float(lr),
        compile=False,
        eval_every_percent=50.0,
        holdout_ratio=0.02,
        seed=int(seed),
    )
    trainer = TeapotTrainer(config)
    trainer.train(fresh=True)
    print(f"SFT checkpoint: {trainer.output_dir}")
    return trainer.output_dir


def run_rl(
    *,
    init_from: str,
    tokenizer: str,
    tool_format: str,
    max_train_episodes: int,
    group_size: int,
    max_new_tokens: int,
    eval_every_episodes: int,
    seed: int,
) -> str:
    config = RlConfig(
        init_from=init_from,
        tokenizer=tokenizer or init_from,
        tool_format=tool_format,
        datasets=[SYNTH_MATH],
        eval_datasets=[],
        algorithm="grpo",
        group_size=int(group_size),
        batch_prompts=1,
        max_turns=3,
        max_new_tokens=int(max_new_tokens),
        temperature=0.8,
        lr=5e-7,
        kl_coef=0.02,
        max_train_episodes=int(max_train_episodes),
        eval_every_episodes=int(eval_every_episodes),
        eval_episodes=16,
        resume_every_episodes=int(eval_every_episodes),
        save_rollouts=True,
        seed=int(seed),
    )
    run_dir = RlTrainer(config).train(fresh=True)
    print(f"RL run complete: {run_dir}")
    return str(run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--init-from",
        default="teapotai/teapotllm",
        help="Hub id or local dir for SFT start (default: teapotai/teapotllm). Use your teapot1alpha path here.",
    )
    parser.add_argument("--tokenizer", default="", help="Tokenizer id/dir; defaults to --init-from / --sft-dir.")
    parser.add_argument("--phase", choices=("all", "sft", "rl", "data"), default="all")
    parser.add_argument("--data-path", default="data/synth_math_hermes.jsonl")
    parser.add_argument("--num-examples", type=int, default=4000)
    parser.add_argument("--sft-tokens", type=int, default=800_000)
    parser.add_argument("--sft-dir", default="", help="Existing SFT run dir for --phase rl.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--sft-lr", type=float, default=2e-5)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--chat-format", default="plain", help="Must match how RL prompts look for your model.")
    parser.add_argument("--tool-format", default="hermes")
    parser.add_argument("--rl-episodes", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-every-episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)

    if args.phase in {"all", "sft", "data"}:
        write_sft_jsonl(data_path, args.num_examples, args.seed)
    if args.phase == "data":
        return

    sft_dir = args.sft_dir
    if args.phase in {"all", "sft"}:
        sft_dir = run_sft(
            init_from=args.init_from,
            tokenizer=args.tokenizer or args.init_from,
            data_path=data_path,
            sft_tokens=args.sft_tokens,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.sft_lr,
            seq_len=args.seq_len,
            seed=args.seed,
            chat_format=args.chat_format,
        )
        marker = Path("runs/last_sft_math_tools.txt")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(sft_dir + "\n", encoding="utf-8")
        print(f"Wrote {marker} -> {sft_dir}")

    if args.phase in {"all", "rl"}:
        if not sft_dir:
            marker = Path("runs/last_sft_math_tools.txt")
            if marker.is_file():
                sft_dir = marker.read_text(encoding="utf-8").strip()
        if not sft_dir:
            raise SystemExit("No SFT dir. Pass --sft-dir or run --phase sft / all first.")
        run_rl(
            init_from=sft_dir,
            tokenizer=args.tokenizer or sft_dir,
            tool_format=args.tool_format,
            max_train_episodes=args.rl_episodes,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            eval_every_episodes=args.eval_every_episodes,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
