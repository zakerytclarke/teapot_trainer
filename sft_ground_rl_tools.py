"""SFT ground on SynthQA + TeapotChat, then RL on math + open web search.

Supervised phase (no tools needed):
  * teapotai/synthqa — context-only answers / exact abstain
  * teapotai/teapot-chat — assistant chat style

RL phase:
  * SYNTH_MATH — calculator
  * GSM8K_RL — calculator
  * NQ_OPEN / WebQuestions / Hotpot / TriviaQA — Brave/Exa/LLM-search
  Web answers are graded containment-first (surrounding prose is OK).

Example::

    python sft_ground_rl_tools.py --init-from Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import os

from transformers import AutoConfig

from teapottrainer import (
    DatasetSpec,
    GSM8K_RL,
    GSM8K_TEST,
    HOTPOT_RL,
    ModelConfig,
    NQ_OPEN_RL,
    RlConfig,
    RlTrainer,
    SearchSettings,
    SYNTH_MATH,
    SYNTHQA,
    SYNTHQA_TEST,
    TEAPOT_CHAT,
    TRIVIA_QA_RL,
    TeapotTrainer,
    TrainingConfig,
    WEB_QUESTIONS_RL,
    bind_tools,
    build_search_tools,
)


def _model_config(path: str, seq_len: int) -> ModelConfig:
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return ModelConfig(
        model_type="llama",
        hidden_size=int(cfg.hidden_size),
        layers=int(cfg.num_hidden_layers),
        heads=int(cfg.num_attention_heads),
        kv_heads=int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        intermediate=int(getattr(cfg, "intermediate_size", cfg.hidden_size * 4)),
        seq_len=min(int(seq_len), int(getattr(cfg, "max_position_embeddings", seq_len))),
    )


def run_sft(
    *,
    init_from: str,
    chat_format: str,
    synthqa_tokens: int,
    chat_tokens: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seq_len: int,
    seed: int,
) -> str:
    config = TrainingConfig(
        model=_model_config(init_from, seq_len),
        init_from=init_from,
        tokenizer=init_from,
        chat_format=chat_format,
        pretrain_datasets=[],
        bridge_datasets=[],
        sft_datasets=[
            DatasetSpec(source=SYNTHQA, target_tokens=int(synthqa_tokens), weight=1.0),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=int(chat_tokens), weight=1.0),
        ],
        eval_datasets=[DatasetSpec(source=SYNTHQA_TEST, target_tokens=0)],
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
    max_train_episodes: int,
    mock_search: bool | None,
    seed: int,
) -> str:
    if mock_search is None:
        mock_search = not any(
            os.environ.get(name)
            for name in (
                "BRAVE_API_KEY",
                "BRAVE_SEARCH_API_KEY",
                "EXA_API_KEY",
                "LLM_SEARCH_API_KEY",
                "OPENAI_API_KEY",
            )
        )
    search = SearchSettings.from_env(mock=bool(mock_search))
    tools = build_search_tools(search)
    if not tools:
        raise SystemExit("No search tools. Set API keys or pass --mock-search.")
    print(f"search tools: {[fn.__name__ for fn in tools]}  mock={search.mock}")

    config = RlConfig(
        init_from=init_from,
        tokenizer=init_from,
        tool_format="hermes",
        search=search,
        datasets=[
            SYNTH_MATH,
            GSM8K_RL,
            bind_tools(NQ_OPEN_RL, tools),
            bind_tools(WEB_QUESTIONS_RL, tools),
            bind_tools(HOTPOT_RL, tools),
            bind_tools(TRIVIA_QA_RL, tools),
        ],
        eval_datasets=[
            DatasetSpec(source=GSM8K_TEST, target_tokens=0),
            DatasetSpec(source=SYNTHQA_TEST, target_tokens=0),
        ],
        eval_example_cap=16,
        algorithm="grpo",
        group_size=4,
        batch_prompts=1,
        max_turns=4,
        max_new_tokens=160,
        temperature=0.8,
        lr=5e-7,
        kl_coef=0.02,
        max_train_episodes=int(max_train_episodes),
        eval_every_episodes=25,
        eval_episodes=8,
        resume_every_episodes=25,
        save_rollouts=True,
        seed=int(seed),
    )
    run_dir = RlTrainer(config).train(fresh=True)
    print(f"RL run complete: {run_dir}")
    return str(run_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-from", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--chat-format", default="chatml")
    p.add_argument("--phase", choices=("all", "sft", "rl"), default="all")
    p.add_argument("--sft-dir", default="", help="Existing SFT dir for --phase rl")
    p.add_argument("--synthqa-tokens", type=int, default=600_000)
    p.add_argument("--chat-tokens", type=int, default=400_000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--sft-lr", type=float, default=2e-5)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--rl-episodes", type=int, default=200)
    p.add_argument("--mock-search", action="store_true", help="Force stub search tools")
    p.add_argument("--no-mock-search", action="store_true", help="Require real API keys")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mock = True if args.mock_search else (False if args.no_mock_search else None)
    sft_dir = args.sft_dir
    if args.phase in {"all", "sft"}:
        sft_dir = run_sft(
            init_from=args.init_from,
            chat_format=args.chat_format,
            synthqa_tokens=args.synthqa_tokens,
            chat_tokens=args.chat_tokens,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.sft_lr,
            seq_len=args.seq_len,
            seed=args.seed,
        )
    if args.phase in {"all", "rl"}:
        if not sft_dir:
            raise SystemExit("Pass --sft-dir or run --phase sft/all first.")
        run_rl(
            init_from=sft_dir if args.phase == "all" else (args.sft_dir or sft_dir),
            max_train_episodes=args.rl_episodes,
            mock_search=mock,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
