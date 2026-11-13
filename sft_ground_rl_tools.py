"""RL-only recipe: tool tasks + SynthQA/TeapotChat exact-match keepalives.

Assumes the model already saw SynthQA / TeapotChat in SFT. This run mixes them
into GRPO alongside calculator math, GSM8K, and open-web search so those skills
do not regress. Keepalive datasets use ``tools=[]`` and exact answer match.

Example::

    python sft_ground_rl_tools.py --init-from Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse

from rl_search import build_full_rl_config
from teapottrainer import RlTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-from", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--rl-episodes", type=int, default=200)
    p.add_argument("--keepalive-weight", type=float, default=1.0)
    p.add_argument("--mock-search", action="store_true")
    p.add_argument("--no-mock-search", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mock = True if args.mock_search else (False if args.no_mock_search else None)
    config = build_full_rl_config(
        init_from=args.init_from,
        tokenizer=args.tokenizer or None,
        max_train_episodes=args.rl_episodes,
        mock_search=mock,
        keepalive_weight=args.keepalive_weight,
    )
    run_dir = RlTrainer(config).train(fresh=True)
    print(f"RL run complete: {run_dir}")


if __name__ == "__main__":
    main()
