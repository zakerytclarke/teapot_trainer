"""Both phases of the real 109M config, with tiny token budgets.

Confirms compiled pretraining and eager SFT both finish, including one
behavior eval. This is the gate before ``train.py``.
"""

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teapottrainer import DatasetSpec, FINEWEB_EDU, SMOLTALK_APIGEN, SMOLTALK_MAGPIE, SQUAD_V2, TeapotTrainer
from train import build_config


def main():
    full = build_config()
    config = full.model_copy(
        update={
            "pretrain_datasets": [
                DatasetSpec(source=FINEWEB_EDU, target_tokens=280_000, weight=1.0),
            ],
            "sft_datasets": [
                DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=24_000, weight=1.0),
                DatasetSpec(source=SQUAD_V2, target_tokens=24_000, weight=1.0),
                DatasetSpec(source=SMOLTALK_APIGEN, target_tokens=24_000, weight=1.0),
            ],
            # End-of-phase eval only. SFT still scores once at the start.
            "eval_every_percent": 100.0,
        }
    )
    trainer = TeapotTrainer(config=config)
    trainer.eval_example_cap = 2
    stats = trainer.train(fresh=True)
    for name in ("pretrain", "sft"):
        phase = stats.get(name)
        if not phase or phase.get("first_loss") is None or phase.get("last_loss") is None:
            raise SystemExit(f"{name} produced no loss: {phase}")
        if not math.isfinite(phase["first_loss"]) or not math.isfinite(phase["last_loss"]):
            raise SystemExit(f"{name} loss was not finite: {phase}")
        if phase["tokens"] <= 0:
            raise SystemExit(f"{name} consumed no tokens")
    print(
        "smoke ok: "
        f"pretrain {stats['pretrain']['first_loss']:.4f} -> {stats['pretrain']['last_loss']:.4f} "
        f"at {stats['pretrain']['tok_per_sec']:.0f} tok/s; "
        f"sft {stats['sft']['first_loss']:.4f} -> {stats['sft']['last_loss']:.4f} "
        f"at {stats['sft']['tok_per_sec']:.0f} tok/s"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
