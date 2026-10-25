"""Short from-scratch pretrain. Confirms the loop learns, then exits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teapottrainer import DatasetSpec, FINEWEB_EDU, ModelConfig, TeapotTrainer, TrainingConfig


def main():
    config = TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=256,
            max_position_embeddings=256,
        ),
        pretrain_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=200_000, weight=1.0),
        ],
        sft_datasets=[],
        eval_datasets=[],
        tokenizer="gpt2",
        batch_size=8,
        grad_accum=2,
        lr=1e-3,
        weight_decay=0.1,
        eval_every_percent=1000.0,
        compile=False,
        seed=42,
    )
    trainer = TeapotTrainer(config=config)
    trainer.qualitative = False
    stats = trainer.train(fresh=True)
    phase = stats["pretrain"]
    print("SMOKE", phase)
    if phase["first_loss"] is None or phase["last_loss"] is None:
        raise SystemExit("smoke produced no loss")
    if not phase["last_loss"] < phase["first_loss"]:
        raise SystemExit(
            f"loss did not fall: {phase['first_loss']:.4f} -> {phase['last_loss']:.4f}"
        )
    print(
        f"smoke ok: loss {phase['first_loss']:.4f} -> {phase['last_loss']:.4f} "
        f"at {phase['tok_per_sec']:.0f} tok/s"
    )


if __name__ == "__main__":
    main()
