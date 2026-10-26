"""Corrective assistant run.

Starts from the finished pretrain-plus-SFT checkpoint and continues on one
tool template, answerable-heavy passages, and a smaller chat replay.
SynthQA and Teapot chat are in both the supervised mix and the evals.
"""

from pathlib import Path

from teapottrainer import (
    DatasetSpec,
    FINEWEB_EDU,
    GSM8K,
    GSM8K_TEST,
    HOTPOT,
    HOTPOT_VAL,
    ModelConfig,
    SMOLTALK_APIGEN_ALIASED,
    SMOLTALK_APIGEN_ALIASED_TEST,
    SMOLTALK_APIGEN_DECISION,
    SMOLTALK_APIGEN_DECISION_TEST,
    SMOLTALK_CONSTRAINTS,
    SMOLTALK_CONSTRAINTS_TEST,
    SMOLTALK_MAGPIE,
    SQUAD_V2_ANSWERABLE,
    SQUAD_V2_UNANSWERABLE,
    SQUAD_V2_VAL,
    SYNTHQA,
    SYNTHQA_TEST,
    TEAPOT_CHAT,
    TeapotTrainer,
    TrainingConfig,
    WHEN2CALL,
    WHEN2CALL_TEST,
)

# The best checkpoint from the 2B pretrain plus the first supervised run.
INIT_FROM = Path(__file__).resolve().parent / "runs" / "teapot-gpt2-770ea2" / "checkpoints" / "best"


def build_config() -> TrainingConfig:
    # Same 109M Llama. This run does not pretrain again.
    # Tool calls, including renamed APIGen tools and GSM as calculator calls,
    # share one system template. SQuAD answerable tokens are 4x the unanswerable ones.
    return TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=512,
            num_hidden_layers=26,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=1408,
            max_position_embeddings=2048,
        ),
        pretrain_datasets=[],
        sft_datasets=[
            # Positive calls on the shared decision template.
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=36_000_000, weight=1.0),
            # Same calls with the function names replaced, so the name has to be read.
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED, target_tokens=12_000_000, weight=1.0),
            # Ask or refuse. This split has almost no positive calls.
            DatasetSpec(source=WHEN2CALL, target_tokens=8_000_000, weight=1.0),
            # Grade-school problems as calculator calls.
            DatasetSpec(source=GSM8K, target_tokens=6_000_000, weight=1.0),
            # Passages. Answerable rows outnumber abstentions four to one.
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=16_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=8_000_000, weight=1.0),
            DatasetSpec(source=SYNTHQA, target_tokens=6_000_000, weight=1.0),
            # A smaller replay of the assistant voice.
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=14_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=8_000_000, weight=1.0),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=4_000_000, weight=1.0),
        ],
        eval_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=1_000_000),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS_TEST, target_tokens=200_000),
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION_TEST, target_tokens=200_000),
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED_TEST, target_tokens=200_000),
            DatasetSpec(source=WHEN2CALL_TEST, target_tokens=200_000),
            DatasetSpec(source=SQUAD_V2_VAL, target_tokens=200_000),
            DatasetSpec(source=HOTPOT_VAL, target_tokens=200_000),
            DatasetSpec(source=SYNTHQA_TEST, target_tokens=200_000),
            DatasetSpec(source=GSM8K_TEST, target_tokens=200_000),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=200_000),
        ],
        tokenizer="gpt2",
        init_from=str(INIT_FROM),
        batch_size=8,
        grad_accum=16,
        lr=5e-5,
        weight_decay=0.0,
        eval_every_percent=10.0,
        compile=True,
    )


def main():
    if not (INIT_FROM / "config.json").is_file():
        raise SystemExit(f"Missing init checkpoint: {INIT_FROM}")
    config = build_config()
    trainer = TeapotTrainer(config=config)
    trainer.eval_example_cap = 32
    trainer.train(fresh=True)


if __name__ == "__main__":
    main()
