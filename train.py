"""Pretrain from scratch, then supervised assistant training.

Tokenizer is HuggingFaceTB/SmolLM2-135M. It is a 49,152-word vocabulary
trained on FineWeb-Edu, math, and code, and it keeps each digit as its own
token, so a year in the passage and the same year in the answer are the same
ids. The GPT-2 checkpoint cannot be loaded with this vocabulary, so pretraining
starts over and is saved to checkpoints/pretrain.
"""

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
    SYNTHQA_ANSWERABLE,
    SYNTHQA_TEST,
    SYNTHQA_UNANSWERABLE,
    TEAPOT_CHAT,
    TeapotTrainer,
    TrainingConfig,
    WHEN2CALL,
    WHEN2CALL_TEST,
)


def build_config() -> TrainingConfig:
    # Same 109M Llama. Pretrain lr is 1e-3; supervised lr is 0.2x that.
    # Passage answer and abstain budgets match. Ask-or-refuse matches the
    # positive tool-call budgets. Hotpot has no unanswerable rows.
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
        pretrain_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=20_000_000_000, weight=1.0),
        ],
        sft_datasets=[
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=12_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED, target_tokens=16_000_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL, target_tokens=24_000_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=14_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=14_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_ANSWERABLE, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_UNANSWERABLE, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=6_000_000, weight=1.0),
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
        tokenizer="HuggingFaceTB/SmolLM2-135M",
        batch_size=8,
        grad_accum=16,
        lr=1e-3,
        weight_decay=0.1,
        eval_every_percent=10.0,
        compile=True,
    )


def main():
    config = build_config()
    trainer = TeapotTrainer(config=config)
    trainer.eval_example_cap = 32
    trainer.train(fresh=True)


if __name__ == "__main__":
    main()
