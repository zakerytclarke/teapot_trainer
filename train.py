"""From-scratch week run in a 16-layer Llama shape.

Hidden size 768, 12 query heads, 4 key/value heads, SwiGLU width 2048.
Sixteen layers stays inside the compute of the 30-layer run this machine
already timed, so 40B tokens still fits in about a week. The tokenizer stays
SmolLM2-135M. The supervised mix pairs each passage with an unanswerable
question about an entity in it, and draws a new tool name on every example.
config.lr is the pretrain peak. After a pretrain phase the trainer uses 0.2x
that value for supervised training.
"""

from teapottrainer import (
    COSMOPEDIA_V2,
    DCLM_EDU,
    DatasetSpec,
    FINEWEB_EDU,
    FINEMATH,
    GSM8K,
    GSM8K_TEST,
    HOTPOT,
    HOTPOT_CONTRASTIVE,
    HOTPOT_VAL,
    INFIWEBMATH,
    ModelConfig,
    SMOLTALK_APIGEN_ALIASED,
    SMOLTALK_APIGEN_ALIASED_TEST,
    SMOLTALK_APIGEN_DECISION,
    SMOLTALK_APIGEN_DECISION_TEST,
    SMOLTALK_APIGEN_MISSING,
    SMOLTALK_APIGEN_UNRELATED,
    SMOLTALK_CONSTRAINTS,
    SMOLTALK_CONSTRAINTS_TEST,
    SMOLTALK_EVERYDAY,
    SMOLTALK_MAGPIE,
    SMOLTALK_REWRITE,
    SMOLTALK_SUMMARIZE,
    SQUAD_V2_ANSWERABLE,
    SQUAD_V2_CONTRASTIVE,
    SQUAD_V2_UNANSWERABLE,
    SQUAD_V2_VAL,
    STACK_EDU,
    SYNTHQA_ANSWERABLE,
    SYNTHQA_CONTRASTIVE,
    SYNTHQA_TEST,
    SYNTHQA_UNANSWERABLE,
    TEAPOT_CHAT,
    TeapotTrainer,
    TrainingConfig,
    WHEN2CALL_ASK,
    WHEN2CALL_REFUSE,
    WHEN2CALL_TEST,
)


def build_config() -> TrainingConfig:
    # 16 layers, hidden 768, 12 query heads, 4 key/value heads.
    # Feed-forward 2048 is 8/3 of the hidden size, the low end of 2048–2304.
    # Sequence length stays 2048. The SmolLM2 tokenizer is unchanged.
    return TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=768,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
            max_position_embeddings=2048,
        ),
        pretrain_datasets=[
            DatasetSpec(source=DCLM_EDU, target_tokens=21_200_000_000, weight=1.0),
            DatasetSpec(source=FINEWEB_EDU, target_tokens=14_000_000_000, weight=1.0),
            DatasetSpec(source=STACK_EDU, target_tokens=2_400_000_000, weight=1.0),
            DatasetSpec(source=INFIWEBMATH, target_tokens=800_000_000, weight=1.0),
            DatasetSpec(source=FINEMATH, target_tokens=700_000_000, weight=1.0),
            DatasetSpec(source=COSMOPEDIA_V2, target_tokens=900_000_000, weight=1.0),
        ],
        sft_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=12_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=12_000_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_ANSWERABLE, target_tokens=8_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_CONTRASTIVE, target_tokens=8_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_CONTRASTIVE, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_UNANSWERABLE, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT_CONTRASTIVE, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED, target_tokens=20_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_MISSING, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_UNRELATED, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_ASK, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_REFUSE, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=8_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_EVERYDAY, target_tokens=8_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_REWRITE, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_SUMMARIZE, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=4_000_000, weight=1.0),
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
        lr=3e-3,
        weight_decay=0.1,
        eval_every_percent=10.0,
        compile=True,
    )


def main():
    config = build_config()
    trainer = TeapotTrainer(config=config)
    trainer.eval_example_cap = 32
    # fresh=False continues checkpoints/resume when a previous launch died.
    # Replay starts at the first sample of that phase so the next draw matches.
    trainer.train(fresh=False)


if __name__ == "__main__":
    main()
