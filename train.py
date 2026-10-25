"""From-scratch assistant run.

The config object is the same TrainingConfig as before. Architecture, tokenizer,
dataset mix, and token budgets are the values that target a small agent model.
This file is the full run. Confidence checks live in scripts/ and do not call this.
"""

from teapottrainer import (
    DatasetSpec,
    FINEWEB_EDU,
    GSM8K,
    GSM8K_TEST,
    HOTPOT,
    HOTPOT_VAL,
    ModelConfig,
    SMOLTALK_APIGEN,
    SMOLTALK_APIGEN_TEST,
    SMOLTALK_CONSTRAINTS,
    SMOLTALK_CONSTRAINTS_TEST,
    SMOLTALK_EVERYDAY,
    SMOLTALK_MAGPIE,
    SMOLTALK_OPENHERMES,
    SMOLTALK_REWRITE,
    SMOLTALK_SUMMARIZE,
    SMOLTALK_SYSTEMCHATS,
    SQUAD_V2,
    SQUAD_V2_VAL,
    TeapotTrainer,
    TrainingConfig,
    WHEN2CALL,
    WHEN2CALL_TEST,
)


def build_config() -> TrainingConfig:
    # Deep-thin Llama, about 110M with tied embeddings and a gpt2 vocab.
    # gpt2 stays the tokenizer so the run does not depend on another checkout.
    # SmolTalk "all" is intentionally not used: it already contains apigen.
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
            DatasetSpec(source=FINEWEB_EDU, target_tokens=2_000_000_000, weight=1.0),
        ],
        sft_datasets=[
            # General assistant behavior.
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=200_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_OPENHERMES, target_tokens=40_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_EVERYDAY, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=20_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_SYSTEMCHATS, target_tokens=15_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_SUMMARIZE, target_tokens=30_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_REWRITE, target_tokens=15_000_000, weight=1.0),
            # Tool use, including a minority of refusals.
            DatasetSpec(source=SMOLTALK_APIGEN, target_tokens=50_000_000, weight=1.0),
            # Ask for missing arguments, or refuse when no tool fits.
            # Positive tool calls come from apigen; this split has almost none.
            DatasetSpec(source=WHEN2CALL, target_tokens=30_000_000, weight=1.0),
            # Answer from a passage, and abstain when it is not in the passage.
            DatasetSpec(source=SQUAD_V2, target_tokens=40_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=40_000_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=10_000_000, weight=1.0),
        ],
        eval_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=1_000_000),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS_TEST, target_tokens=200_000),
            DatasetSpec(source=SMOLTALK_APIGEN_TEST, target_tokens=200_000),
            DatasetSpec(source=WHEN2CALL_TEST, target_tokens=200_000),
            DatasetSpec(source=SQUAD_V2_VAL, target_tokens=200_000),
            DatasetSpec(source=HOTPOT_VAL, target_tokens=200_000),
            DatasetSpec(source=GSM8K_TEST, target_tokens=200_000),
        ],
        tokenizer="gpt2",
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
