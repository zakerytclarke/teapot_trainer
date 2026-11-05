"""One-day check of the SmolLM2-135M teapot model.

The 40B-token mix was about a week on this machine. This run keeps the same
shape, tokenizer, and data sources, and uses about one seventh of those
budgets: 5.7B pretrain tokens and 21.11M supervised tokens.

Thirty layers, hidden size 576, 9 query heads, 3 key/value heads, SwiGLU width
1536. Tied embeddings and the SmolLM vocabulary of 49,152 put the language
model at 134,515,008 parameters. RoPE theta is 100,000 and the position table
is 8,192, matching the released SmolLM2-135M config. Training sequences and
the retrieval pack window stay 2,048.

The tokenizer is HuggingFaceTB/SmolLM2-135M and is used unchanged. Its
vocabulary is 49,152. ``<|endoftext|>`` is id 0 and is the beginning, end,
unknown, and padding token. No special tokens are added, so the embedding
matrix stays the size the recipe trained. ``<|im_start|>`` and ``<|im_end|>``
are already in that vocabulary and are left unused: the supervised mix and
the evals speak System / User / Assistant.

The checkpoint also holds all-MiniLM-L6-v2. That encoder is loaded before
training. Each message is embedded whole, up to the encoder's context length.
``retrieval_loss`` trains the ranking with REINFORCE from the language-model
loss. The ordinary supervised step does not call it.

The supervised mix pairs each passage with an unanswerable question about an
entity in it, and draws a new tool name on every example. config.lr is the
pretrain peak. After a pretrain phase the trainer uses 0.2x that value for
supervised training.
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
    TeapotForCausalLM,
    TeapotTrainer,
    build_smollm_config,
    TrainingConfig,
    WHEN2CALL_ASK,
    WHEN2CALL_REFUSE,
    WHEN2CALL_TEST,
)


def build_config() -> TrainingConfig:
    # Same SmolLM2-135M teapot model as the week run. Token budgets are ~1/7.
    return TrainingConfig(
        model=ModelConfig(
            model_type="custom",
            hidden_size=576,
            num_hidden_layers=30,
            num_attention_heads=9,
            num_key_value_heads=3,
            intermediate_size=1536,
            max_position_embeddings=2048,
            custom_config_cls=build_smollm_config,
            custom_model_cls=TeapotForCausalLM,
        ),
        pretrain_datasets=[
            DatasetSpec(source=DCLM_EDU, target_tokens=3_021_000_000, weight=1.0),
            DatasetSpec(source=FINEWEB_EDU, target_tokens=1_995_000_000, weight=1.0),
            DatasetSpec(source=STACK_EDU, target_tokens=342_000_000, weight=1.0),
            DatasetSpec(source=INFIWEBMATH, target_tokens=114_000_000, weight=1.0),
            DatasetSpec(source=FINEMATH, target_tokens=100_000_000, weight=1.0),
            DatasetSpec(source=COSMOPEDIA_V2, target_tokens=128_000_000, weight=1.0),
        ],
        sft_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=1_700_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=1_700_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_ANSWERABLE, target_tokens=1_140_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=860_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_CONTRASTIVE, target_tokens=1_140_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=570_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_CONTRASTIVE, target_tokens=570_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_UNANSWERABLE, target_tokens=570_000, weight=1.0),
            DatasetSpec(source=HOTPOT_CONTRASTIVE, target_tokens=860_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED, target_tokens=2_860_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=570_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_MISSING, target_tokens=860_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_UNRELATED, target_tokens=860_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_ASK, target_tokens=860_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_REFUSE, target_tokens=860_000, weight=1.0),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=1_140_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_EVERYDAY, target_tokens=1_140_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=860_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_REWRITE, target_tokens=570_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_SUMMARIZE, target_tokens=570_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=280_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=570_000, weight=1.0),
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
        # Each pretrain eval also fine-tunes a copy to completion and saves it
        # next to the base checkpoint. Pretrain itself keeps the base weights.
        sft_at_checkpoints=True,
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
