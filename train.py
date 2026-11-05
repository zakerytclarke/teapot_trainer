"""Continue the 40B checkpoint until pretrain stalls, then teach the loop.

The shape is the 16-layer Llama that just finished 40,000,004,096 tokens:
hidden size 768, 12 query heads, 4 key/value heads, SwiGLU width 2048, RoPE
theta 500,000, position table 2,048. Those weights are loaded into
``TeapotForCausalLM``. Tied embeddings and the SmolLM vocabulary of 49,152
put the language model at 138,437,376 parameters. The run that produced them
trained at 70,463 tokens/s, so another 140B tokens is about 23 days.

Pretrain keeps the same six sources and the same proportions, one decoder
pass. The peak learning rate is 3e-4, which is where the 40B cosine ended.
A fresh cosine at the old 3e-3 peak would rewind a converged model. Every
1B tokens the trainer saves a checkpoint, scores a 64-sequence FineWeb
holdout, and trains a full supervised copy. Pretrain at 70,463 tokens/s
covers that 1B in about 4 hours. The one-hour smoke measured the looped
supervised copy at 4,309 tokens/s, which makes the 272M-token mix about 18
hours, so the checkpoint and its copy still finish inside a day. If the
holdout improves by less than 0.02 nats across a 20B window, pretrain stops.

The bridge is 2B tokens of the same text with the loop drawn from 1 to 4
passes. It runs after pretrain, eager, at 0.1x the pretrain peak, so the
second pass is not new when supervision starts.

Supervised training uses the same task mix as the finished 147M run, seven
times the one-day budgets. FineWeb replay is the same size as all of those
task tokens together, half the mix, after the last run's 8% replay took
FineWeb perplexity from 17.4 to 74. The loop schedule is unchanged: half the
steps are one pass, and the rest are spread across 2 through 10. The
supervised peak stays 6e-4. The same full mix is trained at every pretrain
checkpoint, and once more from the bridge weights at the end. About 139
copies during pretrain, plus the final copy, is the cost of doing that
every day across 140B tokens.

The tokenizer is HuggingFaceTB/SmolLM2-135M and is used unchanged. Its
vocabulary is 49,152. ``<|endoftext|>`` is id 0 and is the beginning, end,
unknown, and padding token. No special tokens are added, so the embedding
matrix stays the size the recipe trained. ``<|im_start|>`` and ``<|im_end|>``
are already in that vocabulary and are left unused: the supervised mix and
the evals speak System / User / Assistant.

The checkpoint also holds teapotai/teapotembedding. That encoder is loaded before
training. Each message is embedded whole, up to the encoder's context length.
``retrieval_loss`` trains the ranking with REINFORCE from the language-model
loss. The ordinary supervised step does not call it.

``before_batch`` sets ``reasoning_effort``. Pretrain is 1. The bridge draws
1 through 4. A phase named ``sft`` or ending in ``_sft`` draws 1 through 10,
with half of those draws at 1. The trainer restores the config after the
batch. Each extra pass is recomputed on the backward. At inference,
``generate(..., reasoning_effort=N)`` sets the count.

The supervised mix pairs each passage with an unanswerable question about an
entity in it, and draws a new tool name on every example.
"""

import random

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
    build_rag_config,
    TrainingConfig,
    WHEN2CALL_ASK,
    WHEN2CALL_REFUSE,
    WHEN2CALL_TEST,
)


def set_reasoning_effort(model, phase: str) -> None:
    """Assign ``reasoning_effort`` for this batch. Any model config can hold it.

    Pretrain and eval stay at one pass. The bridge draws 1 through 4. A phase
    named ``sft`` or ending in ``_sft`` draws 1 through 10, half of them at 1.
    """
    name = str(phase or "")
    if name == "bridge":
        effort = int(random.randint(1, 4))
    elif name == "sft" or name.endswith("_sft"):
        choices = list(range(1, 11))
        # 9 / 18 of the draws are 1. Each of 2..10 is 1 / 18.
        weights = [9, *([1] * (len(choices) - 1))]
        effort = int(random.choices(choices, weights=weights, k=1)[0])
    else:
        effort = 1
    model.config.reasoning_effort = effort


def build_config() -> TrainingConfig:
    # Previous 16-layer Llama shape. The pack window is the whole position table.
    return TrainingConfig(
        model=ModelConfig(
            model_type="custom",
            hidden_size=768,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
            max_position_embeddings=2048,
            custom_config_cls=build_rag_config,
            custom_model_cls=TeapotForCausalLM,
        ),
        # 140B more tokens on the 40B checkpoint. Same 53/35/6/2/1.75/2.25 mix.
        pretrain_datasets=[
            DatasetSpec(source=DCLM_EDU, target_tokens=74_200_000_000, weight=1.0),
            DatasetSpec(source=FINEWEB_EDU, target_tokens=49_000_000_000, weight=1.0),
            DatasetSpec(source=STACK_EDU, target_tokens=8_400_000_000, weight=1.0),
            DatasetSpec(source=INFIWEBMATH, target_tokens=2_800_000_000, weight=1.0),
            DatasetSpec(source=FINEMATH, target_tokens=2_450_000_000, weight=1.0),
            DatasetSpec(source=COSMOPEDIA_V2, target_tokens=3_150_000_000, weight=1.0),
        ],
        # 2B tokens, same mix, so the loop sees raw text before instructions.
        bridge_datasets=[
            DatasetSpec(source=DCLM_EDU, target_tokens=1_060_000_000, weight=1.0),
            DatasetSpec(source=FINEWEB_EDU, target_tokens=700_000_000, weight=1.0),
            DatasetSpec(source=STACK_EDU, target_tokens=120_000_000, weight=1.0),
            DatasetSpec(source=INFIWEBMATH, target_tokens=40_000_000, weight=1.0),
            DatasetSpec(source=FINEMATH, target_tokens=35_000_000, weight=1.0),
            DatasetSpec(source=COSMOPEDIA_V2, target_tokens=45_000_000, weight=1.0),
        ],
        # Task budgets are the finished run. FineWeb replay matches their sum.
        sft_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=135_870_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=11_900_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_ANSWERABLE, target_tokens=7_980_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=6_020_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_CONTRASTIVE, target_tokens=7_980_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=3_990_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_CONTRASTIVE, target_tokens=3_990_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_UNANSWERABLE, target_tokens=3_990_000, weight=1.0),
            DatasetSpec(source=HOTPOT_CONTRASTIVE, target_tokens=6_020_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED, target_tokens=20_020_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=3_990_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_MISSING, target_tokens=6_020_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_UNRELATED, target_tokens=6_020_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_ASK, target_tokens=6_020_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_REFUSE, target_tokens=6_020_000, weight=1.0),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=7_980_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_EVERYDAY, target_tokens=7_980_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=6_020_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_REWRITE, target_tokens=3_990_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_SUMMARIZE, target_tokens=3_990_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=1_960_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=3_990_000, weight=1.0),
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
        init_from="runs/teapot-SmolLM2-135M-4c2004/checkpoints/pretrain",
        batch_size=8,
        grad_accum=16,
        # The 40B run peaked at 3e-3 and finished at 3e-4. Continue from there.
        lr=3e-4,
        # The previous supervised peak. 0.2x the new pretrain peak would be 6e-5.
        sft_lr=6e-4,
        weight_decay=0.1,
        # 1B tokens out of the 140B budget. About 4 hours of pretrain, leaving
        # the supervised copy room to finish inside the same day.
        eval_every_percent=1_000_000_000 / 140_000_000_000 * 100,
        plateau_window_tokens=20_000_000_000,
        plateau_min_loss_drop=0.02,
        sft_at_checkpoints=True,
        compile=True,
        before_batch=[set_reasoning_effort],
    )


def main():
    config = build_config()
    trainer = TeapotTrainer(config=config)
    trainer.eval_example_cap = 32
    # 64 sequences of 2,048 tokens. The plateau decision reads this holdout.
    trainer.lm_eval_batches = 64
    # fresh=False continues checkpoints/resume when a previous launch died.
    # Replay starts at the first sample of that phase so the next draw matches.
    trainer.train(fresh=False)


if __name__ == "__main__":
    main()
