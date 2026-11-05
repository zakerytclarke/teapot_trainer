"""Train a 135M teapot from scratch, then read the assistant at every day.

The shape is the 16-layer Llama: hidden size 768, 12 query heads, 4 key/value
heads, SwiGLU width 2048, RoPE theta 500,000, position table 2,048. Nothing
is loaded from the 40B run. Tied embeddings and the SmolLM vocabulary of
49,152 put the language model at 138,437,376 parameters. The embedder is
teapotai/teapotembedding and is loaded when the model is built.

Pretrain is 66B tokens, one decoder pass, compiled, peaked at 3e-3. The first
40B run was mostly web text and the holdout had already flattened, so this
mix spends the new tokens on reading, math, and structured text: FineWeb-Edu
40%, DCLM-Edu 15%, FineMath 15%, InfiWebMath 10%, Stack-Edu 10%, Cosmopedia
10%. Every 3B tokens is about 12 hours at 70,463 tokens/s. The trainer saves
that base checkpoint, scores the holdouts, trains a full supervised copy, and
then continues the same pretrain optimizer. A copy of the 121.5M-token mix
is about 5 to 8 hours, so the pair finishes inside a day. Across 66B tokens
that is 21 interior copies, plus one more from the finished pretrain weights.
If the FineWeb holdout improves by less than 0.02 nats across 20B tokens,
pretrain stops and that last base still gets its supervised copy.

The bridge is 4B tokens, so it crosses one 3B checkpoint of its own. It is
eager, peaked at 3e-4, and the loop is drawn from 1 to 4. The mix is the text
worth rereading: FineMath 30%, InfiWebMath 20%, Cosmopedia 20%, FineWeb 15%,
Stack 15%. That interior checkpoint gets its own supervised copy. The real
supervised run is the last one, from the finished bridge weights.

Supervised tokens, 121.5M per copy. FineWeb replay is 24M, about 20%, after
an 8% replay let FineWeb perplexity jump from 17 to 74. UltraChat is 8M of
multi-turn dialogue. Teapot chat is 14M and each SynthQA split is weighted 4,
with 10M answerable and 6M each of unanswerable and contrastive, so those
small sets are drawn about as often as the large sources and are not finished
in the first hour. The rest is passage questions, DROP and CoQA, GSM8K and
MetaMath, the tool split, Hermes multi-turn calls, constraints, system chats,
and a thin OpenHermes slice. The loop schedule is unchanged: half the steps
are one pass, and the rest are spread across 2 through 10. The supervised
peak is 6e-4.

A crash continues from checkpoints/resume. The base optimizer and the sample
cursor are there. A supervised copy that already wrote metrics.json is not
trained again. fresh=False is what makes that resume happen. The first launch
still builds a new model, because init_from is empty.

At the measured checkpoint size of 785MB, 21 pretrain pairs plus the finished
pretrain pair, two bridge pairs, and the rolling resume are about 40GB.
"""

import random

from teapottrainer import (
    COSMOPEDIA_V2,
    COQA,
    COQA_VAL,
    DCLM_EDU,
    DROP,
    DROP_VAL,
    DatasetSpec,
    FINEWEB_EDU,
    FINEMATH,
    GSM8K,
    GSM8K_TEST,
    HERMES_FUNCTION_CALLING,
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
    SMOLTALK_METAMATH,
    SMOLTALK_OPENHERMES,
    SMOLTALK_SYSTEMCHATS,
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
    ULTRACHAT,
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
    # From-scratch 16-layer Llama. The pack window is the whole position table.
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
        # 66B tokens. 40/15/15/10/10/10.
        pretrain_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=26_400_000_000, weight=1.0),
            DatasetSpec(source=DCLM_EDU, target_tokens=9_900_000_000, weight=1.0),
            DatasetSpec(source=FINEMATH, target_tokens=9_900_000_000, weight=1.0),
            DatasetSpec(source=INFIWEBMATH, target_tokens=6_600_000_000, weight=1.0),
            DatasetSpec(source=STACK_EDU, target_tokens=6_600_000_000, weight=1.0),
            DatasetSpec(source=COSMOPEDIA_V2, target_tokens=6_600_000_000, weight=1.0),
        ],
        # 4B tokens, so one interior checkpoint fires at 3B. 30/20/20/15/15.
        bridge_datasets=[
            DatasetSpec(source=FINEMATH, target_tokens=1_200_000_000, weight=1.0),
            DatasetSpec(source=INFIWEBMATH, target_tokens=800_000_000, weight=1.0),
            DatasetSpec(source=COSMOPEDIA_V2, target_tokens=800_000_000, weight=1.0),
            DatasetSpec(source=FINEWEB_EDU, target_tokens=600_000_000, weight=1.0),
            DatasetSpec(source=STACK_EDU, target_tokens=600_000_000, weight=1.0),
        ],
        sft_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=24_000_000, weight=1.0),
            DatasetSpec(source=ULTRACHAT, target_tokens=8_000_000, weight=1.0),
            # Small sets. Weight 4 keeps them in the draw until their budgets fill.
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=14_000_000, weight=4.0),
            DatasetSpec(source=SYNTHQA_ANSWERABLE, target_tokens=10_000_000, weight=4.0),
            DatasetSpec(source=SYNTHQA_UNANSWERABLE, target_tokens=6_000_000, weight=4.0),
            DatasetSpec(source=SYNTHQA_CONTRASTIVE, target_tokens=6_000_000, weight=4.0),
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=5_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=2_500_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_CONTRASTIVE, target_tokens=3_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=3_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT_CONTRASTIVE, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=DROP, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=COQA, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_METAMATH, target_tokens=6_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED, target_tokens=4_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_MISSING, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_UNRELATED, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_ASK, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_REFUSE, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=HERMES_FUNCTION_CALLING, target_tokens=3_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_SYSTEMCHATS, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_OPENHERMES, target_tokens=3_000_000, weight=1.0),
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
            DatasetSpec(source=DROP_VAL, target_tokens=200_000),
            DatasetSpec(source=COQA_VAL, target_tokens=200_000),
            DatasetSpec(source=GSM8K_TEST, target_tokens=200_000),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=200_000),
        ],
        tokenizer="HuggingFaceTB/SmolLM2-135M",
        init_from=None,
        batch_size=8,
        grad_accum=16,
        lr=3e-3,
        sft_lr=6e-4,
        weight_decay=0.1,
        # Unused while eval_every_tokens is set. 100 cannot fire a mid-phase save.
        eval_every_percent=100.0,
        eval_every_tokens=3_000_000_000,
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
    # The first launch has no resume, and init_from is empty, so the model is new.
    trainer.train(fresh=False)


if __name__ == "__main__":
    main()
