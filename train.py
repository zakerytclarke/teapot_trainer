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
then continues the same pretrain optimizer. A copy of the 126.5M-token mix
is about 5 to 8 hours, so the pair finishes inside a day. Across 66B tokens
that is 21 interior copies, plus one more from the finished pretrain weights.
If the FineWeb holdout improves by less than 0.02 nats across 20B tokens,
pretrain stops and that last base still gets its supervised copy.

The bridge is 4B tokens, so it crosses one 3B checkpoint of its own. It is
eager, peaked at 3e-4, and the loop is drawn from 1 to 4. The mix is the text
worth rereading: FineMath 30%, InfiWebMath 20%, Cosmopedia 20%, FineWeb 15%,
Stack 15%. That interior checkpoint gets its own supervised copy. The real
supervised run is the last one, from the finished bridge weights.

Supervised tokens, 126.5M per copy. FineWeb replay is 24M, about 19%, after
an 8% replay let FineWeb perplexity jump from 17 to 74. UltraChat is 8M of
multi-turn dialogue. Alpaca is 5M, one pass over the instruction set after
the holdout. Teapot chat is 14M and each SynthQA split is weighted 4,
with 10M answerable and 6M each of unanswerable and contrastive, so those
small sets are drawn about as often as the large sources and are not finished
in the first hour. The rest is passage questions, DROP and CoQA, GSM8K and
MetaMath, the tool split, Hermes multi-turn calls, constraints, system chats,
and a thin OpenHermes slice. The loop schedule is unchanged: half the steps
are one pass, and the rest are spread across 2 through 10. The supervised
peak is 6e-4. Prompts use the SmolLM2-135M-Instruct chat format, the same
template as the tokenizer. When a multi-turn chat does not fit in 2,048
tokens, the supervised step packs it with the retriever: low-scoring earlier
turns are dropped, the turns that remain stay in conversation order, and the
encoder is trained on that choice. A single question with one passage has
nothing earlier to retrieve and stays on the ordinary prompt.

A crash continues from checkpoints/resume. The base optimizer and the sample
cursor are there. A supervised copy that already wrote metrics.json is not
trained again. fresh=False is what makes that resume happen. The first launch
still builds a new model, because init_from is empty.

At the measured checkpoint size of 785MB, 21 pretrain pairs plus the finished
pretrain pair, two bridge pairs, and the rolling resume are about 40GB.
"""

import random

from teapottrainer import (
    CheckpointRule,
    DatasetSpec,
    EarlyStop,
    HERMES_FUNCTION_CALLING,
    HOTPOT_CONTRASTIVE,
    ModelConfig,
    SMOLTALK_APIGEN_ALIASED,
    SMOLTALK_APIGEN_DECISION,
    SMOLTALK_APIGEN_MISSING,
    SMOLTALK_APIGEN_UNRELATED,
    SMOLTALK_CONSTRAINTS,
    SMOLTALK_METAMATH,
    SMOLTALK_OPENHERMES,
    SMOLTALK_SYSTEMCHATS,
    SQUAD_V2_ANSWERABLE,
    SQUAD_V2_CONTRASTIVE,
    SQUAD_V2_UNANSWERABLE,
    SYNTHQA_ANSWERABLE,
    SYNTHQA_CONTRASTIVE,
    SYNTHQA_UNANSWERABLE,
    TeapotForCausalLM,
    TeapotTrainer,
    TrainingConfig,
    TrainingPhase,
    WHEN2CALL_ASK,
    WHEN2CALL_REFUSE,
    alpaca,
    build_rag_config,
    coqa,
    cosmopedia,
    dclm_edu,
    drop,
    fineweb_edu,
    finemath,
    fork_phase,
    gsm8k,
    hotpot,
    infiwebmath,
    smoltalk_apigen_aliased,
    smoltalk_apigen_decision,
    smoltalk_constraints,
    smoltalk_openhermes,
    squad_v2,
    stack_edu,
    synthqa,
    teapot_chat,
    ultrachat,
    when2call,
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
    # Interior checkpoints are every 3B tokens. Percent 100 cannot fire one.
    checkpoints = CheckpointRule(every_tokens=3_000_000_000, every_percent=100.0)
    # 126.5M supervised tokens. FineWeb replay is 24M. The fork below trains
    # this mix from a saved base and leaves that base's optimizer alone.
    supervised = TrainingPhase(
        name="sft",
        token_budget=126_500_000,
        datasets=[
            DatasetSpec(source=fineweb_edu.train, target_tokens=24_000_000, weight=1.0),
            DatasetSpec(source=ultrachat.train, target_tokens=8_000_000, weight=1.0),
            # One pass after the 5% holdout. The set is about 5M supervised tokens.
            DatasetSpec(source=alpaca.train, target_tokens=5_000_000, weight=1.0),
            # Small sets. Weight 4 keeps them in the draw until their budgets fill.
            DatasetSpec(source=teapot_chat.train, target_tokens=14_000_000, weight=4.0),
            DatasetSpec(source=SYNTHQA_ANSWERABLE, target_tokens=10_000_000, weight=4.0),
            DatasetSpec(source=SYNTHQA_UNANSWERABLE, target_tokens=6_000_000, weight=4.0),
            DatasetSpec(source=SYNTHQA_CONTRASTIVE, target_tokens=6_000_000, weight=4.0),
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=5_000_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=2_500_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_CONTRASTIVE, target_tokens=3_000_000, weight=1.0),
            DatasetSpec(source=hotpot.train, target_tokens=3_000_000, weight=1.0),
            DatasetSpec(source=HOTPOT_CONTRASTIVE, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=drop.train, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=coqa.train, target_tokens=2_000_000, weight=1.0),
            DatasetSpec(source=gsm8k.train, target_tokens=4_000_000, weight=1.0),
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
            DatasetSpec(source=fineweb_edu.train, target_tokens=1_000_000),
            DatasetSpec(source=smoltalk_constraints.test, target_tokens=200_000),
            DatasetSpec(source=smoltalk_apigen_decision.test, target_tokens=200_000),
            DatasetSpec(source=smoltalk_apigen_aliased.test, target_tokens=200_000),
            DatasetSpec(source=when2call.test, target_tokens=200_000),
            DatasetSpec(source=squad_v2.validation, target_tokens=200_000),
            DatasetSpec(source=hotpot.validation, target_tokens=200_000),
            DatasetSpec(source=synthqa.test, target_tokens=200_000),
            DatasetSpec(source=drop.validation, target_tokens=200_000),
            DatasetSpec(source=coqa.validation, target_tokens=200_000),
            DatasetSpec(source=gsm8k.test, target_tokens=200_000),
            DatasetSpec(source=teapot_chat.train, target_tokens=200_000),
            # Loss only. UltraChat has a real test split. OpenHermes and Alpaca
            # are train files, so the scorer uses the same 5% holdout training skips.
            DatasetSpec(source=ultrachat.test, target_tokens=200_000),
            DatasetSpec(source=smoltalk_openhermes.train, target_tokens=200_000),
            DatasetSpec(source=alpaca.train, target_tokens=200_000),
        ],
        lr=6e-4,
        compile=False,
        checkpoint=checkpoints,
        before_batch=[set_reasoning_effort],
        fork=True,
        fork_required=True,
        fork_from="latest",
        load_fork=True,
        task_eval=False,
        eval_during=False,
        save=False,
    )
    copy_at_checkpoint = fork_phase(supervised)
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
        phases=[
            # 66B tokens. 40/15/15/10/10/10. Compiled, one pass, peak 3e-3.
            TrainingPhase(
                name="pretrain",
                token_budget=66_000_000_000,
                datasets=[
                    DatasetSpec(source=fineweb_edu.train, target_tokens=26_400_000_000, weight=1.0),
                    DatasetSpec(source=dclm_edu.train, target_tokens=9_900_000_000, weight=1.0),
                    DatasetSpec(source=finemath.train, target_tokens=9_900_000_000, weight=1.0),
                    DatasetSpec(source=infiwebmath.train, target_tokens=6_600_000_000, weight=1.0),
                    DatasetSpec(source=stack_edu.train, target_tokens=6_600_000_000, weight=1.0),
                    DatasetSpec(source=cosmopedia.train, target_tokens=6_600_000_000, weight=1.0),
                ],
                eval_datasets=supervised.eval_datasets,
                lr=3e-3,
                compile=True,
                checkpoint=checkpoints,
                early_stop=EarlyStop(
                    metric="fineweb",
                    window_tokens=20_000_000_000,
                    min_delta=0.02,
                    patience=1,
                ),
                before_batch=[set_reasoning_effort],
                on_checkpoint=[copy_at_checkpoint],
                on_complete=[copy_at_checkpoint],
                task_eval=False,
                reset_step=True,
                save=True,
            ),
            # 4B tokens, so one interior checkpoint fires at 3B. 30/20/20/15/15.
            # Eager, because the loop count changes every batch. Peak is 0.1x.
            TrainingPhase(
                name="bridge",
                token_budget=4_000_000_000,
                datasets=[
                    DatasetSpec(source=finemath.train, target_tokens=1_200_000_000, weight=1.0),
                    DatasetSpec(source=infiwebmath.train, target_tokens=800_000_000, weight=1.0),
                    DatasetSpec(source=cosmopedia.train, target_tokens=800_000_000, weight=1.0),
                    DatasetSpec(source=fineweb_edu.train, target_tokens=600_000_000, weight=1.0),
                    DatasetSpec(source=stack_edu.train, target_tokens=600_000_000, weight=1.0),
                ],
                eval_datasets=supervised.eval_datasets,
                lr_scale=0.1,
                compile=False,
                checkpoint=checkpoints,
                before_batch=[set_reasoning_effort],
                on_checkpoint=[copy_at_checkpoint],
                task_eval=False,
                save=True,
            ),
            # The real supervised run, from the finished bridge weights.
            supervised,
        ],
        tokenizer="HuggingFaceTB/SmolLM2-135M",
        chat_format="smollm",
        init_from=None,
        batch_size=8,
        grad_accum=16,
        weight_decay=0.1,
    )


def main():
    config = build_config()
    trainer = TeapotTrainer(config=config)
    # 1,000 rows from each task split. Smaller splits are scored whole.
    trainer.eval_example_cap = 1000
    # 64 sequences of 2,048 tokens. The plateau decision reads this holdout.
    trainer.lm_eval_batches = 64
    # fresh=False continues checkpoints/resume when a previous launch died.
    # The first launch has no resume, and init_from is empty, so the model is new.
    trainer.train(fresh=False)


if __name__ == "__main__":
    main()
