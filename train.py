"""Three-hour check of the 16-layer teapot model.

The shape is the previous Llama run: 16 layers, hidden size 768, 12 query
heads, 4 key/value heads, SwiGLU width 2048. RoPE theta stays 500,000. The
position table is 2,048, the same window retrieval packs into, so every
trained position is a position the packer can fill. The 40B-token mix was
about a week on this machine, and one seventh of it is the one-day recipe:
5.7B pretrain tokens and 21.11M supervised tokens. This run keeps that
tokenizer, those data sources, and the same proportions, at one twentieth
of the one-day budgets: 285M pretrain tokens and 1.0555M supervised tokens.
The trainer fine-tunes a copy of the base model at each 10% pretrain
checkpoint and once more at the end, so the supervised mix runs 10 times.
The last 135M run on this machine trained at about 70k pretrain tokens/s
and 23k supervised tokens/s for one decoder pass. Supervised steps here
average 3.5 passes, near 6k tokens/s, which puts pretrain near 1.1 hours
and the ten supervised copies near half an hour. Compile and the checkpoint
reloads fill the rest of the three hours. Tied embeddings and the SmolLM
vocabulary of 49,152 put the language model at 138,437,376 parameters.

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

``before_batch`` runs before every training batch and may assign any field on
the model config. This script uses it to set ``reasoning_effort``. Pretraining
sets 1. Supervised training, including the checkpoint copies, draws 1 through
10. Half the supervised steps use one pass. The other half is split evenly
across 2 through 10, so the average pass count is 3.5. The trainer puts the
previous config back after the batch. Each extra pass is recomputed on the
backward, so the memory stays near one pass. At inference,
``generate(..., reasoning_effort=N)`` sets the count.

The supervised mix pairs each passage with an unanswerable question about an
entity in it, and draws a new tool name on every example. config.lr is the
pretrain peak. After a pretrain phase the trainer uses 0.2x that value for
supervised training.
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

    Pretraining and eval leave the block at one pass. A phase named ``sft`` or
    ending in ``_sft`` draws an effort: half of those steps are 1, and the
    other half is split evenly across 2 through 10.
    """
    name = str(phase or "")
    if name == "sft" or name.endswith("_sft"):
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
        pretrain_datasets=[
            DatasetSpec(source=DCLM_EDU, target_tokens=151_050_000, weight=1.0),
            DatasetSpec(source=FINEWEB_EDU, target_tokens=99_750_000, weight=1.0),
            DatasetSpec(source=STACK_EDU, target_tokens=17_100_000, weight=1.0),
            DatasetSpec(source=INFIWEBMATH, target_tokens=5_700_000, weight=1.0),
            DatasetSpec(source=FINEMATH, target_tokens=5_000_000, weight=1.0),
            DatasetSpec(source=COSMOPEDIA_V2, target_tokens=6_400_000, weight=1.0),
        ],
        sft_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=85_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_ANSWERABLE, target_tokens=85_000, weight=1.0),
            DatasetSpec(source=SYNTHQA_ANSWERABLE, target_tokens=57_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=43_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_CONTRASTIVE, target_tokens=57_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2_UNANSWERABLE, target_tokens=28_500, weight=1.0),
            DatasetSpec(source=SYNTHQA_CONTRASTIVE, target_tokens=28_500, weight=1.0),
            DatasetSpec(source=SYNTHQA_UNANSWERABLE, target_tokens=28_500, weight=1.0),
            DatasetSpec(source=HOTPOT_CONTRASTIVE, target_tokens=43_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_ALIASED, target_tokens=143_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=28_500, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_MISSING, target_tokens=43_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_UNRELATED, target_tokens=43_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_ASK, target_tokens=43_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL_REFUSE, target_tokens=43_000, weight=1.0),
            DatasetSpec(source=TEAPOT_CHAT, target_tokens=57_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_EVERYDAY, target_tokens=57_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=43_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_REWRITE, target_tokens=28_500, weight=1.0),
            DatasetSpec(source=SMOLTALK_SUMMARIZE, target_tokens=28_500, weight=1.0),
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=14_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=28_500, weight=1.0),
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
        before_batch=[set_reasoning_effort],
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
