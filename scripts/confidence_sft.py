"""Short SFT on a real mix, starting from Qwen2.5-0.5B-Instruct.

init_from can be set on TrainingConfig or passed to train(). The weights
must match the ModelConfig. This run exists to see whether tool use, refusal,
RAG answer, and RAG abstention move on real rows.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teapottrainer import (
    DatasetSpec,
    GSM8K,
    GSM8K_TEST,
    HOTPOT,
    HOTPOT_VAL,
    ModelConfig,
    SMOLTALK_APIGEN,
    SMOLTALK_APIGEN_DECISION,
    SMOLTALK_APIGEN_TEST,
    SMOLTALK_CONSTRAINTS,
    SMOLTALK_CONSTRAINTS_TEST,
    SMOLTALK_EVERYDAY,
    SMOLTALK_MAGPIE,
    SQUAD_V2,
    SQUAD_V2_VAL,
    TeapotTrainer,
    TrainingConfig,
    WHEN2CALL,
    WHEN2CALL_TEST,
)


def main():
    config = TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=896,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            intermediate_size=4864,
            max_position_embeddings=1024,
        ),
        pretrain_datasets=[],
        sft_datasets=[
            DatasetSpec(source=SMOLTALK_APIGEN, target_tokens=16_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_APIGEN_DECISION, target_tokens=24_000, weight=1.0),
            DatasetSpec(source=WHEN2CALL, target_tokens=16_000, weight=1.0),
            DatasetSpec(source=SQUAD_V2, target_tokens=24_000, weight=1.0),
            DatasetSpec(source=HOTPOT, target_tokens=16_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS, target_tokens=16_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_MAGPIE, target_tokens=16_000, weight=1.0),
            DatasetSpec(source=GSM8K, target_tokens=8_000, weight=1.0),
            DatasetSpec(source=SMOLTALK_EVERYDAY, target_tokens=4_000, weight=1.0),
        ],
        eval_datasets=[
            DatasetSpec(source=SMOLTALK_APIGEN_TEST, target_tokens=50_000),
            DatasetSpec(source=WHEN2CALL_TEST, target_tokens=50_000),
            DatasetSpec(source=SQUAD_V2_VAL, target_tokens=50_000),
            DatasetSpec(source=HOTPOT_VAL, target_tokens=50_000),
            DatasetSpec(source=GSM8K_TEST, target_tokens=50_000),
            DatasetSpec(source=SMOLTALK_CONSTRAINTS_TEST, target_tokens=50_000),
        ],
        tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
        batch_size=2,
        grad_accum=4,
        lr=2e-5,
        weight_decay=0.0,
        eval_every_percent=100.0,
        compile=False,
        seed=42,
    )
    trainer = TeapotTrainer(config=config)
    trainer.eval_example_cap = 16
    trainer.train(
        fresh=True,
        init_from="Qwen/Qwen2.5-0.5B-Instruct",
    )
    print(f"metrics: {trainer.output_dir}/metrics.jsonl")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
