"""Ten million tokens, small enough to try the published library in Colab.

Use a GPU that supports bfloat16 (L4 or A100). A T4 does not.

    !pip install -q teapottrainer
    !python colab_10m.py

8M FineWeb-Edu tokens, then 2M GSM8K tokens. On an L4 this is a few minutes
after the tokenizer and the first dataset stream connect, closer to ten
minutes if the stream is slow. torch.compile stays off so the first step
is not a long compile.
"""

import torch

from teapottrainer import (
    DatasetSpec,
    FINEWEB_EDU,
    GSM8K,
    ModelConfig,
    TeapotTrainer,
    TrainingConfig,
)


def build_config() -> TrainingConfig:
    # About 15M parameters with the GPT-2 vocabulary. Hidden size divides the
    # head count. Sequence length 512 keeps each step small on a Colab GPU.
    return TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
        ),
        pretrain_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=8_000_000),
        ],
        sft_datasets=[
            DatasetSpec(source=GSM8K, target_tokens=2_000_000),
        ],
        eval_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=50_000),
        ],
        tokenizer="gpt2",
        batch_size=8,
        grad_accum=1,
        lr=1e-3,
        weight_decay=0.1,
        # Past 100, so the only eval is the one at the end of each phase.
        eval_every_percent=101.0,
        sft_at_checkpoints=False,
        compile=False,
        seed=42,
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Select a GPU runtime. Runtime → Change runtime type → GPU.")
    if not torch.cuda.is_bf16_supported():
        name = torch.cuda.get_device_name(0)
        raise SystemExit(
            f"{name} cannot train in bfloat16. Switch the runtime to an L4 or A100."
        )
    config = build_config()
    trainer = TeapotTrainer(config)
    trainer.train(fresh=True)


if __name__ == "__main__":
    main()
