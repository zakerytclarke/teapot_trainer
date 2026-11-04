# teapottrainer

`teapottrainer` is a config-driven library for pretraining and supervised fine-tuning of causal language models. You describe the model, the dataset mix, and the token budgets. `TeapotTrainer` runs pretraining, then supervised training, and can resume from the last checkpoint.

```bash
pip install teapottrainer
```

## Train

```python
from teapottrainer import (
    DatasetSpec,
    FINEWEB_EDU,
    GSM8K,
    ModelConfig,
    TeapotTrainer,
    TrainingConfig,
)

config = TrainingConfig(
    model=ModelConfig(
        model_type="llama",
        hidden_size=768,
        layers=12,
        heads=12,
        kv_heads=4,
        intermediate=2048,
        seq_len=2048,
    ),
    pretrain_datasets=[
        DatasetSpec(source=FINEWEB_EDU, target_tokens=1_000_000_000),
    ],
    sft_datasets=[
        DatasetSpec(source=GSM8K, target_tokens=4_000_000),
    ],
    tokenizer="HuggingFaceTB/SmolLM2-135M",
    batch_size=4,
    grad_accum=8,
    lr=5e-4,
)

trainer = TeapotTrainer(config)
trainer.train(fresh=True)
```

`train(fresh=False)` continues from `checkpoints/resume` in the run directory: weights, optimizer, and the sample cursor. Pass `init_from` to start from an existing Hugging Face model or a local checkpoint instead of randomly initialized weights.

Set `sft_at_checkpoints=True` to fine-tune a copy of the base model at each pretrain eval and save that copy beside the base weights. Pretraining keeps going from the base model.

Runs are written under `./runs/`. Loss and task scores go to TensorBoard and to `metrics.jsonl` in the run directory.

## Datasets

Preset sources live on the package: `FINEWEB_EDU`, `DCLM_EDU`, `STACK_EDU`, `FINEMATH`, `GSM8K`, `SQUAD_V2`, `HOTPOT`, `SYNTHQA`, `TEAPOT_CHAT`, the SmolTalk subsets, and the reading, tool-calling, and instruction-following evals. A `DatasetSpec` assigns a source a token budget and a mix weight.

To add a source, build a `DatasetSourceConfig` with a `chat_template_fn` that returns a prompt and an answer. The trainer supervises the answer tokens.

## Develop

The `train.py` script in this repository is one example run. It is not part of the installed package.

```bash
pip install -e ".[dev]"
pytest
```
