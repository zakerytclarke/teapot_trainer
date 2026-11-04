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

## Model

`model_type` chooses how the trainer builds a new model. `llama` and `gpt2` are built in. Any other Hugging Face architecture is `custom`: pass the config class and the model class. The trainer calls the config class with `vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, and `max_position_embeddings`. It does not pass `kv_heads`. A wrapper can set fields the class needs beyond those six.

```python
from transformers import Qwen2Config, Qwen2ForCausalLM

# Llama. kv_heads defaults to 4. heads must divide hidden_size.
ModelConfig(
    model_type="llama",
    hidden_size=768,
    layers=16,
    heads=12,
    kv_heads=4,
    intermediate=2048,
    seq_len=2048,
)

# GPT-2. intermediate and kv_heads are ignored. heads must divide hidden_size.
ModelConfig(
    model_type="gpt2",
    hidden_size=768,
    layers=12,
    heads=12,
    seq_len=1024,
)

# Qwen2, or any other Hugging Face pair with the same constructor fields.
def qwen_config(**kwargs):
    kwargs["num_key_value_heads"] = 2
    return Qwen2Config(**kwargs)

ModelConfig(
    model_type="custom",
    hidden_size=896,
    layers=24,
    heads=14,
    intermediate=4864,
    seq_len=2048,
    custom_config_cls=qwen_config,
    custom_model_cls=Qwen2ForCausalLM,
)
```

`init_from` starts from weights you already have, either a Hugging Face id or a local directory. The `ModelConfig` dimensions have to match that checkpoint.

```python
TrainingConfig(
    model=ModelConfig(
        model_type="llama",
        hidden_size=576,
        layers=30,
        heads=9,
        kv_heads=3,
        intermediate=1536,
        seq_len=2048,
    ),
    init_from="HuggingFaceTB/SmolLM2-135M",
    tokenizer="HuggingFaceTB/SmolLM2-135M",
)
```

## Inference

The finished weights are a normal Transformers checkpoint. After both phases, the run directory is the supervised model. `checkpoints/pretrain` is the base model from before supervised training.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "./runs/teapot-SmolLM2-135M-4c2004"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt)

prompt = "User: What is 2 + 2?\n\nAssistant:\n"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Datasets

Preset sources live on the package: `FINEWEB_EDU`, `DCLM_EDU`, `STACK_EDU`, `FINEMATH`, `GSM8K`, `SQUAD_V2`, `HOTPOT`, `SYNTHQA`, `TEAPOT_CHAT`, the SmolTalk subsets, and the reading, tool-calling, and instruction-following evals. A `DatasetSpec` assigns a source a token budget and a mix weight.

To add a source, build a `DatasetSourceConfig` with a `chat_template_fn` that returns a prompt and an answer. The trainer supervises the answer tokens.

## Develop

The `train.py` script in this repository is one example run. It is not part of the installed package.

```bash
pip install -e ".[dev]"
pytest
```
