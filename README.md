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

## Chat format

`chat_format` wraps supervised examples. Pretrain rows stay raw text, including a raw source that you also put in the supervised mix. The default is `plain`, which is the `System` / `User` / `Assistant` text. The same template is saved with a supervised checkpoint, so Transformers can build the prompt from messages.

```python
config = TrainingConfig(
    chat_format="chatml",
    tokenizer="HuggingFaceTB/SmolLM2-135M",
    sft_datasets=[
        DatasetSpec(source=GSM8K, target_tokens=4_000_000),
    ],
)
```

`chatml` is the SmolLM-Instruct template. SmolLM2 already contains `<|im_start|>` and `<|im_end|>`, and a supervised checkpoint uses `<|im_end|>` as the stop token. `smollm` is the SmolLM2-Instruct template, including its default system line. The other names are `llama2`, `llama3`, `gemma`, `mistral`, `phi`, `zephyr`, `alpaca`, and `vicuna`. `qwen` is another name for `chatml`.

## Model

`model_type` chooses how the trainer builds a new model. `llama` and `gpt2` are built in. Any other Hugging Face architecture is `custom`: pass the config class and the model class. The trainer calls the config class with `vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, and `max_position_embeddings`. A wrapper can set fields the class needs beyond those.

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

The finished weights are a normal Transformers checkpoint. After both phases, the run directory is the supervised model. `checkpoints/pretrain` is the base model from before supervised training, still without a chat template.

With `chat_format="chatml"` or `chat_format="smollm"`, load that supervised directory and pass messages. The pipeline applies the template that was used for supervised training.

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="./runs/teapot-SmolLM2-135M-4c2004")
pipe([{"role": "user", "content": "Who are you?"}])
```

`plain` does not add those control tokens. Format the prompt yourself:

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

`train.py` builds `TeapotForCausalLM`. Loading that checkpoint needs `trust_remote_code=True`.

## Retrieval

`TeapotForCausalLM` is the previous 16-layer Llama shape plus the `teapotai/teapotembedding` encoder: hidden size 768, 12 query heads, 4 key/value heads, SwiGLU width 2048. The language model is trained from scratch. Its position table is the 2,048-token pack window. The embedding model is loaded before training and saved in the same checkpoint.

A chat is packed into 2,048 tokens. The system prompt and the latest user message are always kept. Earlier user and assistant messages are scored by cosine similarity to that latest user message, in one encoder batch. Each message is embedded whole and truncated at the encoder's position table (512 tokens), not split into pieces. The closest messages are placed in chronological order after the system prompt and before the latest user message. One message may fill whatever room is left.

`retrieval_loss(messages, answer, tokenizer)` trains that ranking with REINFORCE. It samples an earlier message from the softmax over those cosine scores, packs it, and uses the language-model loss of `answer` as the reward, compared with the message the scores already rank first. The ordinary supervised step does not call this loss.

`TrainingConfig.before_batch` is a hook the trainer calls before each training batch. It receives the model and the phase name and may assign any field on that model's config, whether the model is Llama, Qwen, GPT-2, or teapot. The trainer restores the config after the batch. `train.py` uses the hook to set `reasoning_effort`, which is how many times the teapot language-model block runs. Pretraining sets it to 1. Half of the supervised steps use one pass, and the other half is split evenly across 2 through 10. `generate(..., reasoning_effort=8)` asks for eight passes at inference. The extra passes are recomputed during training, so the memory stays near one pass.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "./runs/teapot-SmolLM2-135M-4c2004"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, device_map="auto")
messages = [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "The gate code is 1428."},
    {"role": "assistant", "content": "I will remember that."},
    {"role": "user", "content": "What is the gate code?"},
]
outputs = model.generate(messages=messages, tokenizer=tokenizer, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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
