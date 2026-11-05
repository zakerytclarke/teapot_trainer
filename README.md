# teapottrainer

`teapottrainer` is a config-driven library for pretraining and supervised fine-tuning of causal language models. You describe the model, the dataset mix, and the token budgets. `TeapotTrainer` runs pretraining, then supervised training, and can resume from the last checkpoint.

The from-scratch month run in `train.py` is written up in [Month-long 135M run](#month-long-135m-run).

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

`retrieval_loss(messages, answer, tokenizer)` trains that ranking with REINFORCE. It samples an earlier message from the softmax over those cosine scores, packs it, and uses the language-model loss of `answer` as the reward, compared with the message the scores already rank first. The language-model weights stay outside that graph.

A supervised phase packs a multi-turn chat that does not fit the window and adds that retrieval loss, so the encoder learns which earlier turn to keep and the language model trains on the packed prompt. The kept turns stay in conversation order. A short chat, and a single question with one passage, skip the encoder. Pretrain and bridge rows stay raw text.

`TrainingConfig.before_batch` is a hook the trainer calls before each training batch. It receives the model and the phase name and may assign any field on that model's config, whether the model is Llama, Qwen, GPT-2, or teapot. The trainer restores the config after the batch. `train.py` uses the hook to set `reasoning_effort`, which is how many times the teapot language-model block runs. Pretraining and eval stay at one pass. The bridge draws 1 through 4. A supervised phase draws 1 through 10, and half of those draws are 1. `generate(..., reasoning_effort=8)` asks for eight passes at inference. The extra passes are recomputed during training, so the memory stays near one pass.

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

The `train.py` script in this repository is the month-long run below. It is not part of the installed package.

```bash
pip install -e ".[dev]"
pytest
```

## Month-long 135M run

The goal is a 135M assistant that answers from the context and the tools it is given, and that can work a problem. A finished supervised copy is saved at least once a day, so the assistant can be compared as pretraining grows. The run starts from scratch (`init_from` is empty) and is sized to finish inside a month. Nothing is loaded from the earlier 40B checkpoint.

The shape is the 16-layer Llama: hidden size 768, 12 query heads, 4 key/value heads, SwiGLU width 2048, RoPE theta 500,000, position table 2,048. Tied embeddings and the SmolLM vocabulary of 49,152 put the language model at 138,437,376 parameters. The `teapotai/teapotembedding` encoder is loaded when the model is built and saved in every checkpoint. Supervised prompts use `chat_format="smollm"`, the SmolLM2-135M-Instruct template, including its default system line when a chat has no system message. Pretrain checkpoints do not carry that template.

Launch it from the repo with `python train.py`. `fresh=False` continues `checkpoints/resume` after a crash: the base optimizer, the sample cursor, and any supervised copy that already wrote `metrics.json`. The first launch has no resume file, so the model is new.

Measured speeds from the previous run put pretrain near 70,500 tokens/s, the bridge near 10,300 tokens/s, and one supervised copy at about 5 to 8 hours. Long chats also run the retriever, and each checkpoint scores the evals, so a day lands toward the long end of that range. At about 785MB per checkpoint, the saved pairs need about 40GB.

### Pretrain

66B tokens, about 10.8 days of compute. One decoder pass, compiled, learning rate peaked at 3e-3. The mix is web text, math, code, and textbooks.

| Dataset | Tokens | Share |
| --- | ---: | ---: |
| FineWeb-Edu | 26.4B | 40% |
| DCLM-Edu | 9.9B | 15% |
| FineMath | 9.9B | 15% |
| InfiWebMath | 6.6B | 10% |
| Stack-Edu | 6.6B | 10% |
| Cosmopedia | 6.6B | 10% |

Every 3B tokens is about 11.8 hours. The trainer saves that base checkpoint, scores the holdouts, trains one full supervised copy, and then returns to the same pretrain optimizer. That pair is about 17 to 20 hours, so one base checkpoint and one finished assistant land each day. There are 21 interior copies, at 3B through 63B, plus one more from the finished 66B weights. If the FineWeb holdout improves by less than 0.02 nats across any 20B tokens, pretrain stops and that last base still gets its supervised copy.

### Bridge

4B tokens, about 4.5 days. Eager, learning rate peaked at 3e-4, and the loop is drawn from 1 to 4 passes (average 2.5). The same documents as pretrain, narrowed to the text worth a second read.

| Dataset | Tokens | Share |
| --- | ---: | ---: |
| FineMath | 1.2B | 30% |
| InfiWebMath | 0.8B | 20% |
| Cosmopedia | 0.8B | 20% |
| FineWeb-Edu | 0.6B | 15% |
| Stack-Edu | 0.6B | 15% |

One interior checkpoint fires at 3B, about 3.4 days in, and that base gets its own supervised copy. The last 1B is about 27 hours. The final supervised run then starts from the finished bridge weights.

### Supervised copy

121.5M tokens per copy, about 5 to 8 hours. Learning rate peaked at 6e-4. Half the batches are one decoder pass, and the rest are spread from 2 through 10 (average 3.5). A multi-turn chat that does not fit in 2,048 tokens is packed by the retriever: low-scoring earlier turns are dropped, the rest stay in conversation order, and the encoder is trained on that choice.

A full run trains 24 copies: 21 during pretrain, one from the finished pretrain, one from the bridge checkpoint at 3B, and the last one from the finished bridge. Together that is 2.92B supervised tokens.

| Dataset | Tokens | What it teaches |
| --- | ---: | --- |
| FineWeb-Edu | 24M | Replay, so web perplexity stays put |
| Teapot chat | 14M, weight 4 | In-context answers in the teapot style |
| SynthQA answerable | 10M, weight 4 | Answer from the passage |
| SynthQA unanswerable | 6M, weight 4 | Say when the passage is not enough |
| SynthQA contrastive | 6M, weight 4 | Pick the right span against a distractor |
| SQuAD answerable / unanswerable / contrastive | 5M / 2.5M / 3M | The same three skills on Wikipedia passages |
| HotpotQA / Hotpot contrastive | 3M / 2M | Multi-hop reading |
| DROP | 2M | Discrete reasoning over a paragraph |
| CoQA | 2M | Conversational questions |
| GSM8K | 4M | Grade-school problems |
| MetaMath | 6M | More worked math |
| APIGen aliased / decision / missing / unrelated | 4M / 2M / 2M / 2M | Call a tool, pick one, notice a missing argument, refuse a tool that does not apply |
| When2Call ask / refuse | 2M / 2M | Ask for what is missing, or refuse |
| Hermes function calling | 3M | Multi-turn tool traces |
| UltraChat | 8M | Multi-turn dialogue |
| Constraints | 2M | Follow the instruction’s limits |
| System chats | 2M | Obey a system message |
| OpenHermes | 3M | A thin general-assistant slice |

Weight 4 keeps teapot chat and SynthQA in the draw until their budgets fill. Teapot chat is about 11% of the copy. The three SynthQA splits together are about 18%.

Each checkpoint scores FineWeb, constraints, APIGen decision and aliased calls, When2Call, SQuAD, Hotpot, SynthQA, DROP, CoQA, GSM8K, and teapot chat.

### Wall clock

Pretrain plus its 22 supervised copies is about 16 to 18 days. The bridge plus its two supervised copies is about 5 to 5.5 days. A full run is about 3 weeks at the measured speeds, and still inside a month if each supervised copy takes 8 hours.
