# teapot_trainer

A small library for training causal language models, plus a script that fine-tunes one for local assistant work: chat, tool calls, and answers grounded in retrieved passages.

The library does not care which dataset or architecture you bring. A source is any object with `build(tokenizer, seq_len, token_budget, seed)`. A model is a Transformers `PreTrainedModel`, built from a preset, loaded with `init_from`, or returned by `custom_builder`. A finished run is a normal Hugging Face checkpoint.

## Layout of a run

```
runs/<name>/
  config.json
  metrics.jsonl     # every optimizer step
  eval.jsonl        # eval records
  metrics.md        # the same evals as text
  checkpoints/best  # lowest val loss, or highest task score
  checkpoints/last  # latest weights, plus resume.pt
```

Load `checkpoints/best` or `checkpoints/last` with `AutoModelForCausalLM` and `AutoTokenizer`.

## Library

```python
from teapottrainer import (
    ListMessages, MemmapTokens, Stage, TeapotTrainer, TrainConfig,
    preset, tool_messages,
)

cfg = TrainConfig(
    model=preset("100m"),          # or "tiny", "350m", or your own ModelSpec
    tokenizer="path/or/hf-id",
    # init_from="Qwen/Qwen2.5-0.5B-Instruct",  # fine-tune instead of from scratch
    micro_batch=8,
    grad_accum=4,
    seq_len=4096,
    lr=1e-3,
    stages=[
        Stage(
            name="pretrain",
            target_tokens=2_000_000_000,
            sources=[MemmapTokens("train.bin", name="web")],
        ),
    ],
)
TeapotTrainer(cfg).train()
```

Built-in architectures are `llama`, `gpt2`, and `mistral`. A new architecture is a `custom_builder(spec, vocab_size)` that returns a `PreTrainedModel`, so `save_pretrained` still works.

Built-in sources are `MemmapTokens` (flat int32 file), `ListText`, `ListMessages`, `JsonlMessages`, `HuggingFaceText`, and `HuggingFaceSFT`. `tool_messages` and `rag_messages` only format chat rows. They do not train anything by themselves. `fineweb_edu`, `smoltalk`, and `gsm8k` in `teapottrainer.datasets` are optional Hugging Face constructors.

Prompt tokens are labeled `-100`. Pretrain windows are read from disk and are not copied into RAM.

## This repo's training script

`train.py` is the agent run, not the library.

```bash
python train.py --smoke                 # tiny model, confirms the loop
python train.py --confidence-pretrain   # 100M, a few million Ultra-FineWeb tokens
python train.py --confidence-sft        # short SFT of Qwen2.5-0.5B-Instruct on tools and RAG

python train.py --pretrain --model 100m --tokens 2000000000
python train.py --pretrain --model 350m --tokens 20000000000
python train.py --sft --init runs/pretrain-100m/checkpoints/best --tokens 50000000
```

Pretrain expects `../teapot_foundation/tokenized` (`train.bin`, `val.bin`, `test.bin`, `tokenizer/`) or `TEAPOT_TOKEN_DIR`. Resume a stopped run with `--resume runs/<name>/checkpoints/last`.

```bash
python inference.py --model-path runs/confidence-sft/checkpoints/last --chat \
  --system "$(cat runs/confidence-sft/system_prompt.txt)" \
  --message "What is the weather in Reykjavik?"
```

The agent target is a 100M–350M model that emits one JSON tool call, declines a tool call, answers from the passages it was given, and says when those passages do not contain the answer. A from-scratch pretrain on one GPU will not match a trillion-token base model at open-domain knowledge. The SFT metrics to watch are `tools/parse`, `tools/name`, `tools/args`, `tools/restraint`, `rag/answer`, and `rag/abstain`.
