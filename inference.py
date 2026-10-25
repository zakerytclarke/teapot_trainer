#!/usr/bin/env python3
"""Generate from a checkpoint written by TeapotTrainer."""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run a saved teapot checkpoint.")
    parser.add_argument("--model-path", required=True, help="checkpoints/best or checkpoints/last")
    parser.add_argument("--message", required=True)
    parser.add_argument("--system", default=None, help="Optional system turn, used with --chat.")
    parser.add_argument("--chat", action="store_true", help="Wrap --message in the tokenizer chat template.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy.")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    model.eval()

    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.message})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = args.message

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature == 0:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
