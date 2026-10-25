
#!/usr/bin/env python3
"""Inference for models trained with train.py."""

import argparse
import hashlib
import json
import os
import sys

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Llama-style language model."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Training configuration as a JSON string.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to the training JSON configuration.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Exact checkpoint directory. Overrides auto-discovery.")
    parser.add_argument("--message", type=str, required=True,
                        help="Prompt to send to the model.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Set to 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50,
                        help="Set to 0 to disable top-k sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"],
                        default="auto")
    return parser.parse_args()


def read_config(args):
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if args.config:
        try:
            return json.loads(args.config)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in --config: {exc}")

    return None


def _has_weights(path):
    if os.path.isfile(os.path.join(path, "model.safetensors")) or os.path.isfile(
        os.path.join(path, "pytorch_model.bin")
    ):
        return True
    if not os.path.isdir(path):
        return False
    return any(
        filename.startswith("model-") and filename.endswith(".safetensors")
        for filename in os.listdir(path)
    )


def find_checkpoint(config):
    """Find the run directory written by TeapotTrainer."""
    if config is None:
        return None

    from teapottrainer import TrainingConfig

    expected = None
    try:
        cfg = TrainingConfig(**config)
    except Exception as exc:
        print(f"Config did not match TrainingConfig ({exc}). Scanning runs/teapot-*.")
        cfg = None
    if cfg is not None:
        tokenizer_short = cfg.tokenizer.split("/")[-1]
        safe = cfg.model_dump(exclude_defaults=True, mode="json")
        run_hash = hashlib.md5(json.dumps(safe, default=str).encode()).hexdigest()[:6]
        expected = os.path.abspath(f"./runs/teapot-{tokenizer_short}-{run_hash}")
        if os.path.isfile(os.path.join(expected, "config.json")) and _has_weights(expected):
            return expected

    runs_dir = os.path.abspath("./runs")
    if not os.path.isdir(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if not name.startswith("teapot-") or not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "config.json")) and _has_weights(path):
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1:
        print("Multiple checkpoints found; using the newest:")
        for _, path in candidates:
            print(f"  {path}")
    return candidates[0][1]


def main():
    args = parse_args()

    if args.max_new_tokens < 1:
        sys.exit("--max-new-tokens must be at least 1.")
    if args.temperature < 0:
        sys.exit("--temperature cannot be negative.")
    if not 0 < args.top_p <= 1:
        sys.exit("--top-p must be greater than 0 and at most 1.")

    config = read_config(args)

    model_path = args.model_path or find_checkpoint(config)

    if not model_path:
        sys.exit(
            "Could not locate a saved checkpoint.\n"
            "Pass your training config using --config or --config-file, "
            "or specify --model-path ./runs/<checkpoint-directory>."
        )

    if not os.path.isdir(model_path):
        sys.exit(f"Checkpoint directory not found: {model_path}")

    if not os.path.isfile(os.path.join(model_path, "config.json")):
        sys.exit(
            f"No config.json found in {model_path}. "
            "This script expects a Hugging Face save_pretrained checkpoint."
        )

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested, but PyTorch cannot access a CUDA GPU.")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Checkpoint: {os.path.abspath(model_path)}")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (OSError, ValueError):
        tokenizer_name = (config or {}).get(
            "tokenizer", "Qwen/Qwen2.5-0.5B"
        )
        print(f"Tokenizer files not found in checkpoint; loading {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
    ).to(device)
    model.eval()

    inputs = tokenizer(
        args.message,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "repetition_penalty": args.repetition_penalty,
    }

    if args.temperature == 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        })
        if args.top_k > 0:
            generation_kwargs["top_k"] = args.top_k

    print("\n--- Prompt ---")
    print(args.message)
    print("\n--- Generating ---", flush=True)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_length:]

    print("\n--- Response ---")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    print("\n----------------")


if __name__ == "__main__":
    main()
