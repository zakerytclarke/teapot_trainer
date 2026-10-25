"""Eval plugins. Each ``run`` returns metrics plus optional generation samples."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .loss import chunked_cross_entropy


@dataclass
class EvalOutput:
    metrics: dict[str, float] = field(default_factory=dict)
    samples: list[tuple[str, str]] = field(default_factory=list)


def parse_tool_call(text: str) -> dict | None:
    start = text.find("<tool_call>")
    if start < 0:
        return None
    rest = text[start + len("<tool_call>") :]
    end = rest.find("</tool_call>")
    blob = rest if end < 0 else rest[:end]
    try:
        obj = json.loads(blob.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    arguments = obj.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    obj["arguments"] = arguments
    return obj


def _values_equal(pred, gold) -> bool:
    if isinstance(gold, (int, float)) or isinstance(pred, (int, float)):
        try:
            return float(pred) == float(gold)
        except (TypeError, ValueError):
            return False
    return str(pred) == str(gold)


def score_completion(text: str, expect: dict) -> dict[str, float]:
    """Score one greedy completion.

    ``expect["kind"]`` is ``tool``, ``no_tool``, ``contains``, or ``phrase``.
    Metric names default to a short prefix and can be overridden with
    ``metric`` (single score) or ``metric_prefix`` (tool scores).
    """
    kind = expect["kind"]
    call = parse_tool_call(text)
    if kind == "tool":
        prefix = expect.get("metric_prefix", "tools")
        name_ok = bool(call) and call["name"] == expect["name"]
        args_ok = name_ok and all(
            key in call["arguments"] and _values_equal(call["arguments"][key], value)
            for key, value in (expect.get("arguments") or {}).items()
        )
        if args_ok and set(call["arguments"]) != set((expect.get("arguments") or {})):
            args_ok = False
        return {
            f"{prefix}/parse": 1.0 if call else 0.0,
            f"{prefix}/name": 1.0 if name_ok else 0.0,
            f"{prefix}/args": 1.0 if args_ok else 0.0,
        }
    if kind == "no_tool":
        return {expect.get("metric", "tools/restraint"): 0.0 if call else 1.0}
    if kind == "contains":
        return {expect.get("metric", "task/contains"): 1.0 if expect["text"] in text else 0.0}
    if kind == "phrase":
        return {expect.get("metric", "task/phrase"): 1.0 if expect["text"] in text else 0.0}
    raise ValueError(f"unknown expect kind {kind!r}")


@torch.no_grad()
def _loss_on_array(model, tokens: np.ndarray, device, seq_len: int, batch: int) -> float | None:
    model.eval()
    usable = (len(tokens) // seq_len) * seq_len
    if usable < seq_len:
        return None
    total = 0.0
    seen = 0
    for start in range(0, usable, batch * seq_len):
        block = tokens[start : start + batch * seq_len]
        if len(block) < batch * seq_len:
            break
        rows = np.array(block.reshape(batch, seq_len), dtype=np.int32, copy=True)
        x = torch.from_numpy(rows).to(device).long()
        autocast = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(input_ids=x, use_cache=False).logits
            loss = chunked_cross_entropy(logits, x)
        n = x.numel()
        total += float(loss.item()) * n
        seen += n
    model.train()
    if seen == 0:
        return None
    return total / seen


@dataclass
class TokenPerplexity:
    """Next-token loss on a flat int32 memmap. The slice is stable across evals."""

    path: str
    name: str = "val"
    max_tokens: int = 262_144
    batch_size: int = 4

    def run(self, model, tokenizer, device, seq_len: int) -> EvalOutput:
        del tokenizer
        arr = np.memmap(self.path, dtype=np.int32, mode="r")
        n = min(len(arr), self.max_tokens)
        loss = _loss_on_array(model, np.asarray(arr[:n]), device, seq_len, self.batch_size)
        if loss is None or not math.isfinite(loss):
            return EvalOutput()
        return EvalOutput(
            metrics={f"{self.name}/loss": loss, f"{self.name}/ppl": math.exp(min(loss, 20.0))}
        )


@dataclass
class GenerateEval:
    prompts: list[str]
    name: str = "generate"
    max_new_tokens: int = 32
    temperature: float = 0.8

    def run(self, model, tokenizer, device, seq_len: int) -> EvalOutput:
        samples = []
        texts = _generate_batch(
            model, tokenizer, device, self.prompts, self.max_new_tokens, seq_len,
            batch_size=1, temperature=self.temperature,
        )
        samples.extend(zip(self.prompts, texts))
        return EvalOutput(samples=samples)


@dataclass
class TaskEval:
    """Greedy completions scored by :func:`score_completion`.

    Each example is ``{"messages": [...without assistant...], "expect": {...}}``.
    """

    examples: list[dict]
    name: str = "task"
    max_new_tokens: int = 48
    batch_size: int = 8

    def run(self, model, tokenizer, device, seq_len: int) -> EvalOutput:
        from .data import render_prompt

        prompts = [render_prompt(tokenizer, ex["messages"]) for ex in self.examples]
        completions = _generate_batch(
            model,
            tokenizer,
            device,
            prompts,
            self.max_new_tokens,
            seq_len,
            self.batch_size,
        )
        buckets: dict[str, list[float]] = {}
        samples = []
        for example, prompt, completion in zip(self.examples, prompts, completions):
            scores = score_completion(completion, example["expect"])
            for key, value in scores.items():
                buckets.setdefault(key, []).append(value)
            if len(samples) < 6:
                user = ""
                for message in example["messages"]:
                    if message.get("role") == "user":
                        user = message["content"]
                user = user.replace("\n", " ")
                if len(user) > 160:
                    user = user[:157] + "..."
                samples.append((user or prompt[-160:], completion.replace("\n", "\\n")))
        metrics = {key: float(sum(values) / len(values)) for key, values in buckets.items()}
        return EvalOutput(metrics=metrics, samples=samples)


@torch.no_grad()
def _generate_batch(model, tokenizer, device, prompts, max_new: int, seq_len: int, batch_size: int, temperature: float = 0.0):
    model.eval()
    prev_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(8, seq_len - max_new),
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            gen_kwargs = {
                "max_new_tokens": max_new,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                gen_kwargs["do_sample"] = False
            output = model.generate(**encoded, **gen_kwargs)
            prompt_len = encoded["input_ids"].shape[1]
            for row in output:
                texts.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = prev_side
        model.config.use_cache = prev_cache
        model.train()
    return texts
