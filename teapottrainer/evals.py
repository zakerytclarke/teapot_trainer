"""Held-out evals for assistant, tool, and RAG behavior.

Task choice comes from the dataset name and subset, so ``DatasetSpec`` does
not grow a new field. Generation is greedy. Loss is reported beside the task
scores, not instead of them.
"""

from __future__ import annotations

import random
from collections import defaultdict

import torch

from .datasets import DatasetSourceConfig, encode_pair, open_dataset, row_pairs
from .formatting import ABSTAIN_TEXT
from .metrics import (
    best_span_scores,
    classify_decision,
    constraint_checks,
    extract_tool_calls,
    is_abstention,
    score_constraints,
    score_gsm,
    score_tool_prediction,
)


def task_kind(src: DatasetSourceConfig) -> str:
    name = (src.name or "").lower()
    subset = (src.subset or "").lower()
    if "when2call" in name:
        return "when2call"
    if "squad" in name:
        return "squad"
    if "hotpot" in name:
        return "hotpot"
    if name == "gsm8k" or name.endswith("/gsm8k"):
        return "gsm8k"
    if "apigen" in subset:
        return "tools"
    if "constraint" in subset:
        return "constraints"
    if src.is_sft:
        return "sft_loss"
    return "lm"


def source_key(src: DatasetSourceConfig) -> str:
    bits = [src.name.split("/")[-1]]
    if src.subset:
        bits.append(src.subset)
    bits.append(src.split)
    return "/".join(bits)


def _last_user(row: dict) -> str:
    users = [
        (message.get("content") or "")
        for message in (row.get("messages") or [])
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    return users[-1] if users else ""


def _stratified(groups: dict[str, list], n: int, rng: random.Random) -> list:
    keys = [key for key, rows in groups.items() if rows]
    if not keys:
        return []
    rng.shuffle(keys)
    picked: list = []
    pools = {key: rows[:] for key, rows in groups.items() if rows}
    for rows in pools.values():
        rng.shuffle(rows)
    while len(picked) < n and any(pools.values()):
        for key in keys:
            if pools.get(key):
                picked.append(pools[key].pop())
                if len(picked) >= n:
                    break
    return picked


# A full run scores the same slice many times. Keep the rows so Hotpot and
# SQuAD are not reloaded from disk at every checkpoint.
_ROW_CACHE: dict[tuple, list[dict]] = {}


def load_task_rows(src: DatasetSourceConfig, n: int, seed: int, kind: str) -> list[dict]:
    """Load a small, class-balanced slice of an eval split."""
    if kind == "lm" or n <= 0:
        return []
    key = (src.name, src.subset or "", src.split or "", int(n), int(seed), kind)
    cached = _ROW_CACHE.get(key)
    if cached is not None:
        return cached
    dataset = open_dataset(src, streaming=False)
    rows = list(dataset)
    rng = random.Random(seed)
    if kind == "when2call" and rows and "correct_answer" in rows[0]:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get("correct_answer"))].append(row)
        picked = _stratified(groups, n, rng)
        _ROW_CACHE[key] = picked
        return picked
    if kind == "squad":
        groups = {"answerable": [], "unanswerable": []}
        for row in rows:
            texts = (row.get("answers") or {}).get("text") or []
            groups["answerable" if texts else "unanswerable"].append(row)
        picked = _stratified(groups, n, rng)
        _ROW_CACHE[key] = picked
        return picked
    if kind == "tools":
        groups = {"call": [], "refuse": []}
        for row in rows:
            pairs = row_pairs(src, row)
            gold = pairs[-1][1] if pairs else ""
            groups["call" if extract_tool_calls(gold) else "refuse"].append(row)
        picked = _stratified(groups, n, rng)
        _ROW_CACHE[key] = picked
        return picked
    rng.shuffle(rows)
    picked = rows[:n]
    _ROW_CACHE[key] = picked
    return picked


def _generate(model, tokenizer, prompts: list[str], max_new: int, seq_len: int, device) -> list[str]:
    if not prompts:
        return []
    room = max(8, seq_len - max_new)
    encoded = [tokenizer.encode(prompt, add_special_tokens=False)[-room:] for prompt in prompts]
    width = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    input_ids = []
    attention = []
    for ids in encoded:
        pad = width - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attention.append([0] * pad + [1] * len(ids))
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    previous_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
        )
    model.config.use_cache = previous_cache
    if was_training:
        model.train()
    texts = []
    for row in generated:
        texts.append(tokenizer.decode(row[width:], skip_special_tokens=True).strip())
    return texts


def _mean_loss(model, tokenizer, pairs: list[tuple[str, str]], seq_len: int, device) -> float | None:
    encoded = []
    for prompt, answer in pairs[:16]:
        item = encode_pair(tokenizer, prompt, answer, seq_len)
        if item is not None:
            encoded.append(item)
    if not encoded:
        return None
    losses = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for item in encoded:
            input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
            labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(input_ids=input_ids, labels=labels)
            else:
                output = model(input_ids=input_ids, labels=labels)
            if output.loss is not None and not torch.isnan(output.loss):
                losses.append(float(output.loss.item()))
    if was_training:
        model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def _batch_map(fn, items, size):
    for start in range(0, len(items), size):
        yield fn(items[start : start + size])


def evaluate_source(
    model,
    tokenizer,
    src: DatasetSourceConfig,
    *,
    device,
    seq_len: int,
    max_examples: int,
    seed: int,
) -> dict:
    kind = task_kind(src)
    key = source_key(src)
    metrics: dict[str, float] = {}
    samples: list[dict] = []
    if kind == "lm":
        return {"key": key, "kind": kind, "metrics": metrics, "samples": samples}

    rows = load_task_rows(src, max_examples, seed, kind)
    prepared = []
    for row in rows:
        pairs = row_pairs(src, row)
        if not pairs:
            continue
        prompt, answer = pairs[-1]
        prepared.append((row, prompt, answer))
    if not prepared:
        return {"key": key, "kind": kind, "metrics": metrics, "samples": samples}

    loss = _mean_loss(model, tokenizer, [(p, a) for _, p, a in prepared], seq_len, device)
    if loss is not None:
        metrics["loss"] = loss

    max_new = {
        "tools": 160,
        "when2call": 96,
        "squad": 64,
        "hotpot": 48,
        "gsm8k": 256,
        "constraints": 220,
        "sft_loss": 0,
    }.get(kind, 0)
    predictions = [""] * len(prepared)
    if max_new > 0:
        prompts = [prompt for _, prompt, _ in prepared]
        flat: list[str] = []
        for chunk in _batch_map(
            lambda batch: _generate(model, tokenizer, batch, max_new, seq_len, device),
            prompts,
            4,
        ):
            flat.extend(chunk)
        predictions = flat

    if kind == "tools":
        buckets = {"parse": [], "name": [], "args": [], "restraint": []}
        for (row, prompt, answer), prediction in zip(prepared, predictions):
            scored = score_tool_prediction(prediction, answer)
            if scored["gold_is_call"]:
                buckets["parse"].append(scored["parse"])
                buckets["name"].append(scored["name"])
                buckets["args"].append(scored["args"])
            else:
                buckets["restraint"].append(scored["restraint"])
            if len(samples) < 3:
                samples.append({
                    "task": "tools",
                    "user": _last_user(row)[:240],
                    "gold": answer[:300],
                    "prediction": prediction[:300],
                })
        for name, values in buckets.items():
            if values:
                metrics[name] = sum(values) / len(values)
                metrics[f"support_{name}"] = float(len(values))

    elif kind == "when2call":
        correct = []
        recalls = defaultdict(list)
        name_hits = []
        for (row, prompt, answer), prediction in zip(prepared, predictions):
            label = str(row.get("correct_answer") or classify_decision(answer))
            predicted = classify_decision(prediction)
            hit = 1.0 if predicted == label else 0.0
            correct.append(hit)
            recalls[label].append(hit)
            if label == "tool_call":
                gold_calls = extract_tool_calls(str((row.get("answers") or {}).get("tool_call") or answer))
                pred_calls = extract_tool_calls(prediction)
                if gold_calls and pred_calls:
                    name_hits.append(1.0 if gold_calls[0]["name"] == pred_calls[0]["name"] else 0.0)
                else:
                    name_hits.append(0.0)
            if len(samples) < 4:
                samples.append({
                    "task": "when2call",
                    "user": str(row.get("question") or _last_user(row))[:240],
                    "gold": f"{label}: {answer[:220]}",
                    "prediction": f"{predicted}: {prediction[:220]}",
                })
        if correct:
            metrics["decision_accuracy"] = sum(correct) / len(correct)
            metrics["support"] = float(len(correct))
        for label, values in recalls.items():
            metrics[f"recall_{label}"] = sum(values) / len(values)
            metrics[f"support_{label}"] = float(len(values))
        if name_hits:
            metrics["tool_name"] = sum(name_hits) / len(name_hits)

    elif kind in {"squad", "hotpot"}:
        answerable_em, answerable_f1, answerable_contains, answerable_abstain = [], [], [], []
        unanswerable_abstain = []
        ems, f1s = [], []
        for (row, prompt, answer), prediction in zip(prepared, predictions):
            if kind == "squad":
                golds = list((row.get("answers") or {}).get("text") or [])
            else:
                golds = [str(row.get("answer") or "")]
            abstained = is_abstention(prediction)
            if not golds:
                unanswerable_abstain.append(1.0 if abstained else 0.0)
            else:
                em, f1, contained = best_span_scores(prediction, golds)
                ems.append(em)
                f1s.append(f1)
                answerable_em.append(em)
                answerable_f1.append(f1)
                answerable_contains.append(contained)
                answerable_abstain.append(1.0 if abstained else 0.0)
            if len(samples) < 3:
                samples.append({
                    "task": kind,
                    "user": str(row.get("question") or "")[:240],
                    "gold": (golds[0] if golds else ABSTAIN_TEXT)[:200],
                    "prediction": prediction[:200],
                })
        if kind == "squad":
            if answerable_f1:
                metrics["answerable_em"] = sum(answerable_em) / len(answerable_em)
                metrics["answerable_f1"] = sum(answerable_f1) / len(answerable_f1)
                metrics["answerable_contains"] = sum(answerable_contains) / len(answerable_contains)
                metrics["answerable_abstain"] = sum(answerable_abstain) / len(answerable_abstain)
                metrics["support_answerable"] = float(len(answerable_f1))
            if unanswerable_abstain:
                metrics["unanswerable_abstain"] = sum(unanswerable_abstain) / len(unanswerable_abstain)
                metrics["support_unanswerable"] = float(len(unanswerable_abstain))
        elif ems:
            metrics["em"] = sum(ems) / len(ems)
            metrics["f1"] = sum(f1s) / len(f1s)
            metrics["contains"] = sum(answerable_contains) / len(answerable_contains)
            metrics["support"] = float(len(ems))

    elif kind == "gsm8k":
        hits = []
        for (row, prompt, answer), prediction in zip(prepared, predictions):
            hits.append(score_gsm(prediction, answer))
            if len(samples) < 3:
                samples.append({
                    "task": "gsm8k",
                    "user": str(row.get("question") or "")[:240],
                    "gold": answer[-80:],
                    "prediction": prediction[:240],
                })
        if hits:
            metrics["numeric_em"] = sum(hits) / len(hits)
            metrics["support"] = float(len(hits))

    elif kind == "constraints":
        check_scores, example_scores = [], []
        for (row, prompt, answer), prediction in zip(prepared, predictions):
            scored = score_constraints(_last_user(row), prediction)
            if scored is None:
                continue
            check_scores.append(scored["check_pass"])
            example_scores.append(scored["example_pass"])
            if len(samples) < 3:
                gold_checks = constraint_checks(_last_user(row), answer)
                samples.append({
                    "task": "constraints",
                    "user": _last_user(row)[:240],
                    "gold": answer[:220],
                    "prediction": prediction[:220],
                    "gold_checks_passed": sum(1 for _, ok in gold_checks if ok),
                    "gold_checks": len(gold_checks),
                })
        if example_scores:
            metrics["check_pass"] = sum(check_scores) / len(check_scores)
            metrics["example_pass"] = sum(example_scores) / len(example_scores)
            metrics["support"] = float(len(example_scores))

    return {"key": key, "kind": kind, "metrics": metrics, "samples": samples}


def selection_score(flat_metrics: dict[str, float]) -> float | None:
    """Mean of the behavior metrics. Loss is not mixed into this number."""
    wanted = (
        "name",
        "args",
        "restraint",
        "decision_accuracy",
        "answerable_f1",
        "answerable_contains",
        "unanswerable_abstain",
        "f1",
        "contains",
        "numeric_em",
        "example_pass",
    )
    values = []
    for key, value in flat_metrics.items():
        leaf = key.split("/")[-1]
        if leaf in wanted and not key.split("/")[-2:-1] == ["support"]:
            # support_* leaves end with support_name, not the metric itself
            if leaf.startswith("support"):
                continue
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)
