"""Ready-made Hugging Face sources.

These are thin constructors. Pass any of them as a stage source, or build
your own object with a ``build(tokenizer, seq_len, token_budget, seed)`` method.
"""
from __future__ import annotations

from .data import HuggingFaceSFT, HuggingFaceText


def fineweb_edu(cache_tokens: int | None = None, weight: float = 1.0) -> HuggingFaceText:
    return HuggingFaceText(
        dataset="HuggingFaceFW/fineweb-edu",
        name="fineweb-edu",
        split="train",
        text_column="text",
        weight=weight,
        cache_tokens=cache_tokens,
    )


def ultrafineweb(cache_tokens: int | None = None, weight: float = 1.0) -> HuggingFaceText:
    return HuggingFaceText(
        dataset="openbmb/Ultra-FineWeb",
        name="ultrafineweb",
        split="train",
        text_column="content",
        weight=weight,
        cache_tokens=cache_tokens,
    )


def smoltalk(max_rows: int | None = None, weight: float = 1.0, subset: str = "all") -> HuggingFaceSFT:
    def formatter(row):
        return row.get("messages") or []

    return HuggingFaceSFT(
        dataset="HuggingFaceTB/smoltalk",
        name="smoltalk",
        subset=subset,
        split="train",
        formatter=formatter,
        weight=weight,
        max_rows=max_rows,
    )


def gsm8k(split: str = "train", max_rows: int | None = None, weight: float = 1.0) -> HuggingFaceSFT:
    def formatter(row):
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not question or not answer:
            return []
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    return HuggingFaceSFT(
        dataset="gsm8k",
        name="gsm8k",
        subset="main",
        split=split,
        formatter=formatter,
        weight=weight,
        max_rows=max_rows,
    )
