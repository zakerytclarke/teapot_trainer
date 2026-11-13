"""Tiny helpers for Hugging Face-backed RL datasets."""

from __future__ import annotations

import random
from typing import Any, Sequence

from datasets import load_dataset


def load_hf_rows(name: str, *, subset: str | None = None, split: str = "train") -> list[dict]:
    """Materialize a split into a plain list of dicts (fine for mid-size QA sets)."""
    if subset:
        ds = load_dataset(name, subset, split=split)
    else:
        ds = load_dataset(name, split=split)
    return [dict(row) for row in ds]


def pick_row(rows: Sequence[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    if not rows:
        raise ValueError("dataset has no rows")
    return dict(rows[int(rng.randrange(len(rows)))])
