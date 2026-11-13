"""Extensible RL datasets.

An RL dataset is three callables plus optional Python tool functions:

* ``init_fn()`` — download or build whatever the dataset needs once
* ``sample_fn(state, rng)`` — pick one episode seed (messages + meta)
* ``grade_fn(conversation, meta)`` — score a finished conversation JSON

Add new presets as modules under this package (see ``synth_math.py``).
"""

from .base import (
    RlDataset,
    grade_episode,
    normalize_grade,
    prepare_dataset,
    sample_episode,
)
from .synth_math import SYNTH_MATH

__all__ = [
    "RlDataset",
    "SYNTH_MATH",
    "grade_episode",
    "normalize_grade",
    "prepare_dataset",
    "sample_episode",
]
