"""Extensible RL datasets.

An RL dataset is three callables plus optional Python tool functions:

* ``init_fn()`` — download or build whatever the dataset needs once
* ``sample_fn(state, rng)`` — pick one episode seed (messages + meta)
* ``grade_fn(conversation, meta)`` — score a finished conversation JSON

Add new presets as modules under this package (see ``synth_math.py``).
Search-backed presets (``NQ_OPEN_RL``, ``HOTPOT_RL``, …) ship with empty tools;
bind Brave/Exa/LLM-search via ``bind_tools``.
"""

from .base import (
    RlDataset,
    bind_tools,
    grade_episode,
    normalize_grade,
    prepare_dataset,
    sample_episode,
)
from .gsm8k import GSM8K_RL
from .synth_math import SYNTH_MATH
from .synthqa import SYNTHQA_RL
from .teapot_chat import TEAPOT_CHAT_RL
from .web_qa import HOTPOT_RL, NQ_OPEN_RL, TRIVIA_QA_RL, WEB_QUESTIONS_RL, grade_web_qa

__all__ = [
    "RlDataset",
    "SYNTH_MATH",
    "GSM8K_RL",
    "SYNTHQA_RL",
    "TEAPOT_CHAT_RL",
    "NQ_OPEN_RL",
    "HOTPOT_RL",
    "WEB_QUESTIONS_RL",
    "TRIVIA_QA_RL",
    "bind_tools",
    "grade_episode",
    "grade_web_qa",
    "normalize_grade",
    "prepare_dataset",
    "sample_episode",
]
