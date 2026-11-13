"""Reinforcement learning helpers.

``RlTrainer`` lives in ``teapottrainer.rl.trainer`` and is re-exported from the
top-level package. Dataset presets live under ``teapottrainer.rl.datasets``.
This ``__init__`` only exposes leaf modules so dataset presets can import
tools/rewards without circular imports.
"""

from .search import (
    SearchSettings,
    build_search_tools,
    make_brave_search,
    make_exa_search,
    make_llm_search,
    make_llmsearchindex,
)
from .tools import calculator, call_tool, call_tool_by_name, tool_schemas

__all__ = [
    "SearchSettings",
    "build_search_tools",
    "make_brave_search",
    "make_exa_search",
    "make_llm_search",
    "make_llmsearchindex",
    "calculator",
    "call_tool",
    "call_tool_by_name",
    "tool_schemas",
]
