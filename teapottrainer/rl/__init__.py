"""Reinforcement learning helpers.

``RlTrainer`` lives in ``teapottrainer.rl.trainer`` and is re-exported from the
top-level package. Dataset presets live under ``teapottrainer.rl.datasets``.
This ``__init__`` only exposes leaf modules so dataset presets can import
tools/rewards without circular imports.
"""

from .agent import run_agent
from .browser import (
    BrowserSession,
    BrowserSettings,
    BrowserToolkit,
    build_browser_toolkit,
    build_browser_tools,
)
from .rollouts import Trajectory, rollout_episode
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
    "run_agent",
    "Trajectory",
    "rollout_episode",
    "BrowserSession",
    "BrowserSettings",
    "BrowserToolkit",
    "build_browser_toolkit",
    "build_browser_tools",
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
