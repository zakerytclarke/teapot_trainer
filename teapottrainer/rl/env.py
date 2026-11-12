"""RL environment interfaces.

An environment turns a sampled task into an initial message list, executes
tool calls, and scores a finished trajectory. Envs do not own the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .tools import run_tool


@dataclass
class Task:
    """One prompt the policy must solve."""

    task_id: str
    messages: list[dict]
    gold: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] | None = None


@dataclass
class StepResult:
    observation_message: dict
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task: Task
    messages: list[dict]
    actions: list[dict] = field(default_factory=list)
    reward: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    # Token-level training payload filled by the rollout helper.
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    completion_mask: list[int] = field(default_factory=list)
    logprob: float = 0.0


class Env:
    """Base environment."""

    name: str = "base"

    def __init__(self, options: dict | None = None):
        self.options = dict(options or {})

    def sample_task(self, rng) -> Task:
        raise NotImplementedError

    def execute_tool(self, name: str, arguments: dict | None, task: Task) -> StepResult:
        content = run_tool(name, arguments)
        return StepResult(
            observation_message={
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def score(self, trajectory: Trajectory) -> tuple[float, dict[str, float]]:
        raise NotImplementedError


def final_assistant_text(messages: list[dict]) -> str:
    """Last assistant content that is not only a tool call."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if message.get("tool_calls"):
            continue
        if content is None:
            continue
        return str(content)
    return ""


def all_tool_calls(messages: list[dict]) -> list[dict]:
    calls = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict):
                calls.append(
                    {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or {},
                    }
                )
    return calls
