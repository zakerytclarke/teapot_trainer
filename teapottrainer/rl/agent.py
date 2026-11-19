"""Inference-time tool agent loop (same path RL rollouts use).

Load a HF causal LM + tokenizer, pass ordinary Python tool functions, and run
until the model stops calling tools or ``max_turns`` is hit::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from teapottrainer import build_browser_tools, build_search_tools, calculator, run_agent

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tools = [calculator, *build_search_tools(mock=True), *build_browser_tools(mock=True)]
    messages = [{"role": "user", "content": "What is 17*19? Use the calculator."}]
    result = run_agent(model, tokenizer, messages, tools, tool_format="hermes")
    print(result.messages)
"""

from __future__ import annotations

from typing import Any

from ..tool_formats import ToolFormat
from .rollouts import Trajectory, rollout_episode
from .tools import ToolFn


def run_agent(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[ToolFn] | None = None,
    *,
    tool_format: ToolFormat | str = "hermes",
    max_turns: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    sample_id: str = "agent",
    meta: dict[str, Any] | None = None,
) -> Trajectory:
    """Run a full tool-using conversation and return the finished trajectory.

    ``messages`` is a standard chat list (system/user/assistant/tool). ``tools``
    are Python callables (same contract as Hugging Face ``tools=``). The loop
    mirrors Transformers tool use: generate → parse tool_calls → execute →
    append tool results → repeat.
    """
    if not messages:
        raise ValueError("messages must be a non-empty chat list")
    sample = {
        "sample_id": str(sample_id),
        "messages": [dict(m) for m in messages],
        "tools": list(tools or []),
        "meta": dict(meta or {}),
    }
    return rollout_episode(
        model,
        tokenizer,
        sample,
        tool_format,
        max_turns=int(max_turns),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
