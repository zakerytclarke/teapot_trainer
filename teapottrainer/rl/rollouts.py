"""Generate tool-using trajectories and collect token logprobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..tool_formats import ToolFormat, canonical_tool_calls, get_tool_format
from .tools import ToolFn, call_tool_by_name, tool_schemas


@dataclass
class Trajectory:
    """One finished rollout ready for grading / GRPO."""

    sample_id: str
    messages: list[dict]
    meta: dict[str, Any] = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)
    reward: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    completion_mask: list[int] = field(default_factory=list)
    logprob: float = 0.0
    # Tools are injected by the chat template, not stored as chat messages.
    # Keep the rendered first-turn prompt + schemas so rollouts.json shows them.
    tool_schemas: list[dict] = field(default_factory=list)
    rendered_prompt: str = ""
    tool_format: str = ""


def _assistant_has_tool_calls(message: dict) -> bool:
    return bool(message.get("tool_calls"))


def _decode_new_tokens(tokenizer, sequences: torch.Tensor, prompt_len: int) -> str:
    gen = sequences[0, prompt_len:]
    return tokenizer.decode(gen, skip_special_tokens=False)


def _token_logprobs_from_scores(
    scores: tuple[torch.Tensor, ...],
    sequences: torch.Tensor,
    prompt_len: int,
) -> list[float]:
    out: list[float] = []
    gen = sequences[0, prompt_len:]
    for step, step_scores in enumerate(scores):
        if step >= gen.numel():
            break
        log_probs = torch.log_softmax(step_scores[0].float(), dim=-1)
        token_id = int(gen[step].item())
        out.append(float(log_probs[token_id].item()))
    return out


@torch.no_grad()
def generate_assistant_message(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[ToolFn] | None,
    tool_format: ToolFormat | str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[dict, list[int], list[int], list[float], list[int], str]:
    """Sample one assistant turn. Tools are Python functions.

    Returns message, prompt_ids, completion_ids, token_logprobs, completion_mask,
    and the rendered prompt string (including any tool schema injection).
    """
    fmt = get_tool_format(tool_format)
    tool_fns = list(tools or [])

    prompt_text = fmt.build_prompt(
        tokenizer,
        messages,
        tools=tool_fns,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(prompt_text, str):
        prompt_text = str(prompt_text)
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_ids = inputs["input_ids"][0].tolist()

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) > 0,
        "temperature": max(float(temperature), 1e-5),
        "top_p": float(top_p),
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    outputs = model.generate(**inputs, **gen_kwargs)
    sequences = outputs.sequences
    prompt_len = inputs["input_ids"].shape[-1]
    text = _decode_new_tokens(tokenizer, sequences, prompt_len)
    if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
        text = text[: -len(tokenizer.eos_token)]

    prefix = inputs["input_ids"][0]
    message = fmt.parse_response(tokenizer, text, prefix=prefix, tools=tool_fns)
    if "role" not in message:
        message = {"role": "assistant", "content": text}
    if message.get("tool_calls"):
        message["tool_calls"] = canonical_tool_calls(message["tool_calls"])

    completion_ids = sequences[0, prompt_len:].tolist()
    token_logprobs = _token_logprobs_from_scores(outputs.scores or (), sequences, prompt_len)
    completion_mask = [1] * len(completion_ids)
    return message, prompt_ids, completion_ids, token_logprobs, completion_mask, prompt_text


def rollout_episode(
    model,
    tokenizer,
    sample: dict[str, Any],
    tool_format: ToolFormat | str,
    *,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Trajectory:
    """Run one tool-using episode from an RL sample dict.

    ``sample`` comes from ``sample_fn`` / ``sample_episode`` and must include
    ``messages``. Optional ``tools`` is a list of Python functions. The caller
    grades the finished ``trajectory.messages`` conversation JSON.
    """
    fmt = get_tool_format(tool_format)
    tools: list[ToolFn] = list(sample.get("tools") or [])
    messages = [dict(m) for m in sample["messages"]]
    meta = dict(sample.get("meta") or {})
    traj = Trajectory(
        sample_id=str(sample.get("sample_id") or "sample"),
        messages=messages,
        meta=meta,
        tool_schemas=tool_schemas(tools),
        tool_format=fmt.name,
    )

    all_completion_ids: list[int] = []
    all_masks: list[int] = []
    all_logprobs: list[float] = []
    first_prompt_ids: list[int] | None = None

    for _ in range(int(max_turns)):
        message, prompt_ids, completion_ids, token_logprobs, completion_mask, prompt_text = generate_assistant_message(
            model,
            tokenizer,
            messages,
            tools,
            fmt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if first_prompt_ids is None:
            first_prompt_ids = prompt_ids
            traj.rendered_prompt = prompt_text
        messages.append(message)
        traj.actions.append(message)
        all_completion_ids.extend(completion_ids)
        all_masks.extend(completion_mask)
        all_logprobs.extend(token_logprobs)

        if not _assistant_has_tool_calls(message):
            break

        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
            content = call_tool_by_name(tools, name, args)
            obs = {"role": "tool", "name": name, "content": content}
            if fmt.name != "tokenizer":
                obs["content"] = fmt.format_tool_result(name, content, call.get("id"))
            messages.append(obs)

    traj.messages = messages
    traj.prompt_ids = first_prompt_ids or []
    traj.completion_ids = all_completion_ids
    traj.completion_mask = all_masks
    traj.logprob = float(sum(all_logprobs)) if all_logprobs else 0.0
    return traj
