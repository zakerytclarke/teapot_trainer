"""Generate tool-using trajectories and collect token logprobs."""

from __future__ import annotations

from typing import Any

import torch

from ..tool_formats import ToolFormat, canonical_tool_calls, get_tool_format
from .env import Env, Task, Trajectory
from .tools import resolve_tools, run_tool, tool_schemas_for


def _assistant_has_tool_calls(message: dict) -> bool:
    return bool(message.get("tool_calls"))


def _decode_new_tokens(tokenizer, sequences: torch.Tensor, prompt_len: int) -> str:
    gen = sequences[0, prompt_len:]
    return tokenizer.decode(gen, skip_special_tokens=False)


def _token_logprobs_from_scores(scores: tuple[torch.Tensor, ...], sequences: torch.Tensor, prompt_len: int) -> list[float]:
    """Per-generated-token logprob from ``model.generate`` scores."""
    out: list[float] = []
    gen = sequences[0, prompt_len:]
    for step, step_scores in enumerate(scores):
        if step >= gen.numel():
            break
        log_probs = torch.log_softmax(step_scores[0].float(), dim=-1)
        token_id = int(gen[step].item())
        out.append(float(log_probs[token_id].item()))
    return out


def _completion_mask_for_message(tokenizer, message: dict, tool_format: ToolFormat) -> list[int]:
    """Mask of 1s over the serialized assistant/tool-call tokens for this message."""
    if message.get("role") == "assistant" and message.get("tool_calls"):
        text = tool_format.format_assistant_tool_calls(
            message["tool_calls"],
            str(message.get("content") or ""),
        )
    else:
        text = str(message.get("content") or "")
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [1] * len(ids)


@torch.no_grad()
def generate_assistant_message(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[str] | None,
    tool_format: ToolFormat | str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[dict, list[int], list[int], list[float], list[int]]:
    """Sample one assistant turn.

    Returns:
        message, prompt_ids, completion_ids, token_logprobs, completion_mask
    """
    fmt = get_tool_format(tool_format)
    schemas = tool_schemas_for(tools)
    callables = resolve_tools(tools)

    # Prefer an untokenized string from the tool format, then tokenize ourselves.
    # That avoids version-specific apply_chat_template return types (tensor /
    # BatchEncoding / tokenizers.Encoding) when tools are involved.
    prompt_text = fmt.build_prompt(
        tokenizer,
        messages,
        tools=callables or schemas,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(prompt_text, str):
        # Named formats may still return token ids on exotic tokenizers.
        if hasattr(prompt_text, "input_ids"):
            prompt_text = tokenizer.decode(
                prompt_text["input_ids"][0]
                if getattr(prompt_text["input_ids"], "dim", lambda: 1)() > 1
                else prompt_text["input_ids"],
                skip_special_tokens=False,
            )
        else:
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
    # Drop trailing eos for parsing.
    if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
        text = text[: -len(tokenizer.eos_token)]

    prefix = inputs["input_ids"][0]
    message = fmt.parse_response(tokenizer, text, prefix=prefix, tools=callables or schemas)
    if "role" not in message:
        message = {"role": "assistant", "content": text}
    # Ensure tool_calls are canonical when present.
    if message.get("tool_calls"):
        message["tool_calls"] = canonical_tool_calls(message["tool_calls"])

    completion_ids = sequences[0, prompt_len:].tolist()
    token_logprobs = _token_logprobs_from_scores(outputs.scores or (), sequences, prompt_len)
    # Train on all generated tokens for this turn.
    completion_mask = [1] * len(completion_ids)
    return message, prompt_ids, completion_ids, token_logprobs, completion_mask


def rollout_episode(
    model,
    tokenizer,
    env: Env,
    task: Task,
    tool_format: ToolFormat | str,
    *,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Trajectory:
    """Run one tool-using episode and score it."""
    fmt = get_tool_format(tool_format)
    tools = task.allowed_tools
    messages = [dict(m) for m in task.messages]
    traj = Trajectory(task=task, messages=messages)

    all_completion_ids: list[int] = []
    all_masks: list[int] = []
    all_logprobs: list[float] = []
    first_prompt_ids: list[int] | None = None

    for _ in range(int(max_turns)):
        message, prompt_ids, completion_ids, token_logprobs, completion_mask = generate_assistant_message(
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
        messages.append(message)
        traj.actions.append(message)
        all_completion_ids.extend(completion_ids)
        all_masks.extend(completion_mask)
        all_logprobs.extend(token_logprobs)

        if not _assistant_has_tool_calls(message):
            break

        # Execute tools sequentially and append observations.
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
            step = env.execute_tool(name, args, task)
            obs = dict(step.observation_message)
            # Keep result formatting consistent for named wire formats.
            if fmt.name != "tokenizer":
                obs["content"] = fmt.format_tool_result(
                    name,
                    str(obs.get("content") or ""),
                    call.get("id"),
                )
            messages.append(obs)
            # Tool observations are not policy actions; no completion tokens.

    traj.messages = messages
    reward, metrics = env.score(traj)
    traj.reward = float(reward)
    traj.metrics = {k: float(v) for k, v in metrics.items()}
    traj.prompt_ids = first_prompt_ids or []
    traj.completion_ids = all_completion_ids
    traj.completion_mask = all_masks
    traj.logprob = float(sum(all_logprobs)) if all_logprobs else 0.0
    return traj
