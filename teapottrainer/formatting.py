"""Turn dataset rows into one chat surface.

Presets still expose ``chat_template_fn(row) -> (prompt, answer)`` or a list of
those pairs (one pair per assistant turn). The prompt is the supervised prefix.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

ABSTAIN_TEXT = "The context does not contain the answer."

RAG_SYSTEM = (
    "Answer the question using only the context. "
    f"If the context does not contain the answer, reply exactly: {ABSTAIN_TEXT}"
)

DECISION_RULES = (
    "You are a careful assistant with access to tools.\n"
    "If a tool can satisfy the request and every required argument is present, "
    "reply with only a tool call:\n"
    '<tool_call>[{"name": "tool_name", "arguments": {"arg": "value"}}]</tool_call>\n'
    "If a required argument is missing, ask for it in plain text and do not call a tool.\n"
    "If no available tool fits the request, refuse in plain text and do not call a tool.\n"
    "Do not invent tools or arguments."
)


def render_message_pairs(messages: list[dict] | None) -> list[tuple[str, str]]:
    """One ``(prompt, answer)`` per assistant turn, including system text."""
    lines: list[str] = []
    pairs: list[tuple[str, str]] = []
    for message in messages or []:
        role = str(message.get("role") or "").lower()
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if role == "system" and content:
            lines.append(f"System: {content}")
        elif role == "user" and content:
            lines.append(f"User: {content}")
        elif role == "assistant" and content:
            prompt = "\n\n".join(lines + ["Assistant:"]) + "\n"
            pairs.append((prompt, content))
            lines.append(f"Assistant: {content}")
    return pairs


def messages_format(row: dict) -> list[tuple[str, str]]:
    return render_message_pairs(row.get("messages") or [])


def format_apigen_decision(row: dict) -> tuple[str, str]:
    """Same decision prompt as When2Call, with APIGen's tools and gold call.

    When2Call's training split has almost no positive tool calls. Putting
    APIGen calls in this prompt gives the model a call example under the
    prompt used at eval time.
    """
    messages = row.get("messages") or []
    system = ""
    user = ""
    answer = ""
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "system" and content and not system:
            system = content
        elif role == "user" and content and not user:
            user = content
        elif role == "assistant" and content:
            answer = content
    tools = ""
    match = re.search(r"<tools>(.*?)</tools>", system, re.DOTALL)
    if match:
        tools = match.group(1).strip()
    prompt_system = (
        f"{DECISION_RULES}\n\n"
        f"You have access to the following tools:\n<tools>{tools}</tools>"
    )
    prompt = f"System: {prompt_system}\n\nUser: {user}\n\nAssistant:\n"
    return prompt, answer


def tools_blob(tools: Any) -> str:
    parsed = []
    for tool in tools or []:
        if isinstance(tool, str):
            try:
                parsed.append(json.loads(tool))
            except json.JSONDecodeError:
                parsed.append(tool)
        else:
            parsed.append(tool)
    return json.dumps(parsed, ensure_ascii=False)


def format_when2call(row: dict) -> tuple[str, str]:
    messages = row.get("messages") or []
    if messages:
        user = ""
        answer = ""
        for message in messages:
            role = message.get("role")
            content = (message.get("content") or "").strip()
            if role == "user" and content and not user:
                user = content
            if role == "assistant" and content:
                answer = content
    else:
        user = str(row.get("question") or "").strip()
        answers = row.get("answers") or {}
        label = row.get("correct_answer") or ""
        answer = str(answers.get(label) or "").strip()
    system = (
        f"{DECISION_RULES}\n\n"
        f"You have access to the following tools:\n<tools>{tools_blob(row.get('tools'))}</tools>"
    )
    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:\n"
    return prompt, answer


def format_squad(row: dict) -> tuple[str, str]:
    texts = (row.get("answers") or {}).get("text") or []
    answer = texts[0].strip() if texts else ABSTAIN_TEXT
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    prompt = (
        f"System: {RAG_SYSTEM}\n\n"
        f"User: Context:\n{context}\n\nQuestion: {question}\n\nAssistant:\n"
    )
    return prompt, answer


def format_hotpot(row: dict) -> tuple[str, str]:
    context = row.get("context") or {}
    titles = list(context.get("title") or [])
    sentences = list(context.get("sentences") or [])
    support = {
        title
        for title in (row.get("supporting_facts") or {}).get("title") or []
        if isinstance(title, str)
    }
    distractors: list[str] = []
    supporting: list[str] = []
    for title, sents in zip(titles, sentences):
        body = " ".join(sentence for sentence in sents if isinstance(sentence, str))
        block = f"{title}\n{body}".strip()
        if title in support:
            supporting.append(block)
        else:
            distractors.append(block)
    # Distractors go first so a left-truncation keeps the supporting pages and the question.
    rendered_parts: list[str] = []
    used = 0
    for block in distractors + supporting:
        if rendered_parts and used + len(block) > 7000:
            break
        rendered_parts.append(block)
        used += len(block)
    for block in supporting:
        if block not in rendered_parts:
            rendered_parts.append(block)
    rendered = "\n\n".join(rendered_parts)
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    prompt = (
        f"System: {RAG_SYSTEM}\n\n"
        f"User: Context:\n{rendered}\n\nQuestion: {question}\n\nAssistant:\n"
    )
    return prompt, answer


_CALCULATOR_TOOLS = json.dumps(
    [
        {
            "name": "calculator",
            "description": "Evaluate one arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ],
    ensure_ascii=False,
)

_ALIAS_VERBS = (
    "cork", "latch", "braid", "quilt", "hinge", "wick", "parcel", "kindle",
    "frost", "scribe", "ladle", "notch", "plait", "rivet", "temper", "unveil",
)
_ALIAS_NOUNS = (
    "glowworm", "kiln", "lantern", "marble", "cinder", "vellum", "thimble", "comet",
    "cistern", "amber", "lodestone", "quill", "sconce", "keel", "mosaic", "spindle",
)


def _gsm_expressions(answer: str) -> list[str]:
    found = []
    for expr in re.findall(r"<<\s*([^=<>]+?)\s*=", answer or ""):
        cleaned = expr.strip().replace(",", "")
        if cleaned:
            found.append(cleaned)
    return found


def format_gsm8k(row: dict) -> tuple[str, str]:
    """Ask the calculator tool instead of a written solution."""
    question = str(row.get("question") or "").strip()
    exprs = _gsm_expressions(str(row.get("answer") or ""))
    if not exprs:
        exprs = ["0"]
    calls = [{"name": "calculator", "arguments": {"expression": expr}} for expr in exprs]
    answer = "<tool_call>" + json.dumps(calls, ensure_ascii=False) + "</tool_call>"
    prompt = (
        f"System: {DECISION_RULES}\n\n"
        f"You have access to the following tools:\n<tools>{_CALCULATOR_TOOLS}</tools>\n\n"
        f"User: {question}\n\nAssistant:\n"
    )
    return prompt, answer


def _alias_name(name: str, taken: set[str]) -> str:
    digest = hashlib.sha256(name.encode()).digest()
    base = f"{_ALIAS_VERBS[digest[0] % len(_ALIAS_VERBS)]}_{_ALIAS_NOUNS[digest[1] % len(_ALIAS_NOUNS)]}"
    alias = base
    suffix = 2
    while alias in taken:
        alias = f"{base}_{suffix}"
        suffix += 1
    taken.add(alias)
    return alias


def format_apigen_aliased(row: dict) -> tuple[str, str] | None:
    """APIGen call under the shared tool template, with the function names replaced.

    The gold call is renamed to match. The only way to be right is to read the
    tool list in the prompt.
    """
    cloned = []
    for message in row.get("messages") or []:
        cloned.append(dict(message))
    system_index = next((i for i, message in enumerate(cloned) if message.get("role") == "system"), None)
    if system_index is None:
        return format_apigen_decision({"messages": cloned})
    system = str(cloned[system_index].get("content") or "")
    match = re.search(r"<tools>(.*?)</tools>", system, re.DOTALL)
    if match is None:
        return format_apigen_decision({"messages": cloned})
    try:
        tools = json.loads(match.group(1))
    except json.JSONDecodeError:
        return format_apigen_decision({"messages": cloned})
    if isinstance(tools, dict):
        tools = [tools]
    taken: set[str] = set()
    mapping: dict[str, str] = {}
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            old = tool["name"]
            mapping.setdefault(old, _alias_name(old, taken))
            tool["name"] = mapping[old]
    system = system[: match.start(1)] + json.dumps(tools, ensure_ascii=False) + system[match.end(1) :]
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        system = system.replace(f'"{old}"', f'"{new}"')
    cloned[system_index]["content"] = system
    for message in cloned:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
            content = content.replace(f'"{old}"', f'"{new}"')
        message["content"] = content
    return format_apigen_decision({"messages": cloned})


def format_squad_answerable(row: dict) -> tuple[str, str] | None:
    texts = (row.get("answers") or {}).get("text") or []
    if not any(str(text).strip() for text in texts):
        return None
    return format_squad(row)


def format_squad_unanswerable(row: dict) -> tuple[str, str] | None:
    texts = (row.get("answers") or {}).get("text") or []
    if any(str(text).strip() for text in texts):
        return None
    return format_squad(row)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def format_synthqa(row: dict) -> tuple[str, str] | None:
    context = _as_text(row.get("context"))
    question = _as_text(row.get("question"))
    if not context or not question:
        return None
    answer = _as_text(row.get("answer"))
    if row.get("type") == "unanswerable" or not answer:
        answer = ABSTAIN_TEXT
    prompt = (
        f"System: {RAG_SYSTEM}\n\n"
        f"User: Context:\n{context}\n\nQuestion: {question}\n\nAssistant:\n"
    )
    return prompt, answer


def format_teapot_chat(row: dict) -> tuple[str, str] | None:
    system = _as_text(row.get("system_prompt")) or "You are Teapot. If a user asks who you are, reply \"I am Teapot\"."
    query = _as_text(row.get("query"))
    response = _as_text(row.get("response"))
    if not query or not response:
        return None
    context = _as_text(row.get("context"))
    user = f"Context:\n{context}\n\nQuestion: {query}" if context else query
    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:\n"
    return prompt, response


def as_pairs(formatted: Any) -> list[tuple[str, str]]:
    """Accept the original ``(prompt, answer)`` tuple or a list of them."""
    if formatted is None:
        return []
    if (
        isinstance(formatted, tuple)
        and len(formatted) == 2
        and isinstance(formatted[0], str)
        and isinstance(formatted[1], str)
    ):
        prompt, answer = formatted[0].strip(), formatted[1].strip()
        if prompt and answer:
            return [(prompt if prompt.endswith("\n") else prompt + "\n", answer)]
        return []
    if isinstance(formatted, list):
        pairs = []
        for item in formatted:
            pairs.extend(as_pairs(item))
        return pairs
    return []
