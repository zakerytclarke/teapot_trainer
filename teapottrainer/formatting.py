"""Turn dataset rows into one chat surface.

Presets still expose ``chat_template_fn(row) -> (prompt, answer)`` or a list of
those pairs (one pair per assistant turn). The prompt is the supervised prefix.
"""

from __future__ import annotations

import json
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


def format_gsm8k(row: dict) -> tuple[str, str]:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    prompt = (
        "System: Solve the grade-school math problem. "
        "Show brief reasoning, then end with a line of the form #### <number>.\n\n"
        f"User: {question}\n\nAssistant:\n"
    )
    return prompt, answer


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
