"""Turn dataset rows into one chat surface.

Presets still expose ``chat_template_fn(row) -> (prompt, answer)`` or a list of
those pairs (one pair per assistant turn). The prompt is the supervised prefix.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

ABSTAIN_TEXT = "The context does not contain the answer."
TOOLS_PRESENT_REFUSAL = "I cannot answer that with the available tools."

RAG_SYSTEM = (
    "Answer the question using only the context. "
    f"If the context does not contain the answer, reply exactly: {ABSTAIN_TEXT}"
)

# Hotpot rows all have an answer, so this prompt does not offer the abstention sentence.
ANSWER_SYSTEM = "Answer the question using only the context. Reply with the answer only."

DECISION_RULES = (
    "You are a careful assistant with access to tools.\n"
    "If a tool can satisfy the request and every required argument is present, "
    "reply with only a tool call:\n"
    '<tool_call>[{"name": "tool_name", "arguments": {"arg": "value"}}]</tool_call>\n'
    "If a required argument is missing, ask for it in plain text and do not call a tool.\n"
    "If no available tool fits the request, refuse in plain text and do not call a tool.\n"
    "Do not invent tools or arguments."
)


def rewrite_missing_tools_claim(answer: str, tools: str) -> str:
    """Keep the empty-tool refusal only when the tool list is empty.

    APIGen uses ``The query cannot be answered, no tools were provided.`` on
    rows whose prompt still lists tools. Training that sentence teaches the
    model to ignore the tool list.
    """
    if "no tools were provided" not in (answer or "").lower():
        return answer
    if not (tools or "").strip() or tools.strip() == "[]":
        return answer
    return TOOLS_PRESENT_REFUSAL


def _rag_prompt(context: str, question: str, system: str = RAG_SYSTEM) -> str:
    return (
        f"System: {system}\n\n"
        f"User: Context:\n{context}\n\nQuestion: {question}\n\nAssistant:\n"
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
    return prompt, rewrite_missing_tools_claim(answer, tools)


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
    tools = tools_blob(row.get("tools"))
    system = (
        f"{DECISION_RULES}\n\n"
        f"You have access to the following tools:\n<tools>{tools}</tools>"
    )
    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:\n"
    return prompt, rewrite_missing_tools_claim(answer, tools)


def format_squad(row: dict) -> tuple[str, str]:
    texts = (row.get("answers") or {}).get("text") or []
    answer = texts[0].strip() if texts else ABSTAIN_TEXT
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    return _rag_prompt(context, question), answer


def _hotpot_context(row: dict) -> str:
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
    return "\n\n".join(rendered_parts)


def format_hotpot(row: dict) -> tuple[str, str]:
    rendered = _hotpot_context(row)
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    return _rag_prompt(rendered, question, ANSWER_SYSTEM), answer


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


def _alias_name(taken: set[str]) -> str:
    """A new name on every example, so the model has to read the tool list."""
    for _ in range(64):
        alias = f"{random.choice(_ALIAS_VERBS)}_{random.choice(_ALIAS_NOUNS)}"
        if alias not in taken:
            taken.add(alias)
            return alias
    alias = f"rivet_spindle_{len(taken)}"
    taken.add(alias)
    return alias


def _aliased_messages(row: dict) -> list[dict] | None:
    cloned = [dict(message) for message in (row.get("messages") or [])]
    system_index = next((i for i, message in enumerate(cloned) if message.get("role") == "system"), None)
    if system_index is None:
        return None
    system = str(cloned[system_index].get("content") or "")
    match = re.search(r"<tools>(.*?)</tools>", system, re.DOTALL)
    if match is None:
        return None
    try:
        tools = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(tools, dict):
        tools = [tools]
    taken: set[str] = set()
    mapping: dict[str, str] = {}
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            old = tool["name"]
            mapping.setdefault(old, _alias_name(taken))
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
    return cloned


def format_apigen_aliased(row: dict) -> tuple[str, str] | None:
    """APIGen call under the shared tool template, with the function names replaced.

    Names are drawn again for every example. The gold call is renamed to match.
    The only way to be right is to read the tool list in the prompt.
    """
    cloned = _aliased_messages(row)
    if cloned is None:
        return format_apigen_decision(row)
    return format_apigen_decision({"messages": cloned})


def _message(messages: list[dict], role: str) -> dict | None:
    return next((message for message in messages if message.get("role") == role and message.get("content")), None)


def _gold_calls(answer: str) -> list[dict]:
    match = re.search(r"<tool_call>(.*)</tool_call>", answer or "", re.DOTALL)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    return [item for item in payload if isinstance(item, dict)]


def format_apigen_missing_argument(row: dict) -> tuple[str, str] | None:
    """Same tools and request, with one required value removed from the user text."""
    cloned = _aliased_messages(row)
    if not cloned:
        return None
    user = _message(cloned, "user")
    assistant = _message(cloned, "assistant")
    if user is None or assistant is None:
        return None
    calls = _gold_calls(str(assistant.get("content") or ""))
    if not calls:
        return None
    arguments = calls[0].get("arguments") or {}
    if not isinstance(arguments, dict):
        return None
    user_text = str(user.get("content") or "")
    for key, value in arguments.items():
        rendered = str(value).strip()
        if len(rendered) < 2 or rendered not in user_text:
            continue
        trimmed = user_text.replace(rendered, "", 1)
        trimmed = re.sub(r"\s{2,}", " ", trimmed).strip()
        if len(trimmed) < 8:
            continue
        user["content"] = trimmed
        assistant["content"] = f"What should I use for {key}?"
        return format_apigen_decision({"messages": cloned})
    return None


_UNRELATED_TOOL = {
    "description": "Convert a temperature from Celsius to Fahrenheit.",
    "parameters": {
        "type": "object",
        "properties": {"celsius": {"type": "number"}},
        "required": ["celsius"],
    },
}


def format_apigen_unrelated_tool(row: dict) -> tuple[str, str] | None:
    """A real request beside one tool whose description does not fit it."""
    cloned = _aliased_messages(row)
    if not cloned:
        return None
    user = _message(cloned, "user")
    assistant = _message(cloned, "assistant")
    system = _message(cloned, "system")
    if user is None or assistant is None or system is None:
        return None
    if "<tool_call>" not in str(assistant.get("content") or ""):
        return None
    lowered = str(user.get("content") or "").lower()
    if any(word in lowered for word in ("celsius", "fahrenheit", "temperature")):
        return None
    tool = {"name": _alias_name(set()), **_UNRELATED_TOOL}
    content = str(system.get("content") or "")
    match = re.search(r"<tools>.*?</tools>", content, re.DOTALL)
    if match is None:
        return None
    system["content"] = (
        content[: match.start()]
        + "<tools>"
        + json.dumps([tool], ensure_ascii=False)
        + "</tools>"
        + content[match.end() :]
    )
    assistant["content"] = TOOLS_PRESENT_REFUSAL
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


def format_synthqa_answerable(row: dict) -> tuple[str, str] | None:
    if row.get("type") == "unanswerable" or not _as_text(row.get("answer")):
        return None
    return format_synthqa(row)


def format_synthqa_unanswerable(row: dict) -> tuple[str, str] | None:
    if row.get("type") != "unanswerable" and _as_text(row.get("answer")):
        return None
    return format_synthqa(row)


def format_synthqa(row: dict) -> tuple[str, str] | None:
    context = _as_text(row.get("context"))
    question = _as_text(row.get("question"))
    if not context or not question:
        return None
    answer = _as_text(row.get("answer"))
    if row.get("type") == "unanswerable" or not answer:
        answer = ABSTAIN_TEXT
    return _rag_prompt(context, question), answer


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+(?:[A-Z][a-zA-Z0-9]+|of|the|and|de|da))*")
_CONTRASTS = (
    ("Who funded {entity}?", ("fund", "sponsor", "donor", "financ")),
    ("What is the email address for {entity}?", ("email", "@")),
    ("What is the phone number for {entity}?", ("phone", "telephone")),
    ("What was the budget of {entity}?", ("budget",)),
)


def contrastive_question(context: str, answer: str) -> str | None:
    """Ask about an entity that appears in the passage, using a fact that does not.

    The question is unanswerable from the passage, and the tempting entity is
    still sitting in it.
    """
    answer_norm = (answer or "").strip().lower()
    entities: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(context or ""):
        entity = match.group(0).strip()
        while " " in entity and entity.lower().rsplit(" ", 1)[-1] in {"of", "the", "and", "de", "da"}:
            entity = entity.rsplit(" ", 1)[0].strip()
        if len(entity) < 4 or entity in seen:
            continue
        if answer_norm and (entity.lower() in answer_norm or answer_norm in entity.lower()):
            continue
        seen.add(entity)
        entities.append(entity)
    if not entities:
        return None
    lowered = (context or "").lower()
    digest = hashlib.sha256((context or "")[:800].encode()).digest()
    start = digest[0] % len(entities)
    template_shift = digest[1] % len(_CONTRASTS)
    for entity_offset in range(len(entities)):
        entity = entities[(start + entity_offset) % len(entities)]
        for template_offset in range(len(_CONTRASTS)):
            template, cues = _CONTRASTS[(template_shift + template_offset) % len(_CONTRASTS)]
            if any(cue in lowered for cue in cues):
                continue
            return template.format(entity=entity)
    return None


def format_squad_contrastive(row: dict) -> tuple[str, str] | None:
    texts = (row.get("answers") or {}).get("text") or []
    answer = next((str(text).strip() for text in texts if str(text).strip()), "")
    if not answer:
        return None
    context = str(row.get("context") or "").strip()
    question = contrastive_question(context, answer)
    if not question:
        return None
    return _rag_prompt(context, question), ABSTAIN_TEXT


def format_synthqa_contrastive(row: dict) -> tuple[str, str] | None:
    if row.get("type") == "unanswerable" or not _as_text(row.get("answer")):
        return None
    context = _as_text(row.get("context"))
    question = contrastive_question(context, _as_text(row.get("answer")))
    if not context or not question:
        return None
    return _rag_prompt(context, question), ABSTAIN_TEXT


def format_hotpot_contrastive(row: dict) -> tuple[str, str] | None:
    context = _hotpot_context(row)
    question = contrastive_question(context, str(row.get("answer") or ""))
    if not context or not question:
        return None
    return _rag_prompt(context, question), ABSTAIN_TEXT


def _when2call_pair(row: dict) -> tuple[str, str] | None:
    formatted = format_when2call(row)
    if formatted is None:
        return None
    prompt, answer = formatted
    if not prompt or not answer:
        return None
    return prompt, answer


def format_when2call_ask(row: dict) -> tuple[str, str] | None:
    pair = _when2call_pair(row)
    if pair is None:
        return None
    prompt, answer = pair
    if "<tool_call>" in answer or "?" not in answer:
        return None
    return prompt, answer


def format_when2call_refuse(row: dict) -> tuple[str, str] | None:
    pair = _when2call_pair(row)
    if pair is None:
        return None
    prompt, answer = pair
    if "<tool_call>" in answer or "?" in answer:
        return None
    lowered = answer.lower()
    refuses = ("unable", "cannot", "can't", "can not", "sorry", "apolog", "no tool", "do not have", "don't have")
    if not any(phrase in lowered for phrase in refuses):
        return None
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
