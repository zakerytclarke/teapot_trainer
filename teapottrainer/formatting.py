"""Turn dataset rows into one chat surface.

Presets still expose ``chat_template_fn(row) -> (prompt, answer)`` or a list of
those pairs (one pair per assistant turn). The prompt is the supervised prefix.
The active chat format decides how those roles are written. Pretrain text never
comes through here.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from .chat_formats import current_chat_format

ABSTAIN_TEXT = "The context does not contain the answer."
# Benchmark prompts that have no reference reply. Loss is not trained against this.
NO_REFERENCE = "<no_reference>"
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


def render_prompt(messages: list[dict]) -> str:
    """Format role messages up through the assistant prefix for the active format."""
    cleaned = []
    for message in messages:
        role = str(message.get("role") or "").lower()
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            cleaned.append({"role": role, "content": content})
    if not cleaned:
        return ""
    return current_chat_format().render(cleaned, add_generation_prompt=True)


def _rag_prompt(context: str, question: str, system: str = RAG_SYSTEM) -> str:
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return render_prompt(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


def render_message_pairs(messages: list[dict] | None) -> list[tuple[str, str]]:
    """One ``(prompt, answer)`` per assistant turn, including system text."""
    history: list[dict] = []
    pairs: list[tuple[str, str]] = []
    for message in messages or []:
        role = str(message.get("role") or "").lower()
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if role in {"system", "user"} and content:
            history.append({"role": role, "content": content})
        elif role == "assistant" and content:
            prompt = render_prompt(history)
            if prompt:
                pairs.append((prompt, content))
            history.append({"role": "assistant", "content": content})
    return pairs


def messages_format(row: dict) -> list[tuple[str, str]]:
    return render_message_pairs(row.get("messages") or [])


def assistant_histories(messages: list[dict] | None) -> list[tuple[list[dict], str]]:
    """History before each assistant turn, and that turn's answer.

    The history is what the retriever packs when the rendered chat is longer
    than the position window. The answer itself is not part of the pack.
    """
    history: list[dict] = []
    turns: list[tuple[list[dict], str]] = []
    for message in messages or []:
        role = str(message.get("role") or "").lower()
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if role in {"system", "user"} and content:
            history.append({"role": role, "content": content})
        elif role == "assistant" and content:
            turns.append(([dict(item) for item in history], content))
            history.append({"role": "assistant", "content": content})
    return turns


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
    prompt = render_prompt(
        [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": user},
        ]
    )
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
    prompt = render_prompt(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
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
    system = (
        f"{DECISION_RULES}\n\n"
        f"You have access to the following tools:\n<tools>{_CALCULATOR_TOOLS}</tools>"
    )
    prompt = render_prompt(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
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
    prompt = render_prompt(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return prompt, response


def _chat_prompt(user: str) -> str:
    return render_prompt([{"role": "user", "content": user.strip()}])


def format_ifeval(row: dict) -> tuple[str, str] | None:
    prompt = _as_text(row.get("prompt"))
    if not prompt:
        return None
    return _chat_prompt(prompt), NO_REFERENCE


def format_mt_bench(row: dict) -> list[tuple[str, str]] | None:
    """One pair per turn. A turn without a reference stops the chain.

    Later turns include the dataset's earlier reference answers. They do not
    include a previous model generation, which the official second-turn judge
    would use.
    """
    prompts = [_as_text(prompt) for prompt in (row.get("prompt") or [])]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        return None
    references = list(row.get("reference") or [])
    pairs = []
    history: list[dict] = []
    for index, user in enumerate(prompts):
        reference = ""
        if index < len(references) and isinstance(references[index], str):
            reference = references[index].strip()
        history.append({"role": "user", "content": user})
        prompt = render_prompt(history)
        pairs.append((prompt, reference if reference else NO_REFERENCE))
        if not reference:
            break
        history.append({"role": "assistant", "content": reference})
    return pairs


def format_alpaca(row: dict) -> tuple[str, str] | None:
    """Stanford Alpaca instruction, plus the optional input, in the active chat format.

    The published ``text`` column is the old ``### Instruction`` template. The
    supervised mix speaks the chat format instead, so that column is ignored.
    """
    instruction = _as_text(row.get("instruction"))
    context = _as_text(row.get("input"))
    output = _as_text(row.get("output"))
    if not instruction or not output:
        return None
    user = instruction if not context else f"{instruction}\n\n{context}"
    return _chat_prompt(user), output


def format_alpaca_eval(row: dict) -> tuple[str, str] | None:
    instruction = _as_text(row.get("instruction"))
    output = _as_text(row.get("output"))
    if not instruction or not output:
        return None
    return _chat_prompt(instruction), output


def format_arena_hard(row: dict) -> tuple[str, str] | None:
    turns = row.get("turns") or []
    content = ""
    if turns:
        first = turns[0]
        if isinstance(first, dict):
            content = _as_text(first.get("content"))
        else:
            content = _as_text(first)
    if not content:
        return None
    return _chat_prompt(content), NO_REFERENCE


def _choice_prompt(stem: str, options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [f"{letters[index]}. {text}" for index, text in enumerate(options)]
    user = stem.strip() + "\n\n" + "\n".join(lines) + "\n\nReply with the correct option only."
    return _chat_prompt(user)


def _choice_index(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _choice_view(stem: str, options: list[str], index: int | None) -> tuple[str, str, list[str], int] | None:
    cleaned = [text.strip() for text in options]
    if index is None or index < 0 or index >= len(cleaned) or len(cleaned) < 2:
        return None
    if not cleaned[index] or not stem.strip():
        return None
    return _choice_prompt(stem, cleaned), cleaned[index], cleaned, index


def multiple_choice_view(name: str, row: dict) -> tuple[str, str, list[str], int] | None:
    """Prompt, correct option text, every option, and the correct index."""
    lowered = (name or "").lower()
    if "ai2_arc" in lowered:
        choices = row.get("choices") or {}
        texts = [str(text) for text in (choices.get("text") or [])]
        labels = [str(label).strip() for label in (choices.get("label") or [])]
        key = str(row.get("answerKey") or "").strip()
        if key not in labels:
            return None
        return _choice_view(_as_text(row.get("question")), texts, labels.index(key))
    if "hellaswag" in lowered:
        endings = [str(text) for text in (row.get("endings") or [])]
        stem = "Complete the sentence.\n\n" + _as_text(row.get("ctx"))
        return _choice_view(stem, endings, _choice_index(row.get("label")))
    if "piqa" in lowered:
        options = [_as_text(row.get("sol1")), _as_text(row.get("sol2"))]
        return _choice_view(_as_text(row.get("goal")), options, _choice_index(row.get("label")))
    if "winogrande" in lowered:
        sentence = _as_text(row.get("sentence"))
        options = [_as_text(row.get("option1")), _as_text(row.get("option2"))]
        index = _choice_index(row.get("answer"))
        if index is not None:
            index -= 1
        return _choice_view(f"Fill in the blank.\n\n{sentence}", options, index)
    return None


def _format_choice(name: str, row: dict) -> tuple[str, str] | None:
    viewed = multiple_choice_view(name, row)
    if viewed is None:
        return None
    prompt, gold, _options, _index = viewed
    return prompt, gold


def format_arc(row: dict) -> tuple[str, str] | None:
    return _format_choice("allenai/ai2_arc", row)


def format_hellaswag(row: dict) -> tuple[str, str] | None:
    return _format_choice("Rowan/hellaswag", row)


def format_piqa(row: dict) -> tuple[str, str] | None:
    return _format_choice("ybisk/piqa", row)


def format_winogrande(row: dict) -> tuple[str, str] | None:
    return _format_choice("allenai/winogrande", row)


def format_coqa(row: dict) -> list[tuple[str, str]] | None:
    """One pair per question. Later turns include the earlier answers."""
    story = _as_text(row.get("story"))
    questions = [_as_text(question) for question in (row.get("questions") or [])]
    answers = [_as_text(answer) for answer in ((row.get("answers") or {}).get("input_text") or [])]
    if not story or not questions or not answers:
        return None
    pairs = []
    earlier: list[str] = []
    for question, answer in zip(questions, answers):
        if not question or not answer:
            break
        prior = "\n\n".join(earlier)
        user = f"Context:\n{story}\n\n{prior}\n\nQuestion: {question}" if prior else f"Context:\n{story}\n\nQuestion: {question}"
        prompt = render_prompt(
            [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": user},
            ]
        )
        pairs.append((prompt, answer))
        earlier.append(f"Question: {question}\nAnswer: {answer}")
    return pairs or None


def format_drop(row: dict) -> tuple[str, str] | None:
    spans = [_as_text(span) for span in ((row.get("answers_spans") or {}).get("spans") or [])]
    spans = [span for span in spans if span]
    passage = _as_text(row.get("passage"))
    question = _as_text(row.get("question"))
    if not spans or not passage or not question:
        return None
    return _rag_prompt(passage, question, ANSWER_SYSTEM), spans[0]


_TOOL_TAGS = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _calls_from_tags(text: str) -> list[dict]:
    calls = []
    for payload in _TOOL_TAGS.findall(text or ""):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            calls.append({"name": item["name"].strip(), "arguments": arguments})
    return calls


def _tool_call_text(calls: list[dict]) -> str:
    rendered = []
    for call in calls:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        rendered.append({
            "name": call["name"],
            "arguments": {str(key): arguments[key] for key in sorted(arguments, key=str)},
        })
    return "<tool_call>" + json.dumps(rendered, ensure_ascii=False) + "</tool_call>"


def hermes_turns(row: dict) -> list[tuple[list[dict], str]]:
    """History before each Hermes reply, using the rewritten tool-call answer.

    The dataset system text is replaced by the shared tool rules. Tool results
    stay in the next prompt as user turns.
    """
    tools = _as_text(row.get("tools"))
    system = DECISION_RULES if not tools else f"{DECISION_RULES}\nAvailable tools:\n{tools}"
    history = [{"role": "system", "content": system}]
    turns: list[tuple[list[dict], str]] = []
    for message in row.get("conversations") or []:
        who = str(message.get("from") or "").lower()
        value = _as_text(message.get("value"))
        if not value or who == "system":
            continue
        if who == "human":
            history.append({"role": "user", "content": value})
            continue
        if who == "tool":
            history.append({"role": "user", "content": f"Tool result:\n{value}"})
            continue
        if who != "gpt":
            continue
        calls = _calls_from_tags(value)
        answer = _tool_call_text(calls) if calls else value
        turns.append(([dict(item) for item in history], answer))
        history.append({"role": "assistant", "content": answer})
    return turns


def format_hermes(row: dict) -> list[tuple[str, str]] | None:
    """Hermes traces rewritten into the library tool-call format.

    Tool results stay in the next prompt, so the following answer is conditioned
    on them. The dataset's own system text is replaced by the shared tool rules.
    """
    pairs = [
        (render_prompt(history), answer)
        for history, answer in hermes_turns(row)
        if render_prompt(history)
    ]
    return pairs or None


def bfcl_concrete_calls(ground_truth: Any) -> list[dict]:
    """One call per ground-truth entry, using the first listed value of each argument.

    An acceptable value of ``""`` means the argument may be omitted. A training
    target still fills the argument when any other value is listed.
    """
    if not isinstance(ground_truth, list):
        return []
    calls = []
    for entry in ground_truth:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        name, params = next(iter(entry.items()))
        if not isinstance(name, str) or not isinstance(params, dict):
            continue
        arguments = {}
        for key, options in params.items():
            if not isinstance(options, list):
                arguments[str(key)] = options
                continue
            chosen = next((option for option in options if option not in ("", None)), None)
            if chosen is None:
                continue
            arguments[str(key)] = chosen
        calls.append({"name": name, "arguments": arguments})
    return calls


def format_bfcl(row: dict) -> tuple[str, str] | None:
    """BFCL question plus the first acceptable call, or a refusal when no call is valid."""
    users = []
    for turn in row.get("question") or []:
        messages = turn if isinstance(turn, list) else [turn]
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                text = _as_text(message.get("content"))
                if text:
                    users.append(text)
    if not users:
        return None
    tools = json.dumps(row.get("function") or [], ensure_ascii=False)
    system = f"{DECISION_RULES}\nAvailable tools:\n{tools}"
    user = users[0] if len(users) == 1 else "\n".join(users)
    prompt = render_prompt(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    calls = bfcl_concrete_calls(row.get("ground_truth"))
    if not calls:
        return prompt, TOOLS_PRESENT_REFUSAL
    return prompt, _tool_call_text(calls)


def as_pairs(formatted: Any) -> list[tuple[str, str]]:
    """Accept the original ``(prompt, answer)`` tuple or a list of them.

    The plain format keeps its trailing newline. Every other format keeps the
    template text exactly, so it matches ``apply_chat_template``.
    """
    if formatted is None:
        return []
    if (
        isinstance(formatted, tuple)
        and len(formatted) == 2
        and isinstance(formatted[0], str)
        and isinstance(formatted[1], str)
    ):
        plain = current_chat_format().name == "plain"
        prompt = formatted[0].strip() if plain else formatted[0]
        answer = formatted[1] if formatted[1] == NO_REFERENCE else formatted[1].strip()
        if prompt.strip() and answer:
            if plain and not prompt.endswith("\n"):
                prompt += "\n"
            return [(prompt, answer)]
        return []
    if isinstance(formatted, list):
        pairs = []
        for item in formatted:
            pairs.extend(as_pairs(item))
        return pairs
    return []
