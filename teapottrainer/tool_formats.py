"""Configurable tool prompt + call formats.

Internal conversation state always uses the Hugging Face / OpenAI-style message
shape that ``tokenizer.apply_chat_template`` expects:

* tools are JSON schemas (or Python callables converted via ``get_json_schema``)
* assistant tool requests live in ``message["tool_calls"]``
* tool results are ``{"role": "tool", "content": "..."}`` messages

``tool_format="tokenizer"`` (default) lets the tokenizer's chat template decide
how tools are written into the prompt and, when available, uses
``tokenizer.parse_response`` to recover structured assistant messages.

Named formats (``hermes``, ``llama3_json``, ``mistral``, ``json``) override both
sides: how tool schemas are injected into the prompt text, and how model
outputs / tool results are serialized and parsed. Use them when a checkpoint's
template is missing, broken, or you want a fixed wire format across models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union

from transformers.utils.chat_template_utils import get_json_schema

ToolSchema = dict[str, Any]
ToolLike = Union[ToolSchema, Callable[..., Any]]

_ALIASES = {
    "hf": "tokenizer",
    "transformers": "tokenizer",
    "chat_template": "tokenizer",
    "nous": "hermes",
    "qwen": "hermes",
    "qwen2": "hermes",
    "qwen2.5": "hermes",
    "llama3": "llama3_json",
    "llama-3": "llama3_json",
    "llama3.1": "llama3_json",
    "openai_json": "json",
}


def normalize_tool_schemas(tools: Iterable[ToolLike] | None) -> list[ToolSchema]:
    """Convert Python functions or raw schemas into HF tool JSON schemas."""
    if not tools:
        return []
    out: list[ToolSchema] = []
    for tool in tools:
        if callable(tool) and not isinstance(tool, dict):
            out.append(get_json_schema(tool))
            continue
        if not isinstance(tool, dict):
            raise TypeError(f"Tool must be a dict schema or callable, got {type(tool)!r}")
        if "type" in tool and "function" in tool:
            out.append(tool)
        elif "name" in tool and "parameters" in tool:
            out.append({"type": "function", "function": tool})
        else:
            raise ValueError(f"Unrecognized tool schema keys: {sorted(tool)}")
    return out


def tool_names(schemas: list[ToolSchema]) -> list[str]:
    names = []
    for schema in schemas:
        fn = schema.get("function") if isinstance(schema, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def canonical_tool_calls(calls: list[dict] | None) -> list[dict]:
    """Normalize parsed calls into HF assistant ``tool_calls`` entries."""
    out = []
    for raw in calls or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
            fn = dict(raw["function"])
            name = fn.get("name")
            args = fn.get("arguments", fn.get("parameters", {}))
        else:
            name = raw.get("name")
            args = raw.get("arguments", raw.get("parameters", {}))
        if not name:
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        entry = {
            "type": "function",
            "function": {"name": str(name), "arguments": args},
        }
        if raw.get("id"):
            entry["id"] = raw["id"]
        out.append(entry)
    return out


def _simple_calls(calls: list[dict]) -> list[dict]:
    simple = []
    for call in canonical_tool_calls(calls):
        fn = call["function"]
        simple.append({"name": fn["name"], "arguments": fn.get("arguments") or {}})
    return simple


_HERMES_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_MISTRAL_BLOCK = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\])", re.DOTALL)
_LLAMA_PY = re.compile(r"<\|python_tag\|>\s*(\{.*\})", re.DOTALL)


def _loads_json_blob(text: str) -> Any | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _calls_from_json_value(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "tool_calls" in value and isinstance(value["tool_calls"], list):
            return canonical_tool_calls(value["tool_calls"])
        if "name" in value:
            return canonical_tool_calls([value])
        return []
    if isinstance(value, list):
        return canonical_tool_calls(value)
    return []


@dataclass(frozen=True)
class ToolFormat:
    """One wire format for tool schemas in the prompt and tool calls in the reply."""

    name: str
    description: str = ""

    def render_tools_block(self, schemas: list[ToolSchema]) -> str:
        """Human-readable tool listing used by named (non-tokenizer) formats."""
        raise NotImplementedError

    def inject_tools(self, messages: list[dict], schemas: list[ToolSchema]) -> list[dict]:
        """Return messages with tool instructions baked in for named formats."""
        if not schemas:
            return [dict(m) for m in messages]
        block = self.render_tools_block(schemas)
        out = [dict(m) for m in messages]
        if out and out[0].get("role") == "system":
            content = str(out[0].get("content") or "")
            out[0]["content"] = (content + "\n\n" + block).strip()
        else:
            out.insert(0, {"role": "system", "content": block})
        return out

    def format_assistant_tool_calls(self, calls: list[dict], content: str = "") -> str:
        """Serialize canonical tool calls into the text the model is taught to emit."""
        raise NotImplementedError

    def format_tool_result(self, name: str, content: str, call_id: str | None = None) -> str:
        """Serialize a tool result for formats that put results in plain content."""
        return str(content)

    def parse_tool_calls(self, text: str) -> list[dict]:
        """Parse model text into canonical HF ``tool_calls`` entries."""
        raise NotImplementedError

    def parse_assistant_message(self, text: str) -> dict:
        """Parse a generated assistant string into a message dict."""
        calls = self.parse_tool_calls(text)
        if calls:
            # Keep any prose outside the first tool markup as content.
            content = text
            for pattern in (_HERMES_BLOCK, _MISTRAL_BLOCK, _LLAMA_PY):
                content = pattern.sub("", content)
            content = content.strip()
            msg: dict[str, Any] = {"role": "assistant", "tool_calls": calls}
            if content:
                msg["content"] = content
            else:
                msg["content"] = None
            return msg
        return {"role": "assistant", "content": text}

    def build_prompt(
        self,
        tokenizer,
        messages: list[dict],
        tools: Iterable[ToolLike] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs,
    ):
        """Render the chat (+ tools) for generation or training.

        ``tokenizer`` format delegates to ``apply_chat_template``. Named formats
        inject a tools block into messages, then still prefer the tokenizer
        template for role markers when one exists.
        """
        schemas = normalize_tool_schemas(tools)
        if self.name == "tokenizer":
            return tokenizer.apply_chat_template(
                messages,
                tools=schemas or None,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        wired = self.inject_tools(messages, schemas)
        # Strip structured tool_calls into format-specific assistant text so
        # templates that only know role/content still see the call.
        rendered_messages = []
        for message in wired:
            msg = dict(message)
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                text = self.format_assistant_tool_calls(msg["tool_calls"], str(msg.get("content") or ""))
                msg = {"role": "assistant", "content": text}
            elif msg.get("role") == "tool":
                name = str(msg.get("name") or msg.get("tool_name") or "tool")
                body = self.format_tool_result(name, str(msg.get("content") or ""), msg.get("tool_call_id"))
                # Most chat templates expect tool results as user/tool content.
                msg = {"role": msg.get("role") or "tool", "content": body}
                if "name" in message:
                    msg["name"] = message["name"]
            rendered_messages.append(msg)

        try:
            return tokenizer.apply_chat_template(
                rendered_messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )
        except Exception:
            # Last resort for tokenizers without a chat template.
            parts = []
            for message in rendered_messages:
                role = message.get("role", "user")
                content = message.get("content") or ""
                parts.append(f"{role}: {content}")
            if add_generation_prompt:
                parts.append("assistant:")
            text = "\n".join(parts)
            if tokenize:
                return tokenizer(text, return_tensors=kwargs.get("return_tensors"))
            return text

    def parse_response(
        self,
        tokenizer,
        text: str,
        *,
        prefix=None,
        tools: Iterable[ToolLike] | None = None,
    ) -> dict:
        """Parse generated text into an assistant message dict."""
        schemas = normalize_tool_schemas(tools)
        if self.name == "tokenizer" and hasattr(tokenizer, "parse_response"):
            try:
                kwargs = {"tools": schemas} if schemas else {}
                if prefix is None:
                    return tokenizer.parse_response(text, prefix="", **kwargs)
                return tokenizer.parse_response(text, prefix=prefix, **kwargs)
            except Exception:
                pass
        return self.parse_assistant_message(text)


class _TokenizerToolFormat(ToolFormat):
    def render_tools_block(self, schemas: list[ToolSchema]) -> str:
        return json.dumps(schemas, ensure_ascii=False, indent=2)

    def format_assistant_tool_calls(self, calls: list[dict], content: str = "") -> str:
        payload = {"tool_calls": canonical_tool_calls(calls)}
        body = json.dumps(payload, ensure_ascii=False)
        return f"{content}\n{body}".strip() if content else body

    def parse_tool_calls(self, text: str) -> list[dict]:
        return _calls_from_json_value(_loads_json_blob(text))


class _HermesToolFormat(ToolFormat):
    def render_tools_block(self, schemas: list[ToolSchema]) -> str:
        lines = [
            "You may call one or more functions to assist with the user query.",
            "Function signatures are listed below. For each call, reply with:",
            '<tool_call>\n{"name": "function_name", "arguments": {...}}\n</tool_call>',
            "You can emit plain assistant text when no tool is needed.",
            "<tools>",
            json.dumps(schemas, ensure_ascii=False, indent=2),
            "</tools>",
        ]
        return "\n".join(lines)

    def format_assistant_tool_calls(self, calls: list[dict], content: str = "") -> str:
        chunks = []
        if content:
            chunks.append(content)
        for call in _simple_calls(calls):
            chunks.append(
                "<tool_call>\n"
                + json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)
                + "\n</tool_call>"
            )
        return "\n".join(chunks)

    def format_tool_result(self, name: str, content: str, call_id: str | None = None) -> str:
        return f"<tool_response>\n{content}\n</tool_response>"

    def parse_tool_calls(self, text: str) -> list[dict]:
        calls: list[dict] = []
        for payload in _HERMES_BLOCK.findall(text or ""):
            calls.extend(_calls_from_json_value(_loads_json_blob(payload)))
        if calls:
            return calls
        # Tolerate a single bare JSON object after the model forgot the tags.
        return _calls_from_json_value(_loads_json_blob(text))


class _Llama3JsonToolFormat(ToolFormat):
    def render_tools_block(self, schemas: list[ToolSchema]) -> str:
        return (
            "You have access to the following functions. "
            "When you need one, reply with a JSON object "
            '{"name": "...", "parameters": {...}} and nothing else.\n'
            + json.dumps(schemas, ensure_ascii=False, indent=2)
        )

    def format_assistant_tool_calls(self, calls: list[dict], content: str = "") -> str:
        simple = _simple_calls(calls)
        if not simple:
            return content
        # Llama often emits one call; keep parameters key.
        payload = {"name": simple[0]["name"], "parameters": simple[0]["arguments"]}
        body = json.dumps(payload, ensure_ascii=False)
        return f"{content}\n{body}".strip() if content else body

    def parse_tool_calls(self, text: str) -> list[dict]:
        tagged = _LLAMA_PY.search(text or "")
        blob = tagged.group(1) if tagged else text
        value = _loads_json_blob(blob)
        if isinstance(value, dict) and "parameters" in value and "arguments" not in value:
            value = {
                "name": value.get("name"),
                "arguments": value.get("parameters") or {},
            }
        return _calls_from_json_value(value)


class _MistralToolFormat(ToolFormat):
    def render_tools_block(self, schemas: list[ToolSchema]) -> str:
        return (
            "You can call tools using [TOOL_CALLS] followed by a JSON list of "
            '{"name": "...", "arguments": {...}} objects.\n'
            + json.dumps(schemas, ensure_ascii=False, indent=2)
        )

    def format_assistant_tool_calls(self, calls: list[dict], content: str = "") -> str:
        body = "[TOOL_CALLS]" + json.dumps(_simple_calls(calls), ensure_ascii=False)
        return f"{content}\n{body}".strip() if content else body

    def parse_tool_calls(self, text: str) -> list[dict]:
        match = _MISTRAL_BLOCK.search(text or "")
        if match:
            return _calls_from_json_value(_loads_json_blob(match.group(1)))
        return _calls_from_json_value(_loads_json_blob(text))


class _JsonToolFormat(ToolFormat):
    def render_tools_block(self, schemas: list[ToolSchema]) -> str:
        return (
            "Available tools (JSON schemas). To call one, reply with "
            '{"name": "...", "arguments": {...}}.\n'
            + json.dumps(schemas, ensure_ascii=False, indent=2)
        )

    def format_assistant_tool_calls(self, calls: list[dict], content: str = "") -> str:
        simple = _simple_calls(calls)
        body = json.dumps(simple[0] if len(simple) == 1 else simple, ensure_ascii=False)
        return f"{content}\n{body}".strip() if content else body

    def parse_tool_calls(self, text: str) -> list[dict]:
        return _calls_from_json_value(_loads_json_blob(text))


TOOL_FORMATS: dict[str, ToolFormat] = {
    "tokenizer": _TokenizerToolFormat(
        "tokenizer",
        "Hugging Face chat template tools= + parse_response when available",
    ),
    "hermes": _HermesToolFormat("hermes", "ChatML-friendly <tool_call> JSON blocks"),
    "llama3_json": _Llama3JsonToolFormat("llama3_json", "Llama 3 name/parameters JSON"),
    "mistral": _MistralToolFormat("mistral", "[TOOL_CALLS] JSON list"),
    "json": _JsonToolFormat("json", "Bare JSON object/list tool calls"),
}


def tool_format_names() -> list[str]:
    return sorted(TOOL_FORMATS)


def get_tool_format(name: str | ToolFormat | None) -> ToolFormat:
    if isinstance(name, ToolFormat):
        return name
    key = "tokenizer" if name is None or name == "" else str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in TOOL_FORMATS:
        known = ", ".join(tool_format_names())
        raise ValueError(f"Unknown tool_format {name!r}. Expected one of: {known}")
    return TOOL_FORMATS[key]
