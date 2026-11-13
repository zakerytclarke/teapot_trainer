"""Tools are ordinary Python functions.

Hugging Face ``get_json_schema`` / ``apply_chat_template(..., tools=)`` read
type hints and docstrings. The RL loop calls the same functions when the model
emits a tool request.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Iterable

from ..tool_formats import normalize_tool_schemas

ToolFn = Callable[..., Any]


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return float(_ALLOWED_BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return float(_ALLOWED_UNARY[type(node.op)](_eval_ast(node.operand)))
    raise ValueError("Only arithmetic expressions are allowed")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Args:
        expression: An arithmetic expression using +, -, *, /, //, %, **, and parentheses.
    """
    text = str(expression or "").strip()
    if not text:
        return "ERROR: empty expression"
    try:
        parsed = ast.parse(text, mode="eval")
        value = _eval_ast(parsed)
    except Exception as exc:  # noqa: BLE001 - tool errors become observations
        return f"ERROR: {exc}"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def tool_name(fn: ToolFn) -> str:
    return str(getattr(fn, "__name__", "tool"))


def index_tools(tools: Iterable[ToolFn] | None) -> dict[str, ToolFn]:
    """Map tool name -> Python function."""
    out: dict[str, ToolFn] = {}
    for fn in tools or []:
        if not callable(fn):
            raise TypeError(f"tools must be Python callables, got {type(fn)!r}")
        out[tool_name(fn)] = fn
    return out


def tool_schemas(tools: Iterable[ToolFn] | None) -> list[dict]:
    return normalize_tool_schemas(list(tools or []))


def call_tool(fn: ToolFn, arguments: dict | None) -> str:
    """Execute one tool function and always return a string observation."""
    args = arguments if isinstance(arguments, dict) else {}
    try:
        result = fn(**args)
    except TypeError as exc:
        return f"ERROR: bad arguments for {tool_name(fn)}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"
    return str(result)


def call_tool_by_name(tools: Iterable[ToolFn] | None, name: str, arguments: dict | None) -> str:
    table = index_tools(tools)
    key = str(name or "").strip()
    if key not in table:
        known = ", ".join(sorted(table)) or "(none)"
        return f"ERROR: unknown tool {name!r}. Available: {known}"
    return call_tool(table[key], arguments)


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


def conversation_tool_calls(messages: list[dict]) -> list[dict]:
    """Flatten assistant tool_calls in a conversation JSON."""
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
