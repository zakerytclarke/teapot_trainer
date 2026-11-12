"""Built-in tools for RL environments.

Tools are ordinary Python functions with type hints and docstrings so Hugging
Face ``get_json_schema`` / ``apply_chat_template(..., tools=)`` can describe
them. The registry maps stable names used in ``RlConfig.tools``.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from ..tool_formats import normalize_tool_schemas


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


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "calculator": calculator,
}


def resolve_tools(names: list[str] | None) -> list[Callable[..., Any]]:
    """Look up tool callables by config name."""
    if not names:
        return []
    out = []
    for name in names:
        key = str(name).strip()
        if key not in TOOL_REGISTRY:
            known = ", ".join(sorted(TOOL_REGISTRY))
            raise ValueError(f"Unknown tool {name!r}. Registered: {known}")
        out.append(TOOL_REGISTRY[key])
    return out


def tool_schemas_for(names: list[str] | None) -> list[dict]:
    return normalize_tool_schemas(resolve_tools(names))


def run_tool(name: str, arguments: dict | None) -> str:
    """Execute one tool call and return a string observation."""
    key = str(name or "").strip()
    if key not in TOOL_REGISTRY:
        return f"ERROR: unknown tool {name!r}"
    args = arguments if isinstance(arguments, dict) else {}
    try:
        result = TOOL_REGISTRY[key](**args)
    except TypeError as exc:
        return f"ERROR: bad arguments for {key}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"
    return str(result)
