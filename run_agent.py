"""Run a full tool-using agent loop on a loaded HF model.

Wires every built-in tool family:

* ``calculator``
* search (``web_search`` via llmsearchindex by default)
* browser (``browser_goto`` / ``browser_read`` / ``browser_act`` via Playwright)

Example::

    python run_agent.py --prompt "What is 21*19? Use the calculator tool."
    python run_agent.py --prompt "Open https://example.com and summarize the page." --no-mock-browser
    python run_agent.py --messages messages.json

Install extras as needed::

    pip install "teapottrainer[search,browser]"
    python -m playwright install --only-shell chromium   # Colab / headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from teapottrainer import (
    BrowserSettings,
    SearchSettings,
    build_browser_toolkit,
    build_search_tools,
    calculator,
    run_agent,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id or local checkpoint",
    )
    p.add_argument("--tokenizer", default="", help="Defaults to --model")
    p.add_argument(
        "--tool-format",
        default="hermes",
        help="Tool wire format (hermes works well for Qwen instruct)",
    )
    p.add_argument("--prompt", default="", help="Single user turn (simple mode)")
    p.add_argument(
        "--system",
        default=(
            "You are a helpful assistant with calculator, web_search, and browser tools. "
            "Use tools when they help. Prefer calculator for arithmetic, web_search for "
            "facts, and browser_* for interacting with live pages."
        ),
    )
    p.add_argument(
        "--messages",
        default="",
        help="Path to a JSON chat list (overrides --prompt/--system)",
    )
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--mock-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stub web_search (default: stub if llmsearchindex missing)",
    )
    p.add_argument(
        "--mock-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stub browser tools (default: true; pass --no-mock-browser for real Chromium)",
    )
    p.add_argument(
        "--out",
        default="",
        help="Optional path to write the finished conversation JSON",
    )
    return p.parse_args()


def _default_mock_search() -> bool:
    if os.environ.get("LLMSEARCHINDEX_MOCK", "").strip().lower() in {"1", "true", "yes"}:
        return True
    try:
        import llmsearchindex  # noqa: F401
    except Exception:
        return True
    return False


def _load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.messages:
        with open(args.messages, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("--messages JSON must be a list of chat messages")
        return [dict(row) for row in data]
    prompt = str(args.prompt or "").strip()
    if not prompt:
        prompt = "What is 17 * 19? Use the calculator tool, then answer briefly."
    messages: list[dict[str, Any]] = []
    system = str(args.system or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_all_tools(
    *,
    mock_search: bool,
    mock_browser: bool,
):
    """Assemble calculator + search + browser tools (and the browser toolkit for cleanup)."""
    search = SearchSettings(mock=bool(mock_search), default_providers=["llmsearchindex"])
    search_tools = build_search_tools(search)

    browser_settings = BrowserSettings(mock=bool(mock_browser), eager=True)
    toolkit = build_browser_toolkit(browser_settings)

    tools = [calculator, *search_tools, *toolkit.tools]
    return tools, toolkit


def _print_conversation(messages: list[dict[str, Any]]) -> None:
    print("=" * 72)
    for message in messages:
        role = message.get("role", "?")
        if role == "assistant" and message.get("tool_calls"):
            print(f"[{role}] tool_calls:")
            print(json.dumps(message.get("tool_calls"), indent=2)[:4000])
            content = message.get("content")
            if content:
                print(str(content)[:2000])
            continue
        content = message.get("content")
        name = message.get("name")
        prefix = f"[{role}" + (f":{name}" if name else "") + "]"
        text = "" if content is None else str(content)
        print(f"{prefix}\n{text[:4000]}\n")
    print("=" * 72)


def main() -> None:
    args = _parse_args()
    mock_search = _default_mock_search() if args.mock_search is None else bool(args.mock_search)
    mock_browser = bool(args.mock_browser)

    messages = _load_messages(args)
    tools, toolkit = build_all_tools(mock_search=mock_search, mock_browser=mock_browser)
    print(f"tools: {[fn.__name__ for fn in tools]}")
    print(f"mock_search={mock_search}  mock_browser={mock_browser}")

    model_id = args.model
    tok_id = args.tokenizer or model_id
    print(f"loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()

    try:
        traj = run_agent(
            model,
            tokenizer,
            messages,
            tools,
            tool_format=args.tool_format,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            sample_id="run_agent",
        )
    finally:
        toolkit.close()

    _print_conversation(traj.messages)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "messages": traj.messages,
                    "tool_schemas": traj.tool_schemas,
                    "tool_format": traj.tool_format,
                    "rendered_prompt": traj.rendered_prompt,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
