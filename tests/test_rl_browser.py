"""Browser RL tools (mock mode — no Chromium required)."""

from __future__ import annotations

import asyncio

from teapottrainer import (
    BrowserSettings,
    NQ_OPEN_RL,
    bind_tools,
    build_browser_toolkit,
    build_browser_tools,
)
from teapottrainer.rl.tools import call_tool_by_name, tool_schemas


def test_browser_tools_mock_goto_read_act():
    kit = build_browser_toolkit(BrowserSettings(mock=True, eager=True))
    try:
        names = [fn.__name__ for fn in kit.tools]
        assert names == ["browser_goto", "browser_read", "browser_act"]
        schemas = tool_schemas(kit.tools)
        assert {row["function"]["name"] for row in schemas} == set(names)

        assert "OK navigated" in call_tool_by_name(kit.tools, "browser_goto", {"url": "https://example.com"})
        page = call_tool_by_name(kit.tools, "browser_read", {})
        assert "URL: https://example.com" in page
        assert "#e1" in page or "e1" in page
        assert "Interactive elements" in page
        assert "https://example.com/docs" in page

        clicked = call_tool_by_name(
            kit.tools, "browser_act", {"element_id": "e2", "action": "click"}
        )
        assert "OK clicked" in clicked
        typed = call_tool_by_name(
            kit.tools,
            "browser_act",
            {"element_id": "e3", "action": "type", "text": "hello"},
        )
        assert "OK typed" in typed
        assert "hello" in typed
    finally:
        kit.close()


def test_browser_act_requires_known_id():
    tools = build_browser_tools(BrowserSettings(mock=True, eager=True))
    call_tool_by_name(tools, "browser_goto", {"url": "https://example.com"})
    err = call_tool_by_name(tools, "browser_act", {"element_id": "e999", "action": "click"})
    assert err.startswith("ERROR:")


def test_bind_browser_tools_onto_dataset():
    tools = build_browser_tools(BrowserSettings(mock=True, eager=False))
    bound = bind_tools(NQ_OPEN_RL, tools)
    assert NQ_OPEN_RL.tools == []
    assert [fn.__name__ for fn in bound.tools] == [
        "browser_goto",
        "browser_read",
        "browser_act",
    ]


def test_browser_settings_from_env(monkeypatch):
    monkeypatch.setenv("TEAPOT_BROWSER_MOCK", "1")
    monkeypatch.setenv("TEAPOT_BROWSER_HEADED", "1")
    settings = BrowserSettings.from_env()
    assert settings.mock is True
    assert settings.headless is False


def test_ensure_playwright_browsers_is_exported():
    from teapottrainer import ensure_playwright_browsers

    assert callable(ensure_playwright_browsers)


def test_browser_tools_work_inside_running_asyncio_loop():
    """Colab/Jupyter already have a running loop; tools must still work."""

    async def _inside_loop():
        kit = build_browser_toolkit(BrowserSettings(mock=True, eager=True))
        try:
            out = call_tool_by_name(
                kit.tools, "browser_goto", {"url": "https://example.com"}
            )
            assert "OK navigated" in out
            page = call_tool_by_name(kit.tools, "browser_read", {})
            assert "Interactive elements" in page
        finally:
            kit.close()

    asyncio.run(_inside_loop())
