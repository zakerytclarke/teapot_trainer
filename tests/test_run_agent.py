"""Smoke tests for inference agent helper + all-tools assembly."""

from __future__ import annotations

from run_agent import build_all_tools
from teapottrainer import calculator, run_agent
from teapottrainer.rl.tools import tool_schemas


def test_build_all_tools_mock():
    tools, toolkit = build_all_tools(mock_search=True, mock_browser=True)
    try:
        names = {fn.__name__ for fn in tools}
        assert "calculator" in names
        assert "web_search" in names
        assert {"browser_goto", "browser_read", "browser_act"} <= names
        assert calculator in tools
        schemas = tool_schemas(tools)
        assert {row["function"]["name"] for row in schemas} == names
    finally:
        toolkit.close()


def test_run_agent_export_signature():
    assert callable(run_agent)
    assert run_agent.__name__ == "run_agent"
