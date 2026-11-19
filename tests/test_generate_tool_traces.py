"""Unit tests for tool-trace generation helpers and after-the-fact grading."""

from __future__ import annotations

from generate_tool_traces import (
    ABSTAIN_TEXT,
    EpisodeSpec,
    build_system_prompt,
)
from grade_tool_traces import grade_row


def _row(spec: EpisodeSpec, trace: list[dict], available_tools: list[str]) -> dict:
    return {
        "sample_id": spec.sample_id,
        "dataset": spec.dataset,
        "task_type": spec.task_type,
        "question": spec.question,
        "gold_answers": list(spec.gold_answers),
        "tools_available": list(available_tools),
        "tools_required": list(spec.required_tools),
        "messages_input": list(spec.messages_input),
        "messages_trace": list(trace),
        "final_assistant": "",
        "injected_context": spec.injected_context,
        "unanswerable": bool(spec.unanswerable),
    }


def _spec(**kwargs):
    base = dict(
        sample_id="t0",
        dataset="test",
        task_type="search",
        question="What is the capital of France?",
        gold_answers=["Paris"],
        messages_input=[{"role": "user", "content": "What is the capital of France?"}],
        required_tools=["web_search"],
    )
    base.update(kwargs)
    return EpisodeSpec(**base)


def test_grade_requires_search_when_available():
    spec = _spec()
    # Correct answer but no tool call while web_search was available.
    trace = [
        {"role": "user", "content": spec.question},
        {"role": "assistant", "content": "Paris"},
    ]
    score, metrics = grade_row(_row(spec, trace, ["web_search", "calculator"]))
    assert score == 0.0
    assert metrics["used_required_tools"] == 0.0


def test_grade_search_grounded_success():
    spec = _spec()
    trace = [
        {"role": "user", "content": spec.question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"query": "capital of France"}},
                }
            ],
        },
        {"role": "tool", "name": "web_search", "content": "Paris is the capital of France."},
        {"role": "assistant", "content": "Paris"},
    ]
    score, metrics = grade_row(_row(spec, trace, ["web_search"]))
    assert score == 1.0
    assert metrics["grounded"] == 1.0
    assert metrics["answer_contained"] == 1.0
    assert metrics["hallucinated"] == 0.0


def test_grade_hallucination_without_evidence():
    spec = _spec()
    trace = [
        {"role": "user", "content": spec.question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"query": "capital"}},
                }
            ],
        },
        {"role": "tool", "name": "web_search", "content": "No relevant results about France."},
        {"role": "assistant", "content": "Paris"},
    ]
    score, metrics = grade_row(_row(spec, trace, ["web_search"]))
    assert score == 0.0
    assert metrics["hallucinated"] == 1.0


def test_grade_gsm8k_requires_calculator():
    spec = _spec(
        dataset="gsm8k",
        task_type="calc",
        question="What is 17*19?",
        gold_answers=["#### 323"],
        required_tools=["calculator"],
    )
    # Right answer, no calculator while available -> 0
    bad = [
        {"role": "user", "content": spec.question},
        {"role": "assistant", "content": "323"},
    ]
    score, _ = grade_row(_row(spec, bad, ["calculator"]))
    assert score == 0.0

    good = [
        {"role": "user", "content": spec.question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "calculator", "arguments": {"expression": "17*19"}},
                }
            ],
        },
        {"role": "tool", "name": "calculator", "content": "323"},
        {"role": "assistant", "content": "The answer is 323."},
    ]
    score2, metrics = grade_row(_row(spec, good, ["calculator"]))
    assert score2 == 1.0
    assert metrics["grounded"] == 1.0


def test_grade_context_unanswerable_abstain():
    spec = _spec(
        dataset="squad_v2",
        task_type="context_qa",
        gold_answers=[ABSTAIN_TEXT],
        required_tools=[],
        unanswerable=True,
        injected_context="The Nile is a river in Africa.",
    )
    good = [
        {"role": "user", "content": "What is the phone number for the Nile?"},
        {"role": "assistant", "content": ABSTAIN_TEXT},
    ]
    score, metrics = grade_row(_row(spec, good, []))
    assert score == 1.0
    assert metrics["abstained"] == 1.0

    bad = [
        {"role": "user", "content": "What is the phone number for the Nile?"},
        {"role": "assistant", "content": "555-0100"},
    ]
    score2, metrics2 = grade_row(_row(spec, bad, []))
    assert score2 == 0.0
    assert metrics2["hallucinated"] == 1.0


def test_system_prompt_includes_date_and_context_tags():
    text = build_system_prompt(today="2026-11-19")
    assert "2026-11-19" in text
    assert "[INJECTED_CONTEXT]" in text
    assert "browser_goto" in text


def test_mind2web_episode_is_browser_task():
    from generate_tool_traces import make_mind2web_episode

    system = build_system_prompt(today="2026-11-19")
    ep = make_mind2web_episode(
        {
            "confirmed_task": "Find men's running shoes under $50 and add to cart.",
            "website": "kohls",
            "domain": "Shopping",
            "annotation_id": "abc",
            "action_reprs": ["[searchbox] Search -> TYPE: running shoes"],
        },
        system,
        0,
    )
    assert ep is not None
    assert ep.dataset == "mind2web"
    assert ep.task_type == "browser"
    assert ep.required_tools == ["browser"]
    assert "kohls" in ep.question
    assert "www.kohls.com" in ep.question


def test_weblinx_skips_credential_tasks():
    from generate_tool_traces import extract_weblinx_goal, make_weblinx_episode

    hist = (
        'say(speaker="instructor", utterance="Hello") '
        'say(speaker="instructor", utterance="Open site and login with google using the below details: '
        'Id: webtasks.navigator@gmail.com Password: secret")'
    )
    assert extract_weblinx_goal(hist) is None

    good = (
        'say(speaker="instructor", utterance="Hello") '
        'say(speaker="instructor", utterance="Open the Trello website.")'
    )
    goal = extract_weblinx_goal(good)
    assert goal == "Open the Trello website."
    ep = make_weblinx_episode("demo1", goal, build_system_prompt(today="2026-11-19"))
    assert ep is not None
    assert ep.task_type == "browser"
    assert "Trello" in ep.question


def test_sample_tool_bundle_browser_always_includes_browser():
    from generate_tool_traces import sample_tool_bundle

    class _T:
        def __init__(self, name):
            self.__name__ = name

        def __call__(self, *a, **k):
            return "ok"

    rng = __import__("random").Random(0)
    for _ in range(20):
        tools = sample_tool_bundle(
            rng,
            search_tools=[_T("web_search")],
            browser_tools=[_T("browser_goto"), _T("browser_read"), _T("browser_act")],
            task_type="browser",
        )
        names = {t.__name__ for t in tools}
        assert "browser_goto" in names
