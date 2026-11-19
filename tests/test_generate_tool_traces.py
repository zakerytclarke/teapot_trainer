"""Unit tests for tool-trace grading (no teacher model)."""

from __future__ import annotations

from generate_tool_traces import (
    ABSTAIN_TEXT,
    EpisodeSpec,
    build_system_prompt,
    grade_trace,
)


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
    score, metrics = grade_trace(spec, trace, available_tools=["web_search", "calculator"])
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
    score, metrics = grade_trace(spec, trace, available_tools=["web_search"])
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
    score, metrics = grade_trace(spec, trace, available_tools=["web_search"])
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
    score, _ = grade_trace(spec, bad, available_tools=["calculator"])
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
    score2, metrics = grade_trace(spec, good, available_tools=["calculator"])
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
    score, metrics = grade_trace(spec, good, available_tools=[])
    assert score == 1.0
    assert metrics["abstained"] == 1.0

    bad = [
        {"role": "user", "content": "What is the phone number for the Nile?"},
        {"role": "assistant", "content": "555-0100"},
    ]
    score2, metrics2 = grade_trace(spec, bad, available_tools=[])
    assert score2 == 0.0
    assert metrics2["hallucinated"] == 1.0


def test_system_prompt_includes_date_and_context_tags():
    text = build_system_prompt(today="2026-11-19")
    assert "2026-11-19" in text
    assert "[INJECTED_CONTEXT]" in text
