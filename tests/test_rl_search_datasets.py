"""Search tools + GSM8K / web QA RL datasets (no live API calls)."""

from __future__ import annotations

import json
import random

from teapottrainer import (
    GSM8K_RL,
    HOTPOT_RL,
    NQ_OPEN_RL,
    SearchSettings,
    SYNTHQA_RL,
    TEAPOT_CHAT_RL,
    TRIVIA_QA_RL,
    WEB_QUESTIONS_RL,
    bind_tools,
    build_search_tools,
    get_tool_format,
    make_brave_search,
    make_exa_search,
    make_llm_search,
    make_llmsearchindex,
    tool_format_names,
)
from teapottrainer.formatting import ABSTAIN_TEXT, RAG_SYSTEM
from teapottrainer.rl.datasets import grade_episode, sample_episode
from teapottrainer.rl.rewards import (
    score_exact_answer,
    score_gsm8k_answer,
    score_synthqa_answer,
    score_text_answer,
    used_search_tools,
)
from teapottrainer.rl.tools import tool_schemas


def test_search_tool_factories_mock_and_schemas():
    settings = SearchSettings(mock=True)
    tools = build_search_tools(settings)
    names = {fn.__name__ for fn in tools}
    assert names == {"web_search"}
    schemas = tool_schemas(tools)
    assert {row["function"]["name"] for row in schemas} == names

    web = make_llmsearchindex(settings)
    assert "Mock llmsearchindex" in web("who invented the telephone")

    # Opt-in API backends still work when requested explicitly.
    all_tools = build_search_tools(
        settings, providers=["llmsearchindex", "brave", "exa", "llm_search"]
    )
    assert [fn.__name__ for fn in all_tools] == [
        "web_search",
        "brave_search",
        "exa_search",
        "llm_search",
    ]
    assert "Mock Brave" in make_brave_search(settings)("q")
    assert "Mock Exa" in make_exa_search(settings)("q")
    assert "Mock LLM" in make_llm_search(settings)("q")


def test_search_settings_from_env(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test")
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.delenv("LLM_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = SearchSettings.from_env()
    assert settings.brave_api_key == "brave-test"
    assert settings.exa_api_key == "exa-test"
    assert settings.default_providers == ["llmsearchindex"]
    # Default build ignores Brave/Exa keys and uses llmsearchindex only.
    tools = build_search_tools(settings)
    assert [fn.__name__ for fn in tools] == ["web_search"]
    tools = build_search_tools(settings, providers=["brave", "exa"])
    assert [fn.__name__ for fn in tools] == ["brave_search", "exa_search"]


def test_bind_tools_onto_web_datasets():
    tools = build_search_tools(SearchSettings(mock=True))
    bound = bind_tools(NQ_OPEN_RL, tools)
    assert NQ_OPEN_RL.tools == []
    assert [fn.__name__ for fn in bound.tools] == ["web_search"]
    assert bound.name == "nq_open"


def test_gsm8k_sample_and_grade_with_injected_rows():
    state = {
        "rows": [
            {
                "question": "What is 6 times 7?",
                "answer": "Natalia sold clips.\n<<6*7=42>>\n#### 42",
            }
        ]
    }
    sample = sample_episode(GSM8K_RL, state, random.Random(0))
    assert sample["messages"][1]["content"].startswith("What is 6")
    assert any(fn.__name__ == "calculator" for fn in sample["tools"])

    good = [
        *sample["messages"],
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "calculator", "arguments": {"expression": "6*7"}},
                }
            ],
        },
        {"role": "tool", "name": "calculator", "content": "42"},
        {"role": "assistant", "content": "The answer is 42."},
    ]
    reward, metrics = grade_episode(GSM8K_RL, good, sample["meta"])
    assert metrics["accuracy"] == 1.0
    assert reward > 0.0


def test_web_qa_grade_requires_search_by_default():
    meta = {"answers": ["Alexander Graham Bell"], "require_search": True}
    no_tool = [
        {"role": "user", "content": "Who invented the telephone?"},
        {"role": "assistant", "content": "Alexander Graham Bell invented it."},
    ]
    reward, metrics = grade_episode(NQ_OPEN_RL, no_tool, meta)
    assert metrics["contained"] == 1.0
    assert metrics["used_search"] == 0.0
    assert reward == 0.0

    with_tool = [
        {"role": "user", "content": "Who invented the telephone?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"query": "telephone inventor"}},
                }
            ],
        },
        {"role": "tool", "name": "web_search", "content": "Alexander Graham Bell ..."},
        {"role": "assistant", "content": "Alexander Graham Bell invented the telephone."},
    ]
    reward2, metrics2 = grade_episode(NQ_OPEN_RL, with_tool, meta)
    assert metrics2["used_search"] == 1.0
    assert reward2 > 0.0


def test_web_dataset_samplers_use_injected_state():
    rng = random.Random(1)
    nq = sample_episode(
        NQ_OPEN_RL,
        {"rows": [{"question": "Capital of France?", "answer": ["Paris"]}]},
        rng,
    )
    assert nq["meta"]["answers"] == ["Paris"]
    assert nq["meta"]["require_search"] is True

    hotpot = sample_episode(
        HOTPOT_RL,
        {"rows": [{"question": "Who wrote X?", "answer": "Ada", "level": "easy", "type": "bridge"}]},
        rng,
    )
    assert hotpot["meta"]["answers"] == ["Ada"]
    # Context must not be stuffed into the prompt — search is the point.
    assert all("context" not in json.dumps(m).lower() for m in hotpot["messages"])

    webq = sample_episode(
        WEB_QUESTIONS_RL,
        {"rows": [{"question": "Where is the Nile?", "answers": ["Egypt", "Africa"]}]},
        rng,
    )
    assert "Egypt" in webq["meta"]["answers"]

    trivia = sample_episode(
        TRIVIA_QA_RL,
        {
            "rows": [
                {
                    "question": "Largest planet?",
                    "answer": {"value": "Jupiter", "aliases": ["Jupiter", "Jove"]},
                }
            ]
        },
        rng,
    )
    assert "Jupiter" in trivia["meta"]["answers"]


def test_reward_helpers_and_hermes_still_listed():
    # Surrounding prose is fine as long as the gold span is present.
    reward, metrics = score_text_answer(
        "Sure — after checking sources, Paris is the capital of France.",
        ["Paris"],
        used_search=True,
    )
    assert metrics["contained"] == 1.0
    assert reward >= 1.0
    # Verbose answers must not be punished relative to a short exact span.
    short, _ = score_text_answer("Paris", ["Paris"], used_search=True)
    assert reward >= short - 1e-6
    gsm_r, gsm_m = score_gsm8k_answer("#### 42", "foo #### 42", used_calculator=True)
    assert gsm_m["accuracy"] == 1.0
    assert gsm_r > 0.0
    assert used_search_tools([{"name": "web_search", "arguments": {"query": "x"}}])
    assert used_search_tools([{"name": "exa_search", "arguments": {"query": "x"}}])
    assert "hermes" in tool_format_names()
    assert get_tool_format("hermes").name == "hermes"


def test_synthqa_rl_is_context_only_and_rejects_tools():
    answerable_rows = {
        "rows": [
            {
                "type": "qa",
                "context": "Paris is the capital of France.",
                "question": "What is the capital of France?",
                "answer": "Paris",
            }
        ]
    }
    unanswerable_rows = {
        "rows": [
            {
                "type": "unanswerable",
                "context": "The Nile is a river in Africa.",
                "question": "What is the phone number for the Nile?",
                "answer": "",
            }
        ]
    }
    answerable = sample_episode(SYNTHQA_RL, answerable_rows, random.Random(0))
    assert answerable["tools"] == []
    assert answerable["messages"][0]["content"] == RAG_SYSTEM
    assert "Context:" in answerable["messages"][1]["content"]
    assert answerable["meta"]["unanswerable"] is False

    # Even bind_tools cannot stick — sample always clears tools.
    poisoned = bind_tools(SYNTHQA_RL, build_search_tools(SearchSettings(mock=True)))
    forced = sample_episode(poisoned, answerable_rows, random.Random(0))
    assert forced["tools"] == []

    good = [
        *answerable["messages"],
        {"role": "assistant", "content": "Paris"},
    ]
    reward, metrics = grade_episode(SYNTHQA_RL, good, answerable["meta"])
    assert metrics["tool_violation"] == 0.0
    assert reward > 0.0

    with_tool = [
        *answerable["messages"],
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "brave_search", "arguments": {"query": "capital"}}}
            ],
        },
        {"role": "tool", "name": "brave_search", "content": "Paris"},
        {"role": "assistant", "content": "Paris"},
    ]
    bad_reward, bad_metrics = grade_episode(SYNTHQA_RL, with_tool, answerable["meta"])
    assert bad_metrics["tool_violation"] == 1.0
    assert bad_reward == 0.0

    unans = sample_episode(SYNTHQA_RL, unanswerable_rows, random.Random(0))
    assert unans["meta"]["unanswerable"] is True
    assert unans["meta"]["answer"] == ABSTAIN_TEXT
    abstain_conv = [*unans["messages"], {"role": "assistant", "content": ABSTAIN_TEXT}]
    ok, ok_m = grade_episode(SYNTHQA_RL, abstain_conv, unans["meta"])
    assert ok_m["abstained"] == 1.0
    assert ok == 1.0
    invent = [*unans["messages"], {"role": "assistant", "content": "555-0100"}]
    zero, zero_m = grade_episode(SYNTHQA_RL, invent, unans["meta"])
    assert zero_m["abstained"] == 0.0
    assert zero == 0.0

    synth_r, synth_m = score_synthqa_answer("Paris", gold_answer="Paris", unanswerable=False)
    assert synth_m["accuracy"] == 1.0
    assert synth_r == 1.0
    # Soft containment alone is not enough for SynthQA keepalive.
    soft, soft_m = score_synthqa_answer(
        "The capital is Paris of course.",
        gold_answer="Paris",
        unanswerable=False,
    )
    assert soft_m["contained"] == 1.0
    assert soft == 0.0


def test_teapot_chat_rl_exact_match_no_tools():
    state = {
        "rows": [
            {
                "system_prompt": 'You are Teapot. If a user asks who you are, reply "I am Teapot".',
                "query": "Who are you?",
                "response": "I am Teapot",
                "context": "",
            }
        ]
    }
    sample = sample_episode(TEAPOT_CHAT_RL, state, random.Random(0))
    assert sample["tools"] == []
    good = [*sample["messages"], {"role": "assistant", "content": "I am Teapot"}]
    reward, metrics = grade_episode(TEAPOT_CHAT_RL, good, sample["meta"])
    assert metrics["accuracy"] == 1.0
    assert reward == 1.0
    loose = [*sample["messages"], {"role": "assistant", "content": "Hello, I am Teapot today."}]
    zero, zero_m = grade_episode(TEAPOT_CHAT_RL, loose, sample["meta"])
    assert zero_m["accuracy"] == 0.0
    assert zero == 0.0
    assert score_exact_answer("I am Teapot", "I am Teapot")[0] == 1.0
