"""Full RL mix: calculator math, GSM8K, context-only SynthQA, and web search QA.

Tool access is per-dataset:

* ``SYNTH_MATH`` / ``GSM8K_RL`` — calculator only
* ``SYNTHQA_RL`` — **no tools** (answer from context or exact abstain string)
* ``NQ_OPEN_RL`` / ``WEB_QUESTIONS_RL`` / ``HOTPOT_RL`` / ``TRIVIA_QA_RL`` —
  Brave / Exa / LLM-search via ``bind_tools``

API keys (optional; falls back to mock stubs when unset):

* ``BRAVE_API_KEY``
* ``EXA_API_KEY``
* ``LLM_SEARCH_API_KEY`` + ``LLM_SEARCH_BASE_URL`` (+ optional ``LLM_SEARCH_MODEL``)

Example::

    python rl_search.py
"""

from __future__ import annotations

import os

from teapottrainer import (
    DatasetSpec,
    GSM8K_RL,
    GSM8K_TEST,
    HOTPOT_RL,
    NQ_OPEN_RL,
    RlConfig,
    RlTrainer,
    SearchSettings,
    SYNTH_MATH,
    SYNTHQA_RL,
    SYNTHQA_TEST,
    TRIVIA_QA_RL,
    WEB_QUESTIONS_RL,
    bind_tools,
    build_search_tools,
)


def build_full_rl_config(
    *,
    init_from: str = "Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer: str | None = None,
    max_train_episodes: int = 200,
    mock_search: bool | None = None,
) -> RlConfig:
    """Assemble the full multi-dataset RL config (shared by the recipe + Colab)."""
    if mock_search is None:
        mock_search = not any(
            os.environ.get(name)
            for name in (
                "BRAVE_API_KEY",
                "BRAVE_SEARCH_API_KEY",
                "EXA_API_KEY",
                "LLM_SEARCH_API_KEY",
                "OPENAI_API_KEY",
            )
        )
    search = SearchSettings.from_env(mock=bool(mock_search))
    search_tools = build_search_tools(search)
    if not search_tools:
        raise SystemExit(
            "No search tools configured. Set BRAVE_API_KEY / EXA_API_KEY / "
            "LLM_SEARCH_API_KEY+LLM_SEARCH_BASE_URL, or pass mock_search=True."
        )
    print(f"search tools: {[fn.__name__ for fn in search_tools]}  mock={search.mock}")

    return RlConfig(
        init_from=init_from,
        tokenizer=tokenizer or init_from,
        tool_format="hermes",
        search=search,
        datasets=[
            # Math / calculator
            SYNTH_MATH,
            GSM8K_RL,
            # Context-grounded RAG — never bind search tools here
            SYNTHQA_RL,
            # Open-web search QA
            bind_tools(NQ_OPEN_RL, search_tools),
            bind_tools(WEB_QUESTIONS_RL, search_tools),
            bind_tools(HOTPOT_RL, search_tools),
            bind_tools(TRIVIA_QA_RL, search_tools),
        ],
        eval_datasets=[
            DatasetSpec(source=GSM8K_TEST, target_tokens=0),
            DatasetSpec(source=SYNTHQA_TEST, target_tokens=0),
        ],
        eval_example_cap=16,
        algorithm="grpo",
        group_size=4,
        batch_prompts=1,
        max_turns=4,
        max_new_tokens=160,
        temperature=0.8,
        lr=5e-7,
        kl_coef=0.02,
        max_train_episodes=int(max_train_episodes),
        eval_every_episodes=25,
        eval_episodes=8,
        resume_every_episodes=25,
        save_rollouts=True,
        seed=42,
    )


def main() -> None:
    config = build_full_rl_config()
    run_dir = RlTrainer(config).train(fresh=True)
    print(f"RL run complete: {run_dir}")


if __name__ == "__main__":
    main()
