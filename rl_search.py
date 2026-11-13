"""RL recipe: math + open web search with Brave / Exa / LLM-search tools.

Set API keys in the environment (or pass ``SearchSettings`` explicitly):

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
    WEB_QUESTIONS_RL,
    bind_tools,
    build_search_tools,
)


def main() -> None:
    # ``mock=True`` keeps the recipe runnable without keys (stub tool results).
    # Flip to False / omit once BRAVE_API_KEY / EXA_API_KEY / LLM_SEARCH_* are set.
    use_mock = not any(
        os.environ.get(name)
        for name in (
            "BRAVE_API_KEY",
            "BRAVE_SEARCH_API_KEY",
            "EXA_API_KEY",
            "LLM_SEARCH_API_KEY",
            "OPENAI_API_KEY",
        )
    )
    search = SearchSettings.from_env(mock=use_mock)
    search_tools = build_search_tools(search)
    if not search_tools:
        raise SystemExit(
            "No search tools configured. Set BRAVE_API_KEY / EXA_API_KEY / "
            "LLM_SEARCH_API_KEY+LLM_SEARCH_BASE_URL, or SearchSettings(mock=True)."
        )
    print(f"search tools: {[fn.__name__ for fn in search_tools]}  mock={search.mock}")

    config = RlConfig(
        init_from="Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
        tool_format="hermes",
        search=search,
        datasets=[
            SYNTH_MATH,
            GSM8K_RL,
            bind_tools(NQ_OPEN_RL, search_tools),
            bind_tools(WEB_QUESTIONS_RL, search_tools),
            bind_tools(HOTPOT_RL, search_tools),
        ],
        eval_datasets=[DatasetSpec(source=GSM8K_TEST, target_tokens=0)],
        eval_example_cap=16,
        algorithm="grpo",
        group_size=4,
        batch_prompts=1,
        max_turns=4,
        max_new_tokens=160,
        temperature=0.8,
        lr=5e-7,
        kl_coef=0.02,
        max_train_episodes=100,
        eval_every_episodes=25,
        eval_episodes=8,
        resume_every_episodes=25,
        save_rollouts=True,
        seed=42,
    )
    run_dir = RlTrainer(config).train(fresh=True)
    print(f"RL run complete: {run_dir}")


if __name__ == "__main__":
    main()
