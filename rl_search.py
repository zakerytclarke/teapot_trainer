"""Single-phase RL mix: tool tasks + exact-match keepalives (SynthQA / TeapotChat).

No separate SFT step. Math and open-web search get tools; SynthQA and TeapotChat
stay tool-free and grade with exact answer match so GRPO does not wash out
grounding / chat style.

Search backend defaults to local **llmsearchindex** (``web_search`` tool).
Install with ``pip install llmsearchindex``. Brave / Exa are opt-in only.

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
    TEAPOT_CHAT_RL,
    TRIVIA_QA_RL,
    WEB_QUESTIONS_RL,
    bind_tools,
    build_search_tools,
)


def _default_mock_search() -> bool:
    """Use mock stubs when llmsearchindex is not installed (or forced via env)."""
    if os.environ.get("LLMSEARCHINDEX_MOCK", "").strip().lower() in {"1", "true", "yes"}:
        return True
    try:
        import llmsearchindex  # noqa: F401
    except Exception:
        return True
    return False


def build_full_rl_config(
    *,
    init_from: str = "Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer: str | None = None,
    max_train_episodes: int = 200,
    mock_search: bool | None = None,
    keepalive_weight: float = 1.0,
) -> RlConfig:
    """Assemble the mixed RL config (tool tasks + exact-match keepalives)."""
    if mock_search is None:
        mock_search = _default_mock_search()
    search = SearchSettings.from_env(
        mock=bool(mock_search),
        default_providers=["llmsearchindex"],
    )
    search_tools = build_search_tools(search)
    if not search_tools:
        raise SystemExit(
            "No search tools configured. Default is llmsearchindex "
            "(pip install llmsearchindex), or pass mock_search=True."
        )
    print(f"search tools: {[fn.__name__ for fn in search_tools]}  mock={search.mock}")

    keepalive = float(keepalive_weight)
    return RlConfig(
        init_from=init_from,
        tokenizer=tokenizer or init_from,
        tool_format="hermes",
        search=search,
        datasets=[
            # Tool tasks
            SYNTH_MATH.model_copy(update={"weight": 1.0}),
            GSM8K_RL.model_copy(update={"weight": 1.0}),
            bind_tools(NQ_OPEN_RL, search_tools).model_copy(update={"weight": 1.0}),
            bind_tools(WEB_QUESTIONS_RL, search_tools).model_copy(update={"weight": 1.0}),
            bind_tools(HOTPOT_RL, search_tools).model_copy(update={"weight": 1.0}),
            bind_tools(TRIVIA_QA_RL, search_tools).model_copy(update={"weight": 1.0}),
            # Exact-match keepalives (no tools) — run in the same RL loop
            SYNTHQA_RL.model_copy(update={"weight": keepalive}),
            TEAPOT_CHAT_RL.model_copy(update={"weight": keepalive}),
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
