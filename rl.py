"""RL recipe: synthetic math with a calculator tool on a small instruct model.

Assumes supervised fine-tuning is already done. While Teapot SFT checkpoints
are training, this defaults to Qwen2.5-1.5B-Instruct so the RL loop can be
exercised end-to-end. Swap ``init_from`` to a local Teapot SFT directory later.

Tool formatting defaults to Hugging Face ``apply_chat_template(..., tools=)``
via ``tool_format="tokenizer"``. Set ``tool_format`` to ``hermes``,
``llama3_json``, ``mistral``, or ``json`` to force a fixed wire format for both
prompt-side tool listings and call/result parsing.
"""

from teapottrainer import EnvSpec, RlConfig, RlTrainer


def main() -> None:
    config = RlConfig(
        init_from="Qwen/Qwen2.5-1.5B-Instruct",
        tokenizer="Qwen/Qwen2.5-1.5B-Instruct",
        # Prefer the tokenizer chat template + parse_response for tools.
        tool_format="tokenizer",
        tools=["calculator"],
        envs=[
            EnvSpec(
                name="synth_math",
                weight=1.0,
                target_episodes=1_000,
                options={"difficulty": "mixed"},
            )
        ],
        algorithm="grpo",
        group_size=4,
        batch_prompts=1,
        max_turns=3,
        max_new_tokens=96,
        temperature=0.8,
        lr=5e-7,
        kl_coef=0.02,
        max_train_episodes=100,
        eval_every_episodes=20,
        eval_episodes=16,
        resume_every_episodes=20,
        seed=42,
    )
    run_dir = RlTrainer(config).train(fresh=True)
    print(f"RL run complete: {run_dir}")


if __name__ == "__main__":
    main()
