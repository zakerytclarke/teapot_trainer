"""RL recipe: synthetic math with a calculator Python tool on a small instruct model.

Assumes supervised fine-tuning is already done. While Teapot SFT checkpoints
are training, this defaults to Qwen2.5-1.5B-Instruct. Swap ``init_from`` later.

RL datasets are extensible triples under ``teapottrainer.rl.datasets``:
  init_fn() -> state
  sample_fn(state, rng) -> {messages, meta, tools?}
  grade_fn(conversation_json, meta) -> float | {score, ...}

``eval_datasets`` uses the same ``DatasetSpec`` objects as ``TeapotTrainer`` and
is scored before RL, at checkpoint intervals, and after RL.
"""

from teapottrainer import (
    DatasetSpec,
    GSM8K_TEST,
    RlConfig,
    RlTrainer,
    SYNTH_MATH,
)


def main() -> None:
    config = RlConfig(
        init_from="Qwen/Qwen2.5-1.5B-Instruct",
        tokenizer="Qwen/Qwen2.5-1.5B-Instruct",
        tool_format="tokenizer",
        datasets=[SYNTH_MATH],
        eval_datasets=[
            DatasetSpec(source=GSM8K_TEST, target_tokens=0),
        ],
        eval_example_cap=32,
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
        save_rollouts=True,
        seed=42,
    )
    run_dir = RlTrainer(config).train(fresh=True)
    print(f"RL run complete: {run_dir}")


if __name__ == "__main__":
    main()
