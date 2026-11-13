"""RL recipe: synthetic math with a calculator Python tool on a small instruct model.

Assumes supervised fine-tuning is already done. While Teapot SFT checkpoints
are training, this defaults to Qwen2.5-1.5B-Instruct. Swap ``init_from`` later.

RL datasets are extensible triples:
  init_fn() -> state
  sample_fn(state, rng) -> {messages, meta, tools?}
  grade_fn(conversation_json, meta) -> float | {score, ...}

Tools are plain Python functions (HF schemas come from type hints/docstrings).
``tool_format="tokenizer"`` uses Hugging Face chat templates for prompt-side
tool injection and response parsing when available.
"""

from teapottrainer import RlConfig, RlTrainer, SYNTH_MATH


def main() -> None:
    config = RlConfig(
        init_from="Qwen/Qwen2.5-1.5B-Instruct",
        tokenizer="Qwen/Qwen2.5-1.5B-Instruct",
        tool_format="tokenizer",
        datasets=[SYNTH_MATH],
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
