"""RL skeleton: tool formats, math env rewards, and a tiny end-to-end GRPO run."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from teapottrainer import EnvSpec, RlConfig, RlTrainer, get_tool_format, tool_format_names
from teapottrainer.rl.algorithms.grpo import group_advantages
from teapottrainer.rl.env import Trajectory
from teapottrainer.rl.rewards import extract_number, score_numeric_answer
from teapottrainer.rl.tasks import build_env
from teapottrainer.rl.tools import calculator, run_tool, tool_schemas_for
from teapottrainer.tool_formats import normalize_tool_schemas


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def _tiny_tokenizer_and_model(tmp_path: Path):
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(tmp_path / "tok")

    config = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    return tmp_path / "model"


def test_tool_format_names_include_hf_default():
    names = tool_format_names()
    assert "tokenizer" in names
    assert "hermes" in names
    assert "llama3_json" in names
    assert "mistral" in names
    assert "json" in names


def test_hf_schema_from_calculator_function():
    schemas = tool_schemas_for(["calculator"])
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "calculator"
    assert "expression" in schemas[0]["function"]["parameters"]["properties"]


def test_named_tool_formats_roundtrip_prompt_and_calls():
    schemas = tool_schemas_for(["calculator"])
    calls = [{"type": "function", "function": {"name": "calculator", "arguments": {"expression": "2+3"}}}]
    messages = [{"role": "user", "content": "What is 2+3?"}]

    for name in ("hermes", "llama3_json", "mistral", "json"):
        fmt = get_tool_format(name)
        wired = fmt.inject_tools(messages, schemas)
        assert wired[0]["role"] == "system"
        assert "calculator" in wired[0]["content"]

        rendered = fmt.format_assistant_tool_calls(calls)
        parsed = fmt.parse_tool_calls(rendered)
        assert parsed[0]["function"]["name"] == "calculator"
        assert parsed[0]["function"]["arguments"]["expression"] == "2+3"

        result = fmt.format_tool_result("calculator", "5")
        assert "5" in result


def test_calculator_tool_evaluates_expressions():
    assert calculator("6 * 7") == "42"
    assert calculator("(10 - 3) * 2") == "14"
    assert run_tool("calculator", {"expression": "2+2"}) == "4"
    assert "ERROR" in calculator("__import__('os')")


def test_math_env_scores_numeric_answers():
    env = build_env("synth_math", {"difficulty": "easy"})
    rng = random.Random(0)
    task = env.sample_task(rng)
    gold = float(task.gold["value"])

    good = Trajectory(
        task=task,
        messages=task.messages
        + [{"role": "assistant", "content": f"The answer is {int(gold) if gold.is_integer() else gold}."}],
    )
    reward, metrics = env.score(good)
    assert metrics["accuracy"] == 1.0
    assert reward >= 1.0

    bad = Trajectory(
        task=task,
        messages=task.messages + [{"role": "assistant", "content": "The answer is 0."}],
    )
    reward_bad, metrics_bad = env.score(bad)
    # 0 might randomly be correct; force mismatch when needed.
    if metrics_bad["accuracy"] == 1.0:
        bad = Trajectory(
            task=task,
            messages=task.messages + [{"role": "assistant", "content": "The answer is 123456."}],
        )
        reward_bad, metrics_bad = env.score(bad)
    assert metrics_bad["accuracy"] == 0.0
    assert reward_bad == 0.0


def test_group_advantages_zero_mean():
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(adv)) < 1e-5
    assert max(adv) > 0 > min(adv)


def test_rl_config_validates_tool_format_and_group_size():
    cfg = RlConfig(
        init_from="dummy",
        tool_format="qwen",  # alias -> hermes
        envs=[EnvSpec(name="synth_math")],
        group_size=4,
    )
    assert cfg.tool_format == "hermes"
    try:
        RlConfig(init_from="dummy", envs=[EnvSpec(name="synth_math")], group_size=1)
        assert False, "expected group_size validation error"
    except Exception:
        pass


def test_e2e_grpo_synth_math_tiny_model(tmp_path: Path):
    """Full train+eval loop on synthetic math with a tiny random Llama."""
    model_dir = _tiny_tokenizer_and_model(tmp_path)
    run_root = tmp_path / "runs"
    config = RlConfig(
        init_from=str(model_dir),
        tokenizer=str(model_dir),
        tool_format="hermes",
        tools=["calculator"],
        envs=[EnvSpec(name="synth_math", weight=1.0, options={"difficulty": "easy"})],
        algorithm="grpo",
        group_size=2,
        max_turns=2,
        max_new_tokens=24,
        temperature=0.8,
        lr=1e-4,
        kl_coef=0.0,  # skip second model copy on CPU
        max_train_episodes=2,
        eval_every_episodes=2,
        eval_episodes=2,
        resume_every_episodes=2,
        logging_dir=str(run_root),
        seed=0,
    )
    run_dir = RlTrainer(config).train(fresh=True)
    assert run_dir.exists()
    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("reward_mean" in row for row in rows)
    assert any(row.get("split") == "final_eval" for row in rows)
    assert (run_dir / "checkpoints" / "final").exists()
    # Config snapshot written for reproducibility.
    saved = json.loads((run_dir / "rl_config.json").read_text(encoding="utf-8"))
    assert saved["tool_format"] == "hermes"
    assert saved["envs"][0]["name"] == "synth_math"


def test_score_numeric_helper():
    reward, metrics = score_numeric_answer("I think it is 42.", 42)
    assert metrics["accuracy"] == 1.0
    assert reward >= 1.0
    assert extract_number("answer: 3.5 units") == 3.5
