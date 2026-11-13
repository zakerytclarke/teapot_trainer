"""RL skeleton: dataset API, rollouts JSON, eval_datasets, tiny e2e GRPO."""

from __future__ import annotations

import json
import random
from pathlib import Path

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from teapottrainer import (
    DatasetSpec,
    GSM8K_TEST,
    RlConfig,
    RlDataset,
    RlTrainer,
    SYNTH_MATH,
    get_tool_format,
    tool_format_names,
)
from teapottrainer.rl.algorithms.grpo import group_advantages
from teapottrainer.rl.datasets import grade_episode, prepare_dataset, sample_episode
from teapottrainer.rl.rewards import extract_number, score_numeric_answer
from teapottrainer.rl.tools import calculator, call_tool, tool_schemas


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


def test_tools_are_python_functions_with_hf_schemas():
    schemas = tool_schemas([calculator])
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "calculator"
    assert "expression" in schemas[0]["function"]["parameters"]["properties"]
    assert call_tool(calculator, {"expression": "2+2"}) == "4"
    assert "ERROR" in calculator("__import__('os')")


def test_named_tool_formats_roundtrip_prompt_and_calls():
    schemas = tool_schemas([calculator])
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
        assert "5" in fmt.format_tool_result("calculator", "5")


def test_synth_math_dataset_init_sample_grade():
    state = prepare_dataset(SYNTH_MATH)
    rng = random.Random(0)
    sample = sample_episode(SYNTH_MATH, state, rng)
    assert sample["messages"]
    assert "value" in sample["meta"]
    assert all(callable(fn) for fn in sample["tools"])
    assert {getattr(fn, "__name__", "") for fn in sample["tools"]} == {"calculator"}
    # Prompts must require the tool; optional wording lets models skip it.
    assert "must" in sample["messages"][0]["content"].lower()

    gold = float(sample["meta"]["value"])
    answer = f"The answer is {int(gold) if gold.is_integer() else gold}."

    # Correct without calculator → no reward (otherwise GRPO never learns tools).
    no_tool = sample["messages"] + [{"role": "assistant", "content": answer}]
    reward_no_tool, metrics_no_tool = grade_episode(SYNTH_MATH, no_tool, sample["meta"])
    assert metrics_no_tool["accuracy"] == 1.0
    assert metrics_no_tool["used_calculator"] == 0.0
    assert reward_no_tool == 0.0

    # Correct with calculator tool result → rewarded.
    with_tool = sample["messages"] + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": {"expression": sample["meta"]["expression"]},
                    },
                }
            ],
        },
        {"role": "tool", "name": "calculator", "content": str(int(gold) if gold.is_integer() else gold)},
        {"role": "assistant", "content": answer},
    ]
    reward, metrics = grade_episode(SYNTH_MATH, with_tool, sample["meta"])
    assert metrics["accuracy"] == 1.0
    assert metrics["used_calculator"] == 1.0
    assert metrics["calculator_result_ok"] == 1.0
    assert reward >= 1.0

    bad = sample["messages"] + [{"role": "assistant", "content": "The answer is 123456."}]
    reward_bad, metrics_bad = grade_episode(SYNTH_MATH, bad, sample["meta"])
    assert metrics_bad["accuracy"] == 0.0
    assert reward_bad == 0.0


def test_custom_rl_dataset_is_extensible():
    def init_fn():
        return {"answers": {"hi": "hello"}}

    def sample_fn(state, rng):
        return {
            "sample_id": "echo",
            "messages": [{"role": "user", "content": "hi"}],
            "meta": {"expected": state["answers"]["hi"]},
            "tools": [],
        }

    def grade_fn(conversation, meta):
        text = " ".join(
            str(m.get("content") or "") for m in conversation if m.get("role") == "assistant"
        ).lower()
        return 1.0 if meta["expected"] in text else 0.0

    ds = RlDataset(name="echo", init_fn=init_fn, sample_fn=sample_fn, grade_fn=grade_fn, tools=[])
    state = prepare_dataset(ds)
    sample = sample_episode(ds, state, random.Random(0))
    good = sample["messages"] + [{"role": "assistant", "content": "hello there"}]
    score, _ = grade_episode(ds, good, sample["meta"])
    assert score == 1.0


def test_group_advantages_zero_mean():
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(adv)) < 1e-5
    assert max(adv) > 0 > min(adv)


def test_rl_config_validates_tool_format_and_group_size():
    cfg = RlConfig(
        init_from="dummy",
        tool_format="qwen",  # alias -> hermes
        datasets=[SYNTH_MATH],
        group_size=4,
        save_rollouts=False,
    )
    assert cfg.tool_format == "hermes"
    assert cfg.eval_datasets == []
    try:
        RlConfig(init_from="dummy", datasets=[SYNTH_MATH], group_size=1)
        assert False, "expected group_size validation error"
    except Exception:
        pass


def test_e2e_grpo_saves_rollouts_and_runs_eval_hooks(tmp_path: Path):
    """Full train+eval loop: rollouts.json + before/after eval_datasets hooks."""
    model_dir = _tiny_tokenizer_and_model(tmp_path)
    run_root = tmp_path / "runs"
    config = RlConfig(
        init_from=str(model_dir),
        tokenizer=str(model_dir),
        tool_format="hermes",
        datasets=[SYNTH_MATH],
        # Same DatasetSpec type as TeapotTrainer; empty-safe path still exercises hooks.
        eval_datasets=[],
        algorithm="grpo",
        group_size=2,
        max_turns=2,
        max_new_tokens=24,
        temperature=0.8,
        lr=1e-4,
        kl_coef=0.0,
        max_train_episodes=2,
        eval_every_episodes=2,
        eval_episodes=2,
        resume_every_episodes=2,
        save_rollouts=True,
        logging_dir=str(run_root),
        seed=0,
    )
    run_dir = RlTrainer(config).train(fresh=True)
    assert run_dir.exists()

    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = {row.get("split") for row in rows if "split" in row}
    assert "eval_before_rl" in splits
    assert "eval_after_rl" in splits
    assert any(row.get("split") == "final_rl_env_eval" for row in rows)
    train_rows = [row for row in rows if "score_avg" in row]
    assert train_rows, "expected running average score in train metrics"
    env_eval = next(row for row in rows if row.get("split") == "final_rl_env_eval")
    assert "dataset/synth_math/score" in env_eval
    assert "avg_score" in env_eval

    rollouts_path = run_dir / "rollouts.json"
    assert rollouts_path.exists()
    rollouts = json.loads(rollouts_path.read_text(encoding="utf-8"))
    assert isinstance(rollouts, list)
    assert rollouts
    assert "conversation" in rollouts[0]
    assert "score" in rollouts[0]
    assert "metrics" in rollouts[0]
    # Tools show up in the rendered prompt / schemas, not as chat messages.
    assert rollouts[0]["tools"]
    assert rollouts[0]["tools"][0]["function"]["name"] == "calculator"
    assert "calculator" in (rollouts[0].get("rendered_prompt") or "")

    saved = json.loads((run_dir / "rl_config.json").read_text(encoding="utf-8"))
    assert saved["tool_format"] == "hermes"
    assert saved["datasets"][0]["name"] == "synth_math"
    assert saved["save_rollouts"] is True


def test_rl_config_accepts_training_eval_dataset_specs():
    cfg = RlConfig(
        init_from="dummy",
        datasets=[SYNTH_MATH],
        eval_datasets=[DatasetSpec(source=GSM8K_TEST, target_tokens=0)],
        save_rollouts=False,
    )
    assert len(cfg.eval_datasets) == 1
    assert cfg.eval_datasets[0].source.name == GSM8K_TEST.name


def test_score_numeric_helper():
    reward, metrics = score_numeric_answer("I think it is 42.", 42)
    assert metrics["accuracy"] == 1.0
    assert reward >= 1.0
    assert extract_number("answer: 3.5 units") == 3.5

    # Required-tool path: correct answer alone is not enough.
    zero, metrics_req = score_numeric_answer(
        "I think it is 42.",
        42,
        used_calculator=False,
        require_calculator=True,
    )
    assert metrics_req["accuracy"] == 1.0
    assert zero == 0.0
    good, _ = score_numeric_answer(
        "I think it is 42.",
        42,
        used_calculator=True,
        require_calculator=True,
        calculator_result_ok=True,
    )
    assert good >= 1.0
