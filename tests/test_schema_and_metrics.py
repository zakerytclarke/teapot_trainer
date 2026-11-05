"""Schema lock plus scorers that do not need a GPU."""

from teapottrainer.config import DatasetSpec, ModelConfig, TrainingConfig
from teapottrainer.datasets import (
    STOP_TOKEN_LOSS_WEIGHT,
    DatasetSourceConfig,
    GSM8K_TEST,
    SMOLTALK_APIGEN_TEST,
    SQUAD_V2_ANSWERABLE,
    eos_loss_weight,
    encode_pair,
    holdout_membership,
    iterate_encoded,
    row_pairs,
)
from teapottrainer.formatting import format_gsm8k, format_hotpot, format_squad, messages_format
from teapottrainer.metrics import (
    classify_decision,
    constraint_checks,
    extract_tool_calls,
    is_abstention,
    score_gsm,
    score_tool_prediction,
    token_f1,
)


LOCKED_TRAINING_FIELDS = {
    "model",
    "pretrain_datasets",
    "bridge_datasets",
    "sft_datasets",
    "eval_datasets",
    "tokenizer",
    "chat_format",
    "init_from",
    "batch_size",
    "grad_accum",
    "lr",
    "sft_lr",
    "weight_decay",
    "compile",
    "logging_dir",
    "eval_every_percent",
    "plateau_window_tokens",
    "plateau_min_loss_drop",
    "sft_at_checkpoints",
    "holdout_ratio",
    "seed",
    "before_batch",
}


def test_training_config_fields_are_unchanged():
    assert set(TrainingConfig.model_fields) == LOCKED_TRAINING_FIELDS
    assert set(DatasetSpec.model_fields) == {"source", "target_tokens", "weight"}
    assert set(DatasetSourceConfig.model_fields) == {
        "name",
        "subset",
        "split",
        "target_tokens",
        "weight",
        "text_column",
        "is_sft",
        "preprocess_fn",
        "chat_template_fn",
        "token_loss_weight_fn",
    }
    assert set(ModelConfig.model_fields) == {
        "model_type",
        "hidden_size",
        "layers",
        "heads",
        "kv_heads",
        "intermediate",
        "seq_len",
        "custom_config_cls",
        "custom_model_cls",
    }


def test_every_preset_uses_a_namespaced_dataset_id():
    """Current Hugging Face clients reject an id that is not namespace/name."""
    import teapottrainer.datasets as datasets
    from teapottrainer.evals import task_kind

    presets = {
        name: value
        for name, value in vars(datasets).items()
        if isinstance(value, DatasetSourceConfig)
    }
    assert presets
    bare = sorted(f"{name}={value.name}" for name, value in presets.items() if value.name.count("/") != 1)
    assert bare == []
    assert presets["GSM8K"].name == "openai/gsm8k"
    assert presets["GSM8K_TEST"].name == "openai/gsm8k"
    assert presets["HOTPOT"].name == "hotpotqa/hotpot_qa"
    assert presets["HOTPOT_VAL"].name == "hotpotqa/hotpot_qa"
    assert task_kind(presets["GSM8K"]) == "gsm8k"
    assert task_kind(presets["HOTPOT"]) == "hotpot"


def test_train_script_uses_the_same_constructor():
    from train import build_config

    config = build_config()
    dumped = set(config.model_dump().keys())
    assert dumped == LOCKED_TRAINING_FIELDS
    assert config.model_dump()["before_batch"] == ["set_reasoning_effort"]
    assert config.before_batch[0].__name__ == "set_reasoning_effort"

    class Probe:
        def __init__(self):
            self.config = type("Config", (), {"reasoning_effort": 0})()

    probe = Probe()
    config.before_batch[0](probe, "pretrain")
    assert probe.config.reasoning_effort == 1
    assert config.model.model_type == "custom"
    assert config.model.layers == 16
    assert config.model.hidden_size == 768
    assert config.model.heads == 12
    assert config.model.kv_heads == 4
    assert config.model.intermediate == 2048
    assert config.model.seq_len == 2048
    assert config.model.hidden_size % config.model.heads == 0
    assert config.model.custom_model_cls.__name__ == "TeapotForCausalLM"
    built = config.model.custom_config_cls(
        vocab_size=49152,
        hidden_size=config.model.hidden_size,
        num_hidden_layers=config.model.layers,
        num_attention_heads=config.model.heads,
        num_key_value_heads=config.model.kv_heads,
        intermediate_size=config.model.intermediate,
        max_position_embeddings=config.model.seq_len,
    )
    assert built.rope_parameters["rope_theta"] == 500_000
    assert built.max_position_embeddings == 2048
    assert built.pack_context_length == 2048
    assert built.bos_token_id == 0
    assert built.eos_token_id == 0
    assert built.pad_token_id == 0
    assert built.tie_word_embeddings is True
    assert built.rms_norm_eps == 1e-5
    assert built.initializer_range == 0.02
    assert config.tokenizer == "HuggingFaceTB/SmolLM2-135M"
    assert config.init_from == "runs/teapot-SmolLM2-135M-4c2004/checkpoints/pretrain"
    assert TrainingConfig().init_from is None
    assert config.lr == 3e-4
    assert config.sft_lr == 6e-4
    assert config.sft_at_checkpoints is False
    assert config.plateau_window_tokens == 20_000_000_000
    assert config.plateau_min_loss_drop == 0.02
    assert abs(config.eval_every_percent - (10_000_000_000 / 140_000_000_000 * 100)) < 1e-9
    assert sum(spec.target_tokens for spec in config.pretrain_datasets) == 140_000_000_000
    assert sum(spec.target_tokens for spec in config.bridge_datasets) == 2_000_000_000
    pretrain_names = [(spec.source.name, spec.source.subset) for spec in config.pretrain_datasets]
    assert ("HuggingFaceTB/dclm-edu", None) in pretrain_names
    assert ("HuggingFaceFW/fineweb-edu", None) in pretrain_names
    assert ("utter-project/stack-edu-HF", None) in pretrain_names
    assert ("HuggingFaceTB/finemath", "finemath-4plus") in pretrain_names
    assert ("HuggingFaceTB/finemath", "infiwebmath-4plus") in pretrain_names
    assert ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2") in pretrain_names
    names = []
    for spec in config.sft_datasets:
        names.append((spec.source.name, spec.source.subset))
    assert ("HuggingFaceTB/smoltalk", "all") not in names
    assert ("HuggingFaceFW/fineweb-edu", None) in names
    assert ("HuggingFaceTB/smoltalk", "everyday-conversations") in names
    assert ("nvidia/When2Call", "train_sft") in names
    assert ("teapotai/synthqa", None) in names
    assert ("teapotai/teapot-chat", None) in names
    answerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn and spec.source.chat_template_fn.__name__ == "format_squad_answerable")
    unanswerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn and spec.source.chat_template_fn.__name__ == "format_squad_unanswerable")
    contrastive = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn and spec.source.chat_template_fn.__name__ == "format_squad_contrastive")
    assert answerable == 11_900_000
    assert unanswerable == 3_990_000
    assert contrastive == 7_980_000
    synth_answerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn and spec.source.chat_template_fn.__name__ == "format_synthqa_answerable")
    synth_unanswerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn and spec.source.chat_template_fn.__name__ == "format_synthqa_unanswerable")
    assert synth_answerable == 7_980_000
    assert synth_unanswerable == 3_990_000
    replay = next(spec.target_tokens for spec in config.sft_datasets if spec.source.name == "HuggingFaceFW/fineweb-edu")
    task_tokens = sum(spec.target_tokens for spec in config.sft_datasets) - replay
    assert replay == task_tokens
    assert sum(spec.target_tokens for spec in config.sft_datasets) == 271_740_000
    eval_names = [(spec.source.name, spec.source.split) for spec in config.eval_datasets]
    assert ("teapotai/synthqa", "test") in eval_names
    assert ("teapotai/teapot-chat", "train") in eval_names
    assert GSM8K_TEST.split == "test"
    assert SMOLTALK_APIGEN_TEST.chat_template_fn is not None


def test_tool_score_on_exact_call():
    gold = (
        '<tool_call>[{"name": "generate_random_string", '
        '"arguments": {"length": 5, "uppercase": true, "digits": false}}]</tool_call>'
    )
    scored = score_tool_prediction(gold, gold)
    assert scored["parse"] == 1.0
    assert scored["name"] == 1.0
    assert scored["args"] == 1.0
    refused = score_tool_prediction("No tool applies.", gold)
    assert refused["parse"] == 0.0
    restraint = score_tool_prediction("I cannot do that.", "The query cannot be answered, no tools were provided.")
    assert restraint["restraint"] == 1.0
    leaked = score_tool_prediction(gold, "The query cannot be answered, no tools were provided.")
    assert leaked["restraint"] == 0.0


def test_tool_parser_accepts_pretty_and_fenced_calls():
    pretty = """<tool_call>[
{"name": "max_divisor", "arguments": {"N": 48, "M": 18}},
{"name": "max_divisor", "arguments": {"N": 54, "M": 24}}
]</tool_call>"""
    gold = (
        '<tool_call>[{"name": "max_divisor", "arguments": {"N": 48, "M": 18}}, '
        '{"name": "max_divisor", "arguments": {"N": 54, "M": 24}}]</tool_call>'
    )
    assert score_tool_prediction(pretty, gold)["args"] == 1.0
    unclosed = pretty.replace("</tool_call>", "")
    assert score_tool_prediction(unclosed, gold)["name"] == 1.0
    fenced = """You can call it like this:
```json
{"name": "rotateImageAction", "arguments": {"degrees": 90}}
```"""
    assert extract_tool_calls(fenced)[0]["name"] == "rotateImageAction"
    assert classify_decision(fenced) == "tool_call"
    assert extract_tool_calls("None of the provided functions can be used to clean up the comment.") == []


def test_tool_args_ignore_key_order_and_numeric_strings():
    gold = '<tool_call>[{"name": "lookup", "arguments": {"b": 2, "a": 1}}]</tool_call>'
    pred = '<tool_call>[{"name": "lookup", "arguments": {"a": "1", "b": "2"}}]</tool_call>'
    assert score_tool_prediction(pred, gold)["args"] == 1.0
    bare = '{"name": "lookup", "arguments": {"a": 1, "b": 2}}'
    assert extract_tool_calls(bare)[0]["name"] == "lookup"


def test_decision_labels():
    assert classify_decision('{"name": "get_weather", "arguments": {"city": "Paris"}}') == "tool_call"
    assert classify_decision("Could you share the drink id?") == "request_for_info"
    assert classify_decision("Apologies, but I'm unable to provide real-time information.") == "cannot_answer"
    assert classify_decision("Paris is the capital of France.") == "direct"


def test_squad_and_gsm_scores():
    from teapottrainer.metrics import best_span_scores

    assert token_f1("in the late 1990s", "late 1990s") > 0.6
    em, f1, contained = best_span_scores(
        "Tivoli Gardens opened on 15 August 1843.",
        ["15 August 1843"],
    )
    assert em == 0.0 and contained == 1.0 and f1 > 0.2
    assert is_abstention("The context does not contain the answer.")
    assert not is_abstention("Beyonce started in the late 1990s.")
    assert score_gsm("16 - 3 - 4 = 9\n#### 18", "She makes 18.\n#### 18") == 1.0
    assert score_gsm("The answer is 18.0", "#### 18") == 1.0
    assert score_gsm("#### 17", "#### 18") == 0.0
    prompt, answer = format_gsm8k({"question": "What is 2+2?", "answer": "2+2=<<2+2=4>>4\n#### 4"})
    assert prompt.endswith("Assistant:\n")
    assert "calculator" in prompt
    assert '"expression": "2+2"' in answer
    assert score_gsm(
        '<tool_call>[{"name": "calculator", "arguments": {"expression": "2+2"}}]</tool_call>',
        "#### 4",
    ) == 1.0
    squad_prompt, squad_answer = format_squad(
        {"context": "The capital is Paris.", "question": "What is the capital?", "answers": {"text": []}}
    )
    assert "does not contain the answer" in squad_prompt
    assert squad_answer == "The context does not contain the answer."


def test_constraint_checker_on_a_satisfying_reply():
    instruction = (
        "Your response should contain at least 3 sentences. "
        "The response must contain exactly 3 bullet points. "
        "Include keywords [health, reduce, improve]. "
        "At the end of your response, please explicitly add a postscript starting with P.S."
    )
    reply = (
        "Walking helps health. It can reduce stress. It can improve sleep.\n"
        "* Walk every day for health\n"
        "* Stretch to reduce tension\n"
        "* Sleep enough to improve recovery\n"
        "P.S. Start small."
    )
    checks = dict(constraint_checks(instruction, reply))
    assert checks["min_sentences"]
    assert checks["exact_bullets"]
    assert checks["keywords"]
    assert checks["postscript"]


class _CharTokenizer:
    eos_token = ""
    eos_token_id = None
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 97 + 1 for ch in text]


def test_generators_do_not_share_a_source():
    tok = _CharTokenizer()
    left = DatasetSourceConfig(
        name="left",
        split="test",
        is_sft=True,
        chat_template_fn=lambda row: ("User: left\n\nAssistant:\n", row["answer"]),
    )
    right = DatasetSourceConfig(
        name="right",
        split="test",
        is_sft=True,
        chat_template_fn=lambda row: ("User: right\n\nAssistant:\n", row["answer"]),
    )
    gen_left = iterate_encoded(left, tok, seq_len=64, rows=[{"answer": "alpha"}])
    gen_right = iterate_encoded(right, tok, seq_len=64, rows=[{"answer": "beta"}])
    first_left = next(gen_left)
    first_right = next(gen_right)
    assert first_left["labels"] != first_right["labels"]
    assert any(token != -100 for token in first_left["labels"])


def test_holdout_slices_are_disjoint():
    key = "a fineweb document about tides"
    held = holdout_membership(key, 0.05, 42)
    assert held or not held
    # Same key always lands on the same side, and a large sample covers both.
    flags = {holdout_membership(f"doc {i}", 0.05, 42) for i in range(400)}
    assert flags == {True, False}


def test_supervised_tail_contains_the_answer():
    tok = _CharTokenizer()
    encoded = encode_pair(tok, "User: hi\n\nAssistant:\n", "hello there", seq_len=64)
    supervised = [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
    assert supervised
    text = "".join(chr((token - 1) % 97 + 97) if False else "" for token in [])
    del text
    # Char tokenizer is invertible for this alphabet via the same encode of the answer.
    answer_ids = tok.encode("hello there")
    assert supervised[: len(answer_ids)] == answer_ids


def test_prompt_is_masked_with_real_gpt2_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt = "System: be brief\n\nUser: Say hello\n\nAssistant:\n"
    answer = "Hello there."
    encoded = encode_pair(tok, prompt, answer, seq_len=64)
    assert encoded is not None
    supervised = [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
    decoded = tok.decode(supervised)
    assert "Hello" in decoded
    masked = [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label == -100]
    assert masked
    assert "Assistant" in tok.decode(masked)


def test_hotpot_keeps_supporting_pages_near_the_question():
    prompt, answer = format_hotpot({
        "question": "Were they the same nationality?",
        "answer": "yes",
        "supporting_facts": {"title": ["Ada Lovelace"], "sent_id": [0]},
        "context": {
            "title": ["Distractor City", "Ada Lovelace"],
            "sentences": [
                ["Distractor City is unrelated."],
                ["Ada Lovelace was English."],
            ],
        },
    })
    assert answer == "yes"
    assert "does not contain the answer" not in prompt
    assert prompt.index("Ada Lovelace") > prompt.index("Distractor City")
    assert prompt.index("Were they the same nationality?") > prompt.index("Ada Lovelace")


def test_messages_keep_system_and_every_assistant_turn():
    pairs = messages_format({
        "messages": [
            {"role": "system", "content": "Always end with an emoji."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Another"},
            {"role": "assistant", "content": "Sure"},
        ]
    })
    assert len(pairs) == 2
    assert "Always end with an emoji." in pairs[0][0]
    assert pairs[0][1] == "Hello"
    assert "Hello" in pairs[1][0]
    assert pairs[1][1] == "Sure"


def test_apigen_decision_prompt_matches_when2call():
    from datasets import load_dataset

    from teapottrainer.datasets import SMOLTALK_APIGEN_DECISION
    from teapottrainer.formatting import DECISION_RULES

    row = next(iter(load_dataset("HuggingFaceTB/smoltalk", "apigen-80k", split="test", streaming=True)))
    pairs = row_pairs(SMOLTALK_APIGEN_DECISION, row)
    assert len(pairs) == 1
    prompt, answer = pairs[0]
    assert DECISION_RULES.split("\n", 1)[0] in prompt
    assert "<tools>" in prompt
    from teapottrainer.formatting import rewrite_missing_tools_claim

    raw = row["messages"][-1]["content"]
    tools = prompt.split("<tools>", 1)[-1].split("</tools>", 1)[0]
    assert answer == rewrite_missing_tools_claim(raw, tools)
    if "<tool_call>" in answer:
        assert score_tool_prediction(answer, answer)["args"] == 1.0


def test_apigen_row_roundtrip_if_cached():
    from datasets import load_dataset

    row = next(iter(load_dataset("HuggingFaceTB/smoltalk", "apigen-80k", split="test", streaming=True)))
    pairs = row_pairs(SMOLTALK_APIGEN_TEST, row)
    assert pairs
    prompt, answer = pairs[-1]
    assert "Assistant:" in prompt
    assert row["messages"][1]["content"][:40] in prompt
    if "<tool_call>" in answer:
        assert score_tool_prediction(answer, answer)["args"] == 1.0
    else:
        assert score_tool_prediction(answer, answer)["restraint"] == 1.0


def test_when2call_gold_text_matches_its_label():
    from datasets import load_dataset

    from teapottrainer.evals import load_task_rows
    from teapottrainer.datasets import WHEN2CALL_TEST

    rows = load_task_rows(WHEN2CALL_TEST, 60, seed=0, kind="when2call")
    assert rows
    hits = 0
    labels = set()
    for row in rows:
        labels.add(row["correct_answer"])
        gold = row["answers"][row["correct_answer"]]
        hits += int(classify_decision(gold) == row["correct_answer"])
    assert len(labels) >= 3
    assert hits / len(rows) >= 0.9


def test_constraint_gold_replies_pass_the_checker():
    from datasets import load_dataset

    from teapottrainer.datasets import SMOLTALK_CONSTRAINTS_TEST
    from teapottrainer.metrics import score_constraints

    rows = list(load_dataset("HuggingFaceTB/smoltalk", "smol-constraints", split="test", streaming=True).take(40))
    scored = []
    for row in rows:
        pairs = row_pairs(SMOLTALK_CONSTRAINTS_TEST, row)
        if not pairs:
            continue
        user = row["messages"][0]["content"]
        result = score_constraints(user, pairs[-1][1])
        if result is not None:
            scored.append(result["example_pass"])
    assert len(scored) >= 20
    assert sum(scored) / len(scored) >= 0.75


def test_renamed_tool_call_matches_the_prompt():
    from teapottrainer.formatting import format_apigen_aliased
    from teapottrainer.metrics import extract_tool_calls

    row = {
        "messages": [
            {
                "role": "system",
                "content": 'Tools:\n<tools>[{"name": "book_train", "description": "Book a train."}]</tools>',
            },
            {"role": "user", "content": "Book a train from Boston to Portland."},
            {
                "role": "assistant",
                "content": '<tool_call>[{"name": "book_train", "arguments": {"origin": "Boston"}}]</tool_call>',
            },
        ]
    }
    prompt, answer = format_apigen_aliased(row)
    assert "book_train" not in prompt
    assert "book_train" not in answer
    prompted = extract_tool_calls(prompt.split("<tools>", 1)[-1])
    called = extract_tool_calls(answer)
    assert called and prompted
    assert called[0]["name"] == prompted[0]["name"]
    assert called[0]["arguments"]["origin"] == "Boston"
    again, _again_answer = format_apigen_aliased(row)
    assert "book_train" not in again
    # A fresh draw each call. One collision in 256 names is not enough to fail this.
    names = set()
    for _ in range(8):
        prompted_again = format_apigen_aliased(row)[0]
        names.add(extract_tool_calls(prompted_again.split("<tools>", 1)[-1])[0]["name"])
    assert len(names) > 1


def test_init_weights_must_match_the_model_config():
    from types import SimpleNamespace

    from teapottrainer.trainer import architecture_mismatches

    wanted = ModelConfig(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1408,
        max_position_embeddings=2048,
    )
    same = SimpleNamespace(
        hidden_size=512,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1408,
        max_position_embeddings=2048,
    )
    longer_context = SimpleNamespace(**{**same.__dict__, "max_position_embeddings": 4096})
    wrong_width = SimpleNamespace(**{**same.__dict__, "hidden_size": 768})
    too_short = SimpleNamespace(**{**same.__dict__, "max_position_embeddings": 1024})
    assert architecture_mismatches(same, wanted) == []
    assert architecture_mismatches(longer_context, wanted) == []
    assert any(item.startswith("hidden_size") for item in architecture_mismatches(wrong_width, wanted))
    assert any(item.startswith("seq_len") for item in architecture_mismatches(too_short, wanted))
    from teapottrainer.trainer import _resolve_init_from

    assert _resolve_init_from("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        _resolve_init_from("runs/does-not-exist/weights")
    except SystemExit as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing init_from should fail")


def test_mixture_fast_forward_replays_the_same_samples():
    import random

    from teapottrainer.trainer import _python_random_state, take_mixture_sample

    def play(count):
        random.seed(7)
        pools = [
            [{"input_ids": [index]} for index in range(30)],
            [{"input_ids": [100 + index]} for index in range(30)],
        ]
        produced = [0, 0]
        retired = [False, False]
        seen = []

        def take(index):
            if not pools[index]:
                return None
            return pools[index].pop(0)

        while len(seen) < count:
            drawn = take_mixture_sample([1.0, 1.0], [10_000, 10_000], produced, retired, take)
            assert drawn not in (None, "retired")
            seen.append(drawn[1]["input_ids"][0])
        return seen, [list(produced), _python_random_state()]

    full, _state = play(15)
    prefix, checkpoint = play(8)
    assert prefix == full[:8]
    random.seed(7)
    pools = [
        [{"input_ids": [index]} for index in range(30)],
        [{"input_ids": [100 + index]} for index in range(30)],
    ]
    produced = [0, 0]
    retired = [False, False]
    cursor = []

    def take(index):
        if not pools[index]:
            return None
        return pools[index].pop(0)

    while len(cursor) < 8:
        drawn = take_mixture_sample([1.0, 1.0], [10_000, 10_000], produced, retired, take)
        cursor.append(drawn[1]["input_ids"][0])
    assert produced == checkpoint[0]
    assert _python_random_state() == checkpoint[1]
    while len(cursor) < 15:
        drawn = take_mixture_sample([1.0, 1.0], [10_000, 10_000], produced, retired, take)
        cursor.append(drawn[1]["input_ids"][0])
    assert cursor == full


def test_fast_forward_rejects_a_cursor_that_does_not_match():
    import json
    import random
    from types import SimpleNamespace

    from teapottrainer.trainer import TeapotTrainer, _python_random_state, take_mixture_sample

    def play(count):
        random.seed(7)
        pools = [
            [{"input_ids": [index]} for index in range(30)],
            [{"input_ids": [100 + index]} for index in range(30)],
        ]
        produced = [0, 0]
        retired = [False, False]
        restarts = [0, 0]
        seen = []

        def take(index):
            random.choice([0, 1])
            if not pools[index]:
                return None
            sample = pools[index].pop(0)
            seen.append(sample["input_ids"][0])
            return sample

        while len(seen) < count:
            drawn = take_mixture_sample([1.0, 1.0], [10_000, 10_000], produced, retired, take)
            assert drawn not in (None, "retired")
        return seen, produced, restarts, _python_random_state()

    full, produced, restarts, state = play(6)
    payload = json.loads(json.dumps({
        "produced": produced,
        "restarts": restarts,
        "python_random": state,
    }))
    random.seed(7)
    pools = [
        [{"input_ids": [index]} for index in range(30)],
        [{"input_ids": [100 + index]} for index in range(30)],
    ]
    produced = [0, 0]
    retired = [False, False]
    seen = []

    def take(index):
        random.choice([0, 1])
        sample = pools[index].pop(0)
        seen.append(sample["input_ids"][0])
        return sample

    holder = SimpleNamespace(_restarts=[0, 0])
    TeapotTrainer._fast_forward(
        holder, "pretrain", [1.0, 1.0], [10_000, 10_000], produced, retired, take,
        6, payload["produced"], payload["restarts"], payload["python_random"],
    )
    assert seen == full
    try:
        TeapotTrainer._fast_forward(
            holder, "pretrain", [1.0, 1.0], [10_000, 10_000], produced, retired, take,
            6, [0, 0], [0, 0], payload["python_random"],
        )
    except SystemExit as exc:
        assert "did not land" in str(exc)
    else:
        raise AssertionError("a mismatched cursor should stop the resume")


def test_resume_directory_falls_back_when_the_live_dir_is_incomplete():
    import json
    import os
    import shutil
    import tempfile

    from teapottrainer.trainer import TeapotTrainer

    with tempfile.TemporaryDirectory() as tmp:
        trainer = object.__new__(TeapotTrainer)
        trainer.output_dir = tmp
        trainer.best_loss = 1.25
        trainer.best_behavior = None

        def save_weights(model, tokenizer, directory):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
                handle.write('{"hidden_size": 8}')

        trainer._save_weights = save_weights
        trainer._write_resume(None, None, None, {"phase": "pretrain", "phase_done": False, "tokens": 12})
        loaded = trainer._read_resume()
        assert loaded["tokens"] == 12
        assert loaded["dir"] == os.path.join(tmp, "checkpoints", "resume")
        assert loaded["best_loss"] == 1.25
        assert "dir" not in json.load(open(os.path.join(loaded["dir"], "trainer_state.json"), encoding="utf-8"))
        live = trainer._resume_dir()
        os.rename(live, live + ".old")
        os.makedirs(live)
        with open(os.path.join(live, "trainer_state.json"), "w", encoding="utf-8") as handle:
            handle.write('{"tokens": 1}')
        fallback = trainer._read_resume()
        assert fallback["dir"] == live + ".old"
        assert fallback["tokens"] == 12
        shutil.rmtree(live)
        assert trainer._read_resume()["tokens"] == 12


def test_sft_at_checkpoints_trains_a_copy_and_restores_pretrain_state():
    import os
    import random
    import tempfile

    import teapottrainer.trainer as trainer_mod
    from teapottrainer.trainer import TeapotTrainer

    class _Copy:
        def to(self, device):
            return self

        def train(self):
            return self

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "checkpoints", "pretrain_20pct")
        os.makedirs(base)
        with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        trainer = object.__new__(TeapotTrainer)
        trainer.config = TrainingConfig(
            tokenizer="local-test",
            lr=1e-3,
            sft_at_checkpoints=True,
            sft_datasets=[DatasetSpec(source="toy", target_tokens=8)],
            pretrain_datasets=[DatasetSpec(source="toy", target_tokens=8)],
        )
        trainer.output_dir = tmp
        trainer.device = "cpu"
        trainer.best_loss = 1.5
        trainer.best_behavior = 0.25
        trainer._restarts = [4]
        trainer._completed_phases = []
        trainer._checkpointing = False
        trainer.phase_stats = {}
        calls = []
        logged = []

        class _Chart:
            def add_scalar(self, tag, value, step, *args, **kwargs):
                logged.append((tag, value, step))

        trainer.writer = _Chart()

        def train_phase(*args, **kwargs):
            random.seed(123)
            trainer.best_loss = 9.0
            trainer.best_behavior = 0.9
            trainer._completed_phases.append("pretrain_20pct_sft")
            trainer.phase_stats[kwargs.get("final_phase_name") or args[0]] = {
                "last_loss": 0.4,
                "best_train_loss": 0.2,
            }
            calls.append(kwargs)
            directory = os.path.join(tmp, "checkpoints", "pretrain_20pct_sft")
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "metrics.json"), "w", encoding="utf-8") as handle:
                handle.write('{"loss": 1}')

        trainer._train_phase = train_phase
        original = trainer_mod.AutoModelForCausalLM.from_pretrained
        trainer_mod.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: _Copy()
        random.seed(7)
        expected_random = random.getstate()
        try:
            trainer._sft_from_base_checkpoint(_Copy(), None, "pretrain_20pct", 10)
        finally:
            trainer_mod.AutoModelForCausalLM.from_pretrained = original
        assert len(calls) == 1
        assert calls[0]["persist_resume"] is False
        assert calls[0]["record_completion"] is False
        assert calls[0]["update_best"] is False
        assert calls[0]["eval_during"] is False
        assert calls[0]["final_phase_name"] == "pretrain_20pct_sft"
        assert calls[0]["metric_group"] == "sft_checkpoint"
        assert calls[0]["metric_step"] == 10
        assert calls[0]["peak_lr"] == 2e-4
        assert ("sft_checkpoint/best_train_loss", 0.2, 10) in logged
        assert trainer.best_loss == 1.5
        assert trainer.best_behavior == 0.25
        assert trainer._restarts == [4]
        assert trainer._completed_phases == []
        assert random.getstate() == expected_random
        assert os.path.isfile(os.path.join(tmp, "checkpoints", "pretrain_20pct_sft", "train_stats.json"))
        trainer._sft_from_base_checkpoint(object(), None, "pretrain_20pct", 10)
        assert len(calls) == 1
        os.remove(os.path.join(tmp, "checkpoints", "pretrain_20pct_sft", "metrics.json"))
        trainer.config.sft_at_checkpoints = False
        trainer._sft_from_base_checkpoint(_Copy(), None, "pretrain_20pct", 10)
        assert len(calls) == 1
        trainer_mod.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: _Copy()
        try:
            trainer._sft_from_base_checkpoint(_Copy(), None, "pretrain_20pct", 10, required=True)
        finally:
            trainer_mod.AutoModelForCausalLM.from_pretrained = original
        assert len(calls) == 2
        assert calls[1]["metric_group"] == "sft_checkpoint"
        assert calls[1]["metric_step"] == 10


def test_before_batch_name_is_stable_in_the_run_directory():
    from teapottrainer.trainer import TeapotTrainer

    def make():
        def hook(model, phase):
            return None

        return hook

    plain = TeapotTrainer(TrainingConfig(tokenizer="hash-probe-hook", compile=False))
    one = TeapotTrainer(
        TrainingConfig(tokenizer="hash-probe-hook", compile=False, before_batch=make())
    )
    two = TeapotTrainer(
        TrainingConfig(tokenizer="hash-probe-hook", compile=False, before_batch=make())
    )
    try:
        assert one.output_dir == two.output_dir
        assert one.output_dir != plain.output_dir
        assert "hook" in one.config.model_dump()["before_batch"][0]
    finally:
        plain.writer.close()
        one.writer.close()
        two.writer.close()


def test_sft_at_checkpoints_does_not_change_the_run_directory():
    from teapottrainer.trainer import TeapotTrainer

    plain = TeapotTrainer(TrainingConfig(tokenizer="hash-probe-sft-flag", compile=False))
    flagged = TeapotTrainer(
        TrainingConfig(tokenizer="hash-probe-sft-flag", compile=False, sft_at_checkpoints=True)
    )
    try:
        assert plain.output_dir == flagged.output_dir
    finally:
        plain.writer.close()
        flagged.writer.close()


def test_eval_suite_records_task_scores_with_holdout_loss():
    import json
    import math
    import os
    import shutil

    import teapottrainer.trainer as trainer_mod
    from teapottrainer.trainer import TeapotTrainer

    config = TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            max_position_embeddings=8,
        ),
        pretrain_datasets=[],
        sft_datasets=[],
        eval_datasets=[
            DatasetSpec(
                source=DatasetSourceConfig(name="HuggingFaceFW/fineweb-edu", split="train", is_sft=False),
                target_tokens=8,
            ),
            DatasetSpec(
                source=DatasetSourceConfig(
                    name="rajpurkar/squad_v2",
                    split="validation",
                    is_sft=True,
                    chat_template_fn=lambda row: ("q", "a"),
                ),
                target_tokens=8,
            ),
        ],
        tokenizer="local-test",
        compile=False,
        eval_every_percent=101,
        holdout_ratio=0.0,
        seed=1,
    )
    trainer = TeapotTrainer(config)
    created = trainer.output_dir
    trainer.writer.close()

    scalars = []

    class _Quiet:
        def add_scalar(self, tag, value, step, *args, **kwargs):
            scalars.append((tag, value, step))

        def flush(self):
            return None

        def close(self):
            return None

    trainer.writer = _Quiet()
    trainer._checkpointing = False
    seen = []

    def lm_loss(model, tokenizer, src):
        seen.append(("lm", src.name))
        return 2.5

    def task_score(model, tokenizer, src, **kwargs):
        seen.append(("task", src.name))
        return {"metrics": {"answerable_f1": 0.25, "loss": 1.5, "support_answerable": 4.0}, "samples": []}

    trainer._lm_loss = lm_loss
    trainer._save_weights = lambda model, tokenizer, directory: os.makedirs(directory, exist_ok=True)
    original = trainer_mod.evaluate_source
    trainer_mod.evaluate_source = task_score

    class _Model:
        def train(self):
            return None

    try:
        flat = trainer.run_evaluation_suite(_Model(), None, "pretrain_10pct", 3)
        with open(os.path.join(created, "metrics.jsonl"), encoding="utf-8") as handle:
            saved = json.loads(handle.read())
        with open(os.path.join(created, "checkpoints", "best", "metrics.json"), encoding="utf-8") as handle:
            best = json.load(handle)
        with open(os.path.join(created, "checkpoints", "pretrain_10pct", "metrics.json"), encoding="utf-8") as handle:
            snapshot = json.load(handle)
        trainer.run_evaluation_suite(
            _Model(), None, "pretrain_sft", 10, update_best=False, metric_group="sft_checkpoint"
        )
    finally:
        trainer_mod.evaluate_source = original
        trainer.writer.close()
        shutil.rmtree(created, ignore_errors=True)
    assert ("lm", "HuggingFaceFW/fineweb-edu") in seen
    assert ("task", "rajpurkar/squad_v2") in seen
    assert flat["fineweb-edu/train/loss"] == 2.5
    assert flat["squad_v2/validation/answerable_f1"] == 0.25
    assert flat["squad_v2/validation/loss"] == 1.5
    assert flat["loss"] == 2.0
    assert ("eval/loss", 2.0, 3) in scalars
    assert ("val/loss", 2.0, 3) in scalars
    assert ("val/fineweb-edu/train/loss", 2.5, 3) in scalars
    assert ("val/squad_v2/validation/loss", 1.5, 3) in scalars
    assert ("eval/perplexity", math.exp(2.0), 3) in scalars
    assert ("val/perplexity", math.exp(2.0), 3) in scalars
    assert ("val/fineweb-edu/train/perplexity", math.exp(2.5), 3) in scalars
    assert ("val/squad_v2/validation/perplexity", math.exp(1.5), 3) in scalars
    assert ("eval/fineweb-edu/train/perplexity", math.exp(2.5), 3) in scalars
    assert ("sft_checkpoint/loss", 2.0, 10) in scalars
    assert ("sft_checkpoint/perplexity", math.exp(2.0), 10) in scalars
    assert ("sft_checkpoint/fineweb-edu/train/loss", 2.5, 10) in scalars
    assert ("sft_checkpoint/squad_v2/validation/loss", 1.5, 10) in scalars
    assert ("eval/fineweb-edu/train/loss", 2.5, 3) in scalars
    assert all(step != 10 or not tag.startswith("eval/") for tag, _value, step in scalars)
    assert saved["metrics"]["loss"] == 2.0
    assert saved["metrics"]["fineweb-edu/train/loss"] == 2.5
    assert saved["metrics"]["squad_v2/validation/answerable_f1"] == 0.25
    assert best["metrics"]["squad_v2/validation/answerable_f1"] == 0.25
    assert snapshot["phase"] == "pretrain_10pct"
    assert snapshot["step"] == 3
    assert snapshot["metrics"]["squad_v2/validation/answerable_f1"] == 0.25


def test_resume_fast_forward_matches_an_uninterrupted_run():
    """A tiny CPU phase, cut at the first checkpoint, lands on the same weights."""
    import os
    import random
    import shutil

    import torch

    from teapottrainer.config import DatasetSpec, ModelConfig, TrainingConfig
    from teapottrainer.trainer import TeapotTrainer, _capture_torch_rng, _restore_torch_rng

    config = TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            max_position_embeddings=8,
        ),
        pretrain_datasets=[
            DatasetSpec(source="left", target_tokens=96, weight=1.0),
            DatasetSpec(source="right", target_tokens=96, weight=1.0),
        ],
        sft_datasets=[],
        eval_datasets=[],
        tokenizer="local-test",
        batch_size=2,
        grad_accum=1,
        lr=1e-3,
        weight_decay=0.0,
        compile=False,
        eval_every_percent=101,
        holdout_ratio=0.0,
        seed=7,
    )
    trainer = TeapotTrainer(config)
    created = trainer.output_dir
    trainer.device = torch.device("cpu")
    trainer._checkpointing = False
    trainer.resume_every_tokens = 64
    trainer.writer.close()

    class _Quiet:
        def add_scalar(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    trainer.writer = _Quiet()

    class _StopAfterCheckpoint(Exception):
        pass

    def open_iterators(bucket):
        def _open(sources, tokenizer):
            def gen(marker):
                while True:
                    extra = random.choice([0, 1])
                    row = [marker] * 7 + [extra]
                    bucket.append((marker, extra))
                    yield {"input_ids": row, "labels": list(row)}

            return [iter(gen(1)), iter(gen(2))]

        return _open

    class _Tok:
        pad_token_id = 0
        eos_token_id = 0

        def save_pretrained(self, directory):
            os.makedirs(directory, exist_ok=True)

    try:
        torch.manual_seed(7)
        model = trainer._build_model(vocab_size=32)
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        rng_after_init = _capture_torch_rng()
        full_draws = []
        trainer._open_iterators = open_iterators(full_draws)
        trainer._train_phase("pretrain", config.pretrain_datasets, model, _Tok(), 0, config.lr, False, None)
        final = {name: value.detach().clone() for name, value in model.state_dict().items()}

        model.load_state_dict(initial)
        _restore_torch_rng(rng_after_init)
        abort_draws = []
        trainer._open_iterators = open_iterators(abort_draws)
        real_write = trainer._write_resume

        def abort_write(model, tokenizer, optimizer, state):
            real_write(model, tokenizer, optimizer, state)
            if not state.get("phase_done"):
                raise _StopAfterCheckpoint()

        trainer._write_resume = abort_write
        try:
            trainer._train_phase("pretrain", config.pretrain_datasets, model, _Tok(), 0, config.lr, False, None)
        except _StopAfterCheckpoint:
            pass
        else:
            raise AssertionError("the first checkpoint should have stopped this run")
        trainer._write_resume = real_write
        saved = trainer._read_resume()
        assert saved["phase_done"] is False
        assert saved["tokens"] == 64
        assert saved["samples_taken"] == len(abort_draws)
        assert abort_draws == full_draws[: len(abort_draws)]
        assert os.path.isfile(os.path.join(saved["dir"], "optimizer.pt"))

        from transformers import AutoModelForCausalLM

        reloaded = AutoModelForCausalLM.from_pretrained(saved["dir"], dtype=torch.float32)
        for name, value in model.state_dict().items():
            assert torch.equal(reloaded.state_dict()[name], value), name

        resume_draws = []
        trainer._open_iterators = open_iterators(resume_draws)
        trainer._train_phase(
            "pretrain", config.pretrain_datasets, reloaded, _Tok(), 0, config.lr, False, saved,
        )
        model = reloaded
        assert resume_draws == full_draws
        for name, value in final.items():
            assert torch.equal(model.state_dict()[name], value), name
    finally:
        trainer.writer.close()
        shutil.rmtree(created, ignore_errors=True)
        if os.path.isdir(created):
            shutil.rmtree(created, ignore_errors=True)


def test_contrastive_question_uses_an_entity_from_the_passage():
    from teapottrainer.formatting import (
        ABSTAIN_TEXT,
        contrastive_question,
        format_synthqa_contrastive,
    )

    context = "The kiln in Harbor Town opened in 1902. It closed during the 1944 storm."
    question = contrastive_question(context, "1902")
    assert question
    assert "Harbor Town" in question
    assert "1902" not in question
    prompt, answer = format_synthqa_contrastive({
        "context": context,
        "question": "When did the kiln open?",
        "type": "extraction",
        "answer": "1902",
    })
    assert answer == ABSTAIN_TEXT
    assert "Harbor Town" in prompt
    assert "When did the kiln open?" not in prompt
    assert format_synthqa_contrastive({
        "context": context,
        "question": "Who funded it?",
        "type": "unanswerable",
        "answer": None,
    }) is None


def test_missing_argument_asks_and_unrelated_tool_refuses():
    from teapottrainer.formatting import (
        TOOLS_PRESENT_REFUSAL,
        format_apigen_missing_argument,
        format_apigen_unrelated_tool,
        rewrite_missing_tools_claim,
    )
    from teapottrainer.metrics import extract_tool_calls

    row = {
        "messages": [
            {
                "role": "system",
                "content": 'Tools:\n<tools>[{"name": "book_train", "description": "Book a train.", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}}, "required": ["origin"]}}]</tools>',
            },
            {"role": "user", "content": "Book a train from Boston to Portland."},
            {
                "role": "assistant",
                "content": '<tool_call>[{"name": "book_train", "arguments": {"origin": "Boston"}}]</tool_call>',
            },
        ]
    }
    prompt, answer = format_apigen_missing_argument(row)
    assert "<tool_call>" not in answer
    assert answer.endswith("?")
    assert "Boston" not in prompt.split("User:", 1)[-1]
    assert extract_tool_calls(prompt.split("<tools>", 1)[-1])
    refused_prompt, refused = format_apigen_unrelated_tool(row)
    assert refused == TOOLS_PRESENT_REFUSAL
    assert "no tools were provided" not in refused.lower()
    assert "celsius" in refused_prompt
    assert "book_train" not in refused_prompt
    assert rewrite_missing_tools_claim(
        "The query cannot be answered, no tools were provided.",
        '[{"name": "cork_glowworm"}]',
    ) == TOOLS_PRESENT_REFUSAL
    assert "no tools were provided" in rewrite_missing_tools_claim(
        "The query cannot be answered, no tools were provided.",
        "[]",
    )


def test_abstaining_on_an_answerable_question_lowers_the_score():
    from teapottrainer.evals import selection_score

    abstain = selection_score({"squad/answerable_f1": 0.0, "squad/answerable_abstain": 1.0})
    answer = selection_score({"squad/answerable_f1": 0.0, "squad/answerable_abstain": 0.0})
    assert answer > abstain


def test_synthqa_and_teapot_chat_prompts():
    from teapottrainer.formatting import format_synthqa, format_teapot_chat

    prompt, answer = format_synthqa({
        "context": "The kiln opened in 1902.",
        "question": "When did the kiln open?",
        "type": "extraction",
        "answer": "1902",
    })
    assert prompt.endswith("Assistant:\n")
    assert "1902" == answer
    assert "kiln opened in 1902" in prompt
    skipped, abstain = format_synthqa({
        "context": "The kiln opened in 1902.",
        "question": "Who funded it?",
        "type": "unanswerable",
        "answer": None,
    })
    assert abstain == "The context does not contain the answer."
    assert "Who funded it?" in skipped
    chat_prompt, chat_answer = format_teapot_chat({
        "system_prompt": "You are Teapot.",
        "context": "",
        "query": "who are you",
        "response": "I am Teapot",
    })
    assert chat_prompt.endswith("Assistant:\n")
    assert chat_answer == "I am Teapot"
    assert "You are Teapot." in chat_prompt
    from teapottrainer.formatting import format_synthqa_answerable, format_synthqa_unanswerable

    assert format_synthqa_answerable({
        "context": "The kiln opened in 1902.",
        "question": "Who funded it?",
        "type": "unanswerable",
        "answer": None,
    }) is None
    kept, kept_answer = format_synthqa_unanswerable({
        "context": "The kiln opened in 1902.",
        "question": "Who funded it?",
        "type": "unanswerable",
        "answer": None,
    })
    assert kept_answer == "The context does not contain the answer."
    assert "Who funded it?" in kept


def test_generation_stops_on_the_tokenizer_end_token():
    from teapottrainer.evals import generation_stop_ids
    from teapottrainer.trainer import align_tokenizer_ids

    class Tok:
        eos_token_id = 50256
        pad_token_id = 50256

    assert generation_stop_ids(Tok()) == {"pad_token_id": 50256, "eos_token_id": 50256}

    class Config:
        eos_token_id = 2

    class Generation:
        eos_token_id = 2

    class Model:
        def __init__(self):
            self.config = Config()
            self.generation_config = Generation()

    model = Model()
    align_tokenizer_ids(model, Tok())
    assert model.config.eos_token_id == 50256
    assert model.generation_config.eos_token_id == 50256


def test_dataset_can_upweight_the_stop_token():
    class Tok(_CharTokenizer):
        eos_token = "\x01"
        eos_token_id = 2

    source = DatasetSourceConfig(
        name="short",
        split="test",
        is_sft=True,
        chat_template_fn=lambda row: ("User: hi\n\nAssistant:\n", row["answer"]),
        token_loss_weight_fn=eos_loss_weight(STOP_TOKEN_LOSS_WEIGHT),
    )
    encoded = next(iterate_encoded(source, Tok(), seq_len=64, rows=[{"answer": "yes"}]))
    assert encoded["labels"][-1] == Tok.eos_token_id
    assert encoded["loss_weights"][-1] == STOP_TOKEN_LOSS_WEIGHT
    assert all(weight == 1.0 for weight, label in zip(encoded["loss_weights"], encoded["labels"]) if label != Tok.eos_token_id)
    assert SQUAD_V2_ANSWERABLE.token_loss_weight_fn is not None

    class Eos:
        eos_token_id = 2

    assert SQUAD_V2_ANSWERABLE.token_loss_weight_fn({"labels": [-100, 7, 2]}, Eos()) == [1.0, 4.0, 8.0]
    plain = DatasetSourceConfig(
        name="plain",
        split="test",
        is_sft=True,
        chat_template_fn=lambda row: ("User: hi\n\nAssistant:\n", row["answer"]),
    )
    unweighted = next(iterate_encoded(plain, Tok(), seq_len=64, rows=[{"answer": "yes"}]))
    assert "loss_weights" not in unweighted


def test_token_weight_changes_the_loss_and_survives_into_the_trainer():
    import os
    import tempfile

    import torch

    from teapottrainer.trainer import TeapotTrainer, weighted_cross_entropy

    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 0] = 8.0
    logits[0, 1, 2] = 8.0
    labels = torch.tensor([[-100, 1, 2]])
    even = weighted_cross_entropy(logits, labels, torch.ones(1, 3))
    stop_heavier = weighted_cross_entropy(logits, labels, torch.tensor([[1.0, 8.0, 1.0]]))
    content_heavier = weighted_cross_entropy(logits, labels, torch.tensor([[1.0, 1.0, 8.0]]))
    assert stop_heavier > even > content_heavier

    previous = os.getcwd()
    workspace = tempfile.mkdtemp()
    try:
        os.chdir(workspace)
        marked = DatasetSourceConfig(
            name="marked",
            split="train",
            is_sft=True,
            chat_template_fn=lambda row: ("User: hi\n\nAssistant:\n", "yes"),
            token_loss_weight_fn=eos_loss_weight(5),
        )
        config = TrainingConfig(
            model=ModelConfig(model_type="llama", hidden_size=32, layers=1, heads=4, kv_heads=4, intermediate=64, seq_len=32),
            sft_datasets=[DatasetSpec(source=marked, target_tokens=32)],
            tokenizer="gpt2",
        )
        trainer = TeapotTrainer(config)
        materialized = trainer._materialize_sources(config.sft_datasets)
        trainer.writer.close()
    finally:
        os.chdir(previous)
    assert materialized[0].token_loss_weight_fn is marked.token_loss_weight_fn


def test_chat_benchmarks_stay_out_of_the_running_mix():
    from train import build_config
    from teapottrainer import ALPACA_EVAL, ARENA_HARD, IFEVAL, MT_BENCH
    from teapottrainer.datasets import encode_pair
    from teapottrainer.evals import task_kind
    from teapottrainer.formatting import NO_REFERENCE, format_alpaca_eval, format_arena_hard, format_ifeval, format_mt_bench
    from teapottrainer.ifeval import score_ifeval

    config = build_config()
    used = {spec.source.name for spec in config.pretrain_datasets + config.sft_datasets + config.eval_datasets}
    presets = (IFEVAL, MT_BENCH, ALPACA_EVAL, ARENA_HARD)
    assert {src.name for src in presets}.isdisjoint(used)
    assert [task_kind(src) for src in presets] == ["ifeval", "mtbench", "alpaca", "arena"]

    prompt, answer = format_ifeval({"prompt": "Write a note without commas."})
    assert prompt.startswith("User: Write a note")
    assert answer == NO_REFERENCE
    assert encode_pair(_CharTokenizer(), prompt, answer, 64) is None

    turns = format_mt_bench({
        "prompt": ["What is two plus two?", "Now add one."],
        "reference": ["Four.", "Five."],
    })
    assert len(turns) == 2
    assert "Four." in turns[1][0]
    assert turns[1][1] == "Five."
    assert format_mt_bench({"prompt": ["Hello"], "reference": []})[0][1] == NO_REFERENCE

    alpaca_prompt, alpaca_answer = format_alpaca_eval({"instruction": "Name a color.", "output": "Blue."})
    assert "Name a color." in alpaca_prompt
    assert alpaca_answer == "Blue."
    arena_prompt, arena_answer = format_arena_hard({"turns": [{"content": "How do I grep?"}]})
    assert "How do I grep?" in arena_prompt
    assert arena_answer == NO_REFERENCE

    followed = score_ifeval(
        {
            "instruction_id_list": ["punctuation:no_comma", "detectable_format:number_bullet_lists"],
            "kwargs": [{}, {"num_bullets": 2}],
        },
        "Walk every day\n* Stretch gently\n* Sleep enough",
    )
    assert followed["prompt_strict"] == 1.0
    quoted = score_ifeval(
        {"instruction_id_list": ["startend:quotation"], "kwargs": [{}]},
        "Sure.\n\"A quoted answer.\"",
    )
    assert quoted["prompt_strict"] == 0.0
    assert quoted["prompt_loose"] == 1.0
    korean = score_ifeval(
        {"instruction_id_list": ["language:response_language"], "kwargs": [{"language": "ko"}]},
        "안녕하세요 친구",
    )
    assert korean["prompt_strict"] == 1.0
    capitals = score_ifeval(
        {"instruction_id_list": ["change_case:english_capital"], "kwargs": [{}]},
        "ПРИВЕТ",
    )
    assert capitals["prompt_strict"] == 0.0


def test_added_presets_stay_out_of_the_running_mix():
    from train import build_config
    from teapottrainer import (
        ARC_EASY,
        ARC_EASY_VAL,
        BFCL_IRRELEVANCE,
        BFCL_MULTIPLE,
        BFCL_PARALLEL,
        BFCL_SIMPLE,
        COQA,
        COQA_VAL,
        DROP,
        DROP_VAL,
        HELLASWAG,
        HELLASWAG_VAL,
        HERMES_FUNCTION_CALLING,
        PIQA,
        PIQA_VAL,
        WINOGRANDE,
        WINOGRANDE_VAL,
    )
    from teapottrainer.evals import task_kind
    from teapottrainer.formatting import (
        TOOLS_PRESENT_REFUSAL,
        format_bfcl,
        format_coqa,
        format_drop,
        format_hermes,
        multiple_choice_view,
    )
    from teapottrainer.metrics import extract_tool_calls, score_bfcl

    config = build_config()
    used = {spec.source.name for spec in config.pretrain_datasets + config.sft_datasets + config.eval_datasets}
    presets = (
        ARC_EASY,
        ARC_EASY_VAL,
        HELLASWAG,
        HELLASWAG_VAL,
        PIQA,
        PIQA_VAL,
        WINOGRANDE,
        WINOGRANDE_VAL,
        COQA,
        COQA_VAL,
        DROP,
        DROP_VAL,
        HERMES_FUNCTION_CALLING,
        BFCL_SIMPLE,
        BFCL_MULTIPLE,
        BFCL_PARALLEL,
        BFCL_IRRELEVANCE,
    )
    assert {src.name for src in presets}.isdisjoint(used)
    assert [task_kind(src) for src in presets] == [
        "choice", "choice", "choice", "choice", "choice", "choice", "choice", "choice",
        "coqa", "coqa", "drop", "drop",
        "tools",
        "bfcl", "bfcl", "bfcl", "bfcl",
    ]
    assert ARC_EASY_VAL.split == "validation"
    assert BFCL_PARALLEL.subset == "parallel"
    assert HERMES_FUNCTION_CALLING.subset == "func_calling"

    viewed = multiple_choice_view("allenai/ai2_arc", {
        "question": "Which is newest?",
        "choices": {"text": ["phone", "television"], "label": ["A", "B"]},
        "answerKey": "A",
    })
    assert viewed is not None
    prompt, gold, options, index = viewed
    assert "phone" in prompt and "television" in prompt
    assert gold == "phone" and options == ["phone", "television"] and index == 0
    assert multiple_choice_view("ybisk/piqa", {"goal": "Boil eggs.", "sol1": "Use water.", "sol2": "Use sand.", "label": -1}) is None
    assert multiple_choice_view("allenai/winogrande", {
        "sentence": "Sarah was better than Maria so _ got the easy cases.",
        "option1": "Sarah",
        "option2": "Maria",
        "answer": "2",
    })[1] == "Maria"

    turns = format_coqa({
        "story": "Cotton was a white kitten.",
        "questions": ["What color was Cotton?", "Where did she live?"],
        "answers": {"input_text": ["white", "in a barn"]},
    })
    assert turns[0][1] == "white"
    assert "white" in turns[1][0]
    assert turns[1][1] == "in a barn"

    drop_prompt, drop_answer = format_drop({
        "passage": "Chaz Schilens scored, then Vonta Leach scored.",
        "question": "Who scored the first touchdown?",
        "answers_spans": {"spans": ["Chaz Schilens", "JaMarcus Russell"], "types": ["span", "span"]},
    })
    assert "Who scored" in drop_prompt
    assert drop_answer == "Chaz Schilens"

    hermes = format_hermes({
        "tools": "[{\"name\": \"get_feed\"}]",
        "conversations": [
            {"from": "system", "value": "Old system prompt."},
            {"from": "human", "value": "Check the front door."},
            {"from": "gpt", "value": "<tool_call>\n{\"name\": \"get_feed\", \"arguments\": {\"camera_id\": \"front_door\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"record_feed\", \"arguments\": {\"camera_id\": \"front_door\"}}\n</tool_call>"},
            {"from": "tool", "value": "{\"live_feed_url\": \"https://example.com/live\"}"},
            {"from": "gpt", "value": "The feed is ready."},
        ],
    })
    assert len(hermes) == 2
    assert "Old system prompt." not in hermes[0][0]
    assert "get_feed" in hermes[0][0]
    calls = extract_tool_calls(hermes[0][1])
    assert [call["name"] for call in calls] == ["get_feed", "record_feed"]
    assert "Tool result:" in hermes[1][0]
    assert hermes[1][1] == "The feed is ready."

    ground = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    prompt, answer = format_bfcl({
        "question": [[{"role": "user", "content": "Find the area."}]],
        "function": [{"name": "calculate_triangle_area"}],
        "ground_truth": ground,
    })
    assert "Find the area." in prompt
    assert extract_tool_calls(answer)[0]["arguments"]["unit"] == "units"
    alternate = '<tool_call>[{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]</tool_call>'
    assert score_bfcl(alternate, ground)["args"] == 1.0
    swapped = (
        '<tool_call>[{"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}},'
        '{"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}]</tool_call>'
    )
    parallel = [
        {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
    ]
    assert score_bfcl(swapped, parallel)["args"] == 1.0
    assert score_bfcl("I cannot do that.", None)["restraint"] == 1.0
    refused_prompt, refused = format_bfcl({
        "question": [[{"role": "user", "content": "Solve the quadratic."}]],
        "function": [{"name": "math.sum"}],
    })
    assert "Solve the quadratic." in refused_prompt
    assert refused == TOOLS_PRESENT_REFUSAL


def test_chat_format_names_and_aliases():
    import pytest

    from teapottrainer.chat_formats import chat_format_names, get_chat_format

    assert "chatml" in chat_format_names()
    assert "plain" in chat_format_names()
    assert get_chat_format("qwen").name == "chatml"
    assert get_chat_format("smollm2").name == "smollm"
    with pytest.raises(ValueError):
        get_chat_format("not-a-format")
    assert TrainingConfig(chat_format="Qwen").chat_format == "chatml"


def test_every_chat_format_matches_apply_chat_template():
    from transformers import AutoTokenizer

    from teapottrainer.chat_formats import CHAT_FORMATS

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Who are you?"},
    ]
    for name, fmt in CHAT_FORMATS.items():
        tokenizer.chat_template = fmt.template
        assert fmt.render(messages) == tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ), name


def test_plain_format_matches_the_historical_prompt():
    prompt, answer = messages_format({
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Again"},
            {"role": "assistant", "content": "Yes"},
        ]
    })[0]
    assert prompt == "System: Be brief.\n\nUser: Hi\n\nAssistant:\n"
    assert answer == "Hello"
    second = messages_format({
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Again"},
            {"role": "assistant", "content": "Yes"},
        ]
    })[1][0]
    assert second == "System: Be brief.\n\nUser: Hi\n\nAssistant: Hello\n\nUser: Again\n\nAssistant:\n"


def test_chat_format_matches_the_tokenizer_template_and_skips_pretrain():
    from transformers import AutoTokenizer

    from teapottrainer.chat_formats import use_chat_format
    from teapottrainer.formatting import render_prompt

    base = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    instruct = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    smollm2 = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    messages = [{"role": "user", "content": "Who are you?"}]
    with use_chat_format("chatml"):
        prompt = render_prompt(messages)
        assert prompt == instruct.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert "<|im_start|>user" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")
        raw = DatasetSourceConfig(name="web", split="test", is_sft=False, text_column="text")
        document = "a plain document " * 30
        packed = next(iterate_encoded(raw, base, seq_len=8, rows=[{"text": document}]))
        raw_ids = base.encode(document + base.eos_token, add_special_tokens=False)
        assert packed["input_ids"] == raw_ids[:8]
        assert "<|im_start|>" not in base.decode(packed["input_ids"])
        chat = DatasetSourceConfig(
            name="chat",
            split="test",
            is_sft=True,
            chat_template_fn=messages_format,
        )
        encoded = next(iterate_encoded(
            chat,
            base,
            seq_len=64,
            rows=[{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}],
        ))
        supervised = [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
        assert base.decode(supervised).startswith("Hello")
        assert supervised[-1] == base.convert_tokens_to_ids("<|im_end|>")
    with use_chat_format("smollm"):
        rendered = render_prompt(messages)
        assert rendered == smollm2.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert "helpful AI assistant named SmolLM" in rendered
    # The default stays plain after the context exits.
    assert messages_format({
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    })[0][0].endswith("Assistant:\n")


def test_supervised_save_writes_the_chat_template():
    import json
    import os
    import tempfile

    from transformers import AutoTokenizer

    from teapottrainer.trainer import TeapotTrainer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    config = TrainingConfig(chat_format="chatml", tokenizer="HuggingFaceTB/SmolLM2-135M", compile=False)
    trainer = TeapotTrainer(config)
    created = trainer.output_dir
    trainer.writer.close()

    class _Model:
        def __init__(self):
            self.config = type("C", (), {"eos_token_id": tokenizer.eos_token_id})()
            self.generation_config = type("G", (), {"eos_token_id": tokenizer.eos_token_id})()

        def save_pretrained(self, directory):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"eos_token_id": self.config.eos_token_id}, handle)

    model = _Model()
    try:
        with tempfile.TemporaryDirectory() as directory:
            trainer._save_chat = True
            trainer._save_weights(model, tokenizer, directory)
            saved = AutoTokenizer.from_pretrained(directory)
            messages = [{"role": "user", "content": "Who are you?"}]
            assert saved.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).endswith(
                "<|im_start|>assistant\n"
            )
            assert saved.eos_token == "<|im_end|>"
            with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
                assert json.load(handle)["eos_token_id"] == tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert tokenizer.eos_token != "<|im_end|>"
        assert not tokenizer.chat_template
        with tempfile.TemporaryDirectory() as directory:
            trainer._save_weights(model, tokenizer, directory)
            saved = AutoTokenizer.from_pretrained(directory)
            assert not saved.chat_template
    finally:
        trainer.writer.close()
        import shutil
        shutil.rmtree(created, ignore_errors=True)

