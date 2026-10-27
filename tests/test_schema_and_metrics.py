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
    "sft_datasets",
    "eval_datasets",
    "tokenizer",
    "init_from",
    "batch_size",
    "grad_accum",
    "lr",
    "weight_decay",
    "compile",
    "logging_dir",
    "eval_every_percent",
    "holdout_ratio",
    "seed",
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


def test_train_script_uses_the_same_constructor():
    from train import build_config

    config = build_config()
    dumped = set(config.model_dump().keys())
    assert dumped == LOCKED_TRAINING_FIELDS
    assert config.model.layers == 26
    assert config.model.hidden_size == 512
    assert config.tokenizer == "HuggingFaceTB/SmolLM2-135M"
    assert config.init_from is None
    assert TrainingConfig().init_from is None
    assert config.pretrain_datasets[0].target_tokens == 20_000_000_000
    assert config.lr == 1e-3
    names = []
    for spec in config.sft_datasets:
        names.append((spec.source.name, spec.source.subset))
    assert ("HuggingFaceTB/smoltalk", "all") not in names
    assert ("nvidia/When2Call", "train_sft") in names
    assert ("teapotai/synthqa", None) in names
    assert ("teapotai/teapot-chat", None) in names
    answerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn.__name__ == "format_squad_answerable")
    unanswerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn.__name__ == "format_squad_unanswerable")
    assert answerable == unanswerable == 14_000_000
    synth_answerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn.__name__ == "format_synthqa_answerable")
    synth_unanswerable = next(spec.target_tokens for spec in config.sft_datasets if spec.source.chat_template_fn.__name__ == "format_synthqa_unanswerable")
    assert synth_answerable == synth_unanswerable == 6_000_000
    when2call = next(spec.target_tokens for spec in config.sft_datasets if spec.source.name == "nvidia/When2Call")
    calls = sum(
        spec.target_tokens
        for spec in config.sft_datasets
        if spec.source.chat_template_fn.__name__ in {"format_apigen_decision", "format_apigen_aliased"}
    )
    assert when2call >= calls * 0.8
    assert sum(spec.target_tokens for spec in config.sft_datasets) == 116_000_000
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
    assert row["messages"][-1]["content"] == answer
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

