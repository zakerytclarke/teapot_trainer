"""Packing, embedding scores, and the SmolLM2-135M parameter count."""

import json
import os
import tempfile

import torch
from transformers import AutoModelForCausalLM, BertConfig, LlamaConfig, LlamaForCausalLM

from teapottrainer.config import ModelConfig, TrainingConfig
from teapottrainer.modeling_teapot import (
    SMOLLM_PARAMETER_COUNT,
    TeapotConfig,
    TeapotForCausalLM,
    llama_parameter_count,
    pack_conversation,
    split_conversation,
)
from teapottrainer.trainer import TeapotTrainer


class CharTokenizer:
    """One character is one token, so budget tests are exact."""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


class BatchTokenizer:
    """One row per text, truncated to max_length. Ids stay inside a tiny vocab."""

    pad_token_id = 0

    def __call__(
        self,
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        lengths = []
        for text in texts:
            ids = [((ord(ch) % 50) + 1) for ch in text][:max_length] or [1]
            rows.append(ids)
            lengths.append(len(ids))
        width = max(lengths)
        ids = [row + [0] * (width - len(row)) for row in rows]
        mask = [[1] * length + [0] * (width - length) for length in lengths]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids, "attention_mask": mask}


class LmTokenizer:
    eos_token = ""
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [((ord(ch) % 60) + 1) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(32 + ((int(i) - 1) % 60)) for i in ids)


def _tiny_config(**overrides) -> TeapotConfig:
    embedder = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    ).to_dict()
    values = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pack_context_length=48,
        embedder_config=embedder,
        embedder_max_length=8,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    values.update(overrides)
    return TeapotConfig(**values)


def test_parameter_count_matches_a_tied_llama():
    vocab, hidden, layers, heads, kv, intermediate = 128, 32, 2, 4, 2, 64
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv,
            intermediate_size=intermediate,
            tie_word_embeddings=True,
        )
    )
    seen = set()
    total = 0
    for param in model.parameters():
        if param.data_ptr() in seen:
            continue
        seen.add(param.data_ptr())
        total += param.numel()
    assert llama_parameter_count(vocab, hidden, layers, heads, kv, intermediate) == total
    assert llama_parameter_count(49152, 576, 30, 9, 3, 1536) == SMOLLM_PARAMETER_COUNT


def test_latest_user_message_is_the_query():
    system, query, candidates = split_conversation(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Latest question"},
        ]
    )
    assert system == "Be brief."
    assert query == "Latest question"
    assert [item["content"] for item in candidates] == ["Earlier question", "Earlier answer"]


def test_each_message_is_one_truncated_embedding():
    torch.manual_seed(0)
    seen = {}

    class RecordingTokenizer(BatchTokenizer):
        def __call__(self, texts, **kwargs):
            seen["texts"] = list(texts)
            seen["max_length"] = kwargs["max_length"]
            assert "return_overflowing_tokens" not in kwargs
            return super().__call__(texts, **kwargs)

    model = TeapotForCausalLM(_tiny_config(embedder_max_length=8))
    model._embedder_tokenizer_obj = RecordingTokenizer()
    scores = model.score_messages("query " + ("q" * 40), ["alpha " + ("a" * 40), "beta"])
    assert scores.shape == (2,)
    assert seen["max_length"] == 8
    assert len(seen["texts"]) == 3
    assert max(len(text) for text in seen["texts"]) > 8


def test_pack_keeps_system_and_latest_user_and_orders_by_time():
    tokenizer = CharTokenizer()
    messages = [
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": "First fact."},
        {"role": "assistant", "content": "Noted first."},
        {"role": "user", "content": "Question?"},
    ]
    # The later candidate scores higher, and both fit.
    prompt = pack_conversation(messages, tokenizer, [0.1, 0.9], context_length=200)
    assert prompt.index("System: Rules.") < prompt.index("User: First fact.")
    assert prompt.index("User: First fact.") < prompt.index("Assistant: Noted first.")
    assert prompt.index("Assistant: Noted first.") < prompt.index("User: Question?")
    assert prompt.endswith("Assistant:\n")
    assert len(tokenizer.encode(prompt)) <= 200


def test_a_long_message_can_fill_the_remaining_window():
    tokenizer = CharTokenizer()
    messages = [
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": "LOW" * 30},
        {"role": "assistant", "content": "HIGH" + ("x" * 80)},
        {"role": "user", "content": "Question?"},
    ]
    prompt = pack_conversation(messages, tokenizer, [0.1, 0.9], context_length=60)
    assert "System: Rules." in prompt
    assert "Question?" in prompt
    assert "HIG" in prompt
    assert "LOW" not in prompt
    assert len(tokenizer.encode(prompt)) <= 60


def test_user_truncation_keeps_the_end_of_the_question():
    tokenizer = CharTokenizer()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    ]
    prompt = pack_conversation(messages, tokenizer, [], context_length=40)
    assert "SYS" in prompt
    assert prompt.index("Z") > prompt.index("User:")
    assert "ABC" not in prompt
    assert len(tokenizer.encode(prompt)) <= 40


def test_score_length_must_match_candidates():
    try:
        pack_conversation(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}, {"role": "user", "content": "Q"}],
            CharTokenizer(),
            [0.2],
            100,
        )
    except ValueError as exc:
        assert "scores" in str(exc)
    else:
        raise AssertionError("expected a score-length error")


def test_retrieval_loss_trains_the_embedder_only():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "alpha fact"},
        {"role": "assistant", "content": "beta fact"},
        {"role": "user", "content": "What was beta?"},
    ]
    trained = False
    for seed in range(12):
        torch.manual_seed(seed)
        model = TeapotForCausalLM(_tiny_config(max_position_embeddings=128, pack_context_length=128))
        model._embedder_tokenizer_obj = BatchTokenizer()
        model.zero_grad()
        loss = model.retrieval_loss(messages, "beta fact", LmTokenizer())
        loss.backward()
        encoder_grad = model.embedder.embeddings.word_embeddings.weight.grad
        assert encoder_grad is not None
        assert model.lm_head.weight.grad is None
        if torch.count_nonzero(encoder_grad) > 0:
            trained = True
            break
    assert trained


def test_forward_still_returns_a_language_model_loss():
    torch.manual_seed(0)
    model = TeapotForCausalLM(_tiny_config())
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids=input_ids, labels=input_ids)
    assert output.loss is not None
    assert output.logits.shape == (2, 8, 64)


def test_checkpoint_reloads_through_auto_model(tmp_path):
    torch.manual_seed(0)
    model = TeapotForCausalLM(_tiny_config())
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["auto_map"]["AutoConfig"].endswith("TeapotConfig")
    assert config["auto_map"]["AutoModelForCausalLM"].endswith("TeapotForCausalLM")
    assert os.path.isfile(tmp_path / "modeling_teapot.py")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True)
    left = model.embedder.embeddings.word_embeddings.weight.detach()
    right = reloaded.embedder.embeddings.word_embeddings.weight.detach()
    assert torch.equal(left, right)
    assert reloaded.config.pack_context_length == 48


def test_generate_from_messages_stays_inside_the_position_window():
    torch.manual_seed(0)
    model = TeapotForCausalLM(_tiny_config(max_position_embeddings=64, pack_context_length=32))
    model._embedder_tokenizer_obj = BatchTokenizer()
    model.eval()
    messages = [
        {"role": "system", "content": "Rules."},
        {"role": "assistant", "content": "A remembered fact about blue."},
        {"role": "user", "content": "What was blue?"},
    ]

    class LmTokenizer:
        """Map characters into the tiny vocabulary."""

        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [((ord(ch) % 60) + 1) for ch in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(32 + ((int(i) - 1) % 60)) for i in ids)

    output = model.generate_from_messages(messages, LmTokenizer(), max_new_tokens=2)
    assert output.shape[-1] <= 64
    assert output.shape[-1] > 2


def test_custom_builder_receives_kv_heads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def make_config(**kwargs):
        captured.update(kwargs)
        return _tiny_config(
            vocab_size=kwargs["vocab_size"],
            hidden_size=kwargs["hidden_size"],
            num_hidden_layers=kwargs["num_hidden_layers"],
            num_attention_heads=kwargs["num_attention_heads"],
            num_key_value_heads=kwargs["num_key_value_heads"],
            intermediate_size=kwargs["intermediate_size"],
            max_position_embeddings=kwargs["max_position_embeddings"],
        )

    trainer = TeapotTrainer(
        TrainingConfig(
            model=ModelConfig(
                model_type="custom",
                hidden_size=32,
                layers=1,
                heads=4,
                kv_heads=2,
                intermediate=64,
                seq_len=16,
                custom_config_cls=make_config,
                custom_model_cls=TeapotForCausalLM,
            ),
            pretrain_datasets=[],
            sft_datasets=[],
            eval_datasets=[],
        )
    )
    trainer._build_model(vocab_size=50)
    assert captured["num_key_value_heads"] == 2
    assert captured["vocab_size"] == 50
    assert captured["max_position_embeddings"] == 16


def test_pretrain_effort_is_one_and_sft_prefers_one():
    from train import set_reasoning_effort

    class Probe:
        def __init__(self):
            self.config = type("Config", (), {"reasoning_effort": 0})()

    def draw(phase: str) -> int:
        probe = Probe()
        set_reasoning_effort(probe, phase)
        return probe.config.reasoning_effort

    assert draw("pretrain") == 1
    assert draw("eval") == 1
    bridge = [draw("bridge") for _ in range(400)]
    assert set(bridge) <= {1, 2, 3, 4}
    assert min(bridge) == 1 and max(bridge) == 4
    draws = [draw("sft") for _ in range(6000)]
    copies = [draw("pretrain_sft") for _ in range(200)]
    assert set(draws) <= set(range(1, 11))
    assert set(copies) <= set(range(1, 11))
    assert abs(draws.count(1) / len(draws) - 0.5) < 0.05
    assert draws.count(1) > draws.count(2) * 6


def test_before_batch_edits_any_config_for_one_batch_only():
    from transformers import Qwen2Config

    seen = {}

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

    def set_width(model, phase):
        assert phase == "sft"
        model.config.hidden_size = 48
        model.config.rope_parameters["rope_theta"] = 1_000.0

    def set_effort(model, phase):
        model.config.reasoning_effort = 4

    class Model:
        def __init__(self, config):
            self.config = config

        def __call__(self, **kwargs):
            seen["width"] = self.config.hidden_size
            seen["rope"] = self.config.rope_parameters["rope_theta"]
            seen["effort"] = self.config.reasoning_effort
            raise RuntimeError("stop after the hook")

    configs = [
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=8,
            rope_theta=500_000.0,
        ),
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=8,
            rope_theta=500_000.0,
        ),
    ]
    for config in configs:
        seen.clear()
        model = Model(config)
        trainer = TeapotTrainer(
            TrainingConfig(
                tokenizer="hook-probe",
                compile=False,
                before_batch=[set_width, set_effort],
            )
        )
        try:
            try:
                trainer._forward_backward(
                    model,
                    [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}],
                    Tok(),
                    "sft",
                )
            except RuntimeError as exc:
                assert "stop after the hook" in str(exc)
            assert seen["width"] == 48
            assert seen["rope"] == 1_000.0
            assert seen["effort"] == 4
            assert trainer._batch_config_metrics["hidden_size"] == 48.0
            assert trainer._batch_config_metrics["reasoning_effort"] == 4.0
            assert model.config.hidden_size == 32
            assert model.config.rope_parameters["rope_theta"] == 500_000.0
            assert "reasoning_effort" not in model.config.__dict__
        finally:
            trainer.writer.close()


def test_extra_reasoning_effort_trains_the_same_weights():
    torch.manual_seed(0)
    model = TeapotForCausalLM(_tiny_config(max_position_embeddings=32, pack_context_length=32))
    model.train()
    ids = torch.randint(1, 30, (2, 8))
    one = model(input_ids=ids, labels=ids, reasoning_effort=1)
    many = model(input_ids=ids, labels=ids, reasoning_effort=3)
    assert one.loss.shape == many.loss.shape
    many.loss.backward()
    grad = model.model.layers[0].self_attn.q_proj.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad) > 0
    assert model.config.reasoning_effort == 1


def test_plateau_stops_only_after_a_flat_window():
    previous = os.getcwd()
    workspace = tempfile.mkdtemp()
    try:
        os.chdir(workspace)
        trainer = TeapotTrainer(
            TrainingConfig(
                tokenizer="plateau-probe",
                compile=False,
                pretrain_datasets=[],
                plateau_window_tokens=20,
                plateau_min_loss_drop=0.02,
            )
        )
        assert trainer._plateau_reached(10, {"fineweb-edu/train/loss": 3.0}) is False
        assert trainer._plateau_reached(30, {"fineweb-edu/train/loss": 2.90}) is False
        assert trainer._plateau_reached(50, {"fineweb-edu/train/loss": 2.89}) is True
        trainer.writer.close()
    finally:
        os.chdir(previous)


def test_llama_checkpoint_seeds_the_teapot_weights():
    previous = os.getcwd()
    workspace = tempfile.mkdtemp()
    try:
        os.chdir(workspace)

        def make_config(**kwargs):
            return _tiny_config(
                vocab_size=kwargs["vocab_size"],
                hidden_size=kwargs["hidden_size"],
                num_hidden_layers=kwargs["num_hidden_layers"],
                num_attention_heads=kwargs["num_attention_heads"],
                num_key_value_heads=kwargs["num_key_value_heads"],
                intermediate_size=kwargs["intermediate_size"],
                max_position_embeddings=kwargs["max_position_embeddings"],
            )

        llama = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
                max_position_embeddings=64,
            )
        )
        with torch.no_grad():
            llama.model.embed_tokens.weight.fill_(0.25)
        llama.save_pretrained(os.path.join(workspace, "llama"))
        trainer = TeapotTrainer(
            TrainingConfig(
                tokenizer="seed-probe",
                compile=False,
                pretrain_datasets=[],
                model=ModelConfig(
                    model_type="custom",
                    hidden_size=32,
                    layers=1,
                    heads=4,
                    kv_heads=2,
                    intermediate=64,
                    seq_len=64,
                    custom_config_cls=make_config,
                    custom_model_cls=TeapotForCausalLM,
                ),
            )
        )

        class Tok:
            def __len__(self):
                return 64

        model = trainer._load_model_weights(os.path.join(workspace, "llama"), Tok())
        assert isinstance(model, TeapotForCausalLM)
        assert torch.allclose(model.model.embed_tokens.weight, torch.full_like(model.model.embed_tokens.weight, 0.25))
        assert any(name.startswith("embedder.") for name, _ in model.named_parameters())
        trainer.writer.close()
    finally:
        os.chdir(previous)
