"""Build a Transformers causal LM from a :class:`ModelSpec`."""
from __future__ import annotations

from typing import Callable

import torch
from transformers import PreTrainedModel

from .config import ModelSpec


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _check_divisible(spec: ModelSpec) -> None:
    heads = spec.num_attention_heads
    if spec.hidden_size % heads != 0:
        raise ValueError(
            f"hidden_size {spec.hidden_size} is not divisible by "
            f"num_attention_heads {heads}"
        )
    kv = spec.num_key_value_heads
    if heads % kv != 0:
        raise ValueError(
            f"num_attention_heads {heads} is not divisible by "
            f"num_key_value_heads {kv}"
        )


def build_model(spec: ModelSpec, vocab_size: int) -> PreTrainedModel:
    """Return an uninitialized causal LM on CPU in fp32."""
    arch = spec.architecture.lower()
    if spec.custom_builder is not None:
        builder: Callable = spec.custom_builder
        model = builder(spec, vocab_size)
        if not isinstance(model, PreTrainedModel):
            raise TypeError("custom_builder must return a transformers PreTrainedModel")
        return model

    if arch == "custom":
        if spec.custom_config_cls is None or spec.custom_model_cls is None:
            raise ValueError(
                "architecture='custom' needs custom_builder or both "
                "custom_config_cls and custom_model_cls"
            )
        cfg = spec.custom_config_cls(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            intermediate_size=spec.intermediate_size,
            max_position_embeddings=spec.max_position_embeddings,
            tie_word_embeddings=spec.tie_word_embeddings,
        )
        return spec.custom_model_cls(cfg)

    if arch == "llama":
        _check_divisible(spec)
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            max_position_embeddings=spec.max_position_embeddings,
            rms_norm_eps=spec.rms_norm_eps,
            rope_theta=spec.rope_theta,
            tie_word_embeddings=spec.tie_word_embeddings,
            use_cache=False,
            attention_bias=False,
            hidden_act="silu",
        )
        return LlamaForCausalLM(cfg)

    if arch == "mistral":
        _check_divisible(spec)
        from transformers import MistralConfig, MistralForCausalLM

        cfg = MistralConfig(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            max_position_embeddings=spec.max_position_embeddings,
            rms_norm_eps=spec.rms_norm_eps,
            rope_theta=spec.rope_theta,
            tie_word_embeddings=spec.tie_word_embeddings,
            use_cache=False,
        )
        return MistralForCausalLM(cfg)

    if arch == "gpt2":
        if spec.hidden_size % spec.num_attention_heads != 0:
            raise ValueError("gpt2 hidden_size must be divisible by num_attention_heads")
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=spec.hidden_size,
            n_layer=spec.num_hidden_layers,
            n_head=spec.num_attention_heads,
            n_inner=spec.intermediate_size,
            n_positions=spec.max_position_embeddings,
            n_ctx=spec.max_position_embeddings,
            use_cache=False,
        )
        return GPT2LMHeadModel(cfg)

    raise ValueError(
        f"unsupported architecture {spec.architecture!r}; "
        "use llama, gpt2, mistral, or custom"
    )


def enable_checkpointing(model: PreTrainedModel) -> bool:
    fn = getattr(model, "gradient_checkpointing_enable", None)
    if fn is None:
        return False
    try:
        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        fn()
    return True
