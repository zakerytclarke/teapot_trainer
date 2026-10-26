"""Dataset presets and the streaming encoder.

New presets are additional constants. ``chat_template_fn`` may return one
``(prompt, answer)`` pair or a list of pairs (one per assistant turn).
``token_loss_weight_fn`` is optional and, when set, returns one loss weight
per label. The trainer applies those weights; it does not decide them.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field
from datasets import load_dataset

from .formatting import (
    as_pairs,
    format_apigen_aliased,
    format_apigen_decision,
    format_gsm8k,
    format_hotpot,
    format_squad,
    format_squad_answerable,
    format_squad_unanswerable,
    format_synthqa,
    format_synthqa_answerable,
    format_synthqa_unanswerable,
    format_teapot_chat,
    format_when2call,
    messages_format,
)


class DatasetSourceConfig(BaseModel):
    """Configuration for an individual dataset source in a mix."""

    name: str
    subset: Optional[str] = None
    split: str = "train"
    target_tokens: int = 100_000_000
    weight: float = 1.0
    text_column: str = "text"
    is_sft: bool = False

    preprocess_fn: Optional[Callable[[dict], dict]] = Field(default=None, exclude=True)
    chat_template_fn: Optional[Callable[[dict], tuple[str, str]]] = Field(default=None, exclude=True)
    # Maps an encoded example to one weight per label. None means every token counts as 1.
    token_loss_weight_fn: Optional[Callable[[dict, Any], list[float]]] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# End-of-answer token, relative to a normal supervised token.
STOP_TOKEN_LOSS_WEIGHT = 8.0


def eos_loss_weight(weight: float = STOP_TOKEN_LOSS_WEIGHT) -> Callable[[dict, Any], list[float]]:
    """Upweight the tokenizer end token. Every other label stays at 1."""

    def apply(example: dict, tokenizer) -> list[float]:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        return [
            float(weight) if eos_id is not None and label == eos_id else 1.0
            for label in example["labels"]
        ]

    return apply


_STOP_WEIGHT = eos_loss_weight(STOP_TOKEN_LOSS_WEIGHT)


FINEWEB_EDU = DatasetSourceConfig(
    name="HuggingFaceFW/fineweb-edu",
    split="train",
    is_sft=False,
    text_column="text",
)

SMOLTALK = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="all",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)

SMOLTALK_MAGPIE = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="smol-magpie-ultra",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_EVERYDAY = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="everyday-conversations",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_OPENHERMES = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="openhermes-100k",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_CONSTRAINTS = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="smol-constraints",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_SYSTEMCHATS = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="systemchats-30k",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_SUMMARIZE = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="smol-summarize",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_REWRITE = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="smol-rewrite",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_APIGEN = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="apigen-80k",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_APIGEN_DECISION = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="apigen-80k",
    split="train",
    is_sft=True,
    chat_template_fn=format_apigen_decision,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_APIGEN_ALIASED = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="apigen-80k",
    split="train",
    is_sft=True,
    chat_template_fn=format_apigen_aliased,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SMOLTALK_METAMATH = DatasetSourceConfig(
    name="HuggingFaceTB/smoltalk",
    subset="metamathqa-50k",
    split="train",
    is_sft=True,
    chat_template_fn=messages_format,
    token_loss_weight_fn=_STOP_WEIGHT,
)

SMOLTALK_MAGPIE_TEST = SMOLTALK_MAGPIE.model_copy(update={"split": "test"})
SMOLTALK_CONSTRAINTS_TEST = SMOLTALK_CONSTRAINTS.model_copy(update={"split": "test"})
SMOLTALK_APIGEN_TEST = SMOLTALK_APIGEN.model_copy(update={"split": "test"})
SMOLTALK_APIGEN_DECISION_TEST = SMOLTALK_APIGEN_DECISION.model_copy(update={"split": "test"})
SMOLTALK_APIGEN_ALIASED_TEST = SMOLTALK_APIGEN_ALIASED.model_copy(update={"split": "test"})

GSM8K = DatasetSourceConfig(
    name="gsm8k",
    subset="main",
    split="train",
    is_sft=True,
    chat_template_fn=format_gsm8k,
    token_loss_weight_fn=_STOP_WEIGHT,
)
GSM8K_TEST = GSM8K.model_copy(update={"split": "test"})

WHEN2CALL = DatasetSourceConfig(
    name="nvidia/When2Call",
    subset="train_sft",
    split="train",
    is_sft=True,
    chat_template_fn=format_when2call,
    token_loss_weight_fn=_STOP_WEIGHT,
)
WHEN2CALL_TEST = DatasetSourceConfig(
    name="nvidia/When2Call",
    subset="test",
    split="mcq",
    is_sft=True,
    chat_template_fn=format_when2call,
    token_loss_weight_fn=_STOP_WEIGHT,
)

SQUAD_V2 = DatasetSourceConfig(
    name="rajpurkar/squad_v2",
    split="train",
    is_sft=True,
    chat_template_fn=format_squad,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SQUAD_V2_VAL = SQUAD_V2.model_copy(update={"split": "validation"})
SQUAD_V2_ANSWERABLE = SQUAD_V2.model_copy(update={"chat_template_fn": format_squad_answerable})
SQUAD_V2_UNANSWERABLE = SQUAD_V2.model_copy(update={"chat_template_fn": format_squad_unanswerable})

HOTPOT = DatasetSourceConfig(
    name="hotpot_qa",
    subset="distractor",
    split="train",
    is_sft=True,
    chat_template_fn=format_hotpot,
    token_loss_weight_fn=_STOP_WEIGHT,
)
HOTPOT_VAL = HOTPOT.model_copy(update={"split": "validation"})

SYNTHQA = DatasetSourceConfig(
    name="teapotai/synthqa",
    split="train",
    is_sft=True,
    chat_template_fn=format_synthqa,
    token_loss_weight_fn=_STOP_WEIGHT,
)
SYNTHQA_TEST = SYNTHQA.model_copy(update={"split": "test"})
SYNTHQA_ANSWERABLE = SYNTHQA.model_copy(update={"chat_template_fn": format_synthqa_answerable})
SYNTHQA_UNANSWERABLE = SYNTHQA.model_copy(update={"chat_template_fn": format_synthqa_unanswerable})

TEAPOT_CHAT = DatasetSourceConfig(
    name="teapotai/teapot-chat",
    split="train",
    is_sft=True,
    chat_template_fn=format_teapot_chat,
    token_loss_weight_fn=_STOP_WEIGHT,
)


def open_dataset(src: DatasetSourceConfig, streaming: bool):
    kwargs = {"split": src.split, "streaming": streaming}
    try:
        if src.subset:
            return load_dataset(src.name, src.subset, **kwargs)
        return load_dataset(src.name, **kwargs)
    except Exception:
        if src.subset:
            return load_dataset(src.name, **kwargs)
        raise


def holdout_membership(key: str, ratio: float, seed: int) -> bool:
    """True when this example belongs to the train-split holdout."""
    if ratio <= 0:
        return False
    digest = hashlib.md5(f"{seed}:{key[:500]}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return bucket < ratio


def encode_pair(tokenizer, prompt: str, answer: str, seq_len: int) -> dict | None:
    """Mask the prompt. Left-truncate so the supervised answer still fits."""
    answer = (answer or "").strip()
    prompt = prompt or ""
    if not answer or not prompt.strip():
        return None
    if not prompt.endswith("\n"):
        prompt += "\n"
    eos = tokenizer.eos_token or ""
    full_ids = tokenizer.encode(prompt + answer + eos, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        answer_ids = tokenizer.encode(answer + eos, add_special_tokens=False)
        full_ids = prompt_ids + answer_ids
    else:
        answer_ids = full_ids[len(prompt_ids) :]
    if not answer_ids:
        return None
    overflow = len(full_ids) - seq_len
    if overflow > 0:
        if overflow >= len(prompt_ids):
            full_ids = answer_ids[-seq_len:]
            labels = list(full_ids)
        else:
            prompt_ids = prompt_ids[overflow:]
            full_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + list(answer_ids)
    else:
        labels = [-100] * (len(full_ids) - len(answer_ids)) + list(answer_ids)
    if len(labels) != len(full_ids) or all(token == -100 for token in labels):
        return None
    return {"input_ids": list(full_ids), "labels": list(labels)}


def row_text(src: DatasetSourceConfig, row: dict) -> str:
    if src.preprocess_fn:
        row = src.preprocess_fn(row)
    value = row.get(src.text_column, row.get("text", row.get("content", "")))
    return value if isinstance(value, str) else ""


def row_pairs(src: DatasetSourceConfig, row: dict) -> list[tuple[str, str]]:
    if src.preprocess_fn:
        row = src.preprocess_fn(row)
    if not src.chat_template_fn:
        return []
    return as_pairs(src.chat_template_fn(row))


def iterate_encoded(
    src: DatasetSourceConfig,
    tokenizer,
    seq_len: int,
    *,
    is_eval: bool = False,
    holdout_ratio: float = 0.05,
    seed: int = 42,
    rows: list[dict] | None = None,
) -> Iterator[dict]:
    """Yield fixed trainer examples.

    Train and eval of a ``split='train'`` source share a hash holdout so the
    eval slice is not trained on. Test and validation splits are used whole.
    """

    def source_rows():
        if rows is not None:
            yield from rows
            return
        dataset = open_dataset(src, streaming=True)
        if not is_eval:
            # SQuAD v2's first ~2k rows are all answerable, so the buffer has to
            # reach past them or a short run never sees an abstention.
            buffer = 4096 if "squad" in (src.name or "") else 512
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer)
        for row in dataset:
            yield row

    use_holdout = src.split == "train" and holdout_ratio > 0

    if not src.is_sft:
        buffer: list[int] = []
        eos_id = tokenizer.eos_token_id
        for row in source_rows():
            text = row_text(src, row)
            if not text:
                continue
            if use_holdout:
                held = holdout_membership(text, holdout_ratio, seed)
                if held != is_eval:
                    continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if eos_id is not None:
                ids = ids + [eos_id]
            buffer.extend(ids)
            while len(buffer) >= seq_len:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:]
                yield {"input_ids": chunk, "labels": list(chunk)}
        return

    for row in source_rows():
        pairs = row_pairs(src, row)
        for prompt, answer in pairs:
            if use_holdout:
                held = holdout_membership(prompt + "\n" + answer, holdout_ratio, seed)
                if held != is_eval:
                    continue
            encoded = encode_pair(tokenizer, prompt, answer, seq_len)
            if encoded is None:
                continue
            if src.token_loss_weight_fn is not None:
                encoded["loss_weights"] = [float(value) for value in src.token_loss_weight_fn(encoded, tokenizer)]
            yield encoded


class DatasetRegistry:
    """Load, encode, and mix sources. One generator is bound to one source."""

    @staticmethod
    def prepare_dataset_splits(
        sources: list[Any],
        tokenizer,
        seq_len: int,
        is_eval: bool = False,
        holdout_ratio: float = 0.05,
        seed: int = 42,
    ):
        streams = []
        for src in sources:
            streams.append(
                iterate_encoded(
                    src,
                    tokenizer,
                    seq_len,
                    is_eval=is_eval,
                    holdout_ratio=holdout_ratio,
                    seed=seed,
                )
            )
        return streams
