"""Dataset sources.

A source turns text or chat into fixed-length batches. The trainer only
needs ``build(tokenizer, seq_len, token_budget)`` and a ``weight``.

Pretrain tokens can stay on disk (:class:`MemmapTokens`). Chat, tool, and
RAG rows become padded sequences whose prompt labels are ``-100``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


IGNORE = -100


@dataclass
class LabeledDoc:
    ids: list[int]
    labels: list[int]


def chat_turn(user: str, assistant: str, system: str | None = None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return messages


def tool_call_text(name: str, arguments: dict | None = None) -> str:
    payload = {"name": name, "arguments": arguments or {}}
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"<tool_call>{blob}</tool_call>"


def tool_messages(
    user: str,
    tools: list[dict],
    call: dict | None,
    system: str | None = None,
    abstain: str = "I can answer without using a tool.",
) -> list[dict]:
    """One user turn listing the tools, and one assistant turn.

    ``call`` is ``{"name", "arguments"}`` or ``None`` when the assistant
    should refuse to call anything.
    """
    listing = "\n".join(f"- {tool['name']}: {tool.get('description', '')}" for tool in tools)
    prompt = f"{user}\n\nAvailable tools:\n{listing}"
    if call is None:
        answer = abstain
    else:
        answer = tool_call_text(call["name"], call.get("arguments") or {})
    return chat_turn(prompt, answer, system)


def rag_messages(
    passages: list[str],
    question: str,
    answer: str | None,
    system: str | None = None,
    abstain: str = "The context does not contain the answer.",
) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] {passage}" for i, passage in enumerate(passages))
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return chat_turn(prompt, answer if answer else abstain, system)


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Text the model should see before it starts the assistant turn."""
    template = getattr(tokenizer, "chat_template", None)
    if template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = [f"<|{msg['role']}|>\n{msg['content']}\n" for msg in messages]
    return "".join(parts) + "<|assistant|>\n"


def _plain(messages: list[dict]) -> tuple[str, str]:
    head = "".join(
        f"<|{msg['role']}|>\n{msg['content']}\n" for msg in messages[:-1]
    )
    prompt = head + "<|assistant|>\n"
    return prompt, prompt + messages[-1]["content"]


def _find_subseq(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle)
    for start in range(last + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def messages_to_doc(tokenizer, messages: list[dict], seq_len: int) -> LabeledDoc | None:
    """Tokenize a chat, masking every token up to the assistant answer."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("the last message must be the assistant turn")
    template = getattr(tokenizer, "chat_template", None)
    if template and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        prompt, full = _plain(messages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        labels = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    else:
        answer_ids = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        start = _find_subseq(full_ids, answer_ids)
        if start is None:
            return None
        labels = [IGNORE] * len(full_ids)
        for index in range(start, len(full_ids)):
            labels[index] = full_ids[index]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (not full_ids or full_ids[-1] != eos):
        full_ids = full_ids + [eos]
        labels = labels + [eos]
    if len(full_ids) > seq_len or len(full_ids) < 2:
        return None
    if all(label == IGNORE for label in labels[1:]):
        return None
    return LabeledDoc(full_ids, labels)


def _pad_rows(
    docs: list[LabeledDoc], seq_len: int, pad_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ms = [], [], []
    for doc in docs:
        pad = seq_len - len(doc.ids)
        ids = doc.ids + [pad_id] * pad
        labels = doc.labels + [IGNORE] * pad
        mask = [1] * len(doc.ids) + [0] * pad
        xs.append(ids)
        ys.append(labels)
        ms.append(mask)
    return (
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(ms, dtype=np.int32),
    )


def _pack_rows(
    docs: list[LabeledDoc], seq_len: int, eos_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stream_x: list[int] = []
    stream_y: list[int] = []
    for doc in docs:
        stream_x.extend(doc.ids)
        stream_y.extend(doc.labels)
        if not doc.ids or doc.ids[-1] != eos_id:
            stream_x.append(eos_id)
            stream_y.append(eos_id)
    usable = (len(stream_x) // seq_len) * seq_len
    if usable < seq_len:
        raise RuntimeError(
            f"not enough tokens to fill one sequence of length {seq_len}"
        )
    x = np.asarray(stream_x[:usable], dtype=np.int32).reshape(-1, seq_len)
    y = np.asarray(stream_y[:usable], dtype=np.int32).reshape(-1, seq_len)
    mask = np.ones_like(x, dtype=np.int32)
    return x, y, mask


class ExampleBatcher:
    """Shuffled, repeating view over padded or packed rows."""

    def __init__(
        self,
        name: str,
        input_ids: np.ndarray,
        labels: np.ndarray,
        attention_mask: np.ndarray,
        seed: int,
    ):
        if len(input_ids) == 0:
            raise RuntimeError(f"{name}: produced no training sequences")
        self.name = name
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(input_ids))
        self.cursor = 0

    def describe(self) -> str:
        supervised = int((self.labels != IGNORE).sum())
        return (
            f"{self.name}: {len(self.input_ids)} sequences, "
            f"{supervised:,} supervised tokens"
        )

    def take(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        picks = []
        for _ in range(n):
            if self.cursor >= len(self.order):
                self.order = self.rng.permutation(len(self.input_ids))
                self.cursor = 0
            picks.append(int(self.order[self.cursor]))
            self.cursor += 1
        index = np.asarray(picks, dtype=np.int64)
        return self.input_ids[index], self.labels[index], self.attention_mask[index]

    def state_dict(self) -> dict:
        return {
            "cursor": self.cursor,
            "order": self.order.astype(np.int64).tolist(),
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.cursor = int(state["cursor"])
        self.order = np.asarray(state["order"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng"]


class MemmapWindowBatcher:
    """Random windows from a flat int32 token file. No padding."""

    def __init__(self, name: str, path: str | Path, seq_len: int, seed: int):
        self.name = name
        self.path = str(path)
        self.seq_len = seq_len
        self.arr = np.memmap(self.path, dtype=np.int32, mode="r")
        self.n = len(self.arr) // seq_len
        if self.n <= 0:
            raise RuntimeError(f"{path} has no full sequences at seq_len={seq_len}")
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.n)
        self.pos = 0

    def describe(self) -> str:
        return f"{self.name}: {self.n:,} windows of {self.seq_len} from {self.path}"

    def _window(self, which: int) -> np.ndarray:
        start = int(which) * self.seq_len
        return np.asarray(self.arr[start : start + self.seq_len], dtype=np.int32)

    def take(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.empty((n, self.seq_len), dtype=np.int32)
        for i in range(n):
            if self.pos >= len(self.perm):
                self.perm = self.rng.permutation(self.n)
                self.pos = 0
            rows[i] = self._window(self.perm[self.pos])
            self.pos += 1
        mask = np.ones_like(rows, dtype=np.int32)
        return rows, rows.copy(), mask

    def state_dict(self) -> dict:
        return {
            "pos": self.pos,
            "perm": self.perm.astype(np.int64),
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.pos = int(state["pos"])
        self.perm = np.asarray(state["perm"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng"]


def _messages_batcher(name: str, rows: Iterable[list[dict]], tokenizer, seq_len: int, seed: int):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    docs = []
    dropped = 0
    for messages in rows:
        try:
            doc = messages_to_doc(tokenizer, messages, seq_len)
        except ValueError:
            dropped += 1
            continue
        if doc is None:
            dropped += 1
            continue
        docs.append(doc)
    if not docs:
        raise RuntimeError(f"{name}: every example was longer than seq_len={seq_len} or empty")
    x, y, mask = _pad_rows(docs, seq_len, int(pad_id))
    batcher = ExampleBatcher(name, x, y, mask, seed)
    batcher.dropped = dropped  # type: ignore[attr-defined]
    return batcher


@dataclass
class MemmapTokens:
    """Flat int32 token stream, already tokenized with ``tokenizer``."""

    path: str
    name: str = "memmap"
    weight: float = 1.0

    def build(self, tokenizer, seq_len: int, token_budget: int, seed: int = 0):
        del tokenizer, token_budget
        return MemmapWindowBatcher(self.name, self.path, seq_len, seed)


@dataclass
class ListMessages:
    """In-memory chat rows. Each row is a list of ``{role, content}``."""

    rows: list
    name: str = "messages"
    weight: float = 1.0

    def build(self, tokenizer, seq_len: int, token_budget: int, seed: int = 0):
        del token_budget
        return _messages_batcher(self.name, self.rows, tokenizer, seq_len, seed)


@dataclass
class JsonlMessages:
    """JSONL file of ``{"messages": [...]}`` or a bare message list."""

    path: str
    name: str = ""
    weight: float = 1.0

    def build(self, tokenizer, seq_len: int, token_budget: int, seed: int = 0):
        del token_budget
        rows = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(obj["messages"] if isinstance(obj, dict) and "messages" in obj else obj)
        return _messages_batcher(self.name or Path(self.path).stem, rows, tokenizer, seq_len, seed)


@dataclass
class ListText:
    """Raw strings, packed to ``seq_len`` with the eos token between docs."""

    texts: list[str]
    name: str = "text"
    weight: float = 1.0

    def build(self, tokenizer, seq_len: int, token_budget: int, seed: int = 0):
        del token_budget
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer needs an eos token to pack text")
        docs = []
        for text in self.texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                docs.append(LabeledDoc(ids, list(ids)))
        x, y, mask = _pack_rows(docs, seq_len, int(eos))
        return ExampleBatcher(self.name, x, y, mask, seed)


def _iter_hf(dataset: str, subset: str | None, split: str, streaming: bool):
    from datasets import load_dataset

    if subset:
        return load_dataset(dataset, subset, split=split, streaming=streaming)
    return load_dataset(dataset, split=split, streaming=streaming)


@dataclass
class HuggingFaceText:
    """Stream a text column and pack it. Materializes at most ``cache_tokens``."""

    dataset: str
    name: str = ""
    subset: str | None = None
    split: str = "train"
    text_column: str = "text"
    weight: float = 1.0
    cache_tokens: int | None = None
    formatter: Callable[[dict], str] | None = None

    def build(self, tokenizer, seq_len: int, token_budget: int, seed: int = 0):
        cap = self.cache_tokens or token_budget
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer needs an eos token to pack text")
        docs: list[LabeledDoc] = []
        seen = 0
        for row in _iter_hf(self.dataset, self.subset, self.split, streaming=True):
            text = self.formatter(row) if self.formatter else row.get(self.text_column, "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            docs.append(LabeledDoc(ids, list(ids)))
            seen += len(ids)
            if seen >= cap:
                break
        x, y, mask = _pack_rows(docs, seq_len, int(eos))
        return ExampleBatcher(self.name or self.dataset.split("/")[-1], x, y, mask, seed)


@dataclass
class HuggingFaceSFT:
    """Stream rows and turn each into chat messages with ``formatter``."""

    dataset: str
    formatter: Callable[[dict], list[dict]]
    name: str = ""
    subset: str | None = None
    split: str = "train"
    weight: float = 1.0
    max_rows: int | None = None

    def build(self, tokenizer, seq_len: int, token_budget: int, seed: int = 0):
        del token_budget
        rows = []
        for i, row in enumerate(_iter_hf(self.dataset, self.subset, self.split, streaming=True)):
            if self.max_rows is not None and i >= self.max_rows:
                break
            messages = self.formatter(row)
            if messages:
                rows.append(messages)
        return _messages_batcher(self.name or self.dataset.split("/")[-1], rows, tokenizer, seq_len, seed)


def open_token_memmap(path: str | Path, max_tokens: int | None = None) -> np.ndarray:
    arr = np.memmap(path, dtype=np.int32, mode="r")
    if max_tokens is None or max_tokens >= len(arr):
        return arr
    return arr[:max_tokens]
