"""Chat formats for supervised data.

Pretrain rows stay raw text. A format is applied only when a source is
supervised. The same Jinja template is written onto the tokenizer saved with
a supervised checkpoint, so ``pipeline("text-generation")(messages)`` builds
the prompt the model was trained on.

``chatml`` matches SmolLM-Instruct. ``smollm`` matches SmolLM2-Instruct,
including that model's default system line.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from transformers.utils.chat_template_utils import render_jinja_template

# Names that share a template. The stored config value is the canonical name.
_ALIASES = {
    "qwen": "chatml",
    "qwen2": "chatml",
    "openai": "chatml",
    "smollm-instruct": "chatml",
    "smollm2": "smollm",
    "smollm2-instruct": "smollm",
    "llama-2": "llama2",
    "llama-3": "llama3",
    "llama3.1": "llama3",
    "llama-3.1": "llama3",
    "phi3": "phi",
    "phi-3": "phi",
    "gemma2": "gemma",
    "mistral-instruct": "mistral",
}


@dataclass(frozen=True)
class ChatFormat:
    """One way to turn role messages into the string the model sees."""

    name: str
    template: str
    # Appended after a supervised answer when this string is already one vocab token.
    stop_token: str | None = None
    description: str = ""

    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        rendered, _ = render_jinja_template(
            conversations=[messages],
            chat_template=self.template,
            add_generation_prompt=add_generation_prompt,
        )
        return rendered[0]


# The historical System / User / Assistant text. Default, so existing runs stay put.
_PLAIN = (
    "{% for message in messages %}"
    "{% if not loop.first %}{{ '\n\n' }}{% endif %}"
    "{% if message['role'] == 'system' %}System: {{ message['content'] }}"
    "{% elif message['role'] == 'user' %}User: {{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\n\nAssistant:\n' }}{% endif %}"
)

# SmolLM-135M-Instruct. No injected system line. Stop token is already in SmolLM2.
_CHATML = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

# SmolLM2-135M-Instruct, including the default system line when the first
# message is not a system message.
_SMOLLM = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

_LLAMA2 = (
    "{% set ns = namespace(system='') %}"
    "{% if messages and messages[0]['role'] == 'system' %}{% set ns.system = messages[0]['content'] %}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{% if ns.system %}"
    "{{ '<s>[INST] ' + ns.system + '\n\n' + message['content'] + ' [/INST]' }}"
    "{% set ns.system = '' %}"
    "{% else %}"
    "{{ '<s>[INST] ' + message['content'] + ' [/INST]' }}"
    "{% endif %}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content'] + '</s>' }}"
    "{% endif %}"
    "{% endfor %}"
)

_LLAMA3 = (
    "{{ '<|begin_of_text|>' }}"
    "{% for message in messages %}"
    "{% if message['role'] in ['system', 'user', 'assistant'] %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content']|trim + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

_GEMMA = (
    "{{ '<bos>' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<start_of_turn>user\n' + message['content']|trim + '<end_of_turn>\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\n' + message['content']|trim + '<end_of_turn>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\n' + message['content']|trim + '<end_of_turn>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)

_MISTRAL = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<s>[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + '</s>' }}"
    "{% endif %}"
    "{% endfor %}"
)

_PHI = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '<|end|>\n' }}"
    "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)

_ZEPHYR = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '</s>\n' }}"
    "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '</s>\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '</s>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)

_ALPACA = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ '### Instruction:\n' + message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '### Response:\n' + message['content'] + '\n\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '### Response:\n' }}{% endif %}"
)

_VICUNA = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"
)


CHAT_FORMATS: dict[str, ChatFormat] = {
    "plain": ChatFormat(
        "plain",
        _PLAIN,
        description="System / User / Assistant text. This is the default.",
    ),
    "chatml": ChatFormat(
        "chatml",
        _CHATML,
        stop_token="<|im_end|>",
        description="ChatML, the SmolLM-Instruct template. SmolLM2 already has these tokens.",
    ),
    "smollm": ChatFormat(
        "smollm",
        _SMOLLM,
        stop_token="<|im_end|>",
        description="SmolLM2-Instruct, including its default system line.",
    ),
    "llama2": ChatFormat(
        "llama2",
        _LLAMA2,
        stop_token="</s>",
        description="Llama 2 [INST] blocks.",
    ),
    "llama3": ChatFormat(
        "llama3",
        _LLAMA3,
        stop_token="<|eot_id|>",
        description="Llama 3 headers.",
    ),
    "gemma": ChatFormat(
        "gemma",
        _GEMMA,
        stop_token="<end_of_turn>",
        description="Gemma turns. Assistant messages use the model role.",
    ),
    "mistral": ChatFormat(
        "mistral",
        _MISTRAL,
        stop_token="</s>",
        description="Mistral-style [INST] blocks.",
    ),
    "phi": ChatFormat(
        "phi",
        _PHI,
        stop_token="<|end|>",
        description="Phi-style <|user|> and <|assistant|> turns.",
    ),
    "zephyr": ChatFormat(
        "zephyr",
        _ZEPHYR,
        stop_token="</s>",
        description="Zephyr-style <|user|> and <|assistant|> turns.",
    ),
    "alpaca": ChatFormat(
        "alpaca",
        _ALPACA,
        description="### Instruction / ### Response.",
    ),
    "vicuna": ChatFormat(
        "vicuna",
        _VICUNA,
        stop_token="</s>",
        description="USER / ASSISTANT.",
    ),
}


_ACTIVE: ContextVar[str] = ContextVar("teapot_chat_format", default="plain")


def chat_format_names() -> list[str]:
    return list(CHAT_FORMATS)


def get_chat_format(name: str | None) -> ChatFormat:
    key = (name or "plain").strip().lower()
    key = _ALIASES.get(key, key)
    fmt = CHAT_FORMATS.get(key)
    if fmt is None:
        known = ", ".join(chat_format_names())
        raise ValueError(f"Unknown chat_format {name!r}. Choose one of: {known}.")
    return fmt


def current_chat_format() -> ChatFormat:
    return get_chat_format(_ACTIVE.get())


def push_chat_format(name: str | None):
    """Apply ``name`` to supervised rendering until :func:`pop_chat_format`."""
    return _ACTIVE.set(get_chat_format(name).name)


def pop_chat_format(token) -> None:
    _ACTIVE.reset(token)


@contextmanager
def use_chat_format(name: str | None):
    token = push_chat_format(name)
    try:
        yield current_chat_format()
    finally:
        pop_chat_format(token)


def single_token_id(tokenizer, text: str | None) -> int | None:
    """Return the id of ``text`` when the tokenizer stores it as one token."""
    if not text or tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    token_id = tokenizer.convert_tokens_to_ids(text)
    if not isinstance(token_id, int):
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and token_id == unk:
        return None
    if hasattr(tokenizer, "convert_ids_to_tokens") and tokenizer.convert_ids_to_tokens(token_id) != text:
        return None
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        if vocab.get(text) != token_id:
            return None
    return token_id


def supervised_end(tokenizer) -> str | None:
    """Stop string for a supervised answer, or None to keep the tokenizer end token.

    Used for supervised examples only. Pretrain packing does not call this.
    The stop string is used when it is already a single vocabulary token, so
    the saved checkpoint can use it as the generation end token.
    """
    fmt = current_chat_format()
    if not fmt.stop_token:
        return None
    if single_token_id(tokenizer, fmt.stop_token) is None:
        return None
    return fmt.stop_token


def extra_stop_ids(tokenizer) -> list[int]:
    """Generation stop ids for the active format, excluding the tokenizer end token."""
    fmt = current_chat_format()
    token_id = single_token_id(tokenizer, fmt.stop_token)
    if token_id is None:
        return []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if token_id == eos_id:
        return []
    return [token_id]
