"""Llama language model with a Teapot embedding retriever in the same checkpoint.

``build_rag_config`` is the shape ``train.py`` trains: 16 layers, hidden size
768, 12 query heads, 4 key/value heads, SwiGLU width 2048. The position table
is the retrieval pack window, 2,048 tokens. ``build_smollm_config`` is the
separate SmolLM2-135M recipe. ``teapotai/teapotembedding`` is loaded before
training and saved beside the language-model weights.

Each message is embedded once, truncated to the encoder's own context length.
Packing stays a hard choice of the messages the encoder ranks as closest to the
latest user message. ``retrieval_loss`` trains that ranking with REINFORCE:
the reward is the language-model loss of the gold reply under the sampled
message, measured against the message the current scores already rank first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, LlamaConfig
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

EMBEDDER_NAME = "teapotai/teapotembedding"
SMOLLM_VOCAB_SIZE = 49152
SMOLLM_PARAMETER_COUNT = 134_515_008


class TeapotConfig(LlamaConfig):
    """Llama config plus the Teapot embedding encoder settings."""

    model_type = "teapot"

    embedder_name: str = EMBEDDER_NAME
    embedder_config: dict | None = None
    # None uses the encoder position table. Teapot embedding's table is 512.
    embedder_max_length: int | None = None
    pack_context_length: int = 2048
    retrieval_temperature: float = 1.0
    # How many times the language-model block runs. One is a normal pass.
    reasoning_effort: int = 1

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 3
        super().__post_init__(**kwargs)


def llama_parameter_count(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
) -> int:
    """Tied-embedding Llama parameter count. The embedding matrix is counted once."""
    head_dim = hidden_size // num_attention_heads
    query = hidden_size * num_attention_heads * head_dim
    key = hidden_size * num_key_value_heads * head_dim
    output = num_attention_heads * head_dim * hidden_size
    attention = query + key + key + output
    mlp = 3 * hidden_size * intermediate_size
    per_layer = attention + mlp + (2 * hidden_size)
    return (num_hidden_layers * per_layer) + (vocab_size * hidden_size) + hidden_size


def split_conversation(messages: list[dict] | None) -> tuple[str, str, list[dict]]:
    """Return the system text, the latest user text, and earlier candidate messages.

    The latest user message is the retrieval query. System text is kept whole
    as the fixed prompt and is not a candidate. Candidates stay in conversation
    order and remember their original index.
    """
    cleaned = []
    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not role or not content:
            continue
        cleaned.append({"role": role, "content": content, "index": index})
    system = "\n".join(item["content"] for item in cleaned if item["role"] == "system")
    user_positions = [i for i, item in enumerate(cleaned) if item["role"] == "user"]
    if not user_positions:
        return system, "", []
    query_at = user_positions[-1]
    query = cleaned[query_at]["content"]
    candidates = [item for item in cleaned[:query_at] if item["role"] != "system"]
    return system, query, candidates


def role_label(role: str) -> str:
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    return labels.get(role.lower(), (role or "user").capitalize())


def render_prompt(system: str | None, retrieved: list[tuple[str, str]], user: str | None) -> str:
    """The same System / User / Assistant surface the supervised mix already trains."""
    parts = []
    if system:
        parts.append(f"System: {system}")
    for role, content in retrieved:
        parts.append(f"{role_label(role)}: {content}")
    if user:
        parts.append(f"User: {user}")
    return "\n\n".join(parts + ["Assistant:"]) + "\n"


def _encode(tokenizer, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True)


def _prompt_length(tokenizer, system: str | None, retrieved: list[tuple[str, str]], user: str | None) -> int:
    return len(_encode(tokenizer, render_prompt(system, retrieved, user)))


def _fit_fixed(tokenizer, system: str, user: str, context_length: int) -> tuple[str, str]:
    """Keep a system prefix and a user suffix inside the budget.

    The full texts are kept when they fit. Otherwise the system keeps its
    beginning and the user message keeps its end, and both stay present when
    one token of each can fit beside the assistant header.
    """
    if _prompt_length(tokenizer, system or None, [], user or None) <= context_length:
        return system, user

    system_ids = _encode(tokenizer, system)
    user_ids = _encode(tokenizer, user)

    def fits(system_keep: int, user_keep: int) -> bool:
        system_text = _decode(tokenizer, system_ids[:system_keep]).strip()
        user_text = _decode(tokenizer, user_ids[-user_keep:]).strip() if user_keep else ""
        return _prompt_length(tokenizer, system_text or None, [], user_text or None) <= context_length

    best_system = 0
    low, high = 0, len(system_ids)
    while low <= high:
        mid = (low + high) // 2
        if fits(mid, 0):
            best_system = mid
            low = mid + 1
        else:
            high = mid - 1

    best_user = 0
    low, high = 0, len(user_ids)
    while low <= high:
        mid = (low + high) // 2
        if fits(best_system, mid):
            best_user = mid
            low = mid + 1
        else:
            high = mid - 1

    if user_ids and best_user == 0:
        for system_keep in range(best_system, -1, -1):
            if not fits(system_keep, 1):
                continue
            best_system = system_keep
            best_user = 1
            low, high = 1, len(user_ids)
            while low <= high:
                mid = (low + high) // 2
                if fits(system_keep, mid):
                    best_user = mid
                    low = mid + 1
                else:
                    high = mid - 1
            break

    system_text = _decode(tokenizer, system_ids[:best_system]).strip()
    user_text = _decode(tokenizer, user_ids[-best_user:]).strip() if best_user else ""
    return system_text, user_text


def _fit_retrieved(
    tokenizer,
    system: str,
    user: str,
    candidates: list[dict],
    scores: list[float],
    context_length: int,
) -> list[dict]:
    """Add highest-scoring messages until the window is full.

    A message that does not fit whole is cut from the end of its text so it
    can still fill the remaining room. Messages are returned in conversation
    order.
    """
    ranked = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]["index"]))
    chosen: list[dict] = []

    def length_of(items: list[dict]) -> int:
        retrieved = [(item["role"], item["content"]) for item in sorted(items, key=lambda item: item["index"])]
        return _prompt_length(tokenizer, system or None, retrieved, user or None)

    for index in ranked:
        candidate = candidates[index]
        whole = {
            "index": candidate["index"],
            "role": candidate["role"],
            "content": candidate["content"],
        }
        if length_of(chosen + [whole]) <= context_length:
            chosen.append(whole)
            continue
        ids = _encode(tokenizer, candidate["content"])
        best = ""
        low, high = 1, len(ids)
        while low <= high:
            mid = (low + high) // 2
            piece = _decode(tokenizer, ids[:mid]).strip()
            if not piece:
                low = mid + 1
                continue
            trial = {
                "index": candidate["index"],
                "role": candidate["role"],
                "content": piece,
            }
            if length_of(chosen + [trial]) <= context_length:
                best = piece
                low = mid + 1
            else:
                high = mid - 1
        if best:
            chosen.append({"index": candidate["index"], "role": candidate["role"], "content": best})
            break
    chosen.sort(key=lambda item: item["index"])
    return chosen


def pack_conversation(
    messages: list[dict] | None,
    tokenizer,
    scores: list[float] | None,
    context_length: int,
) -> str:
    """Pack a chat into ``context_length`` language-model tokens.

    ``scores`` lines up with the candidate messages from ``split_conversation``.
    ``None`` scores pack the earliest messages that fit.
    """
    if context_length < 1:
        raise ValueError("context_length must be positive.")
    system, query, candidates = split_conversation(messages)
    if scores is None:
        scores = [0.0] * len(candidates)
    if len(scores) != len(candidates):
        raise ValueError(
            f"scores has length {len(scores)} and there are {len(candidates)} candidate messages."
        )
    system_text, user_text = _fit_fixed(tokenizer, system, query, context_length)
    chosen = _fit_retrieved(tokenizer, system_text, user_text, candidates, scores, context_length)
    retrieved = [(item["role"], item["content"]) for item in chosen]
    return render_prompt(system_text or None, retrieved, user_text or None)


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1e-6)
    return F.normalize(summed / counts, p=2, dim=-1)


class TeapotForCausalLM(LlamaForCausalLM):
    """Causal LM whose checkpoint also owns the Teapot embedding retriever."""

    config_class = TeapotConfig

    def __init__(self, config: TeapotConfig):
        super().__init__(config)
        self.embedder = self._build_embedder(config)
        self._embedder_tokenizer_obj = None

    def _build_embedder(self, config: TeapotConfig):
        if config.embedder_config is None:
            pretrained = AutoModel.from_pretrained(config.embedder_name)
            config.embedder_config = pretrained.config.to_dict()
            return pretrained
        data = dict(config.embedder_config)
        model_type = data.pop("model_type")
        return AutoModel.from_config(AutoConfig.for_model(model_type, **data))

    def _embedder_tokenizer(self):
        if self._embedder_tokenizer_obj is None:
            from transformers import AutoTokenizer

            self._embedder_tokenizer_obj = AutoTokenizer.from_pretrained(self.config.embedder_name)
        return self._embedder_tokenizer_obj

    def _encoder_limit(self) -> int:
        table = int(self.embedder.config.max_position_embeddings)
        requested = self.config.embedder_max_length
        if requested is None:
            return table
        return max(1, min(int(requested), table))

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        """One normalized vector per text, truncated to the encoder context."""
        tokenizer = self._embedder_tokenizer()
        encoded = tokenizer(
            [text if text.strip() else " " for text in texts],
            add_special_tokens=True,
            truncation=True,
            max_length=self._encoder_limit(),
            padding=True,
            return_tensors="pt",
        )
        device = next(self.embedder.parameters()).device
        tokens = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        hidden = self.embedder(input_ids=tokens, attention_mask=mask).last_hidden_state
        return _mean_pool(hidden, mask)

    def score_messages(self, query: str, candidates: list[str]) -> torch.Tensor:
        """Cosine of each full message against the query. One encoder pass."""
        if not candidates:
            device = next(self.embedder.parameters()).device
            return torch.empty(0, device=device)
        vectors = self._embed_texts([query, *candidates])
        return vectors[1:] @ vectors[0]

    def _pack_budget(self, max_new_tokens: int) -> int:
        room = int(self.config.max_position_embeddings) - max(0, int(max_new_tokens))
        budget = min(int(self.config.pack_context_length), room)
        return max(1, budget)

    def pack_messages(self, messages: list[dict], tokenizer, max_new_tokens: int = 0) -> str:
        """Score the chat and render a prompt that fits the pack window."""
        _system, query, candidates = split_conversation(messages)
        if candidates and query:
            was_training = self.embedder.training
            self.embedder.eval()
            try:
                with torch.no_grad():
                    scores = self.score_messages(query, [item["content"] for item in candidates])
            finally:
                self.embedder.train(was_training)
            score_list = [float(value) for value in scores.detach().cpu().tolist()]
        else:
            score_list = [0.0] * len(candidates)
        return pack_conversation(messages, tokenizer, score_list, self._pack_budget(max_new_tokens))

    def _pack_one(self, messages: list[dict], tokenizer, index: int, budget: int) -> str:
        """System prompt, one earlier message, and the latest user message."""
        system, query, candidates = split_conversation(messages)
        chosen = [candidates[index]]
        system_text, user_text = _fit_fixed(tokenizer, system, query, budget)
        fitted = _fit_retrieved(tokenizer, system_text, user_text, chosen, [0.0], budget)
        retrieved = [(item["role"], item["content"]) for item in fitted]
        return render_prompt(system_text or None, retrieved, user_text or None)

    def _answer_nll(self, prompt: str, answer: str, tokenizer) -> torch.Tensor:
        """Mean loss of the gold reply. The prompt is masked."""
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        eos = getattr(tokenizer, "eos_token", None) or ""
        answer_ids = list(tokenizer.encode(f"{answer}{eos}", add_special_tokens=False))
        if not answer_ids:
            answer_ids = [int(getattr(tokenizer, "eos_token_id", None) or 0)]
        limit = int(self.config.max_position_embeddings)
        input_ids = prompt_ids + answer_ids
        if len(input_ids) > limit:
            overflow = len(input_ids) - limit
            if overflow >= len(prompt_ids):
                input_ids = answer_ids[-limit:]
                labels = list(input_ids)
            else:
                prompt_ids = prompt_ids[overflow:]
                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + list(answer_ids)
        else:
            labels = [-100] * len(prompt_ids) + list(answer_ids)
        if not any(token != -100 for token in labels):
            return self.lm_head.weight.reshape(-1)[0] * 0
        device = self.device
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        target = torch.tensor([labels], dtype=torch.long, device=device)
        return self(input_ids=ids, labels=target).loss

    def retrieval_loss(self, messages: list[dict], answer: str, tokenizer) -> torch.Tensor:
        """REINFORCE over which earlier message is packed.

        The policy is a softmax over the encoder's cosine scores, so the pretrained
        encoder is the prior. A sampled message is packed with the system
        prompt and the latest user message. The reward is how much that pack
        lowers the language-model loss of ``answer`` relative to the message
        the scores already rank first. The language-model weights are outside
        this graph. Fewer than two candidates returns a zero attached to the
        encoder so ``backward`` is safe.
        """
        anchor = self.embedder.embeddings.word_embeddings.weight.reshape(-1)[0] * 0
        _system, query, candidates = split_conversation(messages)
        if not query or not str(answer or "").strip() or len(candidates) < 2:
            return anchor
        scores = self.score_messages(query, [item["content"] for item in candidates])
        temperature = max(float(self.config.retrieval_temperature), 1e-4)
        log_probs = F.log_softmax(scores / temperature, dim=0)
        action = int(torch.multinomial(log_probs.detach().exp(), 1).item())
        budget = self._pack_budget(0)
        sampled = self._pack_one(messages, tokenizer, action, budget)
        prior = int(scores.detach().argmax().item())
        baseline = self._pack_one(messages, tokenizer, prior, budget)
        with torch.no_grad():
            advantage = self._answer_nll(baseline, answer, tokenizer) - self._answer_nll(sampled, answer, tokenizer)
        return -log_probs[action] * advantage + anchor

    def _decoder_pass(self, hidden_states, causal_mask, position_embeddings, position_ids):
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return hidden_states

    def _forward_with_effort(
        self,
        input_ids,
        attention_mask,
        position_ids,
        inputs_embeds,
        labels,
        logits_to_keep,
        effort: int,
        **kwargs,
    ):
        """Run the decoder block ``effort`` times. Training recomputes each pass."""
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids=position_ids)
        for _ in range(effort):
            if self.training and hidden_states.requires_grad:
                hidden_states = checkpoint(
                    self._decoder_pass,
                    hidden_states,
                    causal_mask,
                    position_embeddings,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._decoder_pass(
                    hidden_states, causal_mask, position_embeddings, position_ids
                )
        hidden_states = self.model.norm(hidden_states)
        from transformers.modeling_utils import slice_logits_to_keep

        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        effort = kwargs.pop("reasoning_effort", None)
        if effort is None:
            effort = getattr(self.config, "reasoning_effort", 1)
        effort = int(effort)
        if effort < 1:
            raise ValueError("reasoning_effort must be at least 1.")
        if effort == 1:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        return self._forward_with_effort(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            labels,
            logits_to_keep,
            effort,
            **kwargs,
        )

    def generate_from_messages(self, messages: list[dict], tokenizer, max_new_tokens: int = 256, **kwargs):
        """Pack the chat to the 2048-token window, then generate the reply."""
        prompt = self.pack_messages(messages, tokenizer, max_new_tokens=max_new_tokens)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [int(getattr(tokenizer, "eos_token_id", None) or 0)]
        device = self.device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        kwargs.setdefault("eos_token_id", eos_id)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        kwargs.setdefault("pad_token_id", eos_id if pad_id is None else pad_id)
        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, *args, **kwargs):
        messages = kwargs.pop("messages", None)
        effort = kwargs.pop("reasoning_effort", None)
        previous = int(getattr(self.config, "reasoning_effort", 1))
        if effort is not None:
            self.config.reasoning_effort = max(1, int(effort))
        try:
            if int(getattr(self.config, "reasoning_effort", 1)) > 1:
                kwargs["use_cache"] = False
            if messages is None:
                return super().generate(*args, **kwargs)
            tokenizer = kwargs.pop("tokenizer")
            max_new_tokens = kwargs.pop("max_new_tokens", 256)
            return self.generate_from_messages(messages, tokenizer, max_new_tokens=max_new_tokens, **kwargs)
        finally:
            self.config.reasoning_effort = previous


def build_rag_config(**kwargs) -> TeapotConfig:
    """Use the previous 16-layer Llama shape, with the pack window as the whole context.

    Retrieval never shows the language model more than ``pack_context_length``
    tokens, so the position table is that same length. RoPE theta stays
    500,000, which is what the trainer's Llama builder already used.
    """
    window = int(kwargs.get("max_position_embeddings") or 2048)
    kwargs["pack_context_length"] = window
    kwargs["max_position_embeddings"] = window
    kwargs.setdefault("hidden_size", 768)
    kwargs.setdefault("num_hidden_layers", 16)
    kwargs.setdefault("num_attention_heads", 12)
    kwargs.setdefault("num_key_value_heads", 4)
    kwargs.setdefault("intermediate_size", 2048)
    kwargs.setdefault("vocab_size", SMOLLM_VOCAB_SIZE)
    kwargs.setdefault("rope_theta", 500_000.0)
    kwargs.setdefault("rms_norm_eps", 1e-5)
    kwargs.setdefault("tie_word_embeddings", True)
    kwargs.setdefault("bos_token_id", 0)
    kwargs.setdefault("eos_token_id", 0)
    kwargs.setdefault("pad_token_id", 0)
    return TeapotConfig(**kwargs)


def build_smollm_config(**kwargs) -> TeapotConfig:
    """Fill in the SmolLM2-135M defaults the trainer does not pass itself.

    The trainer's sequence length is the pack window. The position table stays
    at 8,192, which is the window shipped with SmolLM2-135M, so a 2,048-token
    prompt can still generate.
    """
    hidden = int(kwargs.get("hidden_size") or 576)
    train_window = int(kwargs.get("max_position_embeddings") or 2048)
    kwargs["pack_context_length"] = train_window
    kwargs["max_position_embeddings"] = max(train_window, 8192)
    kwargs.setdefault("hidden_size", 576)
    kwargs.setdefault("num_hidden_layers", 30)
    kwargs.setdefault("num_attention_heads", 9)
    kwargs.setdefault("num_key_value_heads", 3)
    kwargs.setdefault("intermediate_size", 1536)
    kwargs.setdefault("vocab_size", SMOLLM_VOCAB_SIZE)
    kwargs.setdefault("rope_theta", 100_000.0)
    kwargs.setdefault("rms_norm_eps", 1e-5)
    kwargs.setdefault("tie_word_embeddings", True)
    kwargs.setdefault("bos_token_id", 0)
    kwargs.setdefault("eos_token_id", 0)
    kwargs.setdefault("pad_token_id", 0)
    kwargs.setdefault("initializer_range", hidden ** -0.5)
    return TeapotConfig(**kwargs)


AutoConfig.register("teapot", TeapotConfig, exist_ok=True)
AutoModelForCausalLM.register(TeapotConfig, TeapotForCausalLM, exist_ok=True)
TeapotConfig.register_for_auto_class()
TeapotForCausalLM.register_for_auto_class("AutoModelForCausalLM")
