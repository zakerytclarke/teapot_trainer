"""Run a recipe of training phases, then save the last weights.

The public surface is still ``TeapotTrainer(TrainingConfig).train(fresh=True)``.
``config.phases`` is the recipe. When it is empty, the pretrain, bridge, and
supervised dataset lists are that recipe.
``fresh=False`` continues from ``checkpoints/resume`` when that directory
exists: weights, optimizer, and the exact sample cursor. The cursor is
rebuilt by replaying the seeded mixture from the start of the phase, then
checked against the saved counts and random state.
Batch size, cosine schedule, bf16, and per-source token budgets are applied
inside that call. ``init_from`` is an optional method argument for a
confidence run that starts from an existing checkpoint; it is not a config field.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_float32_matmul_precision("high")

from datasets.utils.logging import disable_progress_bar as disable_ds_progress
from huggingface_hub.utils import disable_progress_bars as disable_hub_progress

disable_ds_progress()
disable_hub_progress()

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from .chat_formats import get_chat_format, pop_chat_format, push_chat_format, single_token_id
from .config import TrainingConfig, TrainingPhase
from .phases import phases_from_legacy, phase_view, same_recipe

# A phase hook of None means "use the config hook", which is different from
# a phase that explicitly runs nothing.
_INHERIT = object()
from .datasets import (
    SPAN_TOKEN_LOSS_WEIGHT,
    STOP_TOKEN_LOSS_WEIGHT,
    DatasetRegistry,
    DatasetSourceConfig,
    encode_pair,
    eos_loss_weight,
    iterate_encoded,
    span_loss_weight,
)
from .evals import evaluate_source, selection_score, source_key, task_kind


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _numeric_config_changes(saved, config) -> dict[str, float]:
    """Public numeric config fields a before_batch hook assigned this step."""
    if not saved or config is None:
        return {}
    changes = {}
    for key, value in config.__dict__.items():
        if key.startswith("_") or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if saved.get(key) != value:
            changes[key] = float(value)
    return changes


# A resume re-reads the phase from the first sample so the shuffle buffer is
# rebuilt the same way. This interval only bounds how much optimization is redone.
RESUME_EVERY_TOKENS = 100_000_000


def checkpoint_due(
    tokens: int,
    total_tokens: int,
    next_mark: float,
    every_tokens: int,
    every_percent: float,
) -> tuple[bool, str]:
    """Whether this step should save a mid-phase checkpoint, and its folder suffix.

    Token intervals name the folder by the token count, so each save is its own
    directory. Percent mode keeps the integer-percent suffix.
    """
    if total_tokens <= 0 or tokens <= 0 or tokens >= total_tokens:
        return False, ""
    if every_tokens > 0:
        if tokens < next_mark:
            return False, ""
        return True, f"{int(tokens)}tok"
    if every_percent <= 0:
        return False, ""
    pct = 100.0 * tokens / total_tokens
    if pct < next_mark or pct >= 100.0:
        return False, ""
    return True, f"{int(pct)}pct"


def take_mixture_sample(weights, budgets, produced, retired, take):
    """Draw the next training sample.

    ``take(index)`` returns a sample or ``None`` when that source is exhausted.
    ``None`` from this function means every source is finished. ``"retired"``
    means this draw dropped a source and the caller should draw again.
    """
    alive = [index for index in range(len(weights)) if not retired[index] and produced[index] < budgets[index]]
    if not alive:
        return None
    index = random.choices(alive, weights=[weights[i] for i in alive], k=1)[0]
    sample = take(index)
    if sample is None:
        retired[index] = True
        return "retired"
    produced[index] += len(sample["input_ids"])
    return index, sample


def _python_random_state() -> dict:
    version, state, gauss = random.getstate()
    return {"version": version, "state": list(state), "gauss": gauss}


def _set_python_random_state(payload: dict) -> None:
    random.setstate((payload["version"], tuple(payload["state"]), payload["gauss"]))


def _capture_torch_rng() -> dict:
    payload = {
        "cpu": base64.b64encode(bytes(int(value) for value in torch.get_rng_state().tolist())).decode("ascii")
    }
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
        payload["cuda"] = base64.b64encode(bytes(int(value) for value in cuda_state.tolist())).decode("ascii")
    return payload


def _restore_torch_rng(payload: dict) -> None:
    cpu = torch.tensor(list(base64.b64decode(payload["cpu"])), dtype=torch.uint8)
    torch.set_rng_state(cpu)
    if payload.get("cuda") and torch.cuda.is_available():
        cuda = torch.tensor(list(base64.b64decode(payload["cuda"])), dtype=torch.uint8)
        torch.cuda.set_rng_state(cuda)


def align_tokenizer_ids(model, tokenizer) -> None:
    """Point the checkpoint's stop id at the tokenizer end token.

    A freshly built Llama config uses eos id 2. With the GPT-2 tokenizer that
    id is the character ``#``, while the trained end token is ``<|endoftext|>``.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return
    model.config.eos_token_id = eos_id
    generation = getattr(model, "generation_config", None)
    if generation is not None:
        generation.eos_token_id = eos_id


def weighted_cross_entropy(logits, labels, weights):
    """Mean cross-entropy over supervised tokens, weighted per label.

    ``weights`` lines up with ``labels``. Position ``t`` predicts ``labels[t]``
    from ``logits[t - 1]``, so the weight that applies is ``weights[t]``.
    """
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].to(dtype=shift_logits.dtype)
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    active = shift_labels.ne(-100).to(dtype=token_loss.dtype)
    weighted = shift_weights * active
    return (token_loss * weighted).sum() / weighted.sum().clamp(min=1.0)


def architecture_mismatches(hf_config, model_cfg) -> list[str]:
    """Weight-shape differences between a loaded model and ``ModelConfig``.

    Context may be shorter than the checkpoint. It may not be longer.
    """
    problems = []
    if model_cfg.model_type.lower() == "gpt2":
        compared = (
            ("n_embd", "hidden_size", model_cfg.hidden_size),
            ("n_layer", "layers", model_cfg.layers),
            ("n_head", "heads", model_cfg.heads),
        )
        context_attr = "n_positions"
    else:
        compared = (
            ("hidden_size", "hidden_size", model_cfg.hidden_size),
            ("num_hidden_layers", "layers", model_cfg.layers),
            ("num_attention_heads", "heads", model_cfg.heads),
            ("num_key_value_heads", "kv_heads", model_cfg.kv_heads),
            ("intermediate_size", "intermediate", model_cfg.intermediate),
        )
        context_attr = "max_position_embeddings"
    for attr, label, want in compared:
        got = getattr(hf_config, attr, None)
        if got != want:
            problems.append(f"{label}: checkpoint {got}, config {want}")
    positions = getattr(hf_config, context_attr, None)
    if positions is None or model_cfg.seq_len > positions:
        problems.append(
            f"seq_len: config {model_cfg.seq_len} is longer than checkpoint context {positions}"
        )
    return problems


def _resolve_init_from(path: str) -> str:
    """Return a local directory or a Hugging Face id. Missing local paths error."""
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded
    hub_id = path.count("/") == 1 and not path.startswith((".", "/", "~"))
    if hub_id:
        return path
    raise SystemExit(f"init_from path does not exist: {path}")


class TeapotTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.m_cfg = config.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_example_cap = 32
        self.qualitative = True
        self.phase_stats: dict[str, dict] = {}
        self.best_behavior: float | None = None
        self.best_loss: float | None = None
        self._completed_phases: list[str] = []
        self._checkpointing = False
        self.lm_eval_batches = 16
        self._plateau_history: list[list[float]] = []
        self._plateau_stalls = 0
        self._plateau_phase: str | None = None
        self._active_eval_datasets = None
        self._phase_before_batch = _INHERIT

        tokenizer_short = self.config.tokenizer.split("/")[-1]
        safe_dict = self.config.model_dump(exclude_defaults=True)
        # This flag only decides whether copies are fine-tuned at checkpoints.
        # It must not fork the run directory, or a resume would miss its weights.
        safe_dict.pop("sft_at_checkpoints", None)
        # The same recipe written as phases= keeps the directory the older
        # fields would have hashed. A different order or budget is part of the id.
        safe_dict.pop("phases", None)
        if self.config.phases and not same_recipe(list(self.config.phases), phases_from_legacy(self.config), self.config):
            safe_dict["phases"] = [phase_view(phase, self.config) for phase in self.config.phases]

        def freeze(value):
            if isinstance(value, dict):
                return {key: freeze(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [freeze(item) for item in value]
            if isinstance(value, type):
                return f"{value.__module__}.{value.__qualname__}"
            if callable(value):
                return getattr(value, "__qualname__", type(value).__name__)
            return value

        run_hash = hashlib.md5(json.dumps(freeze(safe_dict), sort_keys=True, default=str).encode()).hexdigest()[:6]
        self.output_dir = f"./runs/teapot-{tokenizer_short}-{run_hash}"
        os.makedirs(self.output_dir, exist_ok=True)
        tb_log_path = self.config.logging_dir or os.path.join(self.output_dir, "runs")
        self.writer = SummaryWriter(log_dir=tb_log_path)

    def _build_model(self, vocab_size: int):
        kind = self.m_cfg.model_type.lower()
        if kind == "custom":
            if not self.m_cfg.custom_config_cls or not self.m_cfg.custom_model_cls:
                raise ValueError("model_type='custom' requires custom_config_cls and custom_model_cls.")
            cfg = self.m_cfg.custom_config_cls(
                vocab_size=vocab_size,
                hidden_size=self.m_cfg.hidden_size,
                num_hidden_layers=self.m_cfg.layers,
                num_attention_heads=self.m_cfg.heads,
                num_key_value_heads=self.m_cfg.kv_heads,
                intermediate_size=self.m_cfg.intermediate,
                max_position_embeddings=self.m_cfg.seq_len,
            )
            return self.m_cfg.custom_model_cls(cfg)
        if kind == "llama":
            cfg = LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=self.m_cfg.hidden_size,
                intermediate_size=self.m_cfg.intermediate,
                num_hidden_layers=self.m_cfg.layers,
                num_attention_heads=self.m_cfg.heads,
                num_key_value_heads=self.m_cfg.kv_heads,
                max_position_embeddings=self.m_cfg.seq_len,
                rms_norm_eps=1e-5,
                rope_theta=500_000.0,
                tie_word_embeddings=True,
                pad_token_id=None,
            )
            return LlamaForCausalLM(cfg)
        if kind == "gpt2":
            cfg = GPT2Config(
                vocab_size=vocab_size,
                n_embd=self.m_cfg.hidden_size,
                n_layer=self.m_cfg.layers,
                n_head=self.m_cfg.heads,
                n_positions=self.m_cfg.seq_len,
                n_ctx=self.m_cfg.seq_len,
            )
            return GPT2LMHeadModel(cfg)
        raise ValueError(f"Unsupported model_type: '{kind}'")

    def _supervised_lr(self) -> float:
        """Peak for a supervised copy. An explicit sft_lr wins."""
        if self.config.sft_lr is not None:
            return float(self.config.sft_lr)
        if self.config.pretrain_datasets:
            return self.config.lr * 0.2
        return self.config.lr

    def _load_model_weights(self, source: str, tokenizer):
        """Load a checkpoint. A plain Llama checkpoint seeds the custom class.

        The language-model tensors are copied. The embedder built with the
        custom class stays as it was loaded, which is teapotai/teapotembedding
        for the teapot model.
        """
        import teapottrainer.modeling_teapot  # noqa: F401  registers the auto classes

        loaded = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, trust_remote_code=True)
        mismatches = architecture_mismatches(loaded.config, self.m_cfg)
        if mismatches:
            detail = "\n  ".join(mismatches)
            raise SystemExit(f"checkpoint does not match the model config:\n  {detail}")
        custom_cls = self.m_cfg.custom_model_cls
        if self.m_cfg.model_type.lower() == "custom" and custom_cls is not None and not isinstance(loaded, custom_cls):
            print(f"Copying {type(loaded).__name__} weights into {custom_cls.__name__}.")
            rows = loaded.get_input_embeddings().weight.shape[0]
            model = self._build_model(vocab_size=max(len(tokenizer), int(rows)))
            missing, unexpected = model.load_state_dict(loaded.state_dict(), strict=False)
            tied = {"lm_head.weight"} if getattr(model.config, "tie_word_embeddings", False) else set()
            bad_missing = [key for key in missing if key not in tied and not key.startswith("embedder.")]
            if bad_missing or unexpected:
                raise SystemExit(
                    "Checkpoint tensors do not match the custom model.\n"
                    f"  missing: {bad_missing}\n  unexpected: {list(unexpected)}"
                )
            kept = [key for key in missing if key.startswith("embedder.")]
            print(f"Embedder stays as loaded ({len(kept)} tensors were not in the checkpoint).")
            del loaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            model = loaded
        embedding = model.get_input_embeddings()
        rows = None if embedding is None else embedding.weight.shape[0]
        if rows is not None and rows < len(tokenizer):
            print(f"Tokenizer is longer than the checkpoint embeddings ({len(tokenizer)} > {rows}). Resizing.")
            model.resize_token_embeddings(len(tokenizer))
        elif rows is not None and rows != len(tokenizer):
            print(f"Checkpoint embeddings have {rows} rows and the tokenizer has {len(tokenizer)}. Leaving the matrix as stored.")
        return model

    def _plateau_reached(self, tokens: int, metrics: dict) -> bool:
        """True when the FineWeb holdout has stopped moving across the window."""
        from .config import EarlyStop

        rule = EarlyStop(
            metric="fineweb",
            window_tokens=int(self.config.plateau_window_tokens or 0),
            min_delta=float(self.config.plateau_min_loss_drop),
            patience=1,
        )
        return self._early_stop(tokens, metrics, rule, "pretrain")

    def _early_stop(self, tokens: int, metrics: dict, rule, phase_name: str) -> bool:
        """True when ``rule`` has seen ``patience`` flat windows in a row."""
        window = 0 if rule is None else int(rule.window_tokens or 0)
        if window <= 0:
            return False
        needle = str(rule.metric or "").lower()
        loss = None
        for key, value in metrics.items():
            if key.endswith("/loss") and needle in key.lower():
                loss = float(value)
                break
        if loss is None:
            return False
        history = self._plateau_history
        history.append([float(tokens), loss])
        baseline = None
        for seen_tokens, seen_loss in history:
            if tokens - seen_tokens >= window:
                baseline = (seen_tokens, seen_loss)
            else:
                break
        if baseline is None:
            return False
        drop = baseline[1] - loss
        if drop >= float(rule.min_delta):
            self._plateau_stalls = 0
            return False
        self._plateau_stalls = int(getattr(self, "_plateau_stalls", 0)) + 1
        patience = max(1, int(rule.patience or 1))
        if self._plateau_stalls < patience:
            print(
                f"{phase_name} holdout moved {drop:.4f} nats over {tokens - baseline[0]:,.0f} tokens "
                f"(from {baseline[1]:.4f} to {loss:.4f}), under {rule.min_delta}. "
                f"Patience {self._plateau_stalls}/{patience}."
            )
            return False
        print(
            f"Pretrain holdout moved {drop:.4f} nats over {tokens - baseline[0]:,.0f} tokens "
            f"(from {baseline[1]:.4f} to {loss:.4f}), under {rule.min_delta}. Stopping pretrain."
            if phase_name == "pretrain"
            else (
                f"{phase_name} holdout moved {drop:.4f} nats over {tokens - baseline[0]:,.0f} tokens "
                f"(from {baseline[1]:.4f} to {loss:.4f}), under {rule.min_delta}. Stopping {phase_name}."
            )
        )
        return True

    def _optimizer(self, model, lr: float) -> AdamW:
        decay, frozen = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and "norm" not in name and not name.endswith(".bias"):
                decay.append(param)
            else:
                frozen.append(param)
        return AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": frozen, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )

    def _set_lr(self, optimizer, progress: float, peak: float) -> float:
        warmup = 0.02
        progress = min(max(progress, 0.0), 1.0)
        if progress < warmup:
            scale = progress / warmup
        else:
            cosine = (progress - warmup) / (1.0 - warmup)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * cosine))
        lr = peak * scale
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _print_model_summary(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg = getattr(model, "config", None)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", self.m_cfg.hidden_size)
        layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", self.m_cfg.layers)
        print("\n" + "=" * 78)
        print(" MODEL")
        print("=" * 78)
        print(f" loaded hidden: {hidden}  layers: {layers}  context used for packing: {self.m_cfg.seq_len}")
        print(f" parameters: {total:,}  trainable: {trainable:,}")
        print("=" * 78 + "\n")
        return total

    def _materialize_sources(self, dataset_specs) -> list[DatasetSourceConfig]:
        sources = []
        for spec in dataset_specs:
            source = spec.source
            if isinstance(source, str):
                src = DatasetSourceConfig(name=source)
            else:
                src = source.model_copy(update={"target_tokens": spec.target_tokens, "weight": spec.weight})
            sources.append(src)
        return sources

    def _before_batch_hooks(self):
        """The functions that may edit model.config for the next batch."""
        hook = self.config.before_batch if self._phase_before_batch is _INHERIT else self._phase_before_batch
        if hook is None:
            return []
        if callable(hook):
            return [hook]
        return list(hook)

    def _restore_config(self, config, saved) -> None:
        if config is None or saved is None:
            return
        for key in list(config.__dict__):
            if key not in saved:
                del config.__dict__[key]
        config.__dict__.update(saved)

    def _reweight_packed(self, sample: dict, encoded: dict, tokenizer) -> None:
        """Keep the source's stop or span weights after the prompt is repacked."""
        old = sample.get("loss_weights")
        if not old:
            return
        if any(abs(float(value) - SPAN_TOKEN_LOSS_WEIGHT) < 1e-6 for value in old):
            encoded["loss_weights"] = span_loss_weight()(encoded, tokenizer)
        elif any(abs(float(value) - STOP_TOKEN_LOSS_WEIGHT) < 1e-6 for value in old):
            encoded["loss_weights"] = eos_loss_weight()(encoded, tokenizer)

    def _pack_supervised_batch(self, model, batch, tokenizer):
        """Replace an over-long chat with the same pack inference will show.

        ``retrieval_loss`` trains the encoder on which earlier turn to keep.
        The language-model step then trains on that packed prompt, so a chat
        that does not fit the 2,048 window is seen with the dropped turns gone
        and the surviving turns still in conversation order.
        """
        live = _unwrap(model)
        if not hasattr(live, "pack_messages"):
            return batch, []
        packed = []
        retrievals = []
        for sample in batch:
            messages = sample.get("messages")
            answer = sample.get("answer")
            if not messages or not isinstance(answer, str) or not answer.strip():
                packed.append(sample)
                continue
            prompt = live.pack_messages(messages, tokenizer)
            encoded = encode_pair(tokenizer, prompt, answer, self.m_cfg.seq_len)
            if encoded is None:
                packed.append(sample)
                continue
            self._reweight_packed(sample, encoded, tokenizer)
            packed.append(encoded)
            if hasattr(live, "retrieval_loss"):
                retrievals.append(live.retrieval_loss(messages, answer, tokenizer))
        return packed, retrievals

    def _collate_batch(self, batch, tokenizer):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0
        width = max(len(sample["input_ids"]) for sample in batch)
        ids, labels, mask, weights = [], [], [], []
        real_tokens = 0
        use_weights = False
        for sample in batch:
            length = len(sample["input_ids"])
            pad = width - length
            ids.append(sample["input_ids"] + [pad_id] * pad)
            labels.append(sample["labels"] + [-100] * pad)
            mask.append([1] * length + [0] * pad)
            sample_weights = sample.get("loss_weights")
            if sample_weights is None:
                sample_weights = [1.0] * length
            elif len(sample_weights) != length:
                raise ValueError("loss_weights must have one value per input token.")
            elif any(value != 1.0 for value in sample_weights):
                use_weights = True
            weights.append([float(value) for value in sample_weights] + [0.0] * pad)
            real_tokens += length
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        kwargs = {"input_ids": input_ids}
        weight_tensor = None
        if use_weights:
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            kwargs["labels"] = label_tensor
        if any(row.count(0) for row in mask):
            kwargs["attention_mask"] = torch.tensor(mask, dtype=torch.long, device=self.device)
        return kwargs, label_tensor, weight_tensor, real_tokens

    def _language_loss(self, model, kwargs, label_tensor, weight_tensor, retrievals):
        output = model(**kwargs)
        loss = (
            weighted_cross_entropy(output.logits, label_tensor, weight_tensor)
            if weight_tensor is not None
            else output.loss
        )
        if retrievals:
            extra = torch.stack([item.reshape(()) for item in retrievals]).mean()
            loss = loss + extra.to(dtype=loss.dtype)
        return loss

    def _forward_backward(self, model, batch, tokenizer, phase_name: str):
        live = _unwrap(model)
        config = getattr(live, "config", None)
        # Deep copy so a hook can replace a field or edit a nested dict, and
        # both are undone after the batch. Assigning is the normal case.
        saved = copy.deepcopy(dict(config.__dict__)) if config is not None else None
        self._batch_config_metrics = {}
        real_tokens = 0
        try:
            for hook in self._before_batch_hooks():
                hook(live, phase_name)
            self._batch_config_metrics = _numeric_config_changes(saved, config)
            retrievals = []
            if self._phase_uses_chat(phase_name):
                batch, retrievals = self._pack_supervised_batch(model, batch, tokenizer)
            kwargs, label_tensor, weight_tensor, real_tokens = self._collate_batch(batch, tokenizer)
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = self._language_loss(model, kwargs, label_tensor, weight_tensor, retrievals)
            else:
                loss = self._language_loss(model, kwargs, label_tensor, weight_tensor, retrievals)
            (loss / self.config.grad_accum).backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(
                    "\nCUDA out of memory. Lower batch_size or max_position_embeddings, "
                    f"or set compile=False. Current batch_size={self.config.batch_size}, "
                    f"seq_len={self.m_cfg.seq_len}."
                )
                sys.exit(1)
            raise
        finally:
            self._restore_config(config, saved)
        return float(loss.detach().float().item()), real_tokens

    def _phase_uses_chat(self, phase_name: str | None) -> bool:
        """Supervised checkpoints carry the chat template. Pretrain checkpoints do not."""
        if not phase_name:
            return False
        return phase_name == "sft" or phase_name.startswith("sft_") or phase_name.endswith("_sft")

    def _apply_chat_tokenizer(self, model, tokenizer):
        """Point a supervised save at the selected template and its stop token."""
        fmt = get_chat_format(self.config.chat_format)
        previous = (
            getattr(tokenizer, "chat_template", None),
            getattr(tokenizer, "eos_token", None),
        )
        tokenizer.chat_template = fmt.template
        if fmt.stop_token and single_token_id(tokenizer, fmt.stop_token) is not None:
            tokenizer.eos_token = fmt.stop_token
            align_tokenizer_ids(model, tokenizer)
        return previous

    def _restore_chat_tokenizer(self, model, tokenizer, previous) -> None:
        template, eos = previous
        tokenizer.chat_template = template
        if eos is not None:
            tokenizer.eos_token = eos
        align_tokenizer_ids(model, tokenizer)

    def _save_weights(self, model, tokenizer, directory: str):
        # ``_save_chat`` is set by the caller. The signature stays three
        # arguments so a resumed pretrain save is unchanged.
        os.makedirs(directory, exist_ok=True)
        chat = bool(getattr(self, "_save_chat", False))
        self._save_chat = False
        previous = self._apply_chat_tokenizer(model, tokenizer) if chat else None
        try:
            unwrapped = _unwrap(model)
            unwrapped.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
        finally:
            if previous is not None:
                self._restore_chat_tokenizer(model, tokenizer, previous)

    def _log_metrics(self, phase: str, step: int, payload: dict, group: str = "eval"):
        path = os.path.join(self.output_dir, "metrics.jsonl")
        record = {"phase": phase, "step": step, "metrics": payload}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        # Base checkpoints use val/. A finished SFT copy uses sft_checkpoint/,
        # plotted at the pretrain step so the copies line up on one chart.
        chart = "sft_checkpoint" if group == "sft_checkpoint" else "val"
        for key, value in payload.items():
            if not isinstance(value, (int, float)):
                continue
            if "support" in key.split("/")[-1]:
                continue
            if group == "eval":
                self.writer.add_scalar(f"eval/{key}", value, step)
            if key == "loss":
                self.writer.add_scalar(f"{chart}/loss", value, step)
            elif key.endswith("/loss"):
                self.writer.add_scalar(f"{chart}/{key}", value, step)
            elif key == "perplexity":
                self.writer.add_scalar(f"{chart}/perplexity", value, step)
            elif key.endswith("/perplexity"):
                self.writer.add_scalar(f"{chart}/{key}", value, step)

    def _lm_loss(self, model, tokenizer, src: DatasetSourceConfig) -> float | None:
        unwrapped = _unwrap(model)
        was_training = unwrapped.training
        unwrapped.eval()
        losses = []
        stream = iterate_encoded(
            src,
            tokenizer,
            self.m_cfg.seq_len,
            is_eval=True,
            holdout_ratio=self.config.holdout_ratio,
            seed=self.config.seed,
        )
        with torch.no_grad():
            for index, sample in enumerate(stream):
                if index >= self.lm_eval_batches:
                    break
                input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=self.device)
                labels = torch.tensor([sample["labels"]], dtype=torch.long, device=self.device)
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = unwrapped(input_ids=input_ids, labels=labels)
                else:
                    output = unwrapped(input_ids=input_ids, labels=labels)
                if output.loss is not None and not torch.isnan(output.loss):
                    losses.append(float(output.loss.item()))
        if was_training:
            unwrapped.train()
        if not losses:
            return None
        return sum(losses) / len(losses)

    def run_evaluation_suite(
        self,
        model,
        tokenizer,
        phase_name: str,
        global_step: int,
        update_best: bool = True,
        metric_group: str = "eval",
    ):
        """Score every eval set. Text holdouts report loss. Task sets report loss and behavior."""
        unwrapped = _unwrap(model)
        flat: dict[str, float] = {}
        print("\n" + "=" * 78)
        print(f" EVAL  {phase_name}  step {global_step}")
        print("=" * 78)
        eval_specs = self._active_eval_datasets if self._active_eval_datasets is not None else self.config.eval_datasets
        if not eval_specs:
            print(" No eval datasets configured.")
            print("=" * 78 + "\n")
            return flat

        for spec in eval_specs:
            sources = self._materialize_sources([spec])
            src = sources[0]
            kind = task_kind(src)
            key = source_key(src)
            if kind == "lm" or not src.is_sft:
                loss = self._lm_loss(model, tokenizer, src)
                if loss is not None:
                    flat[f"{key}/loss"] = loss
                    print(f" {key}  loss {loss:.4f}  ppl {math.exp(min(loss, 20)):.2f}")
                continue
            print(f" scoring {key} ({kind}) on up to {self.eval_example_cap} examples...")
            result = evaluate_source(
                unwrapped,
                tokenizer,
                src,
                device=self.device,
                seq_len=self.m_cfg.seq_len,
                max_examples=self.eval_example_cap,
                seed=self.config.seed,
                holdout_ratio=self.config.holdout_ratio,
            )
            for name, value in result["metrics"].items():
                flat[f"{key}/{name}"] = value
                if not name.startswith("support"):
                    support = result["metrics"].get(f"support_{name}", result["metrics"].get("support"))
                    suffix = f"  n={int(support)}" if support is not None else ""
                    print(f"   {name}: {value:.3f}{suffix}")
            if self.qualitative:
                for sample in result["samples"][:2]:
                    print(f"   sample [{sample['task']}]")
                    print(f"     user: {sample.get('user', '')[:180]}")
                    print(f"     gold: {sample.get('gold', '')[:180]}")
                    print(f"     pred: {sample.get('prediction', '')[:180]}")
        behavior = selection_score(flat)
        losses = [value for name, value in flat.items() if name.endswith("/loss")]
        mean_loss = sum(losses) / len(losses) if losses else None
        for key, value in list(flat.items()):
            if key.endswith("/loss"):
                flat[f"{key[: -len('loss')]}perplexity"] = math.exp(min(value, 20))
        if mean_loss is not None:
            # One series for every eval set. Each set counts once, whatever its size.
            flat["loss"] = mean_loss
            flat["perplexity"] = math.exp(min(mean_loss, 20))
        if behavior is not None:
            print(f" behavior_score: {behavior:.3f}")
        if mean_loss is not None:
            print(f" mean_eval_loss: {mean_loss:.4f}  ppl {flat['perplexity']:.2f}")
        print("=" * 78 + "\n")
        self._log_metrics(phase_name, global_step, flat, group=metric_group)

        improved = False
        if behavior is not None and (self.best_behavior is None or behavior >= self.best_behavior):
            self.best_behavior = behavior
            improved = True
        elif behavior is None and mean_loss is not None and (self.best_loss is None or mean_loss <= self.best_loss):
            self.best_loss = mean_loss
            improved = True
        if flat:
            # Keep every eval's weights. best/ is only the single best score.
            snapshot_dir = os.path.join(self.output_dir, "checkpoints", phase_name)
            self._save_chat = self._phase_uses_chat(phase_name)
            self._save_weights(model, tokenizer, snapshot_dir)
            with open(os.path.join(snapshot_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump({"phase": phase_name, "step": global_step, "metrics": flat}, handle, indent=2)
            print(f" saved eval checkpoint to {snapshot_dir}")
        if update_best and improved and flat:
            best_dir = os.path.join(self.output_dir, "checkpoints", "best")
            self._save_chat = self._phase_uses_chat(phase_name)
            self._save_weights(model, tokenizer, best_dir)
            with open(os.path.join(best_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump({"phase": phase_name, "step": global_step, "metrics": flat}, handle, indent=2)
            print(f" saved best checkpoint to {best_dir}")
        if self._checkpointing:
            _unwrap(model).config.use_cache = False
        unwrapped.train()
        return flat

    def _resume_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints", "resume")

    def _read_resume(self) -> dict | None:
        for folder in (self._resume_dir(), self._resume_dir() + ".old"):
            path = os.path.join(folder, "trainer_state.json")
            if os.path.isfile(path) and os.path.isfile(os.path.join(folder, "config.json")):
                with open(path, encoding="utf-8") as handle:
                    state = json.load(handle)
                state["dir"] = folder
                return state
        return None

    def _restore_phase_boundary_rng(self) -> None:
        """Put torch's generator back where the completed phase left it.

        Supervised training reseeds Python's ``random`` itself. Torch stays
        on the generator from the end of the previous phase, which a new
        process would otherwise have reset with ``manual_seed``.
        """
        boundary = self._read_resume()
        if not boundary or not boundary.get("phase_done") or not boundary.get("torch_rng"):
            return
        _restore_torch_rng(boundary["torch_rng"])

    def _write_resume(self, model, tokenizer, optimizer, state: dict) -> None:
        directory = self._resume_dir()
        temporary = directory + ".tmp"
        # Capture before the save. Writing weights must not move the cursor,
        # so the same state is restored when this call returns.
        python_random = _python_random_state()
        torch_rng = _capture_torch_rng()
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)
        os.makedirs(temporary, exist_ok=True)
        self._save_chat = self._phase_uses_chat(state.get("phase"))
        self._save_weights(model, tokenizer, temporary)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(temporary, "optimizer.pt"))
        payload = dict(state)
        payload.pop("dir", None)
        payload["python_random"] = python_random
        payload["torch_rng"] = torch_rng
        payload["best_loss"] = self.best_loss
        payload["best_behavior"] = self.best_behavior
        payload["plateau_history"] = list(getattr(self, "_plateau_history", []))
        payload["plateau_stalls"] = int(getattr(self, "_plateau_stalls", 0))
        with open(os.path.join(temporary, "trainer_state.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        previous = directory + ".old"
        if os.path.isdir(previous):
            shutil.rmtree(previous)
        if os.path.isdir(directory):
            os.rename(directory, previous)
        os.rename(temporary, directory)
        if os.path.isdir(previous):
            shutil.rmtree(previous)
        _set_python_random_state(python_random)
        _restore_torch_rng(torch_rng)

    def _next_sample(self, index, sources, iterators, restarts):
        try:
            return next(iterators[index])
        except StopIteration:
            restarts[index] += 1
            if restarts[index] > 40:
                return None
            iterators[index] = iter(
                DatasetRegistry.prepare_dataset_splits(
                    [sources[index]],
                    self._tokenizer,
                    self.m_cfg.seq_len,
                    is_eval=False,
                    holdout_ratio=self.config.holdout_ratio,
                    seed=self.config.seed,
                )[0]
            )
            try:
                return next(iterators[index])
            except StopIteration:
                return None

    def _open_iterators(self, sources, tokenizer):
        return [
            iter(
                DatasetRegistry.prepare_dataset_splits(
                    [src],
                    tokenizer,
                    self.m_cfg.seq_len,
                    is_eval=False,
                    holdout_ratio=self.config.holdout_ratio,
                    seed=self.config.seed,
                )[0]
            )
            for src in sources
        ]

    def _fast_forward(self, phase_name, weights, budgets, produced, retired, take, samples_target, produced_target, restarts_target, random_target):
        print(
            f"Fast-forwarding {phase_name} through {samples_target:,} samples "
            f"({sum(produced_target):,} tokens) with the original seed."
        )
        taken = 0
        progress = tqdm(total=samples_target, desc=f"[{phase_name} fast-forward]", unit="sample", unit_scale=True, dynamic_ncols=True)
        try:
            while taken < samples_target:
                drawn = take_mixture_sample(weights, budgets, produced, retired, take)
                if drawn is None:
                    raise SystemExit(
                        f"Resume fast-forward for {phase_name} ran out of data at sample {taken:,} "
                        f"of {samples_target:,}."
                    )
                if drawn == "retired":
                    continue
                taken += 1
                progress.update(1)
        finally:
            progress.close()
        if produced != list(produced_target) or list(self._restarts) != list(restarts_target):
            raise SystemExit(
                f"Resume fast-forward for {phase_name} did not land on the checkpoint cursor.\n"
                f"  produced now {produced}, checkpoint {list(produced_target)}\n"
                f"  restarts now {list(self._restarts)}, checkpoint {list(restarts_target)}"
            )
        if _python_random_state() != random_target:
            raise SystemExit(f"Resume fast-forward for {phase_name} did not land on the checkpoint random state.")
        print(f"Fast-forward complete at {sum(produced):,} tokens.")

    def _sft_from_base_checkpoint(
        self,
        model,
        tokenizer,
        phase_name: str,
        global_step: int,
        required: bool = False,
        mix=None,
        peak_lr: float | None = None,
        allow_any_name: bool = False,
        honor_flag: bool = True,
        label_suffix: str = "sft",
    ) -> None:
        """Fine-tune a copy of a saved base checkpoint on a supervised mix.

        The live pretrain weights, optimizer, mixture cursor, and resume file
        stay where they are. The copy is written to ``checkpoints/<phase>_sft``
        with the held-out metrics from the finished supervised run.
        ``required`` runs the copy even when ``sft_at_checkpoints`` is off, which
        is how the end-of-run SFT stays the same pass as these checkpoints.
        ``mix`` selects a different supervised phase than ``config.sft_datasets``.
        """
        datasets = list(self.config.sft_datasets if mix is None else mix)
        if not datasets:
            return
        if honor_flag and not required and not self.config.sft_at_checkpoints:
            return
        base_name = phase_name == "bridge" or phase_name.startswith("pretrain") or phase_name.startswith("bridge_")
        if not allow_any_name and not base_name:
            return
        label = f"{phase_name}_{label_suffix}"
        directory = os.path.join(self.output_dir, "checkpoints", label)
        if os.path.isfile(os.path.join(directory, "metrics.json")):
            print(f"SFT checkpoint already saved at {directory}")
            return
        base_dir = os.path.join(self.output_dir, "checkpoints", phase_name)
        if not os.path.isfile(os.path.join(base_dir, "config.json")):
            print(f"No base checkpoint at {base_dir}. Skipping side SFT.")
            return

        if required:
            print(f"\nSFT from {phase_name}. Same full pass as the checkpoint copies.")
        else:
            print(f"\nSide SFT from {phase_name}. The pretrain model is left unchanged.")
        python_random = _python_random_state()
        torch_rng = _capture_torch_rng()
        saved_best_loss = self.best_loss
        saved_best_behavior = self.best_behavior
        saved_restarts = getattr(self, "_restarts", None)
        saved_phases = list(self._completed_phases)
        side = None
        try:
            side = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32, trust_remote_code=True)
            align_tokenizer_ids(side, tokenizer)
            side.to(self.device)
            if self._checkpointing and hasattr(side, "gradient_checkpointing_enable"):
                try:
                    side.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    side.gradient_checkpointing_enable()
                side.config.use_cache = False
            sft_lr = self._supervised_lr() if peak_lr is None else float(peak_lr)
            print(f"SFT copy peak lr is {sft_lr:.3e}.")
            self._train_phase(
                label,
                datasets,
                side,
                tokenizer,
                global_step,
                peak_lr=sft_lr,
                task_eval=False,
                resume=None,
                persist_resume=False,
                record_completion=False,
                update_best=False,
                eval_during=False,
                final_phase_name=label,
                metric_group="sft_checkpoint",
                metric_step=global_step,
            )
            stats = self.phase_stats.get(label)
            best_train = None if not stats else stats.get("best_train_loss")
            if best_train is not None and getattr(self, "writer", None) is not None:
                self.writer.add_scalar("sft_checkpoint/best_train_loss", best_train, global_step)
            if stats:
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, "train_stats.json"), "w", encoding="utf-8") as handle:
                    json.dump(stats, handle, indent=2)
            print(f"Saved SFT checkpoint to {directory}")
        finally:
            _set_python_random_state(python_random)
            _restore_torch_rng(torch_rng)
            self.best_loss = saved_best_loss
            self.best_behavior = saved_best_behavior
            self._restarts = saved_restarts
            self._completed_phases = saved_phases
            del side
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            live = _unwrap(model)
            live.train()
            if self._checkpointing:
                live.config.use_cache = False

    def _train_phase(
        self,
        phase_name: str,
        dataset_specs,
        model,
        tokenizer,
        starting_step: int,
        peak_lr: float,
        task_eval: bool,
        resume: dict | None = None,
        persist_resume: bool = True,
        record_completion: bool = True,
        update_best: bool = True,
        eval_during: bool = True,
        final_phase_name: str | None = None,
        metric_group: str = "eval",
        metric_step: int | None = None,
        *,
        every_tokens: int | None = None,
        every_percent: float | None = None,
        on_checkpoint=None,
        early_stop=None,
    ):
        if not dataset_specs:
            return starting_step
        sources = self._materialize_sources(dataset_specs)
        budgets = [max(1, spec.target_tokens) for spec in dataset_specs]
        weights = [max(spec.weight, 0.0) or 1.0 for spec in dataset_specs]
        total_tokens = sum(budgets)
        produced = [0 for _ in sources]
        restarts = [0 for _ in sources]
        retired = [False for _ in sources]
        self._restarts = restarts
        self._tokenizer = tokenizer
        print(
            f"\n{phase_name}: target {total_tokens:,} tokens, peak lr {peak_lr:.3e}, "
            f"batch {self.config.batch_size} x accum {self.config.grad_accum}"
        )
        for src, budget in zip(sources, budgets):
            print(f"  {source_key(src)}: {budget:,} tokens")
        # Reseed immediately before the mixture so resume replays the same draws.
        random.seed(self.config.seed)
        iterators = self._open_iterators(sources, tokenizer)

        def take(index):
            return self._next_sample(index, sources, iterators, restarts)

        tokens = 0
        samples_taken = 0
        micro_steps = 0
        global_step = starting_step
        first_loss = None
        last_loss = None
        best_train_loss = None
        if every_tokens is None:
            every_tokens = int(self.config.eval_every_tokens or 0)
        else:
            every_tokens = int(every_tokens)
        percent_step = float(self.config.eval_every_percent if every_percent is None else every_percent)
        next_eval = float(every_tokens if every_tokens > 0 else percent_step)
        if resume:
            tokens = int(resume["tokens"])
            samples_target = int(resume["samples_taken"])
            self._fast_forward(
                phase_name,
                weights,
                budgets,
                produced,
                retired,
                take,
                samples_target,
                resume["produced"],
                resume["restarts"],
                resume["python_random"],
            )
            _restore_torch_rng(resume["torch_rng"])
            global_step = int(resume["global_step"])
            micro_steps = int(resume["micro_steps"])
            next_eval = float(resume["next_eval"])
            first_loss = resume.get("first_loss")
            best_train_loss = resume.get("best_train_loss")
            samples_taken = samples_target
        optimizer = self._optimizer(_unwrap(model), peak_lr)
        optimizer_path = os.path.join(resume["dir"], "optimizer.pt") if resume else ""
        if resume and os.path.isfile(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device, weights_only=False))
            print("Loaded optimizer state from the resume checkpoint.")
        self._set_lr(optimizer, min(tokens / total_tokens, 1.0) if total_tokens else 0.0, peak_lr)
        model.train()
        if self._checkpointing:
            _unwrap(model).config.use_cache = False

        if task_eval and not resume:
            self.run_evaluation_suite(
                model, tokenizer, f"{phase_name}_start", global_step, update_best=update_best
            )

        started = time.time()
        resume_every = max(1, int(getattr(self, "resume_every_tokens", RESUME_EVERY_TOKENS)))
        next_resume_tokens = (tokens // resume_every + 1) * resume_every
        progress = tqdm(total=total_tokens, initial=tokens, desc=f"[{phase_name}]", unit="tok", unit_scale=True, dynamic_ncols=True)
        try:
            while tokens < total_tokens:
                batch = []
                while len(batch) < self.config.batch_size:
                    drawn = take_mixture_sample(weights, budgets, produced, retired, take)
                    if drawn is None:
                        break
                    if drawn == "retired":
                        continue
                    _index, sample = drawn
                    batch.append(sample)
                    samples_taken += 1
                if not batch:
                    break
                loss, real_tokens = self._forward_backward(model, batch, tokenizer, phase_name)
                if first_loss is None:
                    first_loss = loss
                last_loss = loss
                tokens += real_tokens
                progress.update(real_tokens)
                micro_steps += 1
                if micro_steps % self.config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    lr = self._set_lr(optimizer, tokens / total_tokens, peak_lr)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    elapsed = max(time.time() - started, 1e-6)
                    trained_tokens = tokens - (int(resume["tokens"]) if resume else 0)
                    speed = trained_tokens / elapsed
                    if best_train_loss is None or loss < best_train_loss:
                        best_train_loss = loss
                    self.writer.add_scalar(f"train_{phase_name}/loss", loss, global_step)
                    self.writer.add_scalar(f"train_{phase_name}/lr", lr, global_step)
                    self.writer.add_scalar(f"train_{phase_name}/tokens_per_sec", speed, global_step)
                    for name, value in getattr(self, "_batch_config_metrics", {}).items():
                        self.writer.add_scalar(f"train_{phase_name}/{name}", value, global_step)
                    # One pretrain curve. Side SFT copies keep their own step axis
                    # and must not draw on top of this series.
                    if phase_name == "pretrain":
                        self.writer.add_scalar("train/loss", loss, global_step)
                    progress.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", speed=f"{int(speed)}tok/s")
                    due, suffix = checkpoint_due(
                        tokens,
                        total_tokens,
                        next_eval,
                        every_tokens,
                        percent_step,
                    )
                    if eval_during and due:
                        progress.write(f"\n[eval at {tokens:,} tokens]")
                        checkpoint_name = f"{phase_name}_{suffix}"
                        flat = self.run_evaluation_suite(
                            model, tokenizer, checkpoint_name, global_step, update_best=update_best
                        )
                        if early_stop is not None:
                            plateau = self._early_stop(tokens, flat, early_stop, phase_name)
                        elif phase_name == "pretrain":
                            plateau = self._plateau_reached(tokens, flat)
                        else:
                            plateau = False
                        if on_checkpoint is None:
                            if phase_name in ("pretrain", "bridge"):
                                self._sft_from_base_checkpoint(model, tokenizer, checkpoint_name, global_step)
                        else:
                            self._run_hooks(on_checkpoint, model, tokenizer, checkpoint_name, global_step)
                        if every_tokens > 0:
                            next_eval += every_tokens
                        else:
                            next_eval += percent_step
                        model.train()
                        if plateau:
                            if phase_name == "pretrain":
                                progress.write("\n[pretrain plateau, stopping this phase]")
                            else:
                                progress.write(f"\n[{phase_name} early stop, stopping this phase]")
                            break
                    # After eval, so the saved generator and eval cursor include it.
                    if persist_resume and tokens >= next_resume_tokens:
                        self._write_resume(
                            model,
                            tokenizer,
                            optimizer,
                            {
                                "phase": phase_name,
                                "phase_done": False,
                                "completed_phases": list(self._completed_phases),
                                "tokens": tokens,
                                "samples_taken": samples_taken,
                                "produced": list(produced),
                                "restarts": list(restarts),
                                "global_step": global_step,
                                "micro_steps": micro_steps,
                                "next_eval": next_eval,
                                "first_loss": first_loss,
                                "best_train_loss": best_train_loss,
                                "peak_lr": peak_lr,
                            },
                        )
                        progress.write(f"\n[resume checkpoint at {tokens:,} tokens]")
                        next_resume_tokens = (tokens // resume_every + 1) * resume_every
            if micro_steps % self.config.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self._set_lr(optimizer, min(tokens / total_tokens, 1.0), peak_lr)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if last_loss is not None and (best_train_loss is None or last_loss < best_train_loss):
                    best_train_loss = last_loss
        finally:
            progress.close()

        elapsed = max(time.time() - started, 1e-6)
        resumed_tokens = int(resume["tokens"]) if resume else 0
        trained_tokens = max(tokens - resumed_tokens, 0)
        self.phase_stats[phase_name] = {
            "first_loss": first_loss,
            "last_loss": last_loss,
            "best_train_loss": best_train_loss,
            "tokens": tokens,
            "tokens_per_source": {
                source_key(src): count for src, count in zip(sources, produced)
            },
            "tok_per_sec": trained_tokens / elapsed,
        }
        print(
            f" finished {phase_name}: {tokens:,} tokens, "
            f"loss {first_loss} -> {last_loss}, {trained_tokens / elapsed:,.0f} tok/s"
        )
        self.run_evaluation_suite(
            model,
            tokenizer,
            final_phase_name or f"{phase_name}_end",
            global_step if metric_step is None else metric_step,
            update_best=update_best,
            metric_group=metric_group,
        )
        if record_completion and phase_name not in self._completed_phases:
            self._completed_phases.append(phase_name)
        if persist_resume:
            self._write_resume(
                model,
                tokenizer,
                None,
                {
                    "phase": phase_name,
                    "phase_done": True,
                    "completed_phases": list(self._completed_phases),
                    "tokens": tokens,
                    "samples_taken": samples_taken,
                    "produced": list(produced),
                    "restarts": list(restarts),
                    "global_step": global_step,
                    "micro_steps": micro_steps,
                    "next_eval": next_eval,
                    "first_loss": first_loss,
                    "best_train_loss": best_train_loss,
                    "peak_lr": peak_lr,
                },
            )
        return global_step

    def _active_phases(self) -> list[TrainingPhase]:
        if self.config.phases:
            return list(self.config.phases)
        return phases_from_legacy(self.config)

    def _run_hooks(self, hooks, model, tokenizer, checkpoint_name: str, step: int) -> None:
        if hooks is None:
            return
        if callable(hooks):
            hooks = [hooks]
        for hook in hooks:
            hook(self, model, tokenizer, checkpoint_name, step)

    def _datasets_for_phase(self, phase: TrainingPhase):
        specs = list(phase.datasets)
        budget = phase.token_budget
        if not budget:
            return specs
        total = sum(int(spec.target_tokens) for spec in specs)
        if total == int(budget) or total <= 0:
            return specs
        scaled = []
        remaining = int(budget)
        for index, spec in enumerate(specs):
            if index == len(specs) - 1:
                tokens = remaining
            else:
                tokens = int(round(int(budget) * int(spec.target_tokens) / total))
                remaining -= tokens
            scaled.append(spec.model_copy(update={"target_tokens": max(int(tokens), 1)}))
        return scaled

    def _phase_peak_lr(self, phase: TrainingPhase) -> float:
        if phase.lr is not None:
            return float(phase.lr)
        if phase.lr_scale is not None:
            return float(self.config.lr) * float(phase.lr_scale)
        if phase.fork or phase.name == "sft" or str(phase.name).endswith("_sft"):
            return self._supervised_lr()
        return float(self.config.lr)

    def _use_phase_context(self, phase: TrainingPhase) -> None:
        self._active_eval_datasets = None if phase.eval_datasets is None else list(phase.eval_datasets)
        self._phase_before_batch = _INHERIT if phase.before_batch is None else phase.before_batch

    def _clear_phase_context(self) -> None:
        self._active_eval_datasets = None
        self._phase_before_batch = _INHERIT

    def fork_training_phase(
        self,
        phase: TrainingPhase,
        model,
        tokenizer,
        checkpoint_name: str,
        step: int,
        required: bool = False,
    ) -> None:
        """Train ``phase`` as a copy of ``checkpoint_name`` and leave ``model`` as it was."""
        previous_eval = self._active_eval_datasets
        previous_hook = self._phase_before_batch
        self._use_phase_context(phase)
        try:
            self._sft_from_base_checkpoint(
                model,
                tokenizer,
                checkpoint_name,
                step,
                required=required,
                mix=list(phase.datasets),
                peak_lr=self._phase_peak_lr(phase),
                allow_any_name=True,
                honor_flag=False,
                label_suffix=phase.name,
            )
        finally:
            self._active_eval_datasets = previous_eval
            self._phase_before_batch = previous_hook

    def _done_message(self, name: str) -> str:
        if name == "pretrain":
            return "Pretrain is already complete in the resume checkpoint."
        if name == "bridge":
            return "Bridge is already complete in the resume checkpoint."
        if name == "sft":
            return "Supervised training is already complete in the resume checkpoint."
        return f"{name} is already complete in the resume checkpoint."

    def _ensure_saved(self, model, tokenizer, name: str) -> str:
        directory = os.path.join(self.output_dir, "checkpoints", name)
        if not os.path.isfile(os.path.join(directory, "config.json")):
            self._save_weights(model, tokenizer, directory)
            print(f"Saved {name} weights to {directory}")
        return directory

    def _run_inline_phase(self, phase: TrainingPhase, model, tokenizer, global_step: int, resume):
        if self.config.compile and (phase.name == "sft" or str(phase.name).endswith("_sft")):
            print(
                "SFT stays eager. Variable-length chats crash torch.compile "
                "after it specializes on the first padded width."
            )
        if phase.name in self._completed_phases:
            print(self._done_message(phase.name))
            model = _unwrap(model)
            if phase.on_complete:
                self._ensure_saved(model, tokenizer, phase.name)
                self._run_hooks(phase.on_complete, model, tokenizer, phase.name, global_step)
            return model, global_step

        model = _unwrap(model)
        training_model = model
        if phase.compile and hasattr(torch, "compile") and self.device.type == "cuda":
            print("Compiling model with torch.compile() for fixed-length pretraining.")
            training_model = torch.compile(model)
        elif phase.name == "bridge":
            print("Bridge stays eager so the loop count can change on every batch.")
        peak = self._phase_peak_lr(phase)
        if phase.name == "bridge" and phase.lr_scale == 0.1:
            print(f"Bridge peak lr is 0.1x config.lr ({peak:.3e}).")
        phase_resume = None
        if resume and not resume.get("phase_done") and resume.get("phase") == phase.name:
            phase_resume = resume
        if phase_resume is None and phase.restore_boundary_rng:
            self._restore_phase_boundary_rng()
        # Patience belongs to this phase. A later phase must not inherit a stall
        # counted on an earlier holdout. Resuming this phase keeps its history.
        if phase.early_stop is not None:
            if phase_resume is None and self._plateau_phase not in (None, phase.name):
                self._plateau_history = []
                self._plateau_stalls = 0
            self._plateau_phase = phase.name
        starting = global_step if phase_resume else (0 if phase.reset_step else global_step)
        self._use_phase_context(phase)
        try:
            global_step = self._train_phase(
                phase.name,
                self._datasets_for_phase(phase),
                training_model,
                tokenizer,
                starting,
                peak_lr=peak,
                task_eval=phase.task_eval,
                resume=phase_resume,
                eval_during=phase.eval_during,
                every_tokens=int(phase.checkpoint.every_tokens or 0),
                every_percent=float(phase.checkpoint.every_percent),
                on_checkpoint=list(phase.on_checkpoint),
                early_stop=phase.early_stop,
            )
        finally:
            self._clear_phase_context()
        model = _unwrap(training_model)
        if phase.save:
            directory = os.path.join(self.output_dir, "checkpoints", phase.name)
            self._save_weights(model, tokenizer, directory)
            print(f"Saved {phase.name} weights to {directory}")
        self._run_hooks(phase.on_complete, model, tokenizer, phase.name, global_step)
        return model, global_step

    def _run_fork_phase(self, phase: TrainingPhase, model, tokenizer, global_step: int, resume):
        if self.config.compile and (phase.name == "sft" or str(phase.name).endswith("_sft")):
            print(
                "SFT stays eager. Variable-length chats crash torch.compile "
                "after it specializes on the first padded width."
            )
        if phase.name in self._completed_phases:
            print(self._done_message(phase.name))
            return _unwrap(model), global_step
        if resume and not resume.get("phase_done") and resume.get("phase") == phase.name:
            print("In-progress SFT resume is unused. The final pass reruns the full checkpoint copy.")
        model = _unwrap(model)
        if phase.fork_from in (None, "latest"):
            if any(item.name == "pretrain" and not item.fork for item in self._active_phases()):
                self._ensure_saved(model, tokenizer, "pretrain")
            bridge_dir = os.path.join(self.output_dir, "checkpoints", "bridge")
            base = "bridge" if os.path.isfile(os.path.join(bridge_dir, "config.json")) else "pretrain"
        else:
            base = phase.fork_from
        self.fork_training_phase(phase, model, tokenizer, base, global_step, required=True)
        if not phase.load_fork:
            if phase.name not in self._completed_phases:
                self._completed_phases.append(phase.name)
            return model, global_step
        sft_dir = os.path.join(self.output_dir, "checkpoints", f"{base}_{phase.name}")
        if not os.path.isfile(os.path.join(sft_dir, "config.json")):
            raise SystemExit(f"Final SFT did not write {sft_dir}.")
        model = AutoModelForCausalLM.from_pretrained(sft_dir, dtype=torch.float32, trust_remote_code=True)
        align_tokenizer_ids(model, tokenizer)
        model.to(self.device)
        finished_key = f"{base}_{phase.name}"
        if finished_key in self.phase_stats:
            self.phase_stats[phase.name] = self.phase_stats[finished_key]
        if phase.name not in self._completed_phases:
            self._completed_phases.append(phase.name)
        self._write_resume(
            model,
            tokenizer,
            None,
            {
                "phase": phase.name,
                "phase_done": True,
                "completed_phases": list(self._completed_phases),
                "tokens": 0,
                "samples_taken": 0,
                "produced": [],
                "restarts": [],
                "global_step": global_step,
                "micro_steps": 0,
                "next_eval": float(phase.checkpoint.every_percent),
                "first_loss": None,
                "best_train_loss": (self.phase_stats.get(phase.name) or {}).get("best_train_loss"),
                "peak_lr": self._phase_peak_lr(phase),
            },
        )
        return model, global_step

    def train(self, fresh: bool = True, init_from: str | None = None):
        # Supervised rendering uses this format. Pretrain packing never reads it.
        token = push_chat_format(self.config.chat_format)
        try:
            return self._train_body(fresh=fresh, init_from=init_from)
        finally:
            pop_chat_format(token)

    def _train_body(self, fresh: bool = True, init_from: str | None = None):
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        print(f"Loading tokenizer: {self.config.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Packing splits long documents. Silence the tokenizer's own length warning.
        tokenizer.model_max_length = int(1e9)

        resume = None if fresh else self._read_resume()
        self._completed_phases = []
        if resume:
            self._completed_phases = list(resume.get("completed_phases") or [])
            if resume.get("best_loss") is not None:
                self.best_loss = resume["best_loss"]
            if resume.get("best_behavior") is not None:
                self.best_behavior = resume["best_behavior"]
            finished = " finished" if resume.get("phase_done") else ""
            print(
                f"Resuming from {resume['dir']} "
                f"({resume.get('phase')}{finished}, {int(resume.get('tokens') or 0):,} tokens)."
            )
            self._plateau_history = [list(item) for item in resume.get("plateau_history") or []]
            self._plateau_stalls = int(resume.get("plateau_stalls") or 0)
            if self._plateau_history:
                self._plateau_phase = resume.get("phase")
            model = self._load_model_weights(resume["dir"], tokenizer)
        # A train() argument overrides the config field. Either one is used
        # before pretrain or supervised training starts.
        elif (source := init_from if init_from is not None else self.config.init_from):
            source = _resolve_init_from(source)
            print(f"Loading initial weights from {source} before training.")
            model = self._load_model_weights(source, tokenizer)
        else:
            print("Building a new model from the config.")
            model = self._build_model(vocab_size=len(tokenizer))
        align_tokenizer_ids(model, tokenizer)
        model.to(self.device)
        total_params = self._print_model_summary(model)
        self._checkpointing = total_params >= 50_000_000 and hasattr(model, "gradient_checkpointing_enable")
        if self._checkpointing:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
            model.config.use_cache = False
            print("Gradient checkpointing enabled.")

        print(f"Run directory: {self.output_dir}")
        print(f"TensorBoard: {self.writer.log_dir}")
        global_step = int(resume["global_step"]) if resume else 0
        for phase in self._active_phases():
            if phase.fork:
                model, global_step = self._run_fork_phase(phase, model, tokenizer, global_step, resume)
            else:
                model, global_step = self._run_inline_phase(phase, model, tokenizer, global_step, resume)

        self._save_chat = any(self._phase_uses_chat(name) for name in self._completed_phases)
        self._save_weights(model, tokenizer, self.output_dir)
        with open(os.path.join(self.output_dir, "phase_stats.json"), "w", encoding="utf-8") as handle:
            json.dump(self.phase_stats, handle, indent=2)
        print(f"Saved final weights to {self.output_dir}")
        self.writer.flush()
        self.writer.close()
        return self.phase_stats
