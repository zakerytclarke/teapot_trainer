"""Sequential pretrain then SFT.

The public surface is still ``TeapotTrainer(TrainingConfig).train(fresh=True)``.
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

from .config import TrainingConfig
from .datasets import DatasetRegistry, DatasetSourceConfig, iterate_encoded
from .evals import evaluate_source, selection_score, source_key, task_kind


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# A resume re-reads the phase from the first sample so the shuffle buffer is
# rebuilt the same way. This interval only bounds how much optimization is redone.
RESUME_EVERY_TOKENS = 100_000_000


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

        tokenizer_short = self.config.tokenizer.split("/")[-1]
        try:
            safe_dict = self.config.model_dump(exclude_defaults=True, mode="json")
            run_hash = hashlib.md5(json.dumps(safe_dict, default=str).encode()).hexdigest()[:6]
        except Exception:
            run_hash = hashlib.md5(repr(self.config).encode()).hexdigest()[:6]
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
                src = DatasetSourceConfig(
                    name=source.name,
                    subset=source.subset,
                    split=source.split,
                    text_column=source.text_column,
                    is_sft=source.is_sft,
                    preprocess_fn=source.preprocess_fn,
                    chat_template_fn=source.chat_template_fn,
                    token_loss_weight_fn=source.token_loss_weight_fn,
                    target_tokens=spec.target_tokens,
                    weight=spec.weight,
                )
            sources.append(src)
        return sources

    def _forward_backward(self, model, batch, tokenizer):
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
        if use_weights:
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            kwargs["labels"] = label_tensor
        if any(row.count(0) for row in mask):
            kwargs["attention_mask"] = torch.tensor(mask, dtype=torch.long, device=self.device)
        try:
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(**kwargs)
                    loss = (
                        weighted_cross_entropy(output.logits, label_tensor, weight_tensor)
                        if use_weights
                        else output.loss
                    )
            else:
                output = model(**kwargs)
                loss = (
                    weighted_cross_entropy(output.logits, label_tensor, weight_tensor)
                    if use_weights
                    else output.loss
                )
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
        return float(loss.detach().float().item()), real_tokens

    def _save_weights(self, model, tokenizer, directory: str):
        os.makedirs(directory, exist_ok=True)
        unwrapped = _unwrap(model)
        unwrapped.save_pretrained(directory)
        tokenizer.save_pretrained(directory)

    def _log_metrics(self, phase: str, step: int, payload: dict):
        path = os.path.join(self.output_dir, "metrics.jsonl")
        record = {"phase": phase, "step": step, "metrics": payload}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not key.endswith(tuple(f"/support_{suffix}" for suffix in ("x",))):
                if "support" in key.split("/")[-1]:
                    continue
                self.writer.add_scalar(f"eval/{key}", value, step)

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
                if index >= 16:
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

    def run_evaluation_suite(self, model, tokenizer, phase_name: str, global_step: int, task_eval: bool):
        unwrapped = _unwrap(model)
        flat: dict[str, float] = {}
        print("\n" + "=" * 78)
        print(f" EVAL  {phase_name}  step {global_step}")
        print("=" * 78)
        if not self.config.eval_datasets:
            print(" No eval datasets configured.")
            print("=" * 78 + "\n")
            return flat

        for spec in self.config.eval_datasets:
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
            if not task_eval:
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
        if behavior is not None:
            print(f" behavior_score: {behavior:.3f}")
        if mean_loss is not None:
            print(f" mean_eval_loss: {mean_loss:.4f}")
        print("=" * 78 + "\n")
        self._log_metrics(phase_name, global_step, flat)

        improved = False
        if behavior is not None and (self.best_behavior is None or behavior >= self.best_behavior):
            self.best_behavior = behavior
            improved = True
        elif behavior is None and mean_loss is not None and (self.best_loss is None or mean_loss <= self.best_loss):
            self.best_loss = mean_loss
            improved = True
        if improved and flat:
            best_dir = os.path.join(self.output_dir, "checkpoints", "best")
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
        self._save_weights(model, tokenizer, temporary)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(temporary, "optimizer.pt"))
        payload = dict(state)
        payload.pop("dir", None)
        payload["python_random"] = python_random
        payload["torch_rng"] = torch_rng
        payload["best_loss"] = self.best_loss
        payload["best_behavior"] = self.best_behavior
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

    def _train_phase(self, phase_name: str, dataset_specs, model, tokenizer, starting_step: int, peak_lr: float, task_eval: bool, resume: dict | None = None):
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
        next_eval = self.config.eval_every_percent
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
            self.run_evaluation_suite(model, tokenizer, f"{phase_name}_start", global_step, task_eval=True)

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
                loss, real_tokens = self._forward_backward(model, batch, tokenizer)
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
                    self.writer.add_scalar(f"train_{phase_name}/loss", loss, global_step)
                    self.writer.add_scalar(f"train_{phase_name}/lr", lr, global_step)
                    self.writer.add_scalar(f"train_{phase_name}/tokens_per_sec", speed, global_step)
                    progress.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", speed=f"{int(speed)}tok/s")
                    pct = 100.0 * tokens / total_tokens
                    if pct >= next_eval and pct < 100.0:
                        progress.write(f"\n[eval at {pct:.1f}%]")
                        self.run_evaluation_suite(
                            model, tokenizer, f"{phase_name}_{int(pct)}pct", global_step, task_eval=task_eval
                        )
                        next_eval += self.config.eval_every_percent
                        model.train()
                    # After eval, so the saved generator and eval cursor include it.
                    if tokens >= next_resume_tokens:
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
        finally:
            progress.close()

        elapsed = max(time.time() - started, 1e-6)
        resumed_tokens = int(resume["tokens"]) if resume else 0
        trained_tokens = max(tokens - resumed_tokens, 0)
        self.phase_stats[phase_name] = {
            "first_loss": first_loss,
            "last_loss": last_loss,
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
        self.run_evaluation_suite(model, tokenizer, f"{phase_name}_end", global_step, task_eval=task_eval)
        if phase_name not in self._completed_phases:
            self._completed_phases.append(phase_name)
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
                "peak_lr": peak_lr,
            },
        )
        return global_step

    def train(self, fresh: bool = True, init_from: str | None = None):
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
            model = AutoModelForCausalLM.from_pretrained(resume["dir"], dtype=torch.float32, trust_remote_code=True)
            mismatches = architecture_mismatches(model.config, self.m_cfg)
            if mismatches:
                detail = "\n  ".join(mismatches)
                raise SystemExit(f"resume checkpoint does not match the model config:\n  {detail}")
        # A train() argument overrides the config field. Either one is used
        # before pretrain or supervised training starts.
        elif (source := init_from if init_from is not None else self.config.init_from):
            source = _resolve_init_from(source)
            print(f"Loading initial weights from {source} before training.")
            model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, trust_remote_code=True)
            mismatches = architecture_mismatches(model.config, self.m_cfg)
            if mismatches:
                detail = "\n  ".join(mismatches)
                raise SystemExit(f"init_from does not match the model config:\n  {detail}")
            embedding = model.get_input_embeddings()
            rows = None if embedding is None else embedding.weight.shape[0]
            if rows is not None and rows < len(tokenizer):
                print(f"Tokenizer is longer than the checkpoint embeddings ({len(tokenizer)} > {rows}). Resizing.")
                model.resize_token_embeddings(len(tokenizer))
            elif rows is not None and rows != len(tokenizer):
                print(f"Checkpoint embeddings have {rows} rows and the tokenizer has {len(tokenizer)}. Leaving the matrix as stored.")
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
        pretrain = list(self.config.pretrain_datasets or [])
        sft = list(self.config.sft_datasets or [])
        if pretrain and "pretrain" not in self._completed_phases:
            # Pretrain sequences are always seq_len, so one compiled graph is stable.
            # SFT chats are not, and compile then asserts on a later padded width.
            training_model = model
            if self.config.compile and hasattr(torch, "compile") and self.device.type == "cuda":
                print("Compiling model with torch.compile() for fixed-length pretraining.")
                training_model = torch.compile(model)
            pretrain_resume = resume if resume and not resume.get("phase_done") and resume.get("phase") == "pretrain" else None
            global_step = self._train_phase(
                "pretrain",
                pretrain,
                training_model,
                tokenizer,
                global_step if pretrain_resume else 0,
                peak_lr=self.config.lr,
                task_eval=False,
                resume=pretrain_resume,
            )
            model = _unwrap(training_model)
            pretrain_dir = os.path.join(self.output_dir, "checkpoints", "pretrain")
            self._save_weights(model, tokenizer, pretrain_dir)
            print(f"Saved pretrain weights to {pretrain_dir}")
        elif resume and "pretrain" in self._completed_phases:
            print("Pretrain is already complete in the resume checkpoint.")
            model = _unwrap(model)
        if sft:
            if self.config.compile:
                print(
                    "SFT stays eager. Variable-length chats crash torch.compile "
                    "after it specializes on the first padded width."
                )
            sft_lr = self.config.lr * 0.2 if pretrain else self.config.lr
            if pretrain:
                print(f"SFT peak lr is 0.2x config.lr ({sft_lr:.3e}) because this run already pretrained.")
            if "sft" in self._completed_phases:
                print("Supervised training is already complete in the resume checkpoint.")
            else:
                sft_resume = resume if resume and not resume.get("phase_done") and resume.get("phase") == "sft" else None
                if sft_resume is None:
                    self._restore_phase_boundary_rng()
                global_step = self._train_phase(
                    "sft",
                    sft,
                    model,
                    tokenizer,
                    global_step,
                    peak_lr=sft_lr,
                    task_eval=True,
                    resume=sft_resume,
                )

        self._save_weights(model, tokenizer, self.output_dir)
        with open(os.path.join(self.output_dir, "phase_stats.json"), "w", encoding="utf-8") as handle:
            json.dump(self.phase_stats, handle, indent=2)
        print(f"Saved final weights to {self.output_dir}")
        self.writer.flush()
        self.writer.close()
        return self.phase_stats
