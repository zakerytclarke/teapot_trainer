"""Stage trainer: padded or windowed batches, bf16, cosine schedule, HF checkpoints."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Stage, TrainConfig
from .evals import EvalOutput
from .loss import chunked_cross_entropy
from .metrics import MetricLog, format_eval, human
from .models import build_model, enable_checkpointing, param_count


def cosine_lr(step: int, total: int, peak: float, warmup: int, min_frac: float) -> float:
    """Step is the number of optimizer steps already finished."""
    if total <= 1:
        return peak
    warmup = max(0, min(int(warmup), total - 1))
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_frac + (1.0 - min_frac) * cos)


def adamw_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 2D weights decay, including embeddings. Norms and biases do not.
        if param.ndim < 2 or "norm" in name:
            nodecay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if nodecay:
        groups.append({"params": nodecay, "weight_decay": 0.0})
    return groups


def max_positions(model) -> int:
    cfg = model.config
    for attr in ("max_position_embeddings", "n_positions"):
        value = getattr(cfg, attr, None)
        if value:
            return int(value)
    return 2048


def estimate_step_tokens(batchers, micro_batch: int, grad_accum: int) -> int:
    """Supervised tokens in one optimizer step, used only to size the LR horizon."""
    per_source = []
    for batcher in batchers:
        labels = getattr(batcher, "labels", None)
        if labels is not None and len(labels):
            per_row = float((labels[:, 1:] != -100).sum()) / len(labels)
            per_source.append(per_row * micro_batch)
        else:
            seq_len = int(getattr(batcher, "seq_len"))
            per_source.append((seq_len - 1) * micro_batch)
    mean = sum(per_source) / max(len(per_source), 1)
    return max(1, int(mean * grad_accum))


def selection_value(metrics: dict[str, float]) -> tuple[float | None, bool]:
    """Loss is minimized. Any other metric mix is maximized."""
    loss_keys = [key for key in metrics if key.endswith("loss")]
    preferred = [key for key in loss_keys if "val" in key]
    if preferred:
        return metrics[preferred[0]], False
    if loss_keys:
        return metrics[loss_keys[0]], False
    if metrics:
        return sum(metrics.values()) / len(metrics), True
    return None, False


class TeapotTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(config.output_dir)
        self.log = MetricLog(self.output_dir)
        self.ckpt_root = self.output_dir / "checkpoints"
        self._pending_batcher_state = None

    def train(self, resume: str | None = None) -> dict:
        cfg = self.config
        self._seed(cfg.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer,
            trust_remote_code=cfg.trust_remote_code,
            local_files_only=cfg.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if cfg.init_from:
            print(f"loading init weights: {cfg.init_from}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.init_from,
                torch_dtype=torch.float32,
                trust_remote_code=cfg.trust_remote_code,
                local_files_only=cfg.local_files_only,
            )
        else:
            model = build_model(cfg.model, len(tokenizer))
        model.to(self.device)
        if getattr(model.config, "use_cache", None) is not None:
            model.config.use_cache = False
        model.train()
        if cfg.gradient_checkpointing and self.device.type == "cuda":
            ok = enable_checkpointing(model)
            print(f"gradient checkpointing: {'on' if ok else 'unsupported'}", flush=True)

        n_params = param_count(model)
        print(
            f"model: {n_params / 1e6:.2f}M params  device={self.device}  vocab={len(tokenizer)}",
            flush=True,
        )
        opt = torch.optim.AdamW(
            adamw_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )
        self.log.write_config({**cfg.public_dict(), "params": n_params, "vocab": len(tokenizer)})

        state = {
            "step": 0,
            "stage_step": 0,
            "stage_index": 0,
            "stage_tokens": 0,
            "best": None,
            "best_higher": False,
            "patience": 0,
            "elapsed_s": 0.0,
            "np_rng": None,
        }
        if resume:
            state = self._load_resume(model, opt, resume, state)

        summary = {"params": n_params, "best": None}
        for index in range(state["stage_index"], len(cfg.stages)):
            summary = self._run_stage(index, cfg.stages[index], model, tokenizer, opt, state, n_params)
            if index < len(cfg.stages) - 1:
                state["stage_index"] = index + 1
                state["stage_tokens"] = 0
                state["stage_step"] = 0
                state["patience"] = 0
                state["np_rng"] = None
                self._pending_batcher_state = None
        print(
            f"done. last checkpoint: {self.ckpt_root / 'last'}\n"
            f"best checkpoint: {self.ckpt_root / 'best'}\n"
            f"metrics: {self.log.report_path}",
            flush=True,
        )
        return summary

    def _seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _run_stage(self, index: int, stage: Stage, model, tokenizer, opt, state: dict, n_params: int) -> dict:
        cfg = self.config
        seq_len = stage.seq_len or cfg.seq_len or max_positions(model)
        limit = max_positions(model)
        if seq_len > limit:
            raise ValueError(f"seq_len {seq_len} exceeds model max positions {limit}")
        peak_lr = cfg.lr if stage.lr is None else stage.lr
        warmup = cfg.warmup_steps if stage.warmup_steps is None else stage.warmup_steps

        weights = [float(getattr(src, "weight", 1.0)) for src in stage.sources]
        if any(weight <= 0 for weight in weights):
            raise ValueError(f"stage {stage.name}: weights must be positive")
        probs = np.asarray(weights, dtype=np.float64)
        probs /= probs.sum()

        resuming = self._pending_batcher_state is not None and int(state["stage_index"]) == index
        batchers = []
        for src_i, src in enumerate(stage.sources):
            batcher = src.build(tokenizer, seq_len, stage.target_tokens, seed=cfg.seed + src_i)
            batchers.append(batcher)
            extra = ""
            dropped = getattr(batcher, "dropped", 0)
            if dropped:
                extra = f"  (dropped {dropped} overlong)"
            print(f"  data  {batcher.describe()}{extra}", flush=True)
        if resuming:
            for batcher, saved in zip(batchers, self._pending_batcher_state):
                batcher.load_state_dict(saved)
            self._pending_batcher_state = None

        rng = np.random.default_rng(cfg.seed + 1000 + index)
        if resuming and state.get("np_rng") is not None:
            rng.bit_generator.state = state["np_rng"]

        step_tokens = estimate_step_tokens(batchers, cfg.micro_batch, cfg.grad_accum)
        horizon = max(1, math.ceil(stage.target_tokens / step_tokens))
        print(
            f"\nstage {stage.name}: target {human(stage.target_tokens)} supervised tokens  "
            f"seq={seq_len}  micro={cfg.micro_batch}  accum={cfg.grad_accum}  "
            f"~{human(step_tokens)} tok/step  ~{horizon} steps  lr={peak_lr:.3e}",
            flush=True,
        )

        tokens = int(state["stage_tokens"]) if int(state["stage_index"]) == index else 0
        stage_step = int(state["stage_step"]) if int(state["stage_index"]) == index else 0
        step = int(state["step"])
        best = state["best"]
        best_higher = bool(state["best_higher"])
        patience = int(state["patience"]) if int(state["stage_index"]) == index else 0
        elapsed = float(state["elapsed_s"])
        window_loss = 0.0
        window_tokens = 0
        ema = None
        every = cfg.eval_every_tokens or max(stage.target_tokens // 5, 1)
        next_eval = 0 if (cfg.eval_at_start and tokens == 0) else tokens + every
        last_eval_tokens = -1

        while tokens < stage.target_tokens:
            if tokens >= next_eval:
                train_loss = None if window_tokens == 0 else window_loss / window_tokens
                lr_now = cosine_lr(stage_step, horizon, peak_lr, warmup, cfg.min_lr_frac)
                best, best_higher, patience, stop = self._evaluate(
                    stage, model, tokenizer, seq_len, step, tokens, train_loss,
                    ema, lr_now, elapsed, best, best_higher, patience, n_params,
                    opt, batchers, rng, index, stage_step,
                )
                window_loss = 0.0
                window_tokens = 0
                last_eval_tokens = tokens
                next_eval = tokens + every
                if stop:
                    break

            lr = cosine_lr(stage_step, horizon, peak_lr, warmup, cfg.min_lr_frac)
            for group in opt.param_groups:
                group["lr"] = lr

            opt.zero_grad(set_to_none=True)
            micro_losses = []
            micro_sup = 0
            started = time.time()
            try:
                for _ in range(cfg.grad_accum):
                    pick = int(rng.choice(len(batchers), p=probs))
                    ids, labels, mask = batchers[pick].take(cfg.micro_batch)
                    # Count on CPU. A GPU .item() here would sync every microbatch.
                    micro_sup += int((labels[:, 1:] != -100).sum())
                    batch = self._to_device(ids, labels, mask)
                    micro_losses.append(self._forward_backward(model, batch, cfg.grad_accum))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    raise SystemExit(
                        "CUDA out of memory. Lower micro_batch or seq_len. "
                        f"Resume with --resume {self.ckpt_root / 'last'}"
                    ) from exc
                raise
            if micro_sup <= 0:
                raise SystemExit("a step had no supervised tokens; check labels and seq_len")

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            dt = max(time.time() - started, 1e-6)
            mean_loss = sum(micro_losses) / len(micro_losses)
            if not math.isfinite(mean_loss):
                raise SystemExit(f"non-finite loss at step {step + 1}: {mean_loss}")

            tok_s = micro_sup / dt
            ema = tok_s if ema is None else 0.9 * ema + 0.1 * tok_s
            tokens += micro_sup
            step += 1
            stage_step += 1
            elapsed += dt
            window_loss += mean_loss * micro_sup
            window_tokens += micro_sup
            self.log.append_step(
                {
                    "stage": stage.name,
                    "step": step,
                    "tokens": tokens,
                    "loss": mean_loss,
                    "lr": lr,
                    "tok_s": tok_s,
                }
            )
            if cfg.log_every and stage_step % cfg.log_every == 0:
                print(
                    f"[{stage.name}] step {stage_step}/{horizon}  "
                    f"tokens {human(tokens)}/{human(stage.target_tokens)}  "
                    f"loss {mean_loss:.4f}  {ema:,.0f} tok/s  lr {lr:.2e}",
                    flush=True,
                )

        if last_eval_tokens != tokens:
            train_loss = None if window_tokens == 0 else window_loss / window_tokens
            lr_now = cosine_lr(stage_step, horizon, peak_lr, warmup, cfg.min_lr_frac)
            best, best_higher, patience, _stop = self._evaluate(
                stage, model, tokenizer, seq_len, step, tokens, train_loss,
                ema, lr_now, elapsed, best, best_higher, patience, n_params,
                opt, batchers, rng, index, stage_step, final=True,
            )
        state.update(
            step=step,
            stage_step=stage_step,
            stage_index=index,
            stage_tokens=tokens,
            best=best,
            best_higher=best_higher,
            patience=patience,
            elapsed_s=elapsed,
            np_rng=rng.bit_generator.state,
        )
        return {"params": n_params, "best": best, "tokens": tokens, "step": step}

    def _to_device(self, ids, labels, mask):
        # An all-ones mask disables the flash attention path. Pass one only
        # when some positions are padding.
        batch = {
            "input_ids": torch.from_numpy(np.ascontiguousarray(ids)).to(self.device).long(),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)).to(self.device).long(),
        }
        if int(mask.min()) == 0:
            batch["attention_mask"] = torch.from_numpy(np.ascontiguousarray(mask)).to(self.device).long()
        return batch

    def _forward_backward(self, model, batch, grad_accum: int) -> float:
        autocast = self.device.type == "cuda"
        kwargs = {"input_ids": batch["input_ids"], "use_cache": False}
        if "attention_mask" in batch:
            kwargs["attention_mask"] = batch["attention_mask"]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(**kwargs).logits
            loss = chunked_cross_entropy(logits, batch["labels"])
        (loss / grad_accum).backward()
        return float(loss.detach().item())

    def _evaluate(
        self, stage, model, tokenizer, seq_len, step, tokens, train_loss,
        tok_s, lr, elapsed, best, best_higher, patience, n_params,
        opt, batchers, rng, stage_index, stage_step, final: bool = False,
    ):
        metrics: dict[str, float] = {}
        samples: list[tuple[str, str]] = []
        was_training = model.training
        was_ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
        if was_ckpt and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        try:
            for ev in stage.evals:
                output: EvalOutput = ev.run(model, tokenizer, self.device, seq_len)
                metrics.update(output.metrics)
                samples.extend(output.samples)
        finally:
            if was_ckpt:
                enable_checkpointing(model)
            if was_training:
                model.train()
            if getattr(model.config, "use_cache", None) is not None:
                model.config.use_cache = False
        text = format_eval(
            stage=stage.name + (" final" if final else ""),
            step=step,
            tokens=tokens,
            target=stage.target_tokens,
            train_loss=train_loss,
            metrics=metrics,
            samples=samples,
            tok_s=tok_s,
            lr=lr,
            elapsed_s=elapsed,
        )
        print(text, flush=True)
        self.log.append_eval(
            {
                "stage": stage.name,
                "step": step,
                "tokens": tokens,
                "train_loss": train_loss,
                "lr": lr,
                "tok_s": tok_s,
                "metrics": metrics,
                "final": final,
            },
            text,
        )

        value, higher = selection_value(metrics)
        improved = False
        if value is not None:
            if best is None:
                improved = True
            elif higher and value > best + 1e-4:
                improved = True
            elif (not higher) and value < best - 1e-4:
                improved = True
        if improved:
            best = value
            best_higher = higher
            patience = 0
            self._save_hf(
                model,
                tokenizer,
                self.ckpt_root / "best",
                {
                    "stage": stage.name,
                    "step": step,
                    "tokens": tokens,
                    "metric": value,
                    "higher_is_better": higher,
                    "metrics": metrics,
                    "params": n_params,
                },
            )
            print(f"  saved best ({value:.4f}) -> {self.ckpt_root / 'best'}", flush=True)
        elif value is not None:
            patience += 1

        self._save_hf(
            model,
            tokenizer,
            self.ckpt_root / "last",
            {"stage": stage.name, "step": step, "tokens": tokens, "metrics": metrics, "params": n_params},
        )
        self._save_resume(
            model, opt, self.ckpt_root / "last" / "resume.pt",
            step, stage_step, stage_index, tokens, best, best_higher, patience, elapsed, batchers, rng,
        )
        stop = self.config.patience_evals > 0 and patience >= self.config.patience_evals and tokens > 0
        if stop:
            print(f"early stop: {patience} evals without improvement on {stage.name}", flush=True)
        return best, best_higher, patience, stop

    def _save_hf(self, model, tokenizer, path: Path, extra: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            key: value.detach().to(dtype=torch.bfloat16).cpu()
            for key, value in model.state_dict().items()
        }
        model.save_pretrained(path, state_dict=state, safe_serialization=True)
        tokenizer.save_pretrained(path)
        (path / "extra.json").write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")

    def _save_resume(
        self, model, opt, path, step, stage_step, stage_index, tokens,
        best, best_higher, patience, elapsed, batchers, rng,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "opt": opt.state_dict(),
            "step": step,
            "stage_step": stage_step,
            "stage_index": stage_index,
            "stage_tokens": tokens,
            "best": best,
            "best_higher": best_higher,
            "patience": patience,
            "elapsed_s": elapsed,
            "batchers": [batcher.state_dict() for batcher in batchers],
            "np_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }
        tmp = path.with_suffix(".tmp")
        torch.save(blob, tmp)
        tmp.replace(path)

    def _load_resume(self, model, opt, resume: str, state: dict) -> dict:
        path = Path(resume)
        if path.is_dir():
            path = path / "resume.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        torch.set_rng_state(blob["torch_rng"])
        state.update(
            step=int(blob["step"]),
            stage_step=int(blob.get("stage_step", blob["step"])),
            stage_index=int(blob["stage_index"]),
            stage_tokens=int(blob["stage_tokens"]),
            best=blob["best"],
            best_higher=bool(blob["best_higher"]),
            patience=int(blob["patience"]),
            elapsed_s=float(blob["elapsed_s"]),
            np_rng=blob.get("np_rng"),
        )
        self._pending_batcher_state = blob.get("batchers")
        print(
            f"resumed {path}: step={state['step']} stage={state['stage_index']} "
            f"tokens={state['stage_tokens']}",
            flush=True,
        )
        return state
