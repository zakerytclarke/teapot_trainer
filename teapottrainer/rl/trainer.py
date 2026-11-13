"""Config-driven RL trainer.

Public surface: ``RlTrainer(RlConfig).train(fresh=True)``.

Flow per episode:
1. pick an ``RlDataset`` by weight
2. ``sample_fn(state, rng)`` → messages + meta + Python tool fns
3. roll out tool-using conversation
4. ``grade_fn(conversation_json, meta)`` → numeric reward
5. GRPO update over the group of rollouts for that sample

``eval_datasets`` uses the same ``DatasetSpec`` / ``evaluate_source`` path as
``TeapotTrainer``: scored before RL, at checkpoint intervals, and after RL.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import DatasetSpec
from ..datasets import DatasetSourceConfig
from ..evals import evaluate_source, selection_score, source_key, task_kind
from ..rl_config import RlConfig
from ..tool_formats import get_tool_format
from .algorithms.grpo import build_batch_tensors, group_advantages, grpo_loss
from .datasets import RlDataset, grade_episode, prepare_dataset, sample_episode
from .rollouts import Trajectory, rollout_episode

torch.set_float32_matmul_precision("high")


def _run_hash(config: RlConfig) -> str:
    payload = config.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def _select_dataset(datasets: list[RlDataset], rng: random.Random) -> RlDataset:
    weights = [max(float(ds.weight), 0.0) for ds in datasets]
    if sum(weights) <= 0:
        return datasets[0]
    return rng.choices(datasets, weights=weights, k=1)[0]


def _materialize_source(spec: DatasetSpec) -> DatasetSourceConfig:
    source = spec.source
    if isinstance(source, str):
        return DatasetSourceConfig(name=source)
    if isinstance(source, DatasetSourceConfig):
        return source
    raise TypeError(f"Unsupported eval dataset source type: {type(source)!r}")


class RlTrainer:
    """Reinforcement learning trainer for tool-using language models."""

    def __init__(self, config: RlConfig):
        self.config = config
        self.tool_format = get_tool_format(config.tool_format)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_name = f"rl-{Path(config.init_from).name}-{_run_hash(config)}"
        base = Path(config.logging_dir or "./runs")
        self.run_dir = base / run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.rollouts_path = Path(config.rollouts_path) if config.rollouts_path else self.run_dir / "rollouts.json"
        self.rollouts: list[dict[str, Any]] = []
        self.best_behavior: float | None = None
        self.best_loss: float | None = None

    def train(self, fresh: bool = True) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_rollouts:
            self.rollouts_path.parent.mkdir(parents=True, exist_ok=True)
            if fresh and self.rollouts_path.exists():
                self.rollouts_path.unlink()
            self.rollouts = []
        (self.run_dir / "rl_config.json").write_text(
            json.dumps(self.config.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        rng = random.Random(self.config.seed)

        # Initialize every RL dataset once (downloads, indexes, …).
        dataset_state = {ds.name: prepare_dataset(ds) for ds in self.config.datasets}

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name(), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.config.init_from,
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device)
        model.train()

        ref_model = None
        if float(self.config.kl_coef) > 0:
            ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.init_from,
                trust_remote_code=True,
                dtype=dtype,
            ).to(self.device)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad_(False)

        optimizer = AdamW(
            model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

        start_episodes = 0
        resume_path = self.ckpt_dir / "resume"
        if not fresh and resume_path.exists():
            start_episodes = self._load_resume(model, optimizer, resume_path)
            if self.config.save_rollouts and self.rollouts_path.exists():
                try:
                    self.rollouts = json.loads(self.rollouts_path.read_text(encoding="utf-8"))
                    if not isinstance(self.rollouts, list):
                        self.rollouts = []
                except json.JSONDecodeError:
                    self.rollouts = []

        writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        episodes_done = start_episodes
        best_score = float("-inf")
        pbar = tqdm(total=int(self.config.max_train_episodes), initial=episodes_done, desc="rl")

        # Same idea as TeapotTrainer: score held-out eval sets before training.
        if start_episodes == 0:
            self.run_evaluation_suite(model, tokenizer, tag="before_rl", episode=0, writer=writer)

        while episodes_done < int(self.config.max_train_episodes):
            dataset = _select_dataset(self.config.datasets, rng)
            state = dataset_state[dataset.name]
            sample = sample_episode(dataset, state, rng)

            trajectories: list[Trajectory] = []
            for group_index in range(int(self.config.group_size)):
                self._run_before_batch(model, phase="rl")
                traj = rollout_episode(
                    model,
                    tokenizer,
                    sample,
                    self.tool_format,
                    max_turns=self.config.max_turns,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                reward, metrics = grade_episode(dataset, traj.messages, traj.meta)
                traj.reward = reward
                traj.metrics = metrics
                trajectories.append(traj)
                self._record_rollout(
                    split="train",
                    episode=episodes_done + 1,
                    dataset_name=dataset.name,
                    sample=sample,
                    traj=traj,
                    group_index=group_index,
                )

            rewards = [t.reward for t in trajectories]
            advantages = group_advantages(rewards)
            update_metrics = self._grpo_update(
                model,
                ref_model,
                optimizer,
                tokenizer,
                trajectories,
                advantages,
            )

            episodes_done += 1
            pbar.update(1)
            mean_reward = sum(rewards) / max(len(rewards), 1)
            mean_acc = sum(t.metrics.get("accuracy", t.metrics.get("score", 0.0)) for t in trajectories) / max(
                len(trajectories), 1
            )
            row = {
                "episode": episodes_done,
                "dataset": dataset.name,
                "sample_id": sample.get("sample_id"),
                "reward_mean": mean_reward,
                "accuracy_mean": mean_acc,
                "reward_group": rewards,
                **{f"train/{k}": v for k, v in update_metrics.items()},
            }
            self._append_metrics(row)
            writer.add_scalar("rl/reward_mean", mean_reward, episodes_done)
            writer.add_scalar("rl/accuracy_mean", mean_acc, episodes_done)
            writer.add_scalar("rl/loss", update_metrics.get("loss", 0.0), episodes_done)

            if episodes_done % int(self.config.eval_every_episodes) == 0:
                # RL-env quick eval (same training datasets, greedy).
                env_stats = self.evaluate_rl_envs(model, tokenizer, dataset_state, rng)
                self._append_metrics({"episode": episodes_done, "split": "rl_env_eval", **env_stats})
                writer.add_scalar("rl_env_eval/accuracy", env_stats.get("accuracy", 0.0), episodes_done)
                writer.add_scalar("rl_env_eval/reward", env_stats.get("reward", 0.0), episodes_done)
                if env_stats.get("reward", float("-inf")) >= best_score:
                    best_score = float(env_stats["reward"])
                    self._save_checkpoint(model, optimizer, episodes_done, self.ckpt_dir / "best")

                # Held-out DatasetSpec suite (same as pretrain/SFT trainer).
                self.run_evaluation_suite(
                    model,
                    tokenizer,
                    tag=f"rl_{episodes_done}",
                    episode=episodes_done,
                    writer=writer,
                )
                self._save_checkpoint(model, optimizer, episodes_done, self.ckpt_dir / f"eval_{episodes_done}")

            if episodes_done % int(self.config.resume_every_episodes) == 0:
                self._save_checkpoint(model, optimizer, episodes_done, resume_path)

        pbar.close()
        env_stats = self.evaluate_rl_envs(model, tokenizer, dataset_state, rng)
        self._append_metrics({"episode": episodes_done, "split": "final_rl_env_eval", **env_stats})
        self.run_evaluation_suite(model, tokenizer, tag="after_rl", episode=episodes_done, writer=writer)
        self._save_checkpoint(model, optimizer, episodes_done, resume_path)
        final_dir = self.ckpt_dir / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        self._flush_rollouts()
        writer.close()
        return self.run_dir

    def run_evaluation_suite(
        self,
        model,
        tokenizer,
        *,
        tag: str,
        episode: int,
        writer: SummaryWriter | None = None,
    ) -> dict[str, float]:
        """Score ``eval_datasets`` with the same helpers as ``TeapotTrainer``."""
        flat: dict[str, float] = {}
        print("\n" + "=" * 78)
        print(f" RL EVAL  {tag}  episode {episode}")
        print("=" * 78)
        if not self.config.eval_datasets:
            print(" No eval datasets configured.")
            print("=" * 78 + "\n")
            self._append_metrics({"episode": episode, "split": f"eval_{tag}", "empty": 1.0})
            return flat

        was_training = model.training
        model.eval()
        seq_len = int(getattr(getattr(model, "config", None), "max_position_embeddings", 2048) or 2048)

        for spec in self.config.eval_datasets:
            src = _materialize_source(spec)
            kind = task_kind(src)
            key = source_key(src)
            print(f" scoring {key} ({kind}) on up to {self.config.eval_example_cap} examples...")
            result = evaluate_source(
                model,
                tokenizer,
                src,
                device=self.device,
                seq_len=seq_len,
                max_examples=int(self.config.eval_example_cap),
                seed=self.config.seed,
                holdout_ratio=float(self.config.holdout_ratio),
            )
            for name, value in result["metrics"].items():
                flat[f"{key}/{name}"] = float(value)
                if not name.startswith("support"):
                    support = result["metrics"].get(f"support_{name}", result["metrics"].get("support"))
                    suffix = f"  n={int(support)}" if support is not None else ""
                    print(f"   {name}: {value:.3f}{suffix}")

        behavior = selection_score(flat)
        losses = [value for name, value in flat.items() if name.endswith("/loss")]
        mean_loss = sum(losses) / len(losses) if losses else None
        if mean_loss is not None:
            flat["loss"] = mean_loss
            flat["perplexity"] = math.exp(min(mean_loss, 20))
        if behavior is not None:
            flat["behavior_score"] = float(behavior)
            print(f" behavior_score: {behavior:.3f}")
        if mean_loss is not None:
            print(f" mean_eval_loss: {mean_loss:.4f}  ppl {flat['perplexity']:.2f}")
        print("=" * 78 + "\n")

        self._append_metrics({"episode": episode, "split": f"eval_{tag}", **flat})
        if writer is not None:
            for name, value in flat.items():
                writer.add_scalar(f"eval/{tag}/{name}", value, episode)

        improved = False
        if behavior is not None and (self.best_behavior is None or behavior >= self.best_behavior):
            self.best_behavior = float(behavior)
            improved = True
        elif behavior is None and mean_loss is not None and (self.best_loss is None or mean_loss <= self.best_loss):
            self.best_loss = float(mean_loss)
            improved = True
        if flat:
            snap = self.ckpt_dir / f"eval_{tag}"
            snap.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(snap)
            tokenizer.save_pretrained(snap)
            (snap / "metrics.json").write_text(
                json.dumps({"tag": tag, "episode": episode, "metrics": flat}, indent=2),
                encoding="utf-8",
            )
        if improved and flat:
            best_dir = self.ckpt_dir / "best_eval"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "metrics.json").write_text(
                json.dumps({"tag": tag, "episode": episode, "metrics": flat}, indent=2),
                encoding="utf-8",
            )

        if was_training:
            model.train()
        return flat

    @torch.no_grad()
    def evaluate_rl_envs(self, model, tokenizer, dataset_state: dict[str, Any], rng: random.Random) -> dict[str, float]:
        """Greedy rollouts on the RL training datasets (not DatasetSpec evals)."""
        was_training = model.training
        model.eval()
        rewards = []
        accs = []
        tool_rates = []
        for _ in range(int(self.config.eval_episodes)):
            dataset = _select_dataset(self.config.datasets, rng)
            sample = sample_episode(dataset, dataset_state[dataset.name], rng)
            traj = rollout_episode(
                model,
                tokenizer,
                sample,
                self.tool_format,
                max_turns=self.config.max_turns,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            reward, metrics = grade_episode(dataset, traj.messages, traj.meta)
            traj.reward = reward
            traj.metrics = metrics
            rewards.append(reward)
            accs.append(metrics.get("accuracy", metrics.get("score", 0.0)))
            tool_rates.append(1.0 if metrics.get("used_calculator", 0.0) > 0 else 0.0)
            self._record_rollout(
                split="rl_env_eval",
                episode=-1,
                dataset_name=dataset.name,
                sample=sample,
                traj=traj,
                group_index=0,
            )
        if was_training:
            model.train()
        return {
            "reward": float(sum(rewards) / max(len(rewards), 1)),
            "accuracy": float(sum(accs) / max(len(accs), 1)),
            "calculator_rate": float(sum(tool_rates) / max(len(tool_rates), 1)),
        }

    def _record_rollout(
        self,
        *,
        split: str,
        episode: int,
        dataset_name: str,
        sample: dict[str, Any],
        traj: Trajectory,
        group_index: int,
    ) -> None:
        if not self.config.save_rollouts:
            return
        record = {
            "time": time.time(),
            "split": split,
            "episode": int(episode),
            "dataset": dataset_name,
            "sample_id": sample.get("sample_id") or traj.sample_id,
            "group_index": int(group_index),
            "score": float(traj.reward),
            "metrics": dict(traj.metrics),
            "meta": dict(traj.meta),
            # Tools are injected by the chat template, not as chat messages.
            "tool_format": traj.tool_format,
            "tools": list(traj.tool_schemas),
            "rendered_prompt": traj.rendered_prompt,
            "conversation": list(traj.messages),
        }
        self.rollouts.append(record)
        # Flush often so a crash still leaves a usable JSON file.
        if len(self.rollouts) % 4 == 0:
            self._flush_rollouts()

    def _flush_rollouts(self) -> None:
        if not self.config.save_rollouts:
            return
        self.rollouts_path.parent.mkdir(parents=True, exist_ok=True)
        self.rollouts_path.write_text(
            json.dumps(self.rollouts, indent=2, default=str),
            encoding="utf-8",
        )

    def _grpo_update(self, model, ref_model, optimizer, tokenizer, trajectories, advantages) -> dict[str, float]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_metrics = []
        for traj, adv in zip(trajectories, advantages):
            if not traj.completion_ids:
                continue
            input_ids, attention, completion_mask = build_batch_tensors(
                traj.prompt_ids,
                traj.completion_ids,
                traj.completion_mask,
                pad_id=tokenizer.pad_token_id or 0,
                device=self.device,
            )
            adv_t = torch.tensor([adv], dtype=torch.float32, device=self.device)
            loss, metrics = grpo_loss(
                model,
                ref_model,
                input_ids,
                attention,
                completion_mask,
                adv_t,
                kl_coef=float(self.config.kl_coef),
            )
            (loss / max(len(trajectories), 1)).backward()
            total_metrics.append(metrics)

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config.max_grad_norm))
        optimizer.step()

        if not total_metrics:
            return {"loss": 0.0, "kl": 0.0, "policy_logp": 0.0}
        keys = total_metrics[0].keys()
        return {k: float(sum(m[k] for m in total_metrics) / len(total_metrics)) for k in keys}

    def _run_before_batch(self, model, phase: str) -> None:
        hook = self.config.before_batch
        if hook is None:
            return
        hooks = [hook] if callable(hook) else list(hook)
        for fn in hooks:
            fn(model, phase)

    def _append_metrics(self, row: dict) -> None:
        row = {"time": time.time(), **row}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def _save_checkpoint(self, model, optimizer, episodes: int, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
        (path / "trainer_state.json").write_text(
            json.dumps({"episodes": int(episodes), "seed": self.config.seed}, indent=2),
            encoding="utf-8",
        )
        self._flush_rollouts()

    def _load_resume(self, model, optimizer, path: Path) -> int:
        weights = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        model.load_state_dict(weights.state_dict())
        opt_path = path / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        return int(state.get("episodes") or 0)
