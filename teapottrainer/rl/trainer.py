"""Config-driven RL trainer.

Public surface: ``RlTrainer(RlConfig).train(fresh=True)``.
Assumes supervised fine-tuning is already done and loads ``init_from``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rl_config import EnvSpec, RlConfig
from ..tool_formats import get_tool_format
from .algorithms.grpo import build_batch_tensors, group_advantages, grpo_loss
from .rollouts import rollout_episode
from .tasks import build_env

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _run_hash(config: RlConfig) -> str:
    payload = config.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def _select_env(specs: list[EnvSpec], rng: random.Random) -> EnvSpec:
    weights = [max(float(spec.weight), 0.0) for spec in specs]
    if not specs:
        raise ValueError("RlConfig.envs must contain at least one EnvSpec")
    if sum(weights) <= 0:
        return specs[0]
    return rng.choices(specs, weights=weights, k=1)[0]


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

    def train(self, fresh: bool = True) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "rl_config.json").write_text(
            json.dumps(self.config.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        rng = random.Random(self.config.seed)

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

        writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        env_cache = {spec.name: build_env(spec.name, spec.options) for spec in self.config.envs}

        episodes_done = start_episodes
        best_acc = -1.0
        pbar = tqdm(total=int(self.config.max_train_episodes), initial=episodes_done, desc="rl")

        while episodes_done < int(self.config.max_train_episodes):
            spec = _select_env(self.config.envs, rng)
            env = env_cache[spec.name]
            task = env.sample_task(rng)

            trajectories = []
            for _ in range(int(self.config.group_size)):
                self._run_before_batch(model, phase="rl")
                traj = rollout_episode(
                    model,
                    tokenizer,
                    env,
                    task,
                    self.tool_format,
                    max_turns=self.config.max_turns,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                trajectories.append(traj)

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
            mean_acc = sum(t.metrics.get("accuracy", 0.0) for t in trajectories) / max(len(trajectories), 1)
            row = {
                "episode": episodes_done,
                "env": spec.name,
                "task_id": task.task_id,
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
                eval_stats = self.evaluate(model, tokenizer, env_cache, rng)
                row = {"episode": episodes_done, "split": "eval", **eval_stats}
                self._append_metrics(row)
                writer.add_scalar("eval/accuracy", eval_stats.get("accuracy", 0.0), episodes_done)
                writer.add_scalar("eval/reward", eval_stats.get("reward", 0.0), episodes_done)
                if eval_stats.get("accuracy", -1.0) >= best_acc:
                    best_acc = float(eval_stats["accuracy"])
                    self._save_checkpoint(model, optimizer, episodes_done, self.ckpt_dir / "best")

            if episodes_done % int(self.config.resume_every_episodes) == 0:
                self._save_checkpoint(model, optimizer, episodes_done, resume_path)

        pbar.close()
        # Final eval + save.
        eval_stats = self.evaluate(model, tokenizer, env_cache, rng)
        self._append_metrics({"episode": episodes_done, "split": "final_eval", **eval_stats})
        self._save_checkpoint(model, optimizer, episodes_done, resume_path)
        final_dir = self.ckpt_dir / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        writer.close()
        return self.run_dir

    @torch.no_grad()
    def evaluate(self, model, tokenizer, env_cache, rng: random.Random) -> dict[str, float]:
        was_training = model.training
        model.eval()
        rewards = []
        accs = []
        tool_rates = []
        for _ in range(int(self.config.eval_episodes)):
            spec = _select_env(self.config.envs, rng)
            env = env_cache[spec.name]
            task = env.sample_task(rng)
            traj = rollout_episode(
                model,
                tokenizer,
                env,
                task,
                self.tool_format,
                max_turns=self.config.max_turns,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            rewards.append(traj.reward)
            accs.append(traj.metrics.get("accuracy", 0.0))
            tool_rates.append(1.0 if traj.metrics.get("used_calculator", 0.0) > 0 else 0.0)
        if was_training:
            model.train()
        return {
            "reward": float(sum(rewards) / max(len(rewards), 1)),
            "accuracy": float(sum(accs) / max(len(accs), 1)),
            "calculator_rate": float(sum(tool_rates) / max(len(tool_rates), 1)),
        }

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

    def _load_resume(self, model, optimizer, path: Path) -> int:
        weights = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        model.load_state_dict(weights.state_dict())
        opt_path = path / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        return int(state.get("episodes") or 0)
