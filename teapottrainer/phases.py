"""Phase recipes and the checkpoint hook that trains a supervised copy.

``phases_from_legacy`` is the pretrain, then bridge, then supervised schedule
the older config fields describe. An explicit ``phases`` list that matches it
keeps the same run directory.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import CheckpointRule, EarlyStop, TrainingConfig, TrainingPhase


def fork_phase(child: TrainingPhase, *, required: bool = False):
    """Hook that trains ``child`` from the checkpoint that just saved.

    The live phase keeps its weights. The copy is written to
    ``checkpoints/<checkpoint_name>_<child.name>``.
    """

    def hook(trainer, model, tokenizer, checkpoint_name, step):
        trainer.fork_training_phase(
            child,
            model,
            tokenizer,
            checkpoint_name,
            step,
            required=required,
        )

    hook.__name__ = f"fork_{child.name}"
    hook.__qualname__ = hook.__name__
    return hook


def phases_from_legacy(config: TrainingConfig) -> list[TrainingPhase]:
    """The recipe implied by the pretrain, bridge, and sft dataset lists."""
    rule = CheckpointRule(
        every_tokens=int(config.eval_every_tokens or 0),
        every_percent=float(config.eval_every_percent),
    )
    early = None
    if int(config.plateau_window_tokens or 0) > 0:
        early = EarlyStop(
            metric="fineweb",
            window_tokens=int(config.plateau_window_tokens),
            min_delta=float(config.plateau_min_loss_drop),
            patience=1,
        )
    evals = list(config.eval_datasets or [])
    has_base = bool(config.pretrain_datasets)
    supervised = None
    if config.sft_datasets:
        supervised = TrainingPhase(
            name="sft",
            datasets=list(config.sft_datasets),
            eval_datasets=evals,
            lr=config.sft_lr,
            compile=False,
            checkpoint=rule,
            fork=has_base,
            fork_required=has_base,
            fork_from="latest" if has_base else None,
            load_fork=has_base,
            task_eval=not has_base,
            eval_during=not has_base,
            save=False,
            restore_boundary_rng=not has_base,
        )
    hook = [fork_phase(supervised)] if supervised is not None and config.sft_at_checkpoints else []
    phases: list[TrainingPhase] = []
    if config.pretrain_datasets:
        phases.append(
            TrainingPhase(
                name="pretrain",
                datasets=list(config.pretrain_datasets),
                eval_datasets=evals,
                lr=float(config.lr),
                compile=bool(config.compile),
                checkpoint=rule,
                early_stop=early,
                on_checkpoint=list(hook),
                on_complete=list(hook),
                task_eval=False,
                reset_step=True,
                save=True,
            )
        )
    if config.bridge_datasets:
        phases.append(
            TrainingPhase(
                name="bridge",
                datasets=list(config.bridge_datasets),
                eval_datasets=evals,
                lr_scale=0.1,
                compile=False,
                checkpoint=rule,
                on_checkpoint=list(hook),
                task_eval=False,
                save=True,
            )
        )
    if supervised is not None:
        phases.append(supervised)
    return phases


def _spec_rows(specs) -> list[tuple]:
    rows = []
    for spec in specs or []:
        source = spec.source
        if isinstance(source, str):
            rows.append((source, None, "train", int(spec.target_tokens), float(spec.weight), None))
            continue
        template = getattr(source, "chat_template_fn", None)
        template_name = None if template is None else getattr(template, "__qualname__", type(template).__name__)
        rows.append(
            (
                source.name,
                source.subset,
                source.split,
                int(spec.target_tokens),
                float(spec.weight),
                template_name,
            )
        )
    return rows


def _labels(value: Any) -> list[str]:
    if value is None:
        return []
    hooks = [value] if callable(value) else list(value)
    return [getattr(hook, "__qualname__", type(hook).__name__) for hook in hooks]


def phase_view(phase: TrainingPhase, config: Optional[TrainingConfig] = None) -> dict:
    """The parts of a phase that change what training does.

    A token budget equal to the sum of the mix is the same as omitting it.
    A missing batch hook uses the config hook.
    """
    before = phase.before_batch
    if before is None and config is not None:
        before = config.before_batch
    evals = phase.eval_datasets
    if evals is None and config is not None:
        evals = list(config.eval_datasets or [])
    total = sum(int(spec.target_tokens) for spec in phase.datasets)
    budget = phase.token_budget
    if budget is not None and int(budget) == total:
        budget = None
    early = phase.early_stop
    return {
        "name": phase.name,
        "data": _spec_rows(phase.datasets),
        "eval": _spec_rows(evals),
        "budget": None if budget is None else int(budget),
        "lr": None if phase.lr is None else float(phase.lr),
        "lr_scale": None if phase.lr_scale is None else float(phase.lr_scale),
        "compile": phase.compile,
        "every_tokens": int(phase.checkpoint.every_tokens or 0),
        "every_percent": float(phase.checkpoint.every_percent),
        "early": None
        if early is None
        else (early.metric, int(early.window_tokens), float(early.min_delta), int(early.patience)),
        "on_checkpoint": _labels(phase.on_checkpoint),
        "on_complete": _labels(phase.on_complete),
        "before": _labels(before),
        "task_eval": bool(phase.task_eval),
        "eval_during": bool(phase.eval_during),
        "fork": bool(phase.fork),
        "fork_required": bool(phase.fork_required),
        "fork_from": phase.fork_from,
        "load_fork": bool(phase.load_fork),
        "save": bool(phase.save),
        "reset_step": bool(phase.reset_step),
        "restore_boundary_rng": bool(phase.restore_boundary_rng),
    }


def same_recipe(phases: list[TrainingPhase], other: list[TrainingPhase], config: Optional[TrainingConfig] = None) -> bool:
    return [phase_view(phase, config) for phase in phases] == [phase_view(phase, config) for phase in other]
