"""Run-folder metrics: jsonl for plotting and a text report for tailing."""
from __future__ import annotations

import json
import math
from pathlib import Path


def human(n: float) -> str:
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}Q"


def ppl(loss: float | None) -> str:
    if loss is None or not math.isfinite(loss):
        return "n/a"
    return f"{math.exp(min(loss, 20.0)):.2f}"


class MetricLog:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.eval_path = output_dir / "eval.jsonl"
        self.report_path = output_dir / "metrics.md"

    def append_step(self, record: dict) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def append_eval(self, record: dict, text: str) -> None:
        with self.eval_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        with self.report_path.open("a", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")

    def write_config(self, payload: dict) -> None:
        (self.output_dir / "config.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def format_eval(
    *,
    stage: str,
    step: int,
    tokens: int,
    target: int,
    train_loss: float | None,
    metrics: dict[str, float],
    samples: list[tuple[str, str]],
    tok_s: float | None,
    lr: float,
    elapsed_s: float,
) -> str:
    pct = 100.0 * tokens / max(target, 1)
    lines = [
        "",
        f"================  EVAL {stage}  |  step {step}  ================",
        f"  tokens     {tokens:,} / {target:,}  ({pct:.1f}%)",
        f"  lr         {lr:.3e}",
        f"  tok/s      {0 if tok_s is None else tok_s:,.0f}",
        f"  elapsed    {elapsed_s / 60:.1f} min",
        f"  train loss {train_loss:.4f}   ppl {ppl(train_loss)}" if train_loss is not None else "  train loss n/a",
    ]
    if metrics:
        width = max(len(k) for k in metrics)
        for key in sorted(metrics):
            value = metrics[key]
            extra = ""
            if key.endswith("loss"):
                extra = f"   ppl {ppl(value)}"
            lines.append(f"  {key:<{width}}  {value:.4f}{extra}")
    for prompt, sample in samples:
        lines.append(f"  [prompt] {prompt!r}")
        for row in sample.splitlines() or [""]:
            lines.append(f"    | {row}")
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)
