"""Causal language-model loss that does not upcast the full logits tensor."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean next-token cross entropy.

    ``logits[b, t]`` predicts ``labels[b, t + 1]``. Positions whose shifted
    label is ``ignore_index`` are dropped, which is how prompt tokens and
    padding stay out of the SFT loss.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    total = logits.new_zeros(())
    count = 0
    batch, steps, _vocab = logits.shape
    for row in range(batch):
        row_logits = logits[row]
        row_labels = labels[row]
        for start in range(0, steps, chunk):
            target = row_labels[start : start + chunk]
            n = int((target != ignore_index).sum().item())
            if n == 0:
                continue
            pred = row_logits[start : start + chunk].float()
            total = total + F.cross_entropy(pred, target, reduction="sum", ignore_index=ignore_index)
            count += n
    if count == 0:
        return logits.sum() * 0.0
    return total / count


def supervised_tokens(labels: torch.Tensor, ignore_index: int = -100) -> int:
    """How many tokens the shifted loss can actually train on."""
    if labels.shape[1] < 2:
        return 0
    return int((labels[:, 1:] != ignore_index).sum().item())
