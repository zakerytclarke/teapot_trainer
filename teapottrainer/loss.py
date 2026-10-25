"""Causal language-model loss that does not upcast the full logits tensor."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk: int = 8192,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean next-token cross entropy.

    ``logits[b, t]`` predicts ``labels[b, t + 1]``. Positions whose shifted
    label is ``ignore_index`` are dropped, which is how prompt tokens and
    padding stay out of the SFT loss. The reduction is chunked so the fp32
    cast never covers the whole vocabulary tensor, and it does not sync the
    GPU once per chunk.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].reshape(-1)
    if not shift_logits.is_contiguous():
        shift_logits = shift_logits.contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    total = shift_logits.new_zeros(())
    count = shift_labels.new_zeros(())
    for start in range(0, shift_labels.shape[0], chunk):
        target = shift_labels[start : start + chunk]
        pred = shift_logits[start : start + chunk].float()
        total = total + F.cross_entropy(pred, target, reduction="sum", ignore_index=ignore_index)
        count = count + (target != ignore_index).sum()
    return total / count.clamp(min=1)


def supervised_tokens(labels: torch.Tensor, ignore_index: int = -100) -> int:
    """How many tokens the shifted loss can actually train on."""
    if labels.shape[1] < 2:
        return 0
    return int((labels[:, 1:] != ignore_index).sum().item())
