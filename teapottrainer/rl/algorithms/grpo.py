"""Group Relative Policy Optimization helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Normalize rewards within a prompt group to zero-mean unit-ish scale."""
    if not rewards:
        return []
    vals = torch.tensor(rewards, dtype=torch.float32)
    mean = vals.mean()
    std = vals.std(unbiased=False)
    adv = (vals - mean) / (std + eps)
    return [float(x) for x in adv.tolist()]


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum token logprobs over completion positions (mask=1).

    ``input_ids`` is prompt+completion. ``completion_mask`` aligns to the full
    sequence and is 1 only on completion tokens that should train.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # Predict token t from position t-1.
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_logp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:].to(token_logp.dtype)
    return (token_logp * mask).sum(dim=-1)


def grpo_loss(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO objective with optional KL penalty toward a frozen reference."""
    policy_logp = sequence_logprobs(model, input_ids, attention_mask, completion_mask)
    loss = -(advantages.to(policy_logp.dtype) * policy_logp).mean()

    metrics = {
        "policy_logp": float(policy_logp.detach().mean().item()),
        "advantage_mean": float(advantages.float().mean().item()),
    }

    if ref_model is not None and float(kl_coef) > 0:
        with torch.no_grad():
            ref_logp = sequence_logprobs(ref_model, input_ids, attention_mask, completion_mask)
        # Token-sum KL proxy: log π - log π_ref over completion tokens.
        kl = policy_logp - ref_logp
        loss = loss + float(kl_coef) * kl.mean()
        metrics["kl"] = float(kl.detach().mean().item())
    else:
        metrics["kl"] = 0.0

    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


def build_batch_tensors(
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_mask: list[int],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad one trajectory into model batch tensors of shape ``[1, T]``."""
    ids = list(prompt_ids) + list(completion_ids)
    mask = [0] * len(prompt_ids) + list(completion_mask)
    if len(mask) != len(ids):
        # Fall back to training on every completion token.
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    completion = torch.tensor([mask], dtype=torch.long, device=device)
    if pad_id is not None and False:
        # Single-example path needs no padding; kept for API symmetry.
        pass
    return input_ids, attention, completion
