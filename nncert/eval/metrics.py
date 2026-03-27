from __future__ import annotations

import torch
import torch.nn.functional as F


def distribution_metrics(pred_logits: torch.Tensor, target_probs: torch.Tensor, eps: float = 1e-8):
    """Compute CE/L1/KL per sample for soft distribution targets.

    Returns a tuple `(ce, l1, kl)` where each tensor has shape `[batch]`.
    """
    pred_probs = F.softmax(pred_logits, dim=-1)

    ce = -(target_probs * (pred_probs + eps).log()).sum(dim=-1)
    l1 = torch.abs(pred_probs - target_probs).mean(dim=-1)

    log_pred = F.log_softmax(pred_logits, dim=-1)
    log_target = (target_probs + eps).log()
    kl = (target_probs * (log_target - log_pred)).sum(dim=-1)

    return ce, l1, kl
