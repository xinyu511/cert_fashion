from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import distribution_metrics


def _summarize(values: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
    }


@torch.no_grad()
def evaluate_certifier(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate a certifier over a dataloader of `(image, sigma, target_distribution)`."""
    model.eval()

    ce_values: List[torch.Tensor] = []
    l1_values: List[torch.Tensor] = []
    kl_values: List[torch.Tensor] = []

    for image, sigma, target in loader:
        image = image.to(device)
        sigma = sigma.to(device).view(-1, 1)
        target = target.to(device)

        logits = model(image, sigma)
        ce, l1, kl = distribution_metrics(logits, target)

        ce_values.append(ce.cpu())
        l1_values.append(l1.cpu())
        kl_values.append(kl.cpu())

    ce_all = torch.cat(ce_values).numpy()
    l1_all = torch.cat(l1_values).numpy()
    kl_all = torch.cat(kl_values).numpy()

    out: Dict[str, float] = {"n": int(len(ce_all))}
    out.update(_summarize(ce_all, "ce"))
    out.update(_summarize(l1_all, "l1"))
    out.update(_summarize(kl_all, "kl"))
    return out


def format_metrics_table(rows: Sequence[Tuple[str, Dict[str, float]]]) -> str:
    """Return a plain-text table for side-by-side certifier metrics."""
    header = "{:<24} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "CE(mean)", "L1(mean)", "KL(mean)", "CE(med)", "L1(med)", "KL(med)"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for name, m in rows:
        lines.append(
            "{:<24} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(
                name,
                m["ce_mean"],
                m["l1_mean"],
                m["kl_mean"],
                m["ce_median"],
                m["l1_median"],
                m["kl_median"],
            )
        )

    return "\n".join(lines)
