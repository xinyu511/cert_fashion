"""Backward-compatibility shim for historical data utils path.

The legacy project exposed `NoisyImgDataset` from `utils.data_utils`.
This alias is retained for old experiments while new code should import
from `nncert.data` directly.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class NoisyImgDataset(Dataset):
    """Minimal legacy-compatible dataset wrapper.

    Expects pre-built tensors `images` and `labels`, and applies optional
    additive Gaussian noise sampled with standard deviation `sigma`.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, sigma: float = 0.0):
        self.images = images
        self.labels = labels
        self.sigma = float(sigma)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        x = self.images[index]
        if self.sigma > 0:
            x = x + torch.randn_like(x) * self.sigma
        y = self.labels[index]
        return x, y


__all__ = ["NoisyImgDataset"]
