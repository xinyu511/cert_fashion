from __future__ import annotations

import os
from typing import Type

import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import get_class
from omegaconf import OmegaConf


def _load_classifier_backbone_from_ckpt(clf_path: str) -> nn.Module:
    """Load classifier module from the experiment config next to checkpoint.

    This preserves backward compatibility with historical runs that use
    `utils.pl_module.ImageClassifier` as target.
    """
    # Search upward from checkpoint directory for config.yaml.
    start_dir = os.path.abspath(os.path.dirname(clf_path))
    config_path = None
    cur = start_dir
    for _ in range(8):
        candidate = os.path.join(cur, "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate
            break
        parent = os.path.abspath(os.path.join(cur, ".."))
        if parent == cur:
            break
        cur = parent

    if config_path is None:
        raise FileNotFoundError(
            f"Cannot locate config.yaml while searching upward from checkpoint: {clf_path}"
        )

    conf = OmegaConf.load(config_path)
    target = conf.pl_model._target_
    clf_class: Type = get_class(target)
    clf_module = clf_class.load_from_checkpoint(clf_path)
    return clf_module.clf


class CertNet(nn.Module):
    """Condition classifier by concatenating low-rank channels from scalar sigma."""

    def __init__(self, backbone: nn.Module | None = None, num_mlps: int = 4, clf_path: str | None = None):
        super().__init__()
        if clf_path is not None:
            self.backbone = _load_classifier_backbone_from_ckpt(clf_path)
        elif backbone is not None:
            self.backbone = backbone
        else:
            raise ValueError("Provide `clf_path` or `backbone`.")

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                )
                for _ in range(num_mlps)
            ]
        )

        self.backbone.conv1 = nn.Conv2d(
            self.backbone.conv1.in_channels + num_mlps,
            self.backbone.conv1.out_channels,
            self.backbone.conv1.kernel_size,
            self.backbone.conv1.stride,
            self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias,
        )

    def forward(self, image: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        channels = []
        for mlp in self.mlps:
            x = mlp(sigma)
            x = rearrange(x, "B N -> B 1 N 1") @ rearrange(x, "B N -> B 1 1 N")
            channels.append(x)
        x = torch.cat((image, *channels), dim=1)
        return self.backbone(x)


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layers: int):
        super().__init__()
        seq = [nn.Linear(in_dim, out_dim)]
        for _ in range(layers - 1):
            seq.append(nn.Linear(out_dim, out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        out = rearrange(x, "B N -> B 1 N 1") @ rearrange(x, "B N -> B 1 1 N")
        return x, out


class FuseNet(nn.Module):
    """Feature-wise scalar fusion through multiplicative per-stage masks."""

    def __init__(self, backbone: nn.Module | None = None, clf_path: str | None = None):
        super().__init__()
        if clf_path is not None:
            self.backbone = _load_classifier_backbone_from_ckpt(clf_path)
        elif backbone is not None:
            self.backbone = backbone
        else:
            raise ValueError("Provide `clf_path` or `backbone`.")

        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3

        self.mlp0 = MLPBlock(1, 32, layers=3)
        self.mlp1 = MLPBlock(32, 32, layers=3)
        self.mlp2 = MLPBlock(32, 16, layers=3)
        self.mlp3 = MLPBlock(16, 8, layers=3)

    def forward(self, image: torch.Tensor, scalar: torch.Tensor):
        s0, s0i = self.mlp0(scalar)
        s1, s1i = self.mlp1(s0)
        s2, s2i = self.mlp2(s1)
        s3, s3i = self.mlp3(s2)

        out = self.layer0(image) * s0i
        out = self.layer1(out) * s1i
        out = self.layer2(out) * s2i
        out = self.layer3(out) * s3i

        out = self.backbone.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.backbone.fc(out)
        return out


# Backward-compatible aliases
CertNetOriginal = CertNet
FuseNetOriginal = FuseNet
CertNet_ORIGINAL = CertNetOriginal
FuseNet_ORIGINAL = FuseNetOriginal
