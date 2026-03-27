from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class VotesDatasetBuildConfig:
    dataset_name: str
    classifier_ckpt: str
    output_h5: str
    sigmas: Sequence[float]
    samples_per_sigma: int
    mc_chunk: int
    normalize_inputs: bool
    norm_mean: Sequence[float]
    norm_std: Sequence[float]
    split_seed: int
    global_seed: int


def _chunk1d(n: int, base: int = 1024):
    return (max(1, min(n, base)),)


def _chunk_img(n: int, c: int, h: int, w: int, base: int = 256):
    return (max(1, min(n, base)), c, h, w)


def _chunk_votes(n: int, classes: int, base_n: int = 128):
    return (max(1, min(n, base_n)), 1, classes)


def _build_single_split(
    hf_group: h5py.Group,
    loader: DataLoader,
    classifier: torch.nn.Module,
    device: torch.device,
    sigmas: np.ndarray,
    samples_per_sigma: int,
    mc_chunk: int,
    normalize_inputs: bool,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    n = len(loader.dataset)
    c = classifier(torch.zeros(1, 3, 32, 32, device=device)).shape[1]
    s = len(sigmas)

    image_dset = hf_group.create_dataset(
        "images",
        (n, 3, 32, 32),
        dtype=np.uint8,
        chunks=_chunk_img(n, 3, 32, 32),
    )
    label_dset = hf_group.create_dataset("labels", (n,), dtype=np.int64, chunks=_chunk1d(n))
    dist_dset = hf_group.create_dataset(
        "distributions",
        (n, s, c),
        dtype=np.float32,
        chunks=_chunk_votes(n, c),
    )
    hf_group.create_dataset("sigmas", data=sigmas)

    idx0 = 0
    for imgs, labels in tqdm(loader, desc=f"Building split {hf_group.name}"):
        bsz = imgs.size(0)
        image_dset[idx0 : idx0 + bsz] = imgs.mul(255).byte().numpy()
        label_dset[idx0 : idx0 + bsz] = labels.numpy()

        imgs = imgs.to(device)
        counts = torch.zeros(bsz, s, c, device=device, dtype=torch.int32)

        remaining = samples_per_sigma
        while remaining > 0:
            cur = min(mc_chunk, remaining)
            noise = torch.randn((cur, bsz, 3, 32, 32), device=device)

            for s_idx, sigma in enumerate(sigmas):
                noisy = imgs.unsqueeze(0) + noise * float(sigma)
                noisy = noisy.reshape(cur * bsz, 3, 32, 32)
                if normalize_inputs:
                    noisy = (noisy - mean) / std

                logits = classifier(noisy)
                preds = torch.argmax(logits, dim=1).view(cur, bsz)
                vote_counts = torch.nn.functional.one_hot(preds, num_classes=c).sum(dim=0).to(torch.int32)
                counts[:, s_idx, :] += vote_counts

            remaining -= cur

        dist_dset[idx0 : idx0 + bsz] = (counts.float() / float(samples_per_sigma)).cpu().numpy()
        idx0 += bsz


@torch.inference_mode()
def build_votes_h5(
    config: VotesDatasetBuildConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    classifier: torch.nn.Module,
    device: torch.device,
) -> None:
    """Create offline certifier data with class-distribution targets from MC voting."""
    sigmas = np.asarray(config.sigmas, dtype=np.float32)

    if config.normalize_inputs:
        mean = torch.tensor(config.norm_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(config.norm_std, device=device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)
        std = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1)

    os.makedirs(os.path.dirname(config.output_h5) or ".", exist_ok=True)

    num_classes = classifier(torch.zeros(1, 3, 32, 32, device=device)).shape[1]

    with h5py.File(config.output_h5, "w") as hf:
        hf.attrs["meta"] = json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": config.dataset_name,
                "splits": {
                    "train": len(train_loader.dataset),
                    "val": len(val_loader.dataset),
                    "test": len(test_loader.dataset),
                },
                "num_classes": int(num_classes),
                "samples_per_sigma": int(config.samples_per_sigma),
                "sigmas": sigmas.tolist(),
                "vote_mode": "argmax",
                "normalize": bool(config.normalize_inputs),
                "norm_mean": list(config.norm_mean),
                "norm_std": list(config.norm_std),
                "split_seed": int(config.split_seed),
                "seed": int(config.global_seed),
                "classifier_ckpt": os.path.abspath(config.classifier_ckpt),
                "builder": asdict(config),
                "note": "distributions = argmax vote counts / samples_per_sigma",
            }
        )

        _build_single_split(
            hf_group=hf.create_group("train"),
            loader=train_loader,
            classifier=classifier,
            device=device,
            sigmas=sigmas,
            samples_per_sigma=config.samples_per_sigma,
            mc_chunk=config.mc_chunk,
            normalize_inputs=config.normalize_inputs,
            mean=mean,
            std=std,
        )
        _build_single_split(
            hf_group=hf.create_group("val"),
            loader=val_loader,
            classifier=classifier,
            device=device,
            sigmas=sigmas,
            samples_per_sigma=config.samples_per_sigma,
            mc_chunk=config.mc_chunk,
            normalize_inputs=config.normalize_inputs,
            mean=mean,
            std=std,
        )
        _build_single_split(
            hf_group=hf.create_group("test"),
            loader=test_loader,
            classifier=classifier,
            device=device,
            sigmas=sigmas,
            samples_per_sigma=config.samples_per_sigma,
            mc_chunk=config.mc_chunk,
            normalize_inputs=config.normalize_inputs,
            mean=mean,
            std=std,
        )
