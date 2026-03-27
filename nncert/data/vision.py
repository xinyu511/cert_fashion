from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for supported torchvision datasets."""

    name: str
    torch_dataset_cls: type
    num_classes: int = 10


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "FashionMNIST": DatasetSpec("FashionMNIST", datasets.FashionMNIST, 10),
    "MNIST": DatasetSpec("MNIST", datasets.MNIST, 10),
    "KMNIST": DatasetSpec("KMNIST", datasets.KMNIST, 10),
    "CIFAR10": DatasetSpec("CIFAR10", datasets.CIFAR10, 10),
}


def default_transform(dataset_name: str):
    """Return the canonical transform used in this project."""
    if dataset_name == "CIFAR10":
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

    return transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])


def get_dataset_splits(
    dataset_name: str,
    data_root: str,
    val_size: int,
    split_seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create deterministic train/val/test splits for a torchvision dataset."""
    if dataset_name not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")

    spec = DATASET_REGISTRY[dataset_name]
    tfm = default_transform(dataset_name)

    full_train = spec.torch_dataset_cls(root=data_root, train=True, download=True, transform=tfm)
    test_ds = spec.torch_dataset_cls(root=data_root, train=False, download=True, transform=tfm)

    total_train = len(full_train)
    val_size = min(max(val_size, 0), total_train)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(split_seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    return train_ds, val_ds, test_ds


def _build_loader(ds: Dataset, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_classifier_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    val_size: int,
    split_seed: int,
    pin_memory: bool = True,
):
    """Factory for classifier train/val/test dataloaders."""
    train_ds, val_ds, test_ds = get_dataset_splits(
        dataset_name=dataset_name,
        data_root=data_root,
        val_size=val_size,
        split_seed=split_seed,
    )

    train_loader = _build_loader(train_ds, batch_size, num_workers, pin_memory, shuffle=True)
    val_loader = _build_loader(val_ds, batch_size, num_workers, pin_memory, shuffle=False)
    test_loader = _build_loader(test_ds, batch_size, num_workers, pin_memory, shuffle=False)
    return train_loader, val_loader, test_loader


class VisionDataModule(pl.LightningDataModule):
    """Lightning DataModule for classifier training on torchvision datasets."""

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        batch_size: int,
        num_workers: int,
        val_size: int,
        split_seed: int,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.split_seed = split_seed
        self.pin_memory = pin_memory

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_ds, self.val_ds, _ = get_dataset_splits(
                dataset_name=self.dataset_name,
                data_root=self.data_root,
                val_size=self.val_size,
                split_seed=self.split_seed,
            )

        if stage in (None, "test", "predict"):
            _, _, self.test_ds = get_dataset_splits(
                dataset_name=self.dataset_name,
                data_root=self.data_root,
                val_size=self.val_size,
                split_seed=self.split_seed,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        return _build_loader(self.train_ds, self.batch_size, self.num_workers, self.pin_memory, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        return _build_loader(self.val_ds, self.batch_size, self.num_workers, self.pin_memory, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('test') first.")
        return _build_loader(self.test_ds, self.batch_size, self.num_workers, self.pin_memory, shuffle=False)
