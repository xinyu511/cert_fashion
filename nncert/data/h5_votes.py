from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize


class H5VotesDataset(Dataset):
    """Flattened HDF5 dataset over pairs `(image_idx, sigma_idx)`.

    Returns tuples `(image, sigma, distribution)` where:
      - `image`: `torch.float32`, shape `[3, 32, 32]`
      - `sigma`: `torch.float32`, shape `[1]`
      - `distribution`: `torch.float32`, shape `[num_classes]`
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        use_all_sigmas: bool = True,
        sigma_indices: List[int] | None = None,
        apply_norm_if_meta: bool = True,
        deterministic_single_sigma: bool = False,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.h5 = h5py.File(h5_path, "r")
        if split not in self.h5:
            available = list(self.h5.keys())
            raise KeyError(f"Split '{split}' missing in {h5_path}. Available: {available}")

        self.g = self.h5[split]
        self.imgs = self.g["images"]
        self.labels = self.g["labels"]
        self.dists = self.g["distributions"]
        self.sigmas = self.g["sigmas"][...]

        self.N, self.S, self.C = self.dists.shape

        if use_all_sigmas:
            s_idxs = np.arange(self.S) if sigma_indices is None else np.asarray(sigma_indices, dtype=np.int64)
            self.index_map: List[Tuple[int, int]] = [
                (i, int(s)) for i in range(self.N) for s in s_idxs
            ]
        else:
            rng = np.random.RandomState(split_seed if deterministic_single_sigma else None)
            sampled = rng.randint(0, self.S, size=(self.N,))
            self.index_map = [(i, int(sampled[i])) for i in range(self.N)]

        self.meta: Dict[str, Any] = json.loads(self.h5.attrs.get("meta", "{}"))

        self.normalize = None
        if apply_norm_if_meta and bool(self.meta.get("normalize", False)):
            mean = self.meta.get("norm_mean", [0.4914, 0.4822, 0.4465])
            std = self.meta.get("norm_std", [0.2470, 0.2435, 0.2616])
            self.normalize = Normalize(mean, std)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        i, s = self.index_map[idx]
        img = torch.from_numpy(self.imgs[i].astype(np.float32) / 255.0)
        if self.normalize is not None:
            img = self.normalize(img)

        sigma = torch.tensor([float(self.sigmas[s])], dtype=torch.float32)
        dist = torch.from_numpy(self.dists[i, s].astype(np.float32))
        return img, sigma, dist

    def close(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass


class H5VotesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: str,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_all_sigmas: bool = True,
        apply_norm_if_meta: bool = True,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_all_sigmas = use_all_sigmas
        self.apply_norm_if_meta = apply_norm_if_meta

        self.train_ds: H5VotesDataset | None = None
        self.val_ds: H5VotesDataset | None = None
        self.test_ds: H5VotesDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = H5VotesDataset(
                self.h5_path,
                split="train",
                use_all_sigmas=self.use_all_sigmas,
                apply_norm_if_meta=self.apply_norm_if_meta,
            )
            self.val_ds = H5VotesDataset(
                self.h5_path,
                split="val",
                use_all_sigmas=self.use_all_sigmas,
                apply_norm_if_meta=self.apply_norm_if_meta,
            )

        if stage in (None, "test", "predict"):
            self.test_ds = H5VotesDataset(
                self.h5_path,
                split="test",
                use_all_sigmas=self.use_all_sigmas,
                apply_norm_if_meta=self.apply_norm_if_meta,
            )

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("DataModule not set up. Call setup('test') first.")
        return self._loader(self.test_ds, shuffle=False)

    def teardown(self, stage: str | None = None) -> None:
        for ds in (self.train_ds, self.val_ds, self.test_ds):
            if ds is not None:
                ds.close()


def summarize_h5(h5_path: str) -> Dict[str, Any]:
    """Return and print a concise summary of the offline dataset."""
    summary: Dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        summary["meta"] = json.loads(f.attrs.get("meta", "{}"))
        summary["splits"] = {}
        for split in ["train", "val", "test"]:
            if split not in f:
                continue
            g = f[split]
            summary["splits"][split] = {
                "images": tuple(g["images"].shape),
                "labels": tuple(g["labels"].shape),
                "distributions": tuple(g["distributions"].shape),
                "sigmas": tuple(g["sigmas"].shape),
            }

    print("=== H5 Summary ===")
    print("Meta:", summary["meta"])
    for split, d in summary["splits"].items():
        print(f"[{split}] images={d['images']} labels={d['labels']} dists={d['distributions']} sigmas={d['sigmas']}")
    print("==================")
    return summary
