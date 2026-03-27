from __future__ import annotations

import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nncert.data.vision import get_dataset_splits
from nncert.lightning.modules import ImageClassifier
from nncert.utils.reproducibility import setup_reproducibility
from nncert.workflows.votes_dataset import VotesDatasetBuildConfig, build_votes_h5


@hydra.main(config_path="../../configs", config_name="create_votes_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    print("\n--- Configuration (resolved) ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("--------------------------------\n")

    setup_reproducibility(
        seed=cfg.reproducibility.seed,
        deterministic=cfg.reproducibility.deterministic,
    )

    device = torch.device(cfg.runtime.device if cfg.runtime.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.exists(cfg.classifier_ckpt_path):
        raise FileNotFoundError(f"classifier_ckpt_path not found: {cfg.classifier_ckpt_path}")

    classifier_module = ImageClassifier.load_from_checkpoint(cfg.classifier_ckpt_path, map_location=device)
    classifier = classifier_module.to(device).eval()

    train_ds, val_ds, test_ds = get_dataset_splits(
        dataset_name=cfg.dataset.name,
        data_root=cfg.dataset.path,
        val_size=cfg.dataset.val_size,
        split_seed=cfg.dataset.split_seed,
    )

    common_dl = {
        "batch_size": cfg.runtime.batch_size,
        "num_workers": cfg.runtime.num_workers,
        "pin_memory": cfg.runtime.pin_memory,
        "shuffle": False,
        "drop_last": False,
    }
    train_loader = DataLoader(train_ds, **common_dl)
    val_loader = DataLoader(val_ds, **common_dl)
    test_loader = DataLoader(test_ds, **common_dl)

    sigmas = np.linspace(cfg.votes.sigma_min, cfg.votes.sigma_max, cfg.votes.num_sigmas, dtype=np.float32)

    build_cfg = VotesDatasetBuildConfig(
        dataset_name=cfg.dataset.name,
        classifier_ckpt=cfg.classifier_ckpt_path,
        output_h5=cfg.output_h5_path,
        sigmas=sigmas.tolist(),
        samples_per_sigma=cfg.votes.samples_per_sigma,
        mc_chunk=cfg.votes.mc_chunk,
        normalize_inputs=cfg.votes.normalize_inputs,
        norm_mean=list(cfg.votes.norm_mean),
        norm_std=list(cfg.votes.norm_std),
        split_seed=cfg.dataset.split_seed,
        global_seed=cfg.reproducibility.seed,
    )

    build_votes_h5(
        config=build_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classifier=classifier,
        device=device,
    )

    print(f"Done. Saved HDF5 to {cfg.output_h5_path}")


if __name__ == "__main__":
    main()
