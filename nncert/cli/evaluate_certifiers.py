from __future__ import annotations

import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nncert.data.h5_votes import H5VotesDataset
from nncert.eval.runner import evaluate_certifier, format_metrics_table
from nncert.lightning.modules import ImageCertifier
from nncert.utils.reproducibility import setup_reproducibility


@hydra.main(config_path="../../configs", config_name="eval_certifiers", version_base=None)
def main(cfg: DictConfig) -> None:
    print("\n--- Configuration (resolved) ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("--------------------------------\n")

    setup_reproducibility(seed=cfg.reproducibility.seed, deterministic=False)

    device = torch.device(cfg.runtime.device if cfg.runtime.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = H5VotesDataset(
        h5_path=cfg.h5_path,
        split=cfg.split,
        use_all_sigmas=cfg.runtime.use_all_sigmas,
        apply_norm_if_meta=cfg.runtime.apply_norm_if_meta,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.runtime.batch_size,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
        shuffle=False,
        drop_last=False,
    )

    rows = []
    for item in cfg.models:
        if not os.path.exists(item.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {item.ckpt}")

        model = ImageCertifier.load_from_checkpoint(item.ckpt, map_location=device).to(device)
        metrics = evaluate_certifier(model, loader, device)
        rows.append((item.name, metrics))

    print("\n=== Certifier Evaluation ===")
    print(f"Split: {cfg.split} | Samples: {rows[0][1]['n'] if rows else 0}")
    print(format_metrics_table(rows))
    print("=" * 86)

    if cfg.output_json_path:
        os.makedirs(os.path.dirname(cfg.output_json_path) or ".", exist_ok=True)
        payload = [{"name": name, **metrics} for name, metrics in rows]
        with open(cfg.output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {cfg.output_json_path}")

    dataset.close()


if __name__ == "__main__":
    main()
