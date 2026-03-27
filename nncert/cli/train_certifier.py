from __future__ import annotations

import os

import hydra
from pathlib import Path
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from nncert.data.h5_votes import H5VotesDataModule, summarize_h5
from nncert.utils.reproducibility import setup_reproducibility


@hydra.main(config_path=str(Path(__file__).resolve().parents[2] / "configs"), config_name="train_certifier", version_base=None)
def main(cfg: DictConfig) -> None:
    print("\n--- Configuration (resolved) ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("--------------------------------\n")

    setup_reproducibility(
        seed=cfg.reproducibility.seed,
        deterministic=cfg.reproducibility.deterministic,
    )

    h5_path = cfg.offline_data_path
    if not h5_path or not os.path.exists(h5_path):
        raise FileNotFoundError(f"offline_data_path not found: {h5_path}")
    summarize_h5(h5_path)

    datamodule = H5VotesDataModule(
        h5_path=h5_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        use_all_sigmas=cfg.training.use_all_sigmas,
        apply_norm_if_meta=cfg.training.apply_norm_if_meta,
    )

    model = instantiate(cfg.pl_model, _recursive_=False)

    checkpoint = ModelCheckpoint(
        monitor=cfg.training.monitor,
        mode=cfg.training.monitor_mode,
        save_top_k=1,
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.reproducibility.deterministic,
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=cfg.training.progress_bar_refresh_rate),
        ],
    )

    print("Starting offline certifier training...")
    trainer.fit(model, datamodule=datamodule)

    best = checkpoint.best_model_path if checkpoint.best_model_path else None
    print(f"Training complete! Best checkpoint: {best}")
    trainer.test(model, datamodule=datamodule, ckpt_path=best or None)


if __name__ == "__main__":
    main()
