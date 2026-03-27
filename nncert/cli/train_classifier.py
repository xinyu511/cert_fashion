from __future__ import annotations

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from nncert.data.vision import build_classifier_dataloaders
from nncert.utils.reproducibility import setup_reproducibility


@hydra.main(config_path="../../configs", config_name="train_classifier", version_base=None)
def main(cfg: DictConfig) -> None:
    print("\n--- Configuration (resolved) ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("--------------------------------\n")

    setup_reproducibility(
        seed=cfg.reproducibility.seed,
        deterministic=cfg.reproducibility.deterministic,
    )

    train_loader, val_loader, test_loader = build_classifier_dataloaders(
        dataset_name=cfg.dataset.name,
        data_root=cfg.dataset.path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        val_size=cfg.dataset.val_size,
        split_seed=cfg.dataset.split_seed,
        pin_memory=cfg.training.pin_memory,
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

    trainer.fit(model, train_loader, val_loader)
    best = checkpoint.best_model_path if checkpoint.best_model_path else None
    print(f"Best classifier checkpoint: {best}")
    trainer.test(model, dataloaders=test_loader, ckpt_path=best or None)


if __name__ == "__main__":
    main()
