from __future__ import annotations

import hydra
from pathlib import Path
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from nncert.data.vision import build_classifier_dataloaders
from nncert.utils.reproducibility import setup_reproducibility


def _test_with_checkpoint(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    test_loader,
    ckpt_path: str | None,
) -> None:
    """Run test with robust checkpoint loading across torch/lightning versions."""
    if not ckpt_path:
        trainer.test(model, dataloaders=test_loader)
        return

    # PyTorch 2.6+ changed torch.load default to weights_only=True.
    # For local trusted checkpoints, we explicitly request full load.
    try:
        trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path, weights_only=False)
    except TypeError:
        # Older Lightning versions do not expose `weights_only` in Trainer.test.
        trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)


@hydra.main(config_path=str(Path(__file__).resolve().parents[2] / "configs"), config_name="train_classifier", version_base=None)
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
    _test_with_checkpoint(trainer, model, test_loader, best or None)


if __name__ == "__main__":
    main()
