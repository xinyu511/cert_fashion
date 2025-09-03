# train_cert_offline.py
import os
import json
import h5py
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar

from utils.pl_module import ImageCertifier
from utils.datasets import H5VotesDataset

def _make_loaders(h5_path: str, batch_size: int, num_workers: int, pin_memory: bool = True):
    # Build datasets
    train_ds = H5VotesDataset(h5_path, split="train", use_all_sigmas=True, apply_norm_if_meta=True)
    val_ds   = H5VotesDataset(h5_path, split="val",   use_all_sigmas=True, apply_norm_if_meta=True)
    test_ds  = H5VotesDataset(h5_path, split="test",  use_all_sigmas=True, apply_norm_if_meta=True)

    def _dl(ds, shuffle: bool):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )
    return _dl(train_ds, True), _dl(val_ds, False), _dl(test_ds, False)

def _summarize_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        print("=== H5 Summary ===")
        print("Meta:", meta)
        for split in ["train", "val", "test"]:
            if split in f:
                g = f[split]
                keys = list(g.keys())
                print(f"[{split}] keys={keys}")
                print("  images:", g["images"].shape, g["images"].dtype)
                print("  labels:", g["labels"].shape, g["labels"].dtype)
                print("  dists :", g["distributions"].shape, g["distributions"].dtype)
                print("  sigmas:", g["sigmas"].shape, g["sigmas"].dtype)
        print("==================")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_offline(cfg: DictConfig):
    # 0) Print full resolved config
    print("\n--- Configuration (resolved) ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------\n")

    # 1) H5 sanity
    h5_path = cfg.get("offline_data_path")
    if not h5_path or not os.path.exists(h5_path):
        raise FileNotFoundError(f"offline_data_path not found: {h5_path}")
    _summarize_h5(h5_path)

    # 2) Data loaders
    train_loader, val_loader, test_loader = _make_loaders(
        h5_path=h5_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    from utils.pl_module import ImageCertifier
    from utils.cert_model import FuseNet
    import torch.hub

    # --- inside train_offline(cfg): replace model = instantiate(...) with: ---

    # 1) Build the backbone exactly like your config would
    backbone = torch.hub.load(
        repo_or_dir="chenyaofo/pytorch-cifar-models",
        model="cifar10_resnet20",
        pretrained=False,
    )

    # 2) Build the cert network (FuseNet) with your classifier ckpt
    cert_net = FuseNet(
        backbone=backbone,
        clf_path=cfg.classifier_ckpt_path,   # pass absolute path if Hydra changes run dir
    )

    # 3) Build the Lightning module; optimizer/scheduler are hard-coded inside it
    model = ImageCertifier(
        cert=cert_net,
        optimizer_cfg=None,                  # ignored by your hard-coded configure_optimizers
        scheduler_cfg=None,                  # ignored by your hard-coded configure_optimizers
        loss_type=cfg.training.loss_type,    # "l1" or "ce"
    )

    # 4) Callbacks
    monitor_metric = "val_loss" if len(val_loader.dataset) > 0 else "train_loss"
    ckpt_dir = cfg.get("checkpoints_dir", f"logs/cert/FuseNet_resnet20/{cfg.dataset.name}/offline/{cfg.training.loss_type}")
    ckpt = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=ckpt_dir,
        filename="best-{epoch}-" + monitor_metric + "={"+monitor_metric+":.4f}",
        save_top_k=1,
        mode="min"
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")
    tqdm_bar = TQDMProgressBar(refresh_rate=10)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        devices=cfg.training.devices,
        precision=cfg.training.get("precision", "16-mixed"),
        callbacks=[ckpt, lrmon, tqdm_bar],  # ← TQDM, not Rich
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
    )

    # 6) Train / Validate / Test
    print("Starting offline training...")
    if len(val_loader.dataset) > 0:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)

    best = ckpt.best_model_path if ckpt.best_model_path else None
    print(f"Training complete! Best: {best}")
    if len(test_loader.dataset) > 0:
        trainer.test(model, dataloaders=test_loader, ckpt_path=best or None)

if __name__ == "__main__":
    train_offline()