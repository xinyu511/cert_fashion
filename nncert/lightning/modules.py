from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics.classification import Accuracy


class ImageClassifier(pl.LightningModule):
    """Generic image classifier wrapper for Hydra-instantiated backbones."""

    def __init__(
        self,
        clf: nn.Module,
        num_classes: int,
        optimizer: DictConfig,
        scheduler: DictConfig | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.clf = instantiate(clf)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clf(x)

    def _shared_step(self, batch: Any):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer_factory = instantiate(self.hparams.optimizer)
        optimizer = optimizer_factory(self.parameters())

        if not self.hparams.scheduler:
            return {"optimizer": optimizer}

        scheduler_factory = instantiate(self.hparams.scheduler)
        scheduler = scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class ImageCertifier(pl.LightningModule):
    """Model that predicts a class distribution from image + noise scalar."""

    def __init__(
        self,
        cert: nn.Module,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig | None = None,
        loss_type: str = "l1",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cert = instantiate(cert)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        valid = {"l1", "ce", "kl"}
        if loss_type not in valid:
            raise ValueError(f"loss_type must be one of {valid}, got: {loss_type}")
        self.loss_type = loss_type

    def forward(self, image: torch.Tensor, noise_param: torch.Tensor):
        return self.cert(image, noise_param)

    def _loss(self, pred_logits: torch.Tensor, target_dist: torch.Tensor):
        pred_dist = F.softmax(pred_logits, dim=-1)

        if self.loss_type == "l1":
            loss = F.l1_loss(pred_dist, target_dist)
        elif self.loss_type == "ce":
            eps = 1e-8
            loss = -(target_dist * torch.log(pred_dist + eps)).sum(dim=-1).mean()
        elif self.loss_type == "kl":
            log_pred = F.log_softmax(pred_logits, dim=-1)
            loss = F.kl_div(log_pred, target_dist, reduction="batchmean")
        else:
            raise RuntimeError("Unreachable loss type branch.")

        return loss, pred_dist

    def _shared_step(self, batch: Any):
        imgs, sigmas, targets = batch
        sigmas = sigmas.to(dtype=torch.float32, device=self.device).view(-1, 1)
        logits = self(imgs, sigmas)
        loss, pred_dist = self._loss(logits, targets)
        return loss, pred_dist

    def training_step(self, batch: Any, batch_idx: int):
        loss, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_dist = self._shared_step(batch)
        _, _, targets = batch
        l1 = F.l1_loss(pred_dist, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_l1_loss", l1, prog_bar=True, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, pred_dist = self._shared_step(batch)
        _, _, targets = batch
        l1 = F.l1_loss(pred_dist, targets)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_l1_loss", l1, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer_factory = instantiate(self.optimizer_cfg)
        optimizer = optimizer_factory(self.parameters())

        if not self.scheduler_cfg:
            return {"optimizer": optimizer}

        scheduler_factory = instantiate(self.scheduler_cfg)
        scheduler = scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
