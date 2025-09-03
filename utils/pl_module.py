import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import Accuracy
import torchvision
from einops import rearrange
import os


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        clf: nn.Module,
        num_classes: int,
        optimizer: DictConfig,
        scheduler: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.clf = instantiate(clf)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.clf(x)

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.shared_step(batch)
        self.train_acc(preds, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.shared_step(batch)
        self.val_acc(preds, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return 

    def test_step(self, batch, batch_idx):       
        loss, preds, y = self.shared_step(batch)
        self.test_acc(preds, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc, prog_bar=True)
        return 

    def configure_optimizers(self):

        optimizer = instantiate(self.hparams.optimizer)
        opt = optimizer(self.parameters())

        if self.hparams.scheduler:
            scheduler = instantiate(self.hparams.scheduler)
            sch = scheduler(opt)
            return {'optimizer': opt, 'lr_scheduler':{'scheduler': sch, 'interval': "epoch"}}

        else:
            return {'optimizer': opt}

class ImageCertifier(pl.LightningModule):
    def __init__(self, cert: nn.Module, optimizer_cfg: DictConfig, scheduler_cfg: DictConfig=None,
                 loss_type: str="l1"):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cert = instantiate(cert)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        assert loss_type in ("l1", "ce"), "loss_type must be 'l1' or 'ce'"
        self.loss_type = loss_type

    def forward(self, image, noise_param):
        return self.cert(image, noise_param)

    def _loss(self, pred_logits, target_dist):
        pred_dist = F.softmax(pred_logits, dim=-1)

        if self.loss_type == "l1":
            loss = F.l1_loss(pred_dist, target_dist)

        elif self.loss_type == "ce":
            # Cross entropy with soft labels
            eps = 1e-8
            loss = -(target_dist * torch.log(pred_dist + eps)).sum(dim=-1).mean()

        return loss, pred_dist

    def training_step(self, batch, batch_idx):
        imgs, sigmas, targets = batch
        sigmas = sigmas.to(dtype=torch.float32, device=self.device).view(-1, 1)

        logits = self(imgs, sigmas)
        loss, pred_dist = self._loss(logits, targets)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, sigmas, targets = batch
        sigmas = sigmas.to(dtype=torch.float32, device=self.device).view(-1, 1)

        logits = self(imgs, sigmas)
        loss, pred_dist = self._loss(logits, targets)

        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, sigmas, targets = batch
        sigmas = sigmas.to(dtype=torch.float32, device=self.device).view(-1, 1)

        logits = self(imgs, sigmas)
        loss, pred_dist = self._loss(logits, targets)

        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # Hard-coded Adam + CosineAnnealingLR
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step every epoch
            },
        }