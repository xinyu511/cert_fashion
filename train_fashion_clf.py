# train_fashion_clf.py
import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils.pl_module import ImageClassifier

def make_dataloaders(batch_size=256, num_workers=4):
    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),   # keep [0,1], same as your CIFAR pipeline
    ])
    train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

if __name__ == "__main__":
    # Backbone: reuse cifar10_resnet20 from hub (expects 3x32x32; our tfm matches that)
    clf_cfg = {
        "_target_": "utils.pl_module.ImageClassifier",
        "clf": {
            "_target_": "torch.hub.load",
            "repo_or_dir": "chenyaofo/pytorch-cifar-models",
            "model": "cifar10_resnet20",
            "pretrained": False,
        },
        "num_classes": 10,
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-3, "weight_decay": 1e-5},
        "scheduler": {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 20},
    }
    model = instantiate(clf_cfg, _recursive_=False)

    train_loader, val_loader = make_dataloaders()
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1,
        dirpath="logs/clf/resnet20/fashionmnist/lightning_logs/version_0/checkpoints",
        filename="saved_epoch={epoch}"
    )
    trainer = pl.Trainer(
        max_epochs=20, accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0], precision="16-mixed", callbacks=[ckpt], log_every_n_steps=50
    )
    trainer.fit(model, train_loader, val_loader)