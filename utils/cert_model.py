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

class CertNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_mlps:int, clf_path:str):
        super().__init__()

        exp_dir = os.path.abspath(os.path.join(clf_path, "..", ".."))
        conf = OmegaConf.load(os.path.join(exp_dir, 'config.yaml'))

        clf_class = get_class(conf.pl_model._target_)
        clf_module = clf_class.load_from_checkpoint(clf_path)
                  
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        ) for i in range(num_mlps)]) 

        self.backbone = clf_module.clf
        self.backbone.conv1 = nn.Conv2d(self.backbone.conv1.in_channels + num_mlps,
                                        self.backbone.conv1.out_channels,
                                        self.backbone.conv1.kernel_size,
                                        self.backbone.conv1.stride,
                                        self.backbone.conv1.padding,
                                        bias=self.backbone.conv1.bias)


    def forward(self, mu, std):
        """
        mu: Tensor of shape (B, 3, H, W), mean image
        snr: Tensor of shape (B, 1), noise standard deviation at each pixel
        """

        out = []
        for i in range(len(self.mlps)):
            x = self.mlps[i](std) 
            x = rearrange(x, 'B N -> B 1 N 1') @ rearrange(x, 'B N -> B 1 1 N') 
            out.append(x)
        x = torch.cat((mu, *out), dim=1)
        x = self.backbone(x)
        return x
    

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, out_dim))
        for i in range(layers - 1):
            self.layers.append(nn.Linear(out_dim, out_dim))

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.net(x)
        out = rearrange(x, 'B N -> B 1 N 1') @ rearrange(x, 'B N -> B 1 1 N') 
        return x, out


class FuseNet(nn.Module):
    def __init__(self, backbone: nn.Module, clf_path:str):
        super().__init__()

        exp_dir = os.path.abspath(os.path.join(clf_path, "..", ".."))
        conf = OmegaConf.load(os.path.join(exp_dir, 'config.yaml'))

        clf_class = get_class(conf.pl_model._target_)
        clf_module = clf_class.load_from_checkpoint(clf_path)
        self.backbone = clf_module.clf 
                  
        # Extract ResNet layers manually
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
        )
        self.layer1 = self.backbone.layer1  # 16 channels
        self.layer2 = self.backbone.layer2  # 32 channels
        self.layer3 = self.backbone.layer3  # 64 channels

        self.mlp0 = MLPBlock(1, 32, layers = 3)
        self.mlp1 = MLPBlock(32, 32, layers = 3)
        self.mlp2 = MLPBlock(32, 16, layers = 3)
        self.mlp3 = MLPBlock(16, 8, layers = 3)

    def forward(self, image, scalar):

        s0, s0i = self.mlp0(scalar)
        s1, s1i = self.mlp1(s0)
        s2, s2i = self.mlp2(s1)
        s3, s3i = self.mlp3(s2)

       
        out = self.layer0(image) * s0i
        out = self.layer1(out) * s1i
        out = self.layer2(out) * s2i
        out = self.layer3(out) * s3i
 

        # Continue with avgpool and fc
        out = self.backbone.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.backbone.fc(out)
        return out
