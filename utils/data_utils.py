
from torch.utils.data import Dataset
from hydra.utils import get_class
import torch 
import torch.nn.functional as F
import os 
from omegaconf import OmegaConf

class NoisyImgDataset(Dataset):
    def __init__(self, base_dataset, gpu, pre_transform, post_transform, ckpt_path, max_noise_std, mc_samples, num_classes):
        super().__init__()
        
        self.base_dataset = base_dataset
        self.max_noise_std = max_noise_std
        self.mc_samples = mc_samples

        exp_dir = os.path.abspath(os.path.join(ckpt_path, "..", ".."))
        conf = OmegaConf.load(os.path.join(exp_dir, 'config.yaml'))

        model_class = get_class(conf.pl_model._target_)
        self.model = model_class.load_from_checkpoint(ckpt_path)
        self.device = torch.device('cpu')
        self.model.eval().to(self.device)
        

        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.num_classes = num_classes


    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, idx):
    
        img, label = self.base_dataset[idx]
        if self.pre_transform:
            img = self.pre_transform(img).to(self.device)

        # Add noise        
        img_batch = img.unsqueeze(0).repeat(self.mc_samples, 1, 1, 1)
        noise_std = torch.rand(1, device=self.device)*self.max_noise_std
        noisy_imgs = img_batch + torch.randn_like(img_batch) * noise_std

        if self.post_transform:
            noisy_imgs = self.post_transform(noisy_imgs)

        with torch.no_grad():
            preds = self.model(noisy_imgs)
            preds = torch.argmax(preds, dim=-1)
            preds = F.one_hot(preds, num_classes=self.num_classes).sum(dim=0) / self.mc_samples

        return img, noise_std, preds, label