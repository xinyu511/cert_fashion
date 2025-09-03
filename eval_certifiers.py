# eval_certifiers.py
import os, json, argparse, torch, h5py
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

class H5VotesDataset(Dataset):
    def __init__(self, h5_path, split="val"):
        super().__init__()
        self.f = h5py.File(h5_path, "r")
        self.g = self.f[split]
        self.imgs = self.g["images"]            # (N,3,32,32) uint8
        self.dists = self.g["distributions"]    # (N,S,C) float32
        self.sigmas = self.g["sigmas"][...]     # (S,)
        self.N, self.S, self.C = self.dists.shape

        # flatten (i,s) → list of pairs (use all sigmas)
        self.index = [(i, s) for i in range(self.N) for s in range(self.S)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        x = torch.from_numpy(self.imgs[i]).float() / 255.0     # [3,32,32]
        sigma = torch.tensor([float(self.sigmas[s])], dtype=torch.float32)  # [1]
        target = torch.from_numpy(self.dists[i, s]).float()    # [C]
        return x, sigma, target

    def close(self):
        try: self.f.close()
        except: pass

def soft_ce(pred_logits, target_probs, eps=1e-8):
    pred = torch.softmax(pred_logits, dim=-1)
    return -(target_probs * (pred.add(eps).log())).sum(dim=-1).mean()

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ce_total, l1_total, n = 0.0, 0.0, 0
    for x, sigma, target in loader:
        x = x.to(device)
        sigma = sigma.to(device).view(-1,1)
        target = target.to(device)
        logits = model(x, sigma)                    # model is ImageCertifier
        pred = torch.softmax(logits, dim=-1)
        ce = soft_ce(logits, target)
        l1 = torch.mean(torch.abs(pred - target))
        bs = x.size(0)
        ce_total += ce.item() * bs
        l1_total += l1.item() * bs
        n += bs
    return ce_total / n, l1_total / n

def load_certifier(ckpt_path, device):
    from utils.pl_module import ImageCertifier
    model = ImageCertifier.load_from_checkpoint(ckpt_path, map_location=device)
    return model.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to HDF5 (e.g. ./data/cifar10_certify_gauss_votes.h5)")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--ce_ckpt", required=True, help="Path to CE-trained checkpoint")
    ap.add_argument("--l1_ckpt", required=True, help="Path to L1-trained checkpoint")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed, workers=True)

    ds = H5VotesDataset(args.h5, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)

    ce_model =  (args.ce_ckpt, device)
    l1_model = load_certifier(args.l1_ckpt, device)

    ce_ce, ce_l1 = eval_model(ce_model, dl, device)
    l1_ce, l1_l1 = eval_model(l1_model, dl, device)

    # print a neat 2x2 table
    print("\n=== Loss table (averaged over {} samples of split '{}') ===".format(len(ds), args.split))
    print("{:<22} {:>14} {:>14}".format("", "CE loss", "L1 loss"))
    print("{:<22} {:>14.6f} {:>14.6f}".format("CE-trained model", ce_ce, ce_l1))
    print("{:<22} {:>14.6f} {:>14.6f}".format("L1-trained model", l1_ce, l1_l1))
    print("============================================================\n")

    ds.close()

if __name__ == "__main__":
    main()