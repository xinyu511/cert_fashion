#!/usr/bin/env python3
# create_certifier_dataset.py
import argparse, os, json
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.pl_module import ImageClassifier 

def _chunk1d(n, base=1024):
    return (max(1, min(n, base)),)

def _chunk_img(n, base=256):
    return (max(1, min(n, base)), 3, 32, 32)

def _chunk_votes(n, s, c, base_n=128):
    # we usually keep sigma-chunk = 1 for cache friendliness
    return (max(1, min(n, base_n)), 1, c)


@torch.inference_mode()
def _make_split(hf_group, loader, classifier, device, sigmas, samples_per_sigma, mc_chunk, normalize, mean, std):
    """
    Build datasets for a single split and write into an HDF5 group.
    """
    N = len(loader.dataset)
    C = classifier(torch.zeros(1, 3, 32, 32, device=device)).shape[1]
    S = len(sigmas)

    img_dset  = hf_group.create_dataset(
    "images", (N, 3, 32, 32), dtype=np.uint8,  chunks=_chunk_img(N)
    )
    lbl_dset  = hf_group.create_dataset(
        "labels", (N,), dtype=np.int64, chunks=_chunk1d(N)
    )

    # cnt_dset  = hf_group.create_dataset(
    #     "counts", (N, S, C), dtype=np.int32, chunks=_chunk_votes(N, S, C)
    # )
    dist_dset = hf_group.create_dataset(
        "distributions", (N, S, C), dtype=np.float32, chunks=_chunk_votes(N, S, C)
    )
    hf_group.create_dataset("sigmas", data=np.asarray(sigmas, dtype=np.float32))

    idx0 = 0
    for imgs, labels in tqdm(loader, desc=f"{hf_group.name.strip('/')}"):
        B = imgs.size(0)
        # store clean uint8 images/labels
        img_dset[idx0:idx0+B] = imgs.mul(255).byte().numpy()
        lbl_dset[idx0:idx0+B] = labels.numpy()

        imgs = imgs.to(device)  # [B,3,32,32] in [0,1]
        batch_counts = torch.zeros(B, S, C, device=device, dtype=torch.int32)
        remaining = samples_per_sigma
        while remaining > 0:
            cur = min(mc_chunk, remaining)
            base_noise = torch.randn((cur, B, 3, 32, 32), device=device)
            for s_idx, sigma in enumerate(sigmas):
                noisy = imgs.unsqueeze(0) + base_noise * float(sigma)
                noisy = noisy.reshape(cur * B, 3, 32, 32)         
                if normalize:
                    noisy = (noisy - mean) / std
                logits = classifier(noisy)          
                preds  = torch.argmax(logits, dim=1).view(cur, B)   

                # Vectorized voting: one-hot then sum over trials -> [B, C]
                counts = torch.nn.functional.one_hot(preds, num_classes=C).sum(dim=0).to(torch.int32)
                batch_counts[:, s_idx, :] += counts

            remaining -= cur

        # write counts and normalized distributions
        #cnt_dset[idx0:idx0+B]  = batch_counts.cpu().numpy()
        dist_dset[idx0:idx0+B] = (batch_counts.float() / float(samples_per_sigma)).cpu().numpy()
        idx0 += B


def main():
    p = argparse.ArgumentParser(description="Create CIFAR-10 certifier dataset (train/val/test) via MC argmax voting.")
    p.add_argument("--clf_path", type=str, required=True,
                   help="Path to CIFAR-10 classifier checkpoint (.ckpt)")
    p.add_argument("--out_path", type=str, default="./data/cifar10_certify_gauss_votes.h5")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_size", type=int, default=5000, help="Validation size carved from CIFAR-10 train (default 5k)")
    p.add_argument("--val_seed", type=int, default=42, help="Seed for deterministic train/val split")
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=0.5)
    p.add_argument("--num_sigmas", type=int, default=20)
    p.add_argument("--samples_per_sigma", type=int, default=1000, help="MC trials K per (x, sigma)")
    p.add_argument("--mc_chunk", type=int, default=100, help="How many MC samples to process per forward")
    p.add_argument("--normalize", action="store_true",
                   help="If set, normalize inputs with CIFAR-10 mean/std before classifier inference")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier
    classifier = ImageClassifier.load_from_checkpoint(args.clf_path).to(device).eval()
    C = classifier(torch.zeros(1, 3, 32, 32, device=device)).shape[1]
    if C != 10:
        print(f"[warn] Classifier outputs {C} classes (expected 10 for CIFAR-10). Proceeding with C={C}.")

    # Data
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    full_train = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
    test_ds    = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    total_train = len(full_train) 
    val_size = min(max(0, args.val_size), total_train)
    train_size = total_train - val_size
    g = torch.Generator().manual_seed(args.val_seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=g)


    # Loaders (no shuffle; order preserved in HDF5)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Noise grid
    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.num_sigmas, dtype=np.float32)

    # Optional normalization
    if args.normalize:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1)

    # Write HDF5
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with h5py.File(args.out_path, "w") as hf:
        hf.attrs["meta"] = json.dumps({
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dataset": "CIFAR10",
            "splits": {"train": int(train_size), "val": int(val_size), "test": int(len(test_ds))},
            "num_classes": int(C),
            "samples_per_sigma": int(args.samples_per_sigma),
            "sigmas": sigmas.tolist(),
            "vote_mode": "argmax",
            "normalize": bool(args.normalize),
            "note": "counts = argmax votes over K MC trials; distributions = counts / K",
        })

        g_train = hf.create_group("train")
        g_val   = hf.create_group("val")
        g_test  = hf.create_group("test")

        _make_split(g_train, train_loader, classifier, device, sigmas,
                    samples_per_sigma=args.samples_per_sigma, mc_chunk=args.mc_chunk,
                    normalize=args.normalize, mean=mean, std=std)

        _make_split(g_val,   val_loader,   classifier, device, sigmas,
                    samples_per_sigma=args.samples_per_sigma, mc_chunk=args.mc_chunk,
                    normalize=args.normalize, mean=mean, std=std)

        _make_split(g_test,  test_loader,  classifier, device, sigmas,
                    samples_per_sigma=args.samples_per_sigma, mc_chunk=args.mc_chunk,
                    normalize=args.normalize, mean=mean, std=std)

    print(f"Done. Saved to {args.out_path}")
    print("Groups: /train, /val, /test (each has images, labels, sigmas, distributions)")


if __name__ == "__main__":
    main()