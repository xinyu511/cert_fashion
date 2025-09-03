# h5_datamodule.py
import json, h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

class H5VotesDataset(Dataset):
    """
    Produces tuples: (clean_image_float, sigma_scalar, target_distribution)
    - clean_image_float: [3, 32, 32], in [0,1] (optionally normalized)
    - sigma_scalar: [1], float32
    - target_distribution: [C], float32, sums≈1
    """
    def __init__(self, h5_path, split="train", use_all_sigmas=True, sigma_indices=None,
                 apply_norm_if_meta=True, pin_memory=False):
        super().__init__()
        self.h5 = h5py.File(h5_path, "r")
        self.g  = self.h5[split]
        self.imgs = self.g["images"]          # (N,3,32,32) uint8
        self.labels = self.g["labels"]        # (N,)
        self.dists = self.g["distributions"]  # (N,S,C) float32
        self.sigmas = self.g["sigmas"][...]   # (S,)
        self.N, self.S, self.C = self.dists.shape[0], self.dists.shape[1], self.dists.shape[2]

        # Build (img_idx, s_idx) index mapping to flatten over sigmas
        if use_all_sigmas:
            s_idxs = np.arange(self.S) if sigma_indices is None else np.array(sigma_indices, dtype=np.int64)
        else:
            # sample a single sigma per image (useful for quick debugging)
            s_idxs = np.random.randint(0, self.S, size=(self.N,))
        self.index_map = []
        if use_all_sigmas:
            for i in range(self.N):
                for s in s_idxs:
                    self.index_map.append((i, int(s)))
        else:
            for i in range(self.N):
                self.index_map.append((i, int(s_idxs[i])))

        self.normalize = None
        self.meta = json.loads(self.h5.attrs["meta"])
        if apply_norm_if_meta and bool(self.meta.get("normalize", False)):
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std  = torch.tensor([0.2470, 0.2435, 0.2616])
            self.normalize = Normalize(mean.tolist(), std.tolist())

        self.pin_memory = pin_memory

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        i, s = self.index_map[idx]
        img = torch.from_numpy(self.imgs[i].astype(np.float32) / 255.0)     # [3,32,32] in [0,1]
        if self.normalize is not None:
            img = self.normalize(img)
        sigma = torch.tensor([float(self.sigmas[s])], dtype=torch.float32)   # [1]
        dist  = torch.from_numpy(self.dists[i, s].astype(np.float32))        # [C], sums≈1
        return img, sigma, dist

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass


class H5DataModule(torch.utils.data.Dataset):
    """
    Simple factory; use loaders() function below for Lightning-friendly loaders.
    """
    pass


def loaders(h5_path, batch_size=128, num_workers=8, pin_memory=True):
    train_ds = H5VotesDataset(h5_path, "train", use_all_sigmas=True, pin_memory=pin_memory)
    val_ds   = H5VotesDataset(h5_path, "val",   use_all_sigmas=True, pin_memory=pin_memory)
    test_ds  = H5VotesDataset(h5_path, "test",  use_all_sigmas=True, pin_memory=pin_memory)

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return _dl(train_ds, True), _dl(val_ds, False), _dl(test_ds, False)