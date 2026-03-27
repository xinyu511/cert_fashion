"""Backward-compatibility shim for historical dataset import paths."""

from nncert.data.h5_votes import H5VotesDataModule as H5DataModule
from nncert.data.h5_votes import H5VotesDataset


def loaders(h5_path, batch_size=128, num_workers=8, pin_memory=True):
    """Legacy helper that returns train/val/test dataloaders."""
    dm = H5DataModule(
        h5_path=h5_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_all_sigmas=True,
        apply_norm_if_meta=True,
    )
    dm.setup("fit")
    dm.setup("test")
    return dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()


__all__ = ["H5VotesDataset", "H5DataModule", "loaders"]
