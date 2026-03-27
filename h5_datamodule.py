"""Compatibility module.

Use `nncert.data.h5_votes` for canonical implementations.
"""

from nncert.data.h5_votes import H5VotesDataModule, H5VotesDataset, summarize_h5

__all__ = ["H5VotesDataset", "H5VotesDataModule", "summarize_h5"]