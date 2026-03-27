from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch


def setup_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Set all commonly used RNG seeds.

    Args:
        seed: Global seed.
        deterministic: If True, request deterministic CUDA behavior.
    """
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not have deterministic implementations.
            pass
