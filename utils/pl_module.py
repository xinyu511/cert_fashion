"""Backward-compatibility shim for historical Lightning module paths.

Legacy checkpoints may reference `utils.pl_module.*` in Hydra config.
Canonical implementations live in `nncert.lightning.modules`.
"""

from nncert.lightning.modules import ImageCertifier, ImageClassifier

__all__ = ["ImageClassifier", "ImageCertifier"]
