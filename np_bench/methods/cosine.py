from __future__ import annotations
import numpy as np
from .base import BaseMethod


class CosineMethod(BaseMethod):
    name = "Cosine"
    needs_weights = False
    needs_seed = False

    def fit(self, H0_train: np.ndarray, H1_train: np.ndarray, *, weights=None, seed=None) -> "CosineMethod":
        return self  # no-op

    def score(self, X: np.ndarray) -> np.ndarray:
        return X.sum(axis=1).astype(np.float32)
