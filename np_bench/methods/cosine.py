from __future__ import annotations

import numpy as np
from .base import BaseMethod


class CosineMethod(BaseMethod):
    name = "Cosine"
    needs_weights = False
    needs_seed = False

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: float = 0.05,
    ) -> "CosineMethod":
        return self  # no-op

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"CosineMethod.score expects 2D array, got shape={X.shape}")

        # If X is Hadamard(u, v) with u,v L2-normalized, then sum(X) == cosine(u,v).
        s = np.sum(X.astype(np.float64, copy=False), axis=1)

        # Numerical safety
        s = np.clip(s, -1.0, 1.0)

        # Ensure finite outputs (avoid poisoning quantiles)
        s = np.nan_to_num(s, nan=-1.0, posinf=1.0, neginf=-1.0)

        return s.astype(np.float32, copy=False)
