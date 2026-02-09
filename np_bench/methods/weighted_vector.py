from __future__ import annotations
import numpy as np
from .base import BaseMethod
from typing import Optional


class VectorWeightedMethod(BaseMethod):
    name = "Vec (Wgt)"
    needs_weights = True
    needs_seed = False

    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights
        self.w_normalized = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "VectorWeightedMethod":
        if weights is None:
            raise ValueError("weights is required for VectorWeightedMethod")

        w = weights.astype(np.float32)
        self.w_normalized = w / (w.mean() + 1e-9)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w_normalized is None:
            raise ValueError("fit() must be called before score()")
        return (X @ self.w_normalized).astype(np.float32)
