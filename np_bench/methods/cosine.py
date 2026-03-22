from __future__ import annotations

import numpy as np
from .base import BaseMethod


class CosineMethod(BaseMethod):
    name = "Cosine"
    needs_weights = False
    needs_seed = False
    input_space = "embedding"

    def __init__(self) -> None:
        self.prototype: np.ndarray | None = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: float = 0.05,
    ) -> "CosineMethod":
        del H0_train, weights, seed, alpha
        H1 = np.asarray(H1_train, dtype=np.float64)
        if H1.ndim != 2:
            raise ValueError(f"CosineMethod.fit expects 2D H1_train, got shape={H1.shape}")
        if H1.shape[1] <= 1:
            raise ValueError(
                "CosineMethod.fit expects embedding inputs with dim>1; "
                "for scalar precomputed cosine use PrecomputedCosineMethod"
            )
        if H1.shape[0] == 0:
            self.prototype = None
            return self

        p = np.mean(H1, axis=0)
        p_norm = float(np.linalg.norm(p))
        if p_norm <= 1e-12 or not np.isfinite(p_norm):
            self.prototype = None
        else:
            self.prototype = (p / p_norm).astype(np.float64, copy=False)
        return self

    def _raw_cosine(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"CosineMethod.score expects 2D array, got shape={X.shape}")
        if X.shape[1] <= 1:
            raise ValueError(
                "CosineMethod.score expects embedding inputs with dim>1; "
                "for scalar precomputed cosine use PrecomputedCosineMethod"
            )
        if self.prototype is None:
            return np.zeros(X.shape[0], dtype=np.float32)
        if self.prototype.shape[0] != X.shape[1]:
            raise ValueError(
                f"CosineMethod.score input dim mismatch: expected {self.prototype.shape[0]}, got {X.shape[1]}"
            )

        x_norm = np.linalg.norm(X, axis=1)
        s = (X @ self.prototype) / np.maximum(x_norm, 1e-12)

        s = np.clip(s, -1.0, 1.0)
        s = np.nan_to_num(s, nan=-1.0, posinf=1.0, neginf=-1.0)
        return s.astype(np.float32, copy=False)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._raw_cosine(X)
