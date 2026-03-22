from __future__ import annotations

import numpy as np

from .base import BaseMethod


class PrecomputedCosineMethod(BaseMethod):
    """Explicit scalar-score baseline using precomputed cosine_to_anchor."""

    name = "PrecomputedCosine"
    needs_weights = False
    needs_seed = False
    input_space = "scalar_score"

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: float = 0.05,
    ) -> "PrecomputedCosineMethod":
        del weights, seed, alpha
        self._validate_scalar_matrix(H0_train, where="fit(H0_train)")
        self._validate_scalar_matrix(H1_train, where="fit(H1_train)")
        return self

    @staticmethod
    def _validate_scalar_matrix(X: np.ndarray, *, where: str) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(
                f"PrecomputedCosine expects scalar score matrix (N,1) in {where}, got shape={X.shape}"
            )

    def score(self, X: np.ndarray) -> np.ndarray:
        self._validate_scalar_matrix(X, where="score")
        s = np.asarray(X[:, 0], dtype=np.float64)
        s = np.clip(s, -1.0, 1.0)
        s = np.nan_to_num(s, nan=-1.0, posinf=1.0, neginf=-1.0)
        return s.astype(np.float32, copy=False)
