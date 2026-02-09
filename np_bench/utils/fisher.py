from __future__ import annotations
import numpy as np


def get_fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fisher score per feature:
      (mu1 - mu0)^2 / (var0 + var1 + eps)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows != y size: {X.shape[0]} != {y.shape[0]}")

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.size == 0 or X1.size == 0:
        return np.zeros(X.shape[1], dtype=np.float32)

    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)
    var0 = X0.var(axis=0)
    var1 = X1.var(axis=0)

    scores = ((mean1 - mean0) ** 2) / (var0 + var1 + 1e-9)
    return scores.astype(np.float32)
