from __future__ import annotations

from typing import Optional

from np_bench.methods.base import BaseMethod
import numpy as np


class LocalThreshold:
    def __init__(
        self,
        method: BaseMethod,
        alpha: float,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        tie_mode: str = "ge",
    ):
        self.method = method
        self.alpha = alpha
        self.weights = weights
        self.seed = seed
        self.tie_mode = tie_mode
        self.tau: float = float("inf")

    def fit(self, H0_train: np.ndarray, H1_train: np.ndarray) -> "LocalThreshold":
        self.method.fit(H0_train, H1_train, weights=self.weights, seed=self.seed)
        s0_train = np.asarray(self.method.score(H0_train), dtype=np.float32).reshape(-1)
        if s0_train.size == 0:
            self.tau = float("inf")
        else:
            self.tau = float(np.quantile(s0_train, 1.0 - float(self.alpha)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # Raw continuous scores (higher => more positive)
        return np.asarray(self.method.score(X), dtype=np.float32).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        if self.tie_mode == "gt":
            return (s > self.tau).astype(np.int32)
        return (s >= self.tau).astype(np.int32)

    def run(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        H0_eval: np.ndarray,
        H1_eval: np.ndarray,
    ) -> tuple[float, float]:
        self.fit(H0_train, H1_train)
        s0_eval = self.score(H0_eval)
        s1_eval = self.score(H1_eval)
        if self.tie_mode == "gt":
            tpr = float(np.mean(s1_eval > self.tau))
            fpr = float(np.mean(s0_eval > self.tau))
        else:
            tpr = float(np.mean(s1_eval >= self.tau))
            fpr = float(np.mean(s0_eval >= self.tau))
        return tpr, fpr


# Backward compatibility for old import/name usage.
local_threshold = LocalThreshold
