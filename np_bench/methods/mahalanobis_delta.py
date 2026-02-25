import numpy as np
from .base import OnlineBaseMethod
from .whitened_cosine import _inv_sqrt_cov

class MahalanobisDeltaMethod(OnlineBaseMethod):
    def __init__(self, name="MahalanobisDelta", eps=1e-6, max_rank: int = 128):
        super().__init__()
        self.name = name
        self.eps = eps
        self.max_rank = max_rank
        self.W = None

    def fit(self, H0_train: np.ndarray, H1_train: np.ndarray, *,
            weights=None, seed=None) -> "MahalanobisDeltaMethod":
        # Within-class whitening: center each class, pool residuals
        D0 = H0_train - H0_train.mean(axis=0, keepdims=True)
        D1 = H1_train - H1_train.mean(axis=0, keepdims=True)
        D = np.concatenate([D0, D1], axis=0)
        self.W = _inv_sqrt_cov(D, eps=self.eps, max_rank=self.max_rank)
        self.mem_H0 = H0_train.copy()
        self.mem_H1 = H1_train.copy()
        self._refit_mahal()
        return self

    def _refit_mahal(self) -> None:
        WH0 = self.mem_H0 @ self.W
        WH1 = self.mem_H1 @ self.W
        mu0 = WH0.mean(axis=0)
        mu1 = WH1.mean(axis=0)
        d_w = mu1 - mu0
        self.w = self.W @ d_w
        self.b = -0.5 * float((mu0 + mu1) @ d_w)

    def refit(self) -> None:
        if self.W is not None and self.mem_H0 is not None and self.mem_H1 is not None:
            self._refit_mahal()
        else:
            super().refit()

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        D = (A - B) @ self.W.T
        d = np.linalg.norm(D, axis=1)
        return -d