from typing import Optional

import numpy as np

from .base import OnlineBaseMethod
from .whitening_core import (
    RankMode,
    WhiteningType,
    _select_rank as _select_rank,
    compute_whitening_matrix,
)


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def _inv_sqrt_cov(
    X: np.ndarray,
    *,
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-6,
    max_rank: Optional[int] = 128,
    rank_mode: RankMode = "explained_variance",
    explained_variance: float = 0.99,
) -> np.ndarray:
    """
    Compute a truncated inverse-square-root of the covariance of X.

    Returns W of shape (d, d) such that X @ W is approximately whitened in the
    retained PCA subspace. Directions outside the retained subspace are zeroed
    out rather than amplified.
    """
    W, _, _ = compute_whitening_matrix(
        X,
        whitening_type="zca",
        abs_eps=abs_eps,
        rel_eps=rel_eps,
        max_rank=max_rank,
        rank_mode=rank_mode,
        explained_variance=explained_variance,
        shrinkage=0.0,
    )
    return W


class WhitenedCosineMethod(OnlineBaseMethod):
    def __init__(
        self,
        name: str = "WhitenedCosine",
        *,
        eps: float = 1e-6,
        rel_eps: float = 1e-6,
        max_rank: Optional[int] = 128,
        rank_mode: RankMode = "explained_variance",
        explained_variance: float = 0.99,
        norm_eps: float = 1e-12,
        whitening_type: WhiteningType = "zca",
        shrinkage: float = 0.0,
        abs_eps: Optional[float] = None,
        pca_whiten_abs_eps: Optional[float] = None,
        pca_whiten_rel_eps: Optional[float] = None,
        pca_whiten_max_rank: Optional[int] = None,
        pca_whiten_rank_mode: Optional[RankMode] = None,
        pca_whiten_explained_variance: Optional[float] = None,
        pca_whiten_norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.name = name
        if abs_eps is not None:
            eps = abs_eps
        if pca_whiten_abs_eps is not None:
            eps = pca_whiten_abs_eps
        if pca_whiten_rel_eps is not None:
            rel_eps = pca_whiten_rel_eps
        if pca_whiten_max_rank is not None:
            max_rank = pca_whiten_max_rank
        if pca_whiten_rank_mode is not None:
            rank_mode = pca_whiten_rank_mode
        if pca_whiten_explained_variance is not None:
            explained_variance = pca_whiten_explained_variance
        if pca_whiten_norm_eps is not None:
            norm_eps = pca_whiten_norm_eps

        if whitening_type not in {"zca", "pca", "zca_cor", "pca_cor"}:
            raise ValueError(f"Unknown whitening_type: {whitening_type!r}")
        self.eps = float(eps)
        self.abs_eps = self.eps
        self.rel_eps = float(rel_eps)
        self.max_rank = max_rank
        self.rank_mode = rank_mode
        self.explained_variance = float(explained_variance)
        self.norm_eps = float(norm_eps)
        self.whitening_type = whitening_type
        self.shrinkage = float(shrinkage)
        self.W = None
        self.mean_ = None
        self.selected_rank_ = 0

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
    ) -> "WhitenedCosineMethod":
        del weights, seed

        if H0_train.ndim != 2 or H1_train.ndim != 2:
            raise ValueError("H0_train and H1_train must have shape (n, d).")
        if H0_train.shape[1] != H1_train.shape[1]:
            raise ValueError(
                "H0_train and H1_train must have the same feature dimension. "
                f"Got {H0_train.shape[1]} and {H1_train.shape[1]}."
            )
        if H0_train.shape[0] == 0 or H1_train.shape[0] == 0:
            raise ValueError("H0_train and H1_train must both contain at least one row.")

        all_data = np.concatenate([H0_train, H1_train], axis=0)
        self.W, self.mean_, self.selected_rank_ = compute_whitening_matrix(
            all_data,
            whitening_type=self.whitening_type,
            abs_eps=self.eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank,
            rank_mode=self.rank_mode,
            explained_variance=self.explained_variance,
            shrinkage=self.shrinkage,
        )
        self.mem_H0 = H0_train.copy()
        self.mem_H1 = H1_train.copy()
        self._refit_whitened()
        return self

    def _refit_whitened(self) -> None:
        if self.W is None:
            raise RuntimeError("Missing whitening matrix W. Call fit(...) first.")
        if self.mem_H0 is None or self.mem_H1 is None:
            raise RuntimeError("Missing stored training data. Call fit(...) first.")

        if self.whitening_type == "zca":
            WH0 = self.mem_H0 @ self.W
            WH1 = self.mem_H1 @ self.W
        else:
            WH0 = self.mem_H0 @ self.W.T
            WH1 = self.mem_H1 @ self.W.T
        mu0 = WH0.mean(axis=0)
        mu1 = WH1.mean(axis=0)
        d_w = mu1 - mu0
        if self.whitening_type == "zca":
            self.w = self.W @ d_w
        else:
            self.w = self.W.T @ d_w
        self.b = -0.5 * float((mu0 + mu1) @ d_w)

    def refit(self) -> None:
        if self.W is not None and self.mem_H0 is not None and self.mem_H1 is not None:
            self._refit_whitened()
        else:
            super().refit()

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Missing whitening matrix W. Call fit(...) first.")
        WA = _l2_normalize(A @ self.W.T, eps=self.norm_eps)
        WB = _l2_normalize(B @ self.W.T, eps=self.norm_eps)
        return np.sum(WA * WB, axis=1)

    def linear_form(self) -> tuple[str, np.ndarray, float]:
        if self.w is None:
            raise RuntimeError("WhitenedCosineMethod.linear_form() called before fit().")
        return ("hadamard_linear", np.asarray(self.w, dtype=np.float64), float(self.b))
