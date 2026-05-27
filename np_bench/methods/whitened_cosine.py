from typing import Literal, Optional

import numpy as np

from .base import OnlineBaseMethod


RankMode = Literal["fixed", "explained_variance", "threshold"]


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def _select_rank(
    eigvals_desc: np.ndarray,
    *,
    n: int,
    d: int,
    max_rank: Optional[int],
    rank_mode: RankMode,
    explained_variance: float,
    abs_eps: float,
    rel_eps: float,
) -> int:
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be positive or None.")
    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    rank_cap = min(max(n - 1, 1), d)
    if max_rank is not None:
        rank_cap = min(rank_cap, max_rank)

    if eigvals_desc.size == 0:
        return 1

    max_eigval = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eigval)
    valid = eigvals_desc > threshold

    if not np.any(valid):
        return 1

    valid_vals = eigvals_desc[valid]
    valid_count = int(valid_vals.size)

    if rank_mode == "fixed":
        k = rank_cap
    elif rank_mode == "threshold":
        k = valid_count
    elif rank_mode == "explained_variance":
        total = float(np.sum(valid_vals))
        if total <= 0.0:
            return 1
        ratios = np.cumsum(valid_vals) / total
        k = int(np.searchsorted(ratios, explained_variance) + 1)
    else:
        raise ValueError(f"Unknown rank_mode: {rank_mode!r}")

    return max(1, min(k, valid_count, rank_cap))


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
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d). Got {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one row.")

    n, d = X.shape
    Xf = np.asarray(X, dtype=np.float64)
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, n - 1)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_desc = np.maximum(eigvals[::-1], 0.0)
    eigvecs_desc = eigvecs[:, ::-1]

    k = _select_rank(
        eigvals_desc,
        n=n,
        d=d,
        max_rank=max_rank,
        rank_mode=rank_mode,
        explained_variance=explained_variance,
        abs_eps=abs_eps,
        rel_eps=rel_eps,
    )

    top_vals = eigvals_desc[:k]
    top_vecs = eigvecs_desc[:, :k]

    max_eigval = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eigval)
    mask = top_vals > threshold

    if not np.any(mask):
        return np.zeros((d, d), dtype=np.float64)

    top_vals = top_vals[mask]
    top_vecs = top_vecs[:, mask]
    inv_sqrt = 1.0 / np.sqrt(top_vals)
    return (top_vecs * inv_sqrt[None, :]) @ top_vecs.T


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
    ):
        super().__init__()
        self.name = name
        self.eps = float(eps)
        self.rel_eps = float(rel_eps)
        self.max_rank = max_rank
        self.rank_mode = rank_mode
        self.explained_variance = float(explained_variance)
        self.norm_eps = float(norm_eps)
        self.W = None

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
        self.W = _inv_sqrt_cov(
            all_data,
            abs_eps=self.eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank,
            rank_mode=self.rank_mode,
            explained_variance=self.explained_variance,
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

        WH0 = self.mem_H0 @ self.W
        WH1 = self.mem_H1 @ self.W
        mu0 = WH0.mean(axis=0)
        mu1 = WH1.mean(axis=0)
        d_w = mu1 - mu0
        self.w = self.W @ d_w
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