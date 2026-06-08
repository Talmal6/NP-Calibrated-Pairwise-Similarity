from __future__ import annotations

from typing import Literal, Optional

import numpy as np


RankMode = Literal["fixed", "explained_variance", "threshold"]
WhiteningType = Literal["zca", "pca", "zca_cor", "pca_cor"]


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


def _validate_2d_data(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d). Got {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one row.")
    return np.asarray(X, dtype=np.float64)


def _covariance_from_data(
    X: np.ndarray,
    *,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(1, n - 1)
    cov = 0.5 * (cov + cov.T)

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")
    if shrinkage > 0.0:
        diag = np.diag(np.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag
        cov = 0.5 * (cov + cov.T)

    return cov, mean.reshape(-1)


def _sorted_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_desc = np.maximum(eigvals[::-1], 0.0)
    eigvecs_desc = eigvecs[:, ::-1]
    return eigvals_desc, eigvecs_desc


def _retained_eigenpairs(
    eigvals_desc: np.ndarray,
    eigvecs_desc: np.ndarray,
    *,
    n: int,
    d: int,
    max_rank: Optional[int],
    rank_mode: RankMode,
    explained_variance: float,
    abs_eps: float,
    rel_eps: float,
) -> tuple[np.ndarray, np.ndarray, int]:
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
        return top_vals[:0], top_vecs[:, :0], 0

    vals = top_vals[mask]
    vecs = top_vecs[:, mask]
    return vals, vecs, int(vals.shape[0])


def _zca_from_eigenpairs(vals: np.ndarray, vecs: np.ndarray, d: int) -> np.ndarray:
    if vals.size == 0:
        return np.zeros((d, d), dtype=np.float64)
    inv_sqrt = 1.0 / np.sqrt(vals)
    return (vecs * inv_sqrt[None, :]) @ vecs.T


def _pca_from_eigenpairs(vals: np.ndarray, vecs: np.ndarray, d: int) -> np.ndarray:
    if vals.size == 0:
        return np.zeros((0, d), dtype=np.float64)
    inv_sqrt = 1.0 / np.sqrt(vals)
    return inv_sqrt[:, None] * vecs.T


def compute_whitening_matrix(
    X: np.ndarray,
    *,
    whitening_type: WhiteningType,
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-6,
    max_rank: Optional[int] = 128,
    rank_mode: RankMode = "explained_variance",
    explained_variance: float = 0.99,
    shrinkage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(W, mean, selected_rank)`` for canonical whitening variants."""
    if whitening_type not in {"zca", "pca", "zca_cor", "pca_cor"}:
        raise ValueError(f"Unknown whitening_type: {whitening_type!r}")

    Xf = _validate_2d_data(X)
    n, d = Xf.shape
    cov, mean = _covariance_from_data(Xf, shrinkage=shrinkage)

    # W whitens centered inputs (X - mean); legacy callers may intentionally score uncentered rows for backward compatibility.
    if whitening_type in {"zca", "pca"}:
        eigvals_desc, eigvecs_desc = _sorted_eigh(cov)
        vals, vecs, rank = _retained_eigenpairs(
            eigvals_desc,
            eigvecs_desc,
            n=n,
            d=d,
            max_rank=max_rank,
            rank_mode=rank_mode,
            explained_variance=explained_variance,
            abs_eps=abs_eps,
            rel_eps=rel_eps,
        )
        if whitening_type == "zca":
            return _zca_from_eigenpairs(vals, vecs, d), mean, rank
        return _pca_from_eigenpairs(vals, vecs, d), mean, rank

    diag = np.maximum(np.diag(cov), 0.0)
    max_diag = max(float(np.max(diag)), 0.0) if diag.size else 0.0
    diag_threshold = max(abs_eps, rel_eps * max_diag)

    inv_std = np.zeros(d, dtype=np.float64)
    valid_std = diag > diag_threshold
    inv_std[valid_std] = 1.0 / np.sqrt(diag[valid_std])

    corr = (inv_std[:, None] * cov) * inv_std[None, :]
    corr = 0.5 * (corr + corr.T)

    eigvals_desc, eigvecs_desc = _sorted_eigh(corr)
    vals, vecs, rank = _retained_eigenpairs(
        eigvals_desc,
        eigvecs_desc,
        n=n,
        d=d,
        max_rank=max_rank,
        rank_mode=rank_mode,
        explained_variance=explained_variance,
        abs_eps=abs_eps,
        rel_eps=rel_eps,
    )

    if whitening_type == "zca_cor":
        return _zca_from_eigenpairs(vals, vecs, d) @ np.diag(inv_std), mean, rank
    return _pca_from_eigenpairs(vals, vecs, d) * inv_std[None, :], mean, rank
