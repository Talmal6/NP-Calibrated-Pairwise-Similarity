import numpy as np
from typing import Literal, Optional

from .base import OnlineBaseMethod


RankMode = Literal["fixed", "explained_variance", "threshold"]


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _validate_2d(name: str, X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d). Got shape {X.shape}.")


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
    """
    Select the number of PCA directions to keep.

    eigvals_desc must be sorted descending.
    """

    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be positive or None.")

    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    # The centered covariance rank cannot exceed n - 1.
    rank_cap = min(max(n - 1, 1), d)

    if max_rank is not None:
        rank_cap = min(rank_cap, max_rank)

    if eigvals_desc.size == 0:
        return 1

    max_eigval = max(float(eigvals_desc[0]), 0.0)

    # Scale-aware threshold. This is better than using only an absolute eps.
    threshold = max(abs_eps, rel_eps * max_eigval)

    valid = eigvals_desc > threshold

    if not np.any(valid):
        return 1

    valid_vals = eigvals_desc[valid]
    valid_count = len(valid_vals)

    if rank_mode == "fixed":
        k = rank_cap

    elif rank_mode == "threshold":
        k = valid_count

    elif rank_mode == "explained_variance":
        total = float(np.sum(valid_vals))

        if total <= 0.0:
            return 1

        cumulative = np.cumsum(valid_vals)
        ratios = cumulative / total
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
    Compute a truncated inverse-square-root covariance transform.

    Returns W with shape (d, d), where X @ W is whitened in the selected
    PCA subspace.

    Rank selection modes:

    - "fixed":
        Keep up to max_rank valid components.

    - "explained_variance":
        Keep the smallest number of valid components explaining
        `explained_variance` of retained covariance variance.

    - "threshold":
        Keep all components whose eigenvalues exceed the absolute/relative
        threshold.

    Directions outside the selected PCA subspace are zeroed out.
    """

    _validate_2d("X", X)

    n, d = X.shape

    if n == 0:
        raise ValueError("X must contain at least one row.")

    X_float = np.asarray(X, dtype=np.float64)
    X_centered = X_float - X_float.mean(axis=0, keepdims=True)

    # Use n - 1 because the covariance rank after centering is at most n - 1.
    denom = max(n - 1, 1)
    cov = (X_centered.T @ X_centered) / denom

    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns ascending order. Flip to descending.
    eigvals_desc = eigvals[::-1]
    eigvecs_desc = eigvecs[:, ::-1]

    # Numerical safety: covariance eigenvalues should be nonnegative, but tiny
    # negative values can appear from floating-point error.
    eigvals_desc = np.maximum(eigvals_desc, 0.0)

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

    vals = eigvals_desc[:k]
    vecs = eigvecs_desc[:, :k]

    max_eigval = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eigval)

    mask = vals > threshold

    if not np.any(mask):
        # Degenerate case: no reliable variance directions.
        # Returning zeros is safer than amplifying noise with identity / sqrt(eps).
        return np.zeros((d, d), dtype=X.dtype)

    vals = vals[mask]
    vecs = vecs[:, mask]

    inv_sqrt = 1.0 / np.sqrt(vals)

    # W = V_k diag(lambda_k^-1/2) V_k.T
    W = (vecs * inv_sqrt[None, :]) @ vecs.T

    return W.astype(X.dtype, copy=False)


class WhitenedCosineMethod(OnlineBaseMethod):
    """
    PCA-whitened cosine scorer.

    Learns a whitening transform from pooled H0/H1 data, then scores pairs
    by cosine similarity after whitening.
    """

    def __init__(
        self,
        name: str = "PCAWhitenedCosine",
        *,
        abs_eps: float = 1e-6,
        rel_eps: float = 1e-6,
        max_rank: Optional[int] = 128,
        rank_mode: RankMode = "explained_variance",
        explained_variance: float = 0.99,
        norm_eps: float = 1e-12,
    ):
        super().__init__()

        self.name = name

        self.abs_eps = abs_eps
        self.rel_eps = rel_eps
        self.max_rank = max_rank
        self.rank_mode = rank_mode
        self.explained_variance = explained_variance
        self.norm_eps = norm_eps

        self.W: Optional[np.ndarray] = None

        self.mem_H0: Optional[np.ndarray] = None
        self.mem_H1: Optional[np.ndarray] = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
    ) -> "WhitenedCosineMethod":
        _validate_2d("H0_train", H0_train)
        _validate_2d("H1_train", H1_train)

        if H0_train.shape[1] != H1_train.shape[1]:
            raise ValueError(
                "H0_train and H1_train must have the same feature dimension. "
                f"Got {H0_train.shape[1]} and {H1_train.shape[1]}."
            )

        if H0_train.shape[0] == 0 or H1_train.shape[0] == 0:
            raise ValueError("H0_train and H1_train must both contain at least one row.")

        pooled = np.concatenate([H0_train, H1_train], axis=0)

        self.W = _inv_sqrt_cov(
            pooled,
            abs_eps=self.abs_eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank,
            rank_mode=self.rank_mode,
            explained_variance=self.explained_variance,
        )

        self.mem_H0 = H0_train.copy()
        self.mem_H1 = H1_train.copy()

        self._refit_whitened()

        return self

    def _check_is_fitted(self) -> None:
        if self.W is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")

    def _refit_whitened(self) -> None:
        self._check_is_fitted()

        if self.mem_H0 is None or self.mem_H1 is None:
            raise RuntimeError("Missing stored training data. Call fit(...) first.")

        WH0 = self.mem_H0 @ self.W
        WH1 = self.mem_H1 @ self.W

        mu0 = WH0.mean(axis=0)
        mu1 = WH1.mean(axis=0)

        d_w = mu1 - mu0

        # Linear score in original coordinates:
        #
        # score(x) = (x @ W) @ d_w + b
        #          = x @ (W @ d_w) + b
        #
        # W is symmetric because it is built as V diag(...) V.T.
        self.w = self.W @ d_w
        self.b = -0.5 * float((mu0 + mu1) @ d_w)

    def refit(self) -> None:
        if self.W is not None and self.mem_H0 is not None and self.mem_H1 is not None:
            self._refit_whitened()
        else:
            super().refit()

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        self._check_is_fitted()

        _validate_2d("A", A)
        _validate_2d("B", B)

        if A.shape != B.shape:
            raise ValueError(f"A and B must have the same shape. Got {A.shape} and {B.shape}.")

        if A.shape[1] != self.W.shape[0]:
            raise ValueError(
                "A/B feature dimension does not match fitted whitening matrix. "
                f"Got {A.shape[1]}, expected {self.W.shape[0]}."
            )

        WA = _l2_normalize(A @ self.W, eps=self.norm_eps)
        WB = _l2_normalize(B @ self.W, eps=self.norm_eps)

        return np.sum(WA * WB, axis=1)