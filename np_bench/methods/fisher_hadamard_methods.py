from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .base import OnlineBaseMethod


RankMode = Literal["fixed", "explained_variance", "threshold"]


def _validate_2d(name: str, X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d). Got {arr.shape}.")
    return arr


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


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
    if eigvals_desc.ndim != 1:
        raise ValueError("eigvals_desc must be 1D.")
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be positive or None.")
    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    if eigvals_desc.size == 0:
        return 0

    rank_cap = min(max(n - 1, 1), d)
    if max_rank is not None:
        rank_cap = min(rank_cap, max_rank)

    max_eig = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eig)
    valid_mask = eigvals_desc > threshold
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return 0

    valid_vals = eigvals_desc[valid_mask]

    if rank_mode == "fixed":
        k = rank_cap
    elif rank_mode == "threshold":
        k = valid_count
    elif rank_mode == "explained_variance":
        total = float(np.sum(valid_vals))
        if total <= 0.0:
            return 0
        ratios = np.cumsum(valid_vals) / total
        k = int(np.searchsorted(ratios, explained_variance) + 1)
    else:
        raise ValueError(f"Unknown rank_mode: {rank_mode!r}")

    return max(1, min(k, valid_count, rank_cap))


def _compute_whitening_matrix(
    X_cov: np.ndarray,
    *,
    center: bool,
    abs_eps: float,
    rel_eps: float,
    max_rank: Optional[int],
    rank_mode: RankMode,
    explained_variance: float,
    shrinkage: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    X_cov = _validate_2d("X_cov", X_cov).astype(np.float64, copy=False)
    n, d = X_cov.shape
    if n == 0:
        return np.zeros((d, d), dtype=np.float64), 0, np.zeros(d, dtype=np.float64)

    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must be in [0, 1).")

    mean = X_cov.mean(axis=0)
    if center:
        X_work = X_cov - mean[None, :]
    else:
        X_work = X_cov

    cov = (X_work.T @ X_work) / max(n - 1, 1)

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
    if k == 0:
        return np.zeros((d, d), dtype=np.float64), 0, mean

    vals = eigvals_desc[:k]
    vecs = eigvecs_desc[:, :k]

    max_eig = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eig)
    valid = vals > threshold
    if not np.any(valid):
        return np.zeros((d, d), dtype=np.float64), 0, mean

    vals = vals[valid]
    vecs = vecs[:, valid]

    mean_valid = float(np.mean(vals))
    vals_used = (1.0 - shrinkage) * vals + shrinkage * mean_valid
    vals_used = np.maximum(vals_used, threshold)

    inv_sqrt = 1.0 / np.sqrt(vals_used)
    W = (vecs * inv_sqrt[None, :]) @ vecs.T

    return W, int(vals_used.shape[0]), mean


class HadamardCosineMethod(OnlineBaseMethod):
    """
    Cosine is a fixed linear score over Hadamard features:
    score(q, c) = 1^T (q * c). This class keeps that fixed scorer.
    """

    def __init__(
        self,
        name: str = "HadamardCosine",
        *,
        normalize_pair_inputs: bool = False,
        norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.name = name
        self.normalize_pair_inputs = bool(normalize_pair_inputs)
        self.norm_eps = float(norm_eps)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "HadamardCosineMethod":
        del weights, seed
        H0 = _validate_2d("H0_train", H0_train)
        H1 = _validate_2d("H1_train", H1_train)
        if H0.shape[1] != H1.shape[1]:
            raise ValueError(
                "H0_train and H1_train must have the same feature dimension. "
                f"Got {H0.shape[1]} and {H1.shape[1]}."
            )

        d = H0.shape[1]
        self.w = np.ones(d, dtype=np.float64)
        self.b = 0.0
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")
        X = _validate_2d("X", X)
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"X feature dimension mismatch: expected {self.w.shape[0]}, got {X.shape[1]}."
            )
        return X @ self.w + self.b

    def linear_form(self) -> tuple[str, np.ndarray, float]:
        if self.w is None:
            raise RuntimeError("HadamardCosineMethod.linear_form() called before fit().")
        if self.normalize_pair_inputs:
            raise RuntimeError("normalized pair-input HadamardCosine is not a clean q*anchor linear form")
        return ("hadamard_linear", np.asarray(self.w, dtype=np.float64), float(self.b))

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")

        A = _validate_2d("A", A)
        B = _validate_2d("B", B)
        if A.shape != B.shape:
            raise ValueError(f"A and B must have the same shape. Got {A.shape} and {B.shape}.")
        if A.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"A/B feature dimension mismatch: expected {self.w.shape[0]}, got {A.shape[1]}."
            )

        if self.normalize_pair_inputs:
            A = _l2_normalize(A.astype(np.float64, copy=False), eps=self.norm_eps)
            B = _l2_normalize(B.astype(np.float64, copy=False), eps=self.norm_eps)

        H = A * B
        return self.score(H)


class StabilizedFisherHadamardMethod(OnlineBaseMethod):
    """
    Fisher-whitened Hadamard scorer.

    Cosine on normalized pairs is a fixed linear score on Hadamard features;
    this method learns discriminative Hadamard weights with stabilized
    whitening and LDA-style mean separation.
    """

    def __init__(
        self,
        name: str = "FWHS",
        *,
        covariance_mode: Literal["pooled", "within"] = "within",
        abs_eps: float = 1e-6,
        rel_eps: float = 1e-6,
        max_rank: Optional[int] = 128,
        rank_mode: RankMode = "fixed",
        explained_variance: float = 0.99,
        shrinkage: float = 0.01,
        fallback_to_cosine: bool = True,
        normalize_pair_inputs: bool = False,
        norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if covariance_mode not in {"pooled", "within"}:
            raise ValueError("covariance_mode must be 'pooled' or 'within'.")

        self.name = name
        self.covariance_mode = covariance_mode
        self.abs_eps = float(abs_eps)
        self.rel_eps = float(rel_eps)
        self.max_rank = max_rank
        self.rank_mode = rank_mode
        self.explained_variance = float(explained_variance)
        self.shrinkage = float(shrinkage)
        self.fallback_to_cosine = bool(fallback_to_cosine)
        self.normalize_pair_inputs = bool(normalize_pair_inputs)
        self.norm_eps = float(norm_eps)

        self.W: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.mu0_: Optional[np.ndarray] = None
        self.mu1_: Optional[np.ndarray] = None
        self.mu0_raw_: Optional[np.ndarray] = None
        self.mu1_raw_: Optional[np.ndarray] = None
        self.selected_rank_: int = 0
        self.fallback_used: bool = False

    def _set_cosine_fallback(self, d: int) -> None:
        self.W = np.eye(d, dtype=np.float64)
        self.w = np.ones(d, dtype=np.float64)
        self.b = 0.0
        self.mu0_ = None
        self.mu1_ = None
        self.selected_rank_ = 0
        self.fallback_used = True

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "StabilizedFisherHadamardMethod":
        del weights, seed
        H0 = _validate_2d("H0_train", H0_train).astype(np.float64, copy=False)
        H1 = _validate_2d("H1_train", H1_train).astype(np.float64, copy=False)

        if H0.shape[1] != H1.shape[1]:
            raise ValueError(
                "H0_train and H1_train must have the same feature dimension. "
                f"Got {H0.shape[1]} and {H1.shape[1]}."
            )

        d = H0.shape[1]
        if H0.shape[0] == 0 or H1.shape[0] == 0:
            if self.fallback_to_cosine:
                self._set_cosine_fallback(d)
                return self
            raise ValueError("H0_train and H1_train must both be non-empty.")

        self.mu0_raw_ = H0.mean(axis=0)
        self.mu1_raw_ = H1.mean(axis=0)

        if self.covariance_mode == "pooled":
            X_cov = np.concatenate([H0, H1], axis=0)
            center = True
        else:
            R0 = H0 - self.mu0_raw_[None, :]
            R1 = H1 - self.mu1_raw_[None, :]
            X_cov = np.concatenate([R0, R1], axis=0)
            center = False

        W, rank, _ = _compute_whitening_matrix(
            X_cov,
            center=center,
            abs_eps=self.abs_eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank,
            rank_mode=self.rank_mode,
            explained_variance=self.explained_variance,
            shrinkage=self.shrinkage,
        )

        if rank <= 0:
            if self.fallback_to_cosine:
                self._set_cosine_fallback(d)
                return self
            raise ValueError("No reliable whitening directions were found.")

        Z0 = H0 @ W
        Z1 = H1 @ W

        mu0 = Z0.mean(axis=0)
        mu1 = Z1.mean(axis=0)
        d_vec = mu1 - mu0

        w = W @ d_vec
        b = -0.5 * float((mu0 + mu1) @ d_vec)

        if (not np.all(np.isfinite(w))) or (not np.isfinite(b)):
            if self.fallback_to_cosine:
                self._set_cosine_fallback(d)
                return self
            raise ValueError("Computed non-finite parameters during fit.")

        self.W = W
        self.w = w
        self.b = b
        self.mu0_ = mu0
        self.mu1_ = mu1
        self.selected_rank_ = int(rank)
        self.fallback_used = False
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")
        X = _validate_2d("X", X)
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"X feature dimension mismatch: expected {self.w.shape[0]}, got {X.shape[1]}."
            )
        return X @ self.w + self.b

    def linear_form(self) -> tuple[str, np.ndarray, float]:
        if self.w is None:
            raise RuntimeError("StabilizedFisherHadamardMethod.linear_form() called before fit().")
        if self.normalize_pair_inputs:
            raise RuntimeError("normalized pair-input FisherHadamard is not a clean q*anchor linear form")
        return ("hadamard_linear", np.asarray(self.w, dtype=np.float64), float(self.b))

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")

        A = _validate_2d("A", A)
        B = _validate_2d("B", B)
        if A.shape != B.shape:
            raise ValueError(f"A and B must have the same shape. Got {A.shape} and {B.shape}.")
        if A.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"A/B feature dimension mismatch: expected {self.w.shape[0]}, got {A.shape[1]}."
            )

        if self.normalize_pair_inputs:
            A = _l2_normalize(A.astype(np.float64, copy=False), eps=self.norm_eps)
            B = _l2_normalize(B.astype(np.float64, copy=False), eps=self.norm_eps)

        # Important: Hadamard-space scorer. Build pair feature first, then score.
        H = A * B
        return self.score(H)


class FisherWhitenedHadamardPooledMethod(StabilizedFisherHadamardMethod):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("covariance_mode", "pooled")
        kwargs.setdefault("name", "FWHS-pooled")
        super().__init__(**kwargs)


class FisherWhitenedHadamardWithinMethod(StabilizedFisherHadamardMethod):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("covariance_mode", "within")
        kwargs.setdefault("name", "FWHS-within")
        super().__init__(**kwargs)


class StabilizedWhitenedCosineMethod(OnlineBaseMethod):
    """
    Separate raw-vector whitening baseline.

    This class learns whitening in raw embedding space and computes cosine
    between whitened vectors. It is intentionally separate from Hadamard
    feature scorers.
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
        shrinkage: float = 0.01,
        norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.name = name
        self.abs_eps = float(abs_eps)
        self.rel_eps = float(rel_eps)
        self.max_rank = max_rank
        self.rank_mode = rank_mode
        self.explained_variance = float(explained_variance)
        self.shrinkage = float(shrinkage)
        self.norm_eps = float(norm_eps)

        self.W: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.prototype_: Optional[np.ndarray] = None
        self.selected_rank_: int = 0

    def fit(
        self,
        A_train: np.ndarray,
        B_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "StabilizedWhitenedCosineMethod":
        del weights, seed
        A = _validate_2d("A_train", A_train).astype(np.float64, copy=False)
        B = _validate_2d("B_train", B_train).astype(np.float64, copy=False)

        if A.shape[1] != B.shape[1]:
            raise ValueError(
                "A_train and B_train must have the same feature dimension. "
                f"Got {A.shape[1]} and {B.shape[1]}."
            )

        pooled = np.concatenate([A, B], axis=0)
        W, rank, mean = _compute_whitening_matrix(
            pooled,
            center=True,
            abs_eps=self.abs_eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank,
            rank_mode=self.rank_mode,
            explained_variance=self.explained_variance,
            shrinkage=self.shrinkage,
        )
        if rank <= 0:
            raise ValueError("No reliable whitening directions were found for raw embeddings.")

        self.W = W
        self.mean_ = mean
        self.selected_rank_ = int(rank)

        Z_pos = (B - self.mean_[None, :]) @ self.W
        proto = Z_pos.mean(axis=0)
        pnorm = float(np.linalg.norm(proto))
        self.prototype_ = proto / max(pnorm, self.norm_eps)
        return self

    def _check_fitted(self) -> None:
        if self.W is None or self.mean_ is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _validate_2d("X", X).astype(np.float64, copy=False)
        if X.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"X feature dimension mismatch: expected {self.W.shape[0]}, got {X.shape[1]}."
            )
        Z = (X - self.mean_[None, :]) @ self.W
        Z = _l2_normalize(Z, eps=self.norm_eps)
        if self.prototype_ is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        return Z @ self.prototype_

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        self._check_fitted()
        A = _validate_2d("A", A).astype(np.float64, copy=False)
        B = _validate_2d("B", B).astype(np.float64, copy=False)
        if A.shape != B.shape:
            raise ValueError(f"A and B must have the same shape. Got {A.shape} and {B.shape}.")
        if A.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"A/B feature dimension mismatch: expected {self.W.shape[0]}, got {A.shape[1]}."
            )

        WA = _l2_normalize((A - self.mean_[None, :]) @ self.W, eps=self.norm_eps)
        WB = _l2_normalize((B - self.mean_[None, :]) @ self.W, eps=self.norm_eps)
        return np.sum(WA * WB, axis=1)
