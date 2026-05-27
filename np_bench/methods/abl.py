import numpy as np
from typing import Dict, Iterable, Literal, Optional, Tuple

from .base import OnlineBaseMethod


RankMode = Literal["fixed", "explained_variance", "threshold"]
ShrinkageTarget = Literal["diag", "identity"]

AblationMode = Literal[
    "raw_cosine",
    "current_whitened_cosine",
    "pca_whitened_cosine",
    "raw_hadamard_linear",
    "center_only_hadamard_linear",
    "pca_whitened_hadamard_linear",
    "diag_whitened_hadamard_linear",
    "h0_only_pca_whitened_hadamard_linear",
    "h1_only_pca_whitened_hadamard_linear",
    "within_class_pca_whitened_hadamard_linear",
    "shuffle_labels_pca_whitened_hadamard_linear",
    "no_rank_truncation_pca_whitened_hadamard_linear",
    # Short experiment-table aliases.
    "A0_raw_cosine",
    "A1_hadamard_linear",
    "A2_current_whitened_cosine",
    "A3_whitened_hadamard_linear",
    "A4_no_whitening_hadamard_centroid",
    "A5_diag_whitened_hadamard_linear",
    "A6_full_whitened_hadamard_linear",
    "A7_shuffle_labels_whitened_linear",
]


_ABLATION_ALIASES: Dict[str, str] = {
    # Canonical cosine aliases.
    "pca_whitened_cosine": "current_whitened_cosine",

    # Compact ablation-table aliases.
    "A0_raw_cosine": "raw_cosine",
    "A1_hadamard_linear": "raw_hadamard_linear",
    "A2_current_whitened_cosine": "current_whitened_cosine",
    "A3_whitened_hadamard_linear": "pca_whitened_hadamard_linear",
    "A4_no_whitening_hadamard_centroid": "raw_hadamard_linear",
    "A5_diag_whitened_hadamard_linear": "diag_whitened_hadamard_linear",
    "A6_full_whitened_hadamard_linear": "no_rank_truncation_pca_whitened_hadamard_linear",
    "A7_shuffle_labels_whitened_linear": "shuffle_labels_pca_whitened_hadamard_linear",
}


CANONICAL_ABLATIONS = (
    "raw_cosine",
    "current_whitened_cosine",
    "raw_hadamard_linear",
    "center_only_hadamard_linear",
    "pca_whitened_hadamard_linear",
    "diag_whitened_hadamard_linear",
    "h0_only_pca_whitened_hadamard_linear",
    "h1_only_pca_whitened_hadamard_linear",
    "within_class_pca_whitened_hadamard_linear",
    "shuffle_labels_pca_whitened_hadamard_linear",
    "no_rank_truncation_pca_whitened_hadamard_linear",
)


def available_ablations() -> tuple[str, ...]:
    """Return the canonical ablation names supported by this file."""
    return CANONICAL_ABLATIONS


def _resolve_ablation(ablation: str) -> str:
    return _ABLATION_ALIASES.get(ablation, ablation)


def _validate_2d(name: str, X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d). Got shape {X.shape}.")


def _validate_same_feature_dim(H0: np.ndarray, H1: np.ndarray) -> None:
    if H0.shape[1] != H1.shape[1]:
        raise ValueError(
            "H0_train and H1_train must have the same feature dimension. "
            f"Got {H0.shape[1]} and {H1.shape[1]}."
        )


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _identity(d: int, dtype: np.dtype = np.float64) -> np.ndarray:
    return np.eye(d, dtype=dtype)


def _pooled(H0: np.ndarray, H1: np.ndarray) -> np.ndarray:
    return np.concatenate([H0, H1], axis=0)


def _select_rank(
    eigvals_desc: np.ndarray,
    *,
    rank_cap: int,
    max_rank: Optional[int],
    rank_mode: RankMode,
    explained_variance: float,
    abs_eps: float,
    rel_eps: float,
) -> int:
    """
    Select the number of PCA directions to retain.

    eigvals_desc must be sorted in descending order.
    """
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be positive or None.")

    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    rank_cap = max(1, int(rank_cap))
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


def _cov_from_centered(Xc: np.ndarray, *, denom: int) -> np.ndarray:
    denom = max(int(denom), 1)
    return (Xc.T @ Xc) / denom


def _pooled_covariance(H0: np.ndarray, H1: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Global/pooled covariance of H0 ∪ H1.

    Returns:
        cov, rank_cap
    """
    X = _pooled(H0, H1).astype(np.float64, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    rank_cap = min(max(X.shape[0] - 1, 1), X.shape[1])
    return _cov_from_centered(Xc, denom=X.shape[0] - 1), rank_cap


def _class_covariance(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Covariance of one class, centered by its own class mean.

    Returns:
        cov, rank_cap
    """
    X = X.astype(np.float64, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    rank_cap = min(max(X.shape[0] - 1, 1), X.shape[1])
    return _cov_from_centered(Xc, denom=X.shape[0] - 1), rank_cap


def _within_class_covariance(H0: np.ndarray, H1: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Fisher/WCCN-style within-class covariance.

    This removes the H0/H1 mean difference before estimating covariance, which
    is usually the right covariance estimate for a linear reuse classifier.

    Returns:
        cov, rank_cap
    """
    H0 = H0.astype(np.float64, copy=False)
    H1 = H1.astype(np.float64, copy=False)

    H0c = H0 - H0.mean(axis=0, keepdims=True)
    H1c = H1 - H1.mean(axis=0, keepdims=True)

    denom = max(H0.shape[0] + H1.shape[0] - 2, 1)
    cov = (H0c.T @ H0c + H1c.T @ H1c) / denom

    # Centered within-class rank is at most (n0 - 1) + (n1 - 1).
    rank_cap = min(max(H0.shape[0] + H1.shape[0] - 2, 1), H0.shape[1])
    return cov, rank_cap


def _shrink_covariance(
    cov: np.ndarray,
    *,
    shrinkage: float = 0.05,
    shrinkage_target: ShrinkageTarget = "diag",
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Stabilize a covariance matrix before inversion.

    For target="diag":
        Sigma' = (1 - lambda) Sigma + lambda diag(Sigma) + ridge * scale * I

    For target="identity":
        Sigma' = (1 - lambda) Sigma + lambda scale * I + ridge * scale * I

    The diagonal target keeps per-feature variance while shrinking
    cross-feature covariance. This is usually the safer default for your setup.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative.")

    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    d = cov.shape[0]
    diag = np.diag(cov)
    scale = float(np.mean(diag)) if diag.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    if shrinkage_target == "diag":
        target = np.diag(np.maximum(diag, 0.0))
    elif shrinkage_target == "identity":
        target = scale * np.eye(d, dtype=np.float64)
    else:
        raise ValueError(f"Unknown shrinkage_target: {shrinkage_target!r}")

    shrunk = (1.0 - shrinkage) * cov + shrinkage * target

    if ridge > 0.0:
        shrunk = shrunk + ridge * scale * np.eye(d, dtype=np.float64)

    return 0.5 * (shrunk + shrunk.T)


def _inv_sqrt_cov_matrix(
    cov: np.ndarray,
    *,
    rank_cap: int,
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-6,
    max_rank: Optional[int] = 128,
    rank_mode: RankMode = "explained_variance",
    explained_variance: float = 0.99,
) -> np.ndarray:
    """
    Compute a truncated inverse-square-root covariance transform from cov.

    Returns W where X @ W is approximately whitened in the retained PCA
    subspace. Directions outside the retained subspace are zeroed out.
    """
    _validate_2d("cov", cov)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square. Got {cov.shape}.")

    d = cov.shape[0]
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns ascending order. Flip to descending.
    eigvals_desc = eigvals[::-1]
    eigvecs_desc = eigvecs[:, ::-1]

    # Numerical safety.
    eigvals_desc = np.maximum(eigvals_desc, 0.0)

    k = _select_rank(
        eigvals_desc,
        rank_cap=rank_cap,
        max_rank=max_rank,
        rank_mode=rank_mode,
        explained_variance=explained_variance,
        abs_eps=abs_eps,
        rel_eps=rel_eps,
    )

    vals = eigvals_desc[:k]
    vecs = eigvecs_desc[:, :k]

    max_eigval = max(float(eigvals_desc[0]), 0.0) if eigvals_desc.size else 0.0
    threshold = max(abs_eps, rel_eps * max_eigval)
    mask = vals > threshold

    if not np.any(mask):
        # Safer than returning I / sqrt(eps), which amplifies pure noise.
        return np.zeros((d, d), dtype=np.float64)

    vals = vals[mask]
    vecs = vecs[:, mask]

    inv_sqrt = 1.0 / np.sqrt(vals)
    W = (vecs * inv_sqrt[None, :]) @ vecs.T
    return W.astype(np.float64, copy=False)


def _inv_sqrt_from_covariance(
    cov: np.ndarray,
    *,
    rank_cap: int,
    abs_eps: float,
    rel_eps: float,
    max_rank: Optional[int],
    rank_mode: RankMode,
    explained_variance: float,
    shrinkage: float,
    shrinkage_target: ShrinkageTarget,
    ridge: float,
) -> np.ndarray:
    cov = _shrink_covariance(
        cov,
        shrinkage=shrinkage,
        shrinkage_target=shrinkage_target,
        ridge=ridge,
    )
    return _inv_sqrt_cov_matrix(
        cov,
        rank_cap=rank_cap,
        abs_eps=abs_eps,
        rel_eps=rel_eps,
        max_rank=max_rank,
        rank_mode=rank_mode,
        explained_variance=explained_variance,
    )


def _diag_inv_sqrt_cov(
    X: np.ndarray,
    *,
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-6,
) -> np.ndarray:
    """
    Diagonal-only whitening.

    This ablates away cross-dimensional covariance and keeps only per-feature
    variance normalization.
    """
    _validate_2d("X", X)

    n, d = X.shape
    if n == 0:
        raise ValueError("X must contain at least one row.")

    X_float = np.asarray(X, dtype=np.float64)
    X_centered = X_float - X_float.mean(axis=0, keepdims=True)
    denom = max(n - 1, 1)
    var = np.sum(X_centered * X_centered, axis=0) / denom
    var = np.maximum(var, 0.0)

    max_var = max(float(np.max(var)), 0.0)
    threshold = max(abs_eps, rel_eps * max_var)

    inv_sqrt = np.zeros_like(var)
    mask = var > threshold
    inv_sqrt[mask] = 1.0 / np.sqrt(var[mask])

    return np.diag(inv_sqrt).astype(np.float64, copy=False)


def _shuffle_labels_preserve_counts(
    H0: np.ndarray,
    H1: np.ndarray,
    *,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle H0/H1 labels while preserving the original class sizes.

    This is a sanity-check ablation. If performance stays high after this,
    the supervised H0/H1 signal is probably not doing the real work.
    """
    pooled = _pooled(H0, H1)
    labels = np.concatenate(
        [
            np.zeros(H0.shape[0], dtype=np.int8),
            np.ones(H1.shape[0], dtype=np.int8),
        ]
    )

    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)

    return pooled[shuffled == 0], pooled[shuffled == 1]


def _fit_centroid_linear(
    H0: np.ndarray,
    H1: np.ndarray,
    W: np.ndarray,
    *,
    feature_mean: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray]:
    """
    Fit a nearest-centroid linear separator after a linear transform W.

    The separator is defined in whitened space:

        z = (h - m) @ W
        d = mean(H1_z) - mean(H0_z)
        score(h) = z @ d + b_z

    and converted back to original feature coordinates:

        score(h) = h @ w + b

    Returns:
        w, b, d_w, b_w, feature_mean
    """
    _validate_2d("H0", H0)
    _validate_2d("H1", H1)
    _validate_same_feature_dim(H0, H1)

    H0 = np.asarray(H0, dtype=np.float64)
    H1 = np.asarray(H1, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if feature_mean is None:
        feature_mean = np.zeros(H0.shape[1], dtype=np.float64)
    else:
        feature_mean = np.asarray(feature_mean, dtype=np.float64)

    Z0 = (H0 - feature_mean) @ W
    Z1 = (H1 - feature_mean) @ W

    mu0 = Z0.mean(axis=0)
    mu1 = Z1.mean(axis=0)

    d_w = mu1 - mu0
    b_w = -0.5 * float((mu0 + mu1) @ d_w)

    # score(h) = ((h - m) @ W) @ d_w + b_w
    #          = h @ (W @ d_w) + (b_w - m @ (W @ d_w))
    w = W @ d_w
    b = b_w - float(feature_mean @ w)

    return w, b, d_w, b_w, feature_mean


class WhitenedCosineMethod(OnlineBaseMethod):
    """
    Ablation-ready scorer for semantic-cache pair scoring.

    Important naming note:
    The strong methods here are not true cosine methods. Under
    --hadamard_preprocess, the benchmark usually passes precomputed pair
    features X, commonly:

        X = emb(query) * emb(candidate)

    The useful score is a whitened linear score over those features:

        score(X) = X @ w + b

    This file keeps the class name WhitenedCosineMethod only to avoid breaking
    existing registry/import code.
    """

    def __init__(
        self,
        name: str = "PCAWhitenedCosine",
        *,
        ablation: AblationMode = "current_whitened_cosine",
        abs_eps: float = 1e-6,
        rel_eps: float = 1e-6,
        max_rank: Optional[int] = 128,
        rank_mode: RankMode = "explained_variance",
        explained_variance: float = 0.99,
        shrinkage: float = 0.05,
        shrinkage_target: ShrinkageTarget = "diag",
        ridge: float = 1e-8,
        norm_eps: float = 1e-12,
    ):
        super().__init__()

        self.name = name
        self.ablation = ablation
        self.resolved_ablation = _resolve_ablation(ablation)

        if self.resolved_ablation not in CANONICAL_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}. "
                f"Available canonical ablations: {CANONICAL_ABLATIONS}."
            )

        self.abs_eps = abs_eps
        self.rel_eps = rel_eps
        self.max_rank = max_rank
        self.rank_mode = rank_mode
        self.explained_variance = explained_variance
        self.shrinkage = shrinkage
        self.shrinkage_target = shrinkage_target
        self.ridge = ridge
        self.norm_eps = norm_eps

        self.score_mode: str = "unfitted"

        self.W: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None
        self.d_w: Optional[np.ndarray] = None
        self.b_w: Optional[float] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.prototype_: Optional[np.ndarray] = None

        self.mem_H0: Optional[np.ndarray] = None
        self.mem_H1: Optional[np.ndarray] = None
        self._fit_seed: Optional[int] = None

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
        _validate_same_feature_dim(H0_train, H1_train)

        if H0_train.shape[0] == 0 or H1_train.shape[0] == 0:
            raise ValueError("H0_train and H1_train must both contain at least one row.")

        # Store float64 copies for stable covariance work and refit().
        self.mem_H0 = np.asarray(H0_train, dtype=np.float64).copy()
        self.mem_H1 = np.asarray(H1_train, dtype=np.float64).copy()
        self._fit_seed = seed

        self._fit_from_memory(seed=seed)
        return self

    def _make_whitening_from_cov(self, cov: np.ndarray, rank_cap: int, *, max_rank: Optional[int] = None, rank_mode: Optional[RankMode] = None, explained_variance: Optional[float] = None) -> np.ndarray:
        return _inv_sqrt_from_covariance(
            cov,
            rank_cap=rank_cap,
            abs_eps=self.abs_eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank if max_rank is None else max_rank,
            rank_mode=self.rank_mode if rank_mode is None else rank_mode,
            explained_variance=self.explained_variance if explained_variance is None else explained_variance,
            shrinkage=self.shrinkage,
            shrinkage_target=self.shrinkage_target,
            ridge=self.ridge,
        )

    def _fit_from_memory(self, *, seed=None) -> None:
        if self.mem_H0 is None or self.mem_H1 is None:
            raise RuntimeError("Missing stored training data. Call fit(...) first.")

        H0 = self.mem_H0
        H1 = self.mem_H1
        d = H0.shape[1]
        pooled = _pooled(H0, H1)

        self.W = None
        self.w = None
        self.b = None
        self.d_w = None
        self.b_w = None
        self.prototype_ = None
        self.feature_mean = np.zeros(d, dtype=np.float64)
        self.score_mode = "unfitted"

        if self.resolved_ablation == "raw_cosine":
            # score_pairs(A, B): real cosine(A, B)
            # score(X): direct dot-product proxy if X is already A * B.
            self.W = _identity(d)
            self.score_mode = "raw_cosine"
            return

        if self.resolved_ablation == "current_whitened_cosine":
            # True pairwise whitened cosine is only available in score_pairs(A, B).
            # In score(X), where only precomputed pair features exist, we use a
            # diagnostic feature-space prototype cosine. This avoids silently
            # pretending that feature vectors contain separable raw pairs.
            cov, rank_cap = _pooled_covariance(H0, H1)
            self.W = self._make_whitening_from_cov(cov, rank_cap)

            WH1 = H1 @ self.W
            proto = WH1.mean(axis=0)
            proto_norm = float(np.linalg.norm(proto))
            if proto_norm <= self.norm_eps:
                self.prototype_ = np.zeros_like(proto)
            else:
                self.prototype_ = proto / proto_norm

            self.score_mode = "feature_prototype_cosine"
            return

        # From this point onward, every ablation is a linear scorer over
        # precomputed pair features, usually Hadamard features.
        H0_for_separator = H0
        H1_for_separator = H1
        self.score_mode = "linear_hadamard_features"

        if self.resolved_ablation == "raw_hadamard_linear":
            W = _identity(d)

        elif self.resolved_ablation == "center_only_hadamard_linear":
            W = _identity(d)
            self.feature_mean = pooled.mean(axis=0)

        elif self.resolved_ablation == "pca_whitened_hadamard_linear":
            # Pooled covariance + shrinkage + rank truncation.
            cov, rank_cap = _pooled_covariance(H0, H1)
            W = self._make_whitening_from_cov(cov, rank_cap)

        elif self.resolved_ablation == "diag_whitened_hadamard_linear":
            # Explicitly no cross-dimensional covariance.
            W = _diag_inv_sqrt_cov(
                pooled,
                abs_eps=self.abs_eps,
                rel_eps=self.rel_eps,
            )

        elif self.resolved_ablation == "h0_only_pca_whitened_hadamard_linear":
            cov, rank_cap = _class_covariance(H0)
            W = self._make_whitening_from_cov(cov, rank_cap)

        elif self.resolved_ablation == "h1_only_pca_whitened_hadamard_linear":
            cov, rank_cap = _class_covariance(H1)
            W = self._make_whitening_from_cov(cov, rank_cap)

        elif self.resolved_ablation == "within_class_pca_whitened_hadamard_linear":
            # This should match the new strong implementation:
            # within-class covariance + shrinkage + PCA inverse sqrt + centroid head.
            cov, rank_cap = _within_class_covariance(H0, H1)
            W = self._make_whitening_from_cov(cov, rank_cap)

        elif self.resolved_ablation == "shuffle_labels_pca_whitened_hadamard_linear":
            # Keep geometry real, destroy the supervised label signal.
            # This tests whether the H0/H1 separator is doing real work.
            cov, rank_cap = _within_class_covariance(H0, H1)
            W = self._make_whitening_from_cov(cov, rank_cap)
            H0_for_separator, H1_for_separator = _shuffle_labels_preserve_counts(
                H0,
                H1,
                seed=seed,
            )

        elif self.resolved_ablation == "no_rank_truncation_pca_whitened_hadamard_linear":
            # Within-class covariance with shrinkage, but keep every numerically
            # valid eigen-direction. This isolates rank truncation as the factor.
            cov, rank_cap = _within_class_covariance(H0, H1)
            W = self._make_whitening_from_cov(
                cov,
                rank_cap,
                max_rank=None,
                rank_mode="threshold",
                explained_variance=1.0,
            )

        else:
            raise ValueError(f"Unknown resolved ablation: {self.resolved_ablation!r}")

        self.W = W
        self.w, self.b, self.d_w, self.b_w, self.feature_mean = _fit_centroid_linear(
            H0_for_separator,
            H1_for_separator,
            W,
            feature_mean=self.feature_mean,
        )

    def _check_is_fitted(self) -> None:
        if self.W is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Score a single precomputed feature matrix.

        This is the path used by global-mode benchmarks when
        --hadamard_preprocess has already built pair features.
        """
        self._check_is_fitted()
        _validate_2d("X", X)

        X = np.asarray(X, dtype=np.float64)
        assert self.W is not None

        if X.shape[1] != self.W.shape[0]:
            raise ValueError(
                "X feature dimension does not match fitted transform. "
                f"Got {X.shape[1]}, expected {self.W.shape[0]}."
            )

        if self.resolved_ablation == "raw_cosine":
            # If X = A * B and embeddings are normalized, sum(X) equals dot/cosine.
            # If they are not normalized, this is still the direct-dot baseline.
            return np.sum(X, axis=1)

        if self.resolved_ablation == "current_whitened_cosine":
            if self.prototype_ is None:
                raise RuntimeError("Feature-cosine prototype is missing. Call fit(...) first.")
            Z = _l2_normalize(X @ self.W, eps=self.norm_eps)
            return np.sum(Z * self.prototype_[None, :], axis=1)

        if self.w is None or self.b is None:
            raise RuntimeError("Linear ablation is missing w/b. Call fit(...) first.")

        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                "X feature dimension does not match fitted linear scorer. "
                f"Got {X.shape[1]}, expected {self.w.shape[0]}."
            )

        return X @ self.w + self.b

    def score_hadamard_features(self, H: np.ndarray) -> np.ndarray:
        """
        Explicit alias for scoring already-built Hadamard/pair features.
        """
        return self.score(H)

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Score raw embedding pairs.

        - raw_cosine:
            cosine(A, B)

        - current_whitened_cosine:
            true whitened cosine cosine(A @ W, B @ W)
            This is diagnostic only if W was trained on Hadamard features.

        - all Hadamard-linear ablations:
            H = A * B
            score(H) = H @ w + b
        """
        self._check_is_fitted()

        _validate_2d("A", A)
        _validate_2d("B", B)

        if A.shape != B.shape:
            raise ValueError(f"A and B must have the same shape. Got {A.shape} and {B.shape}.")

        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        assert self.W is not None

        if self.resolved_ablation == "raw_cosine":
            NA = _l2_normalize(A, eps=self.norm_eps)
            NB = _l2_normalize(B, eps=self.norm_eps)
            return np.sum(NA * NB, axis=1)

        if A.shape[1] != self.W.shape[0]:
            raise ValueError(
                "A/B feature dimension does not match fitted transform. "
                f"Got {A.shape[1]}, expected {self.W.shape[0]}."
            )

        if self.resolved_ablation == "current_whitened_cosine":
            WA = _l2_normalize(A @ self.W, eps=self.norm_eps)
            WB = _l2_normalize(B @ self.W, eps=self.norm_eps)
            return np.sum(WA * WB, axis=1)

        H = A * B
        return self.score(H)

    def refit(self) -> None:
        if self.mem_H0 is not None and self.mem_H1 is not None:
            self._fit_from_memory(seed=self._fit_seed)
        else:
            super().refit()


def make_ablation_methods(
    *,
    ablations: Optional[Iterable[str]] = None,
    name_prefix: str = "ablation",
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-6,
    max_rank: Optional[int] = 128,
    rank_mode: RankMode = "explained_variance",
    explained_variance: float = 0.99,
    shrinkage: float = 0.05,
    shrinkage_target: ShrinkageTarget = "diag",
    ridge: float = 1e-8,
    norm_eps: float = 1e-12,
) -> Dict[str, WhitenedCosineMethod]:
    """
    Convenience factory for creating the full ablation suite.

    Example:

        methods = make_ablation_methods()
        for name, method in methods.items():
            method.fit(H0_train, H1_train, seed=0)
            scores = method.score(H_eval)  # if H_eval is precomputed pair features

    For raw embedding pairs:

        scores = method.score_pairs(A_eval, B_eval)
    """
    selected = tuple(ablations) if ablations is not None else CANONICAL_ABLATIONS

    out: Dict[str, WhitenedCosineMethod] = {}
    for ablation in selected:
        resolved = _resolve_ablation(ablation)
        if resolved not in CANONICAL_ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}. "
                f"Available canonical ablations: {CANONICAL_ABLATIONS}."
            )

        out[ablation] = WhitenedCosineMethod(
            name=f"{name_prefix}:{ablation}",
            ablation=ablation,  # type: ignore[arg-type]
            abs_eps=abs_eps,
            rel_eps=rel_eps,
            max_rank=max_rank,
            rank_mode=rank_mode,
            explained_variance=explained_variance,
            shrinkage=shrinkage,
            shrinkage_target=shrinkage_target,
            ridge=ridge,
            norm_eps=norm_eps,
        )

    return out
