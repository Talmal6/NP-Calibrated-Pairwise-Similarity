from __future__ import annotations

import numpy as np

from .base import BaseMethod


def _pairwise_l2_distances(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    Y_arr = np.asarray(Y, dtype=np.float32)
    out = np.empty((X_arr.shape[0], Y_arr.shape[0]), dtype=np.float32)
    y_norm = np.sum(Y_arr * Y_arr, axis=1, dtype=np.float32)

    for start in range(0, X_arr.shape[0], batch_size):
        stop = min(start + batch_size, X_arr.shape[0])
        xb = X_arr[start:stop]
        x_norm = np.sum(xb * xb, axis=1, dtype=np.float32)
        d2 = x_norm[:, None] + y_norm[None, :] - 2.0 * (xb @ Y_arr.T)
        np.maximum(d2, 0.0, out=d2)
        out[start:stop] = np.sqrt(d2, dtype=np.float32)

    return out


def _fast_from_distances(d_pos: np.ndarray, d_neg: np.ndarray) -> np.ndarray:
    d_pos_min = np.min(d_pos, axis=1)
    d_neg_min = np.min(d_neg, axis=1)
    return ((d_neg_min - d_pos_min) / 2.0).astype(np.float32, copy=False)


def _fit_lda_projector(
    H0: np.ndarray,
    H1: np.ndarray,
    *,
    shrink: float,
) -> tuple[np.ndarray, np.ndarray]:
    H0_arr = np.asarray(H0, dtype=np.float32)
    H1_arr = np.asarray(H1, dtype=np.float32)
    mu0 = np.mean(H0_arr, axis=0)
    mu1 = np.mean(H1_arr, axis=0)

    Z0 = H0_arr - mu0
    Z1 = H1_arr - mu1
    sw = (Z0.T @ Z0 + Z1.T @ Z1) / max(1.0, float(H0_arr.shape[0] + H1_arr.shape[0] - 2))
    sw = sw + float(shrink) * np.eye(sw.shape[0], dtype=np.float32)

    w = np.linalg.solve(sw, (mu1 - mu0).astype(np.float32))
    w = (w / max(float(np.linalg.norm(w)), 1e-12)).astype(np.float32)
    mean = ((mu0 + mu1) * 0.5).astype(np.float32)
    return mean, w.reshape(-1, 1)


def _fit_pca_whiten_projector(
    H0: np.ndarray,
    H1: np.ndarray,
    *,
    dim: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([H0, H1], axis=0).astype(np.float32, copy=False)
    mean = np.mean(X, axis=0).astype(np.float32)
    Z = X - mean
    _, singular_values, vt = np.linalg.svd(Z, full_matrices=False)

    k = int(max(1, min(dim, vt.shape[0], vt.shape[1])))
    components = vt[:k].T.astype(np.float32, copy=False)
    eigvals = (singular_values[:k] ** 2) / max(1.0, float(X.shape[0]))
    scales = (1.0 / np.sqrt(np.maximum(eigvals, float(eps)))).astype(np.float32)
    W = components * scales[None, :]
    return mean, W.astype(np.float32, copy=False)


def _exact_one(
    d_pos: np.ndarray,
    d_neg: np.ndarray,
    pos_neg_dist: np.ndarray,
) -> float:
    if d_pos.shape[0] == 1 and d_neg.shape[0] == 1:
        return float((d_neg[0] - d_pos[0]) / 2.0)

    pos_order = np.argsort(d_pos)
    neg_order = np.argsort(d_neg)
    d_pos_sorted = d_pos[pos_order]
    d_neg_sorted = d_neg[neg_order]

    closest_pos_dist = float(d_pos_sorted[0])
    min_r = closest_pos_dist + 2.0 * float(d_neg_sorted[0])
    sep_neg = min_r

    for o_dist_raw, o_idx_raw in zip(d_neg_sorted, neg_order):
        o_dist = float(o_dist_raw)
        if o_dist > min_r:
            break

        if o_dist > closest_pos_dist:
            pos_limit = min(min_r, o_dist)
            n_pos_considered = int(np.searchsorted(d_pos_sorted, pos_limit, side="right"))
        else:
            n_pos_considered = int(d_pos_sorted.shape[0])

        if n_pos_considered == 0:
            continue

        pos_idx = pos_order[:n_pos_considered]
        d_so = pos_neg_dist[pos_idx, int(o_idx_raw)].astype(np.float64, copy=False)
        numer = (o_dist ** 2) - np.square(d_pos_sorted[:n_pos_considered].astype(np.float64))
        sep_vals = np.zeros_like(d_so, dtype=np.float64)
        valid = d_so > 1e-12
        sep_vals[valid] = numer[valid] / (2.0 * d_so[valid])
        sep_pos = float(np.max(sep_vals))
        sep_neg = min(sep_pos, sep_neg)
        min_r = closest_pos_dist + 2.0 * max(0.0, sep_neg)

    return float(sep_neg)


class SeparationScoreMethod(BaseMethod):
    """Binary class-support separation score on embedding features.

    Positive support is H1, negative support is H0, and higher scores mean
    closer to the H1 support relative to H0.
    """

    needs_weights = False
    needs_seed = False
    input_space = "embedding"

    def __init__(
        self,
        *,
        exact: bool = False,
        batch_size: int = 256,
    ) -> None:
        self.exact = bool(exact)
        self.batch_size = int(batch_size)
        self.name = "ExactSeparation" if self.exact else "FastSeparation"
        self.X_pos: np.ndarray | None = None
        self.X_neg: np.ndarray | None = None
        self._pos_neg_dist: np.ndarray | None = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: float = 0.05,
    ) -> "SeparationScoreMethod":
        del weights, seed, alpha
        X_neg = np.asarray(H0_train, dtype=np.float32)
        X_pos = np.asarray(H1_train, dtype=np.float32)
        if X_pos.ndim != 2 or X_neg.ndim != 2:
            raise ValueError(
                f"{self.name}.fit expects 2D train arrays; got H0={X_neg.shape}, H1={X_pos.shape}"
            )
        if X_pos.shape[0] == 0:
            raise ValueError(f"{self.name}.fit requires non-empty H1_train")
        if X_neg.shape[0] == 0:
            raise ValueError(f"{self.name}.fit requires non-empty H0_train")
        if X_pos.shape[1] != X_neg.shape[1]:
            raise ValueError(
                f"{self.name}.fit dimension mismatch: H0 dim={X_neg.shape[1]}, H1 dim={X_pos.shape[1]}"
            )

        self.X_pos = X_pos
        self.X_neg = X_neg
        self._pos_neg_dist = (
            _pairwise_l2_distances(X_pos, X_neg, batch_size=self.batch_size)
            if self.exact
            else None
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.X_pos is None or self.X_neg is None:
            raise RuntimeError(f"{self.name} must be fit before score")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"{self.name}.score expects 2D X, got shape={X_arr.shape}")
        if X_arr.shape[1] != self.X_pos.shape[1]:
            raise ValueError(
                f"{self.name}.score dimension mismatch: expected {self.X_pos.shape[1]}, got {X_arr.shape[1]}"
            )

        d_pos = _pairwise_l2_distances(X_arr, self.X_pos, batch_size=self.batch_size)
        d_neg = _pairwise_l2_distances(X_arr, self.X_neg, batch_size=self.batch_size)
        if not self.exact:
            return _fast_from_distances(d_pos, d_neg)

        if self._pos_neg_dist is None:
            raise RuntimeError(f"{self.name} missing cached positive/negative distances")
        scores = np.empty(X_arr.shape[0], dtype=np.float32)
        for i in range(X_arr.shape[0]):
            scores[i] = _exact_one(d_pos[i], d_neg[i], self._pos_neg_dist)
        return scores


class ProjectedSeparationScoreMethod(SeparationScoreMethod):
    """Separation after learning a metric-aligned linear projection."""

    def __init__(
        self,
        *,
        projection: str,
        exact: bool = False,
        dim: int = 64,
        lda_shrink: float = 1e-2,
        whiten_eps: float = 1e-6,
        batch_size: int = 256,
    ) -> None:
        super().__init__(exact=exact, batch_size=batch_size)
        projection_norm = str(projection).lower()
        if projection_norm not in {"lda", "pca_whiten"}:
            raise ValueError("projection must be 'lda' or 'pca_whiten'")
        self.projection = projection_norm
        self.dim = int(dim)
        self.lda_shrink = float(lda_shrink)
        self.whiten_eps = float(whiten_eps)
        self.mean_: np.ndarray | None = None
        self.W_: np.ndarray | None = None

        prefix = "Exact" if self.exact else "Fast"
        if self.projection == "lda":
            self.name = f"{prefix}LDASeparation"
        else:
            self.name = f"{prefix}PCAWhitenedSeparation"

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.W_ is None:
            raise RuntimeError(f"{self.name} must be fit before score")
        X_arr = np.asarray(X, dtype=np.float32)
        return ((X_arr - self.mean_) @ self.W_).astype(np.float32, copy=False)

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: float = 0.05,
    ) -> "ProjectedSeparationScoreMethod":
        del weights, seed, alpha
        H0 = np.asarray(H0_train, dtype=np.float32)
        H1 = np.asarray(H1_train, dtype=np.float32)
        if H0.ndim != 2 or H1.ndim != 2:
            raise ValueError(f"{self.name}.fit expects 2D train arrays")
        if H0.shape[0] == 0 or H1.shape[0] == 0:
            raise ValueError(f"{self.name}.fit requires non-empty H0_train and H1_train")

        if self.projection == "lda":
            self.mean_, self.W_ = _fit_lda_projector(H0, H1, shrink=self.lda_shrink)
        else:
            self.mean_, self.W_ = _fit_pca_whiten_projector(
                H0,
                H1,
                dim=self.dim,
                eps=self.whiten_eps,
            )

        H0p = self._project(H0)
        H1p = self._project(H1)
        SeparationScoreMethod.fit(self, H0p, H1p)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return SeparationScoreMethod.score(self, self._project(X))
