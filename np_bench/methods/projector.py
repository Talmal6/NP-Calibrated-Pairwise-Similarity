from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class Projector:
    kind: str                 # "lda" | "pca"
    mean: np.ndarray          # shape (d,)
    W: np.ndarray             # shape (d, k)


def _pca_projector(X: np.ndarray, k: int) -> Projector:
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0)
    Z = X - mu
    # SVD on centered data
    # Z = U S Vt, PCs are rows of Vt
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    W = Vt[:k].T.astype(np.float32)  # (d,k)
    return Projector(kind="pca", mean=mu.astype(np.float32), W=W)


def _lda_binary_projector(X0: np.ndarray, X1: np.ndarray, shrink: float = 1e-2) -> Projector:
    """
    Binary LDA -> 1D direction.
    w = (Sw + shrink*I)^(-1) (m1 - m0)
    """
    X0 = np.asarray(X0, dtype=np.float32)
    X1 = np.asarray(X1, dtype=np.float32)

    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)

    Z0 = X0 - m0
    Z1 = X1 - m1

    # within-class scatter (cov up to scaling)
    Sw = (Z0.T @ Z0 + Z1.T @ Z1) / max(1.0, float(X0.shape[0] + X1.shape[0] - 2))
    d = Sw.shape[0]

    Sw = Sw + (shrink * np.eye(d, dtype=np.float32))

    w = np.linalg.solve(Sw, (m1 - m0).astype(np.float32))
    nrm = float(np.linalg.norm(w) + 1e-12)
    w = (w / nrm).astype(np.float32)

    W = w.reshape(-1, 1)  # (d,1)
    mu = ((m0 + m1) * 0.5).astype(np.float32)
    return Projector(kind="lda", mean=mu, W=W)


def fit_projector(
    kind: str,
    X0: np.ndarray,
    X1: np.ndarray,
    k: int,
    *,
    lda_shrink: float = 1e-2,
) -> Projector:
    kind = kind.lower()
    if kind == "pca":
        X = np.concatenate([X0, X1], axis=0)
        return _pca_projector(X, k)
    if kind == "lda":
        # binary LDA is always 1D
        return _lda_binary_projector(X0, X1, shrink=lda_shrink)
    raise ValueError(f"Unknown projector kind: {kind}")


def apply_projector(P: Projector, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return (X - P.mean) @ P.W


def _var_ratio_weights(H0: np.ndarray, H1: np.ndarray) -> np.ndarray:
    v0 = np.var(H0, axis=0)
    v1 = np.var(H1, axis=0)
    return (v1 / (v0 + 1e-12)).astype(np.float32)


class ProjectedMethod:
    """
    Wrap any existing method:
      - Learn a projector on (X0,X1) inside fit()
      - Then call base.fit() on projected data (if base has fit)
      - score() always runs base.score() on projected X

    For methods that need weights (AndBox/Vec), we recompute weights in projected space.
    """
    def __init__(
        self,
        name: str,
        base_method: Any,
        *,
        proj_kind: str,
        proj_dim: int,
        lda_shrink: float = 1e-2,
    ):
        self.name = name
        self.base = base_method
        self.proj_kind = proj_kind
        self.proj_dim = int(proj_dim)
        self.lda_shrink = float(lda_shrink)
        self.P: Optional[Projector] = None

        self.needs_weights = bool(getattr(base_method, "needs_weights", False))
        self.needs_seed = bool(getattr(base_method, "needs_seed", False))

    def fit(
        self,
        H0: np.ndarray,
        H1: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        alpha: float = 0.05,
    ):
        k = 1 if self.proj_kind.lower() == "lda" else self.proj_dim
        self.P = fit_projector(self.proj_kind, H0, H1, k, lda_shrink=self.lda_shrink)

        H0p = apply_projector(self.P, H0)
        H1p = apply_projector(self.P, H1)

        if hasattr(self.base, "fit"):
            kwargs: dict[str, Any] = {}
            if getattr(self.base, "needs_seed", False):
                kwargs["seed"] = seed
            kwargs["alpha"] = alpha

            if getattr(self.base, "needs_weights", False):
                kwargs["weights"] = _var_ratio_weights(H0p, H1p)

            # only pass kwargs the base method actually accepts
            sig = inspect.signature(self.base.fit)
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            self.base.fit(H0p, H1p, **kwargs)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.P is None:
            raise ValueError("fit() must be called before score()")

        Xp = apply_projector(self.P, X)

        if not hasattr(self.base, "score"):
            raise ValueError(f"Base method {type(self.base).__name__} has no score()")

        s = self.base.score(Xp)

        # normalize output to float32 1D
        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 2 and s.shape[1] == 1:
            s = s.reshape(-1)
        elif s.ndim != 1:
            # last-resort flatten (keeps behavior deterministic)
            s = s.reshape(-1)

        return s

