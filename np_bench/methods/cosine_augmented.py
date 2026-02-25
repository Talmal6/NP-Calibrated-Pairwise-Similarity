"""Cosine-augmented methods: combine raw cosine score with PCA/LDA projections."""
from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np

from .base import BaseMethod
from .projector import Projector, fit_projector, apply_projector


def _cosine_score_1d(X: np.ndarray) -> np.ndarray:
    """Raw cosine: sum along axis=1 (same as CosineMethod.score)."""
    s = np.sum(X.astype(np.float64, copy=False), axis=1)
    s = np.clip(s, -1.0, 1.0)
    return np.nan_to_num(s, nan=-1.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def _var_ratio_weights(H0: np.ndarray, H1: np.ndarray) -> np.ndarray:
    v0 = np.var(H0, axis=0)
    v1 = np.var(H1, axis=0)
    return (v1 / (v0 + 1e-12)).astype(np.float32)


class CosineAugmentedMethod:
    """
    Augment the raw cosine score with PCA or LDA projections of the full
    embedding, then feed the combined feature vector to a learned classifier.

    Feature vector per sample:
        [cosine_score, proj_1, proj_2, ..., proj_k]

    This lets the classifier leverage cosine (the dominant signal) while
    also using geometric structure captured by PCA/LDA.
    """

    def __init__(
        self,
        name: str,
        base_method: Any,
        *,
        proj_kind: str = "pca",
        proj_dim: int = 8,
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

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Construct [cosine, projected_features] for each row."""
        cos = _cosine_score_1d(X).reshape(-1, 1)          # (N, 1)
        if self.P is not None:
            proj = apply_projector(self.P, X)               # (N, k)
            return np.hstack([cos, proj]).astype(np.float32)
        return cos.astype(np.float32)

    def fit(
        self,
        H0: np.ndarray,
        H1: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        alpha: float = 0.05,
    ):
        # Fit projector on full embedding
        k = 1 if self.proj_kind.lower() == "lda" else self.proj_dim
        self.P = fit_projector(self.proj_kind, H0, H1, k, lda_shrink=self.lda_shrink)

        # Build augmented features
        H0a = self._augment(H0)
        H1a = self._augment(H1)

        # Fit the base classifier on augmented features
        if hasattr(self.base, "fit"):
            kwargs: dict[str, Any] = {}
            if getattr(self.base, "needs_seed", False):
                kwargs["seed"] = seed
            kwargs["alpha"] = alpha

            if getattr(self.base, "needs_weights", False):
                kwargs["weights"] = _var_ratio_weights(H0a, H1a)

            # Only pass kwargs the base method actually accepts
            sig = inspect.signature(self.base.fit)
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            self.base.fit(H0a, H1a, **kwargs)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.P is None:
            raise ValueError("fit() must be called before score()")

        Xa = self._augment(X)

        if not hasattr(self.base, "score"):
            raise ValueError(f"Base method {type(self.base).__name__} has no score()")

        s = self.base.score(Xa)
        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 2 and s.shape[1] == 1:
            s = s.reshape(-1)
        elif s.ndim != 1:
            s = s.reshape(-1)

        return s
