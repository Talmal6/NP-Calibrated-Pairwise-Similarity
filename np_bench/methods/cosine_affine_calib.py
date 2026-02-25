from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import BaseMethod


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_affine_logistic_numpy(
    s: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-2,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> Tuple[float, float]:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if s.size == 0 or y.size == 0:
        return 1.0, 0.0

    a = 1.0
    b = 0.0
    X0 = np.ones_like(s)

    for _ in range(max_iter):
        z = a * s + b
        p = _sigmoid(z)
        r = p - y
        w = p * (1.0 - p)

        g_a = float(np.sum(r * s) + l2 * a)
        g_b = float(np.sum(r) + l2 * b)

        h_aa = float(np.sum(w * s * s) + l2)
        h_ab = float(np.sum(w * s * X0))
        h_bb = float(np.sum(w) + l2)

        det = h_aa * h_bb - h_ab * h_ab
        if not np.isfinite(det) or abs(det) < 1e-12:
            break

        d_a = (h_bb * g_a - h_ab * g_b) / det
        d_b = (-h_ab * g_a + h_aa * g_b) / det

        a_new = a - d_a
        b_new = b - d_b

        if abs(a_new - a) + abs(b_new - b) < tol:
            a, b = float(a_new), float(b_new)
            break
        a, b = float(a_new), float(b_new)

    if not np.isfinite(a) or not np.isfinite(b):
        return 1.0, 0.0
    return float(a), float(b)


class CosineAffineCalibMethod(BaseMethod):
    name = "CosineAffineCalib"
    needs_weights = False
    needs_seed = False

    def __init__(
        self,
        *,
        l2: float = 1e-2,
        max_iter: int = 80,
        min_group_samples: int = 30,
        min_group_pos: int = 3,
        min_group_neg: int = 3,
    ) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.min_group_samples = int(min_group_samples)
        self.min_group_pos = int(min_group_pos)
        self.min_group_neg = int(min_group_neg)

        self.a_global = 1.0
        self.b_global = 0.0
        self.a_by_group: Dict[int, Tuple[float, float]] = {}
        self.active_group: Optional[int] = None

    def _raw_cosine(self, X: np.ndarray) -> np.ndarray:
        s = np.sum(np.asarray(X, dtype=np.float64), axis=1)
        s = np.clip(s, -1.0, 1.0)
        s = np.nan_to_num(s, nan=-1.0, posinf=1.0, neginf=-1.0)
        return s.astype(np.float32, copy=False)

    def _fit_affine(self, s: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore

            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=max(200, self.max_iter),
                C=max(1e-6, 1.0 / max(self.l2, 1e-6)),
                fit_intercept=True,
                class_weight="balanced",
                random_state=0,
            )
            clf.fit(s.reshape(-1, 1), y.astype(np.int32, copy=False))
            a = float(clf.coef_.reshape(-1)[0])
            b = float(clf.intercept_.reshape(-1)[0])
            if np.isfinite(a) and np.isfinite(b):
                return a, b
        except Exception:
            pass

        return _fit_affine_logistic_numpy(
            s,
            y,
            l2=self.l2,
            max_iter=self.max_iter,
        )

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: float = 0.05,
    ) -> "CosineAffineCalibMethod":
        del weights, seed, alpha
        s0 = self._raw_cosine(H0_train)
        s1 = self._raw_cosine(H1_train)
        s = np.concatenate([s0, s1], axis=0)
        y = np.concatenate(
            [np.zeros(s0.shape[0], dtype=np.int32), np.ones(s1.shape[0], dtype=np.int32)],
            axis=0,
        )
        if s.size > 0 and np.unique(y).size > 1:
            self.a_global, self.b_global = self._fit_affine(s, y)
        else:
            self.a_global, self.b_global = 1.0, 0.0
        self.a_by_group = {}
        self.active_group = None
        return self

    def fit_group_calibrators(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        group_ids: np.ndarray,
    ) -> None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.int32).reshape(-1)
        group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if scores.shape[0] != y.shape[0] or scores.shape[0] != group_ids.shape[0]:
            raise ValueError("scores, y, and group_ids must have matching lengths")

        self.a_by_group = {}
        for gid in np.unique(group_ids):
            mask = group_ids == gid
            n = int(mask.sum())
            if n < self.min_group_samples:
                continue
            y_g = y[mask]
            n_pos = int(np.sum(y_g == 1))
            n_neg = int(np.sum(y_g == 0))
            if n_pos < self.min_group_pos or n_neg < self.min_group_neg:
                continue
            a, b = self._fit_affine(scores[mask], y_g)
            self.a_by_group[int(gid)] = (float(a), float(b))

    def set_active_group(self, group_id: Optional[int]) -> None:
        self.active_group = int(group_id) if group_id is not None else None

    def score(self, X: np.ndarray) -> np.ndarray:
        s = self._raw_cosine(X).astype(np.float64)
        a, b = self.a_global, self.b_global
        if self.active_group is not None:
            a, b = self.a_by_group.get(int(self.active_group), (self.a_global, self.b_global))
        out = a * s + b
        return out.astype(np.float32, copy=False)
