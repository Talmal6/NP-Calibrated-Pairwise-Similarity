from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import BaseMethod
from typing import Optional


@dataclass(frozen=True)
class AndBoxModel:
    t: np.ndarray          # shape (d,)
    active: np.ndarray     # shape (k,)


def _fit_andbox_core(
    H0: np.ndarray,
    H1: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    seed: int,
    weighted_pick: bool,
    steps: int = 4000,
) -> AndBoxModel:
    """
    Fit AND-box on the provided data (intended: TRAIN ONLY).
    Returns a model (threshold vector t and active dims).
    """
    rng = np.random.default_rng(seed)
    N0, d = H0.shape
    max_fp = int(np.floor(alpha * N0))
    global_min = float(min(H0.min(), H1.min()) - 1.0)

    n_active = max(64, int(d * 0.25))
    if d > n_active:
        active = np.argsort(weights)[-n_active:]
    else:
        active = np.arange(d)

    if weighted_pick:
        w = weights.astype(np.float64).copy()
        w = w - w.min() + 1e-12
        p = w[active] / np.sum(w[active])
        pick = lambda: int(rng.choice(active, p=p))
    else:
        pick = lambda: int(rng.choice(active))

    t = np.full(d, global_min, dtype=np.float32)

    # heuristic init: pick beta_star so that ALL(active) constraint has approx FPR<=alpha (on H0)
    beta_star = 0.001
    lo, hi = 0.0, 1.0
    H0_a = H0[:, active]
    for _ in range(20):
        mid = (lo + hi) / 2.0
        t_val = np.quantile(H0_a, 1.0 - mid, axis=0)
        fpr_mid = float(np.mean(np.all(H0_a >= t_val, axis=1)))
        if fpr_mid <= alpha:
            beta_star = mid
            lo = mid
        else:
            hi = mid

    t[active] = np.quantile(H0_a, 1.0 - beta_star, axis=0).astype(np.float32)

    qs = np.linspace(0.0, 0.999, 40)
    cand = np.quantile(H0, qs, axis=0).astype(np.float32)
    cancel = np.full((1, d), global_min, dtype=np.float32)
    cand = np.vstack([cancel, cand])
    cand = np.maximum.accumulate(cand, axis=0)

    def metrics(curr_t: np.ndarray):
        a0 = np.all(H0 >= curr_t, axis=1)
        fp = int(np.sum(a0))
        if fp > max_fp:
            return -1, fp
        tp = int(np.sum(np.all(H1 >= curr_t, axis=1)))
        return tp, fp

    current_tp, current_fp = metrics(t)
    best_tp, best_fp = current_tp, current_fp
    best_t = t.copy()
    no_improve = 0

    for _ in range(steps):
        i = pick()
        col = cand[:, i]
        curr_idx = np.searchsorted(col, t[i], side="right") - 1
        curr_idx = int(np.clip(curr_idx, 0, len(col) - 1))

        if rng.random() < 0.05:
            new_idx = 0
        else:
            jump = int(rng.integers(-4, 5))
            new_idx = int(np.clip(curr_idx + jump, 0, len(col) - 1))

        old = t[i]
        t[i] = col[new_idx]
        new_tp, new_fp = metrics(t)

        if new_tp != -1:
            better = (new_tp > current_tp) or (new_tp == current_tp and new_fp < current_fp)
            if better:
                current_tp, current_fp = new_tp, new_fp
                no_improve = 0
                if (new_tp > best_tp) or (new_tp == best_tp and new_fp < best_fp):
                    best_tp, best_fp = new_tp, new_fp
                    best_t = t.copy()
            else:
                t[i] = old
                no_improve += 1
        else:
            t[i] = old
            no_improve += 1

        if no_improve >= 600:
            break

    return AndBoxModel(t=best_t.astype(np.float32), active=active.astype(np.int32))


def score_andbox_margin(model: AndBoxModel, X: np.ndarray) -> np.ndarray:
    """
    Continuous score for NP calibration:
        margin(x) = min_{i in active} (x_i - t_i)
    Larger is "more positive". The natural accept boundary is margin>=0,
    but NP calibration will learn a threshold on calib(H0).
    """
    a = model.active
    return np.min(X[:, a] - model.t[a], axis=1).astype(np.float32)


class AndBoxHCMethod(BaseMethod):
    """AND-Box with hard-coded (HC) variant."""
    name = "AndBox-HC"
    needs_weights = True
    needs_seed = True

    def __init__(self):
        self.model = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "AndBoxHCMethod":
        # Note: alpha must be passed via the run() method
        # This fit method is a no-op placeholder
        if weights is None:
            raise ValueError("weights is required for AndBoxHCMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxHCMethod")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit() must be called before score()")
        return score_andbox_margin(self.model, X)

    def run(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        H0_calib: np.ndarray,
        H0_eval: np.ndarray,
        H1_eval: np.ndarray,
        alpha: float,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        tie_mode: str = "ge",
    ):
        """Override run to handle alpha-dependent fitting."""
        from .base import MethodResult, _np_eval_from_calib
        import time

        if weights is None:
            raise ValueError("weights is required for AndBoxHCMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxHCMethod")

        # Fit with alpha
        self.model = _fit_andbox_core(
            H0_train, H1_train, weights, alpha, seed, weighted_pick=False, steps=4000
        )

        # Score
        s0_cal = self.score(H0_calib)
        s0_ev = self.score(H0_eval)
        s1_ev = self.score(H1_eval)

        # NP metrics (eval)
        tpr, fpr, thresh = _np_eval_from_calib(s0_cal, s0_ev, s1_ev, alpha, tie_mode=tie_mode)

        # Train metrics (reuse same threshold from calib)
        s0_tr = self.score(H0_train)
        s1_tr = self.score(H1_train)
        if tie_mode == "gt":
            train_tpr = float(np.mean(s1_tr > thresh))
            train_fpr = float(np.mean(s0_tr > thresh))
        else:
            train_tpr = float(np.mean(s1_tr >= thresh))
            train_fpr = float(np.mean(s0_tr >= thresh))

        # Inference timing
        t0 = time.perf_counter()
        _ = self.score(H1_eval)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return MethodResult(tpr=tpr, fpr=fpr, time_ms=float(dt_ms),
                            train_tpr=train_tpr, train_fpr=train_fpr)


class AndBoxWgtMethod(BaseMethod):
    """AND-Box with weighted (WGT) variant."""
    name = "AndBox-Wgt"
    needs_weights = True
    needs_seed = True

    def __init__(self):
        self.model = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "AndBoxWgtMethod":
        # Note: alpha must be passed via the run() method
        # This fit method is a no-op placeholder
        if weights is None:
            raise ValueError("weights is required for AndBoxWgtMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxWgtMethod")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit() must be called before score()")
        return score_andbox_margin(self.model, X)

    def run(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        H0_calib: np.ndarray,
        H0_eval: np.ndarray,
        H1_eval: np.ndarray,
        alpha: float,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        tie_mode: str = "ge",
    ):
        """Override run to handle alpha-dependent fitting."""
        from .base import MethodResult, _np_eval_from_calib
        import time

        if weights is None:
            raise ValueError("weights is required for AndBoxWgtMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxWgtMethod")

        # Fit with alpha
        self.model = _fit_andbox_core(
            H0_train, H1_train, weights, alpha, seed, weighted_pick=True, steps=4000
        )

        # Score
        s0_cal = self.score(H0_calib)
        s0_ev = self.score(H0_eval)
        s1_ev = self.score(H1_eval)

        # NP metrics (eval)
        tpr, fpr, thresh = _np_eval_from_calib(s0_cal, s0_ev, s1_ev, alpha, tie_mode=tie_mode)

        # Train metrics (reuse same threshold from calib)
        s0_tr = self.score(H0_train)
        s1_tr = self.score(H1_train)
        if tie_mode == "gt":
            train_tpr = float(np.mean(s1_tr > thresh))
            train_fpr = float(np.mean(s0_tr > thresh))
        else:
            train_tpr = float(np.mean(s1_tr >= thresh))
            train_fpr = float(np.mean(s0_tr >= thresh))

        # Inference timing
        t0 = time.perf_counter()
        _ = self.score(H1_eval)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return MethodResult(tpr=tpr, fpr=fpr, time_ms=float(dt_ms),
                            train_tpr=train_tpr, train_fpr=train_fpr)
