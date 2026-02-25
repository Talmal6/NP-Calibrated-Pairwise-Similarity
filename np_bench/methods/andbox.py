from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import time
from typing import Optional

from .base import BaseMethod, MethodResult


# =============================================================================
# 1. Original AndBox Logic (Hard & Weighted)
# =============================================================================

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
    Fit AND-box on the provided data (TRAIN).
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

    # Heuristic init
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
    Continuous score: margin(x) = min_{i in active} (x_i - t_i)
    """
    a = model.active
    return np.min(X[:, a] - model.t[a], axis=1).astype(np.float32)


# =============================================================================
# 2. Robust AndBox Logic (Soft Boundaries & Dimensionality Reduction)
# =============================================================================

@dataclass(frozen=True)
class RobustAndBoxModel:
    t: np.ndarray           # shape (d,)
    active: np.ndarray      # shape (n_active,)
    allowed_violations: int # k-out-of-n logic


def score_robust_box(model: RobustAndBoxModel, X: np.ndarray) -> np.ndarray:
    """
    Score based on the k-th worst margin, allowing 'allowed_violations' dimensions to fail.
    """
    a = model.active
    margins = X[:, a] - model.t[a]  # shape (N, n_active)
    
    k = model.allowed_violations
    n_active = len(a)

    if k >= n_active:
        return np.max(margins, axis=1).astype(np.float32)
    
    if k == 0:
        return np.min(margins, axis=1).astype(np.float32)

    # Use partition to find the k-th smallest element.
    # The element at index k represents the boundary where k items are smaller.
    partitioned = np.partition(margins, k, axis=1)
    return partitioned[:, k].astype(np.float32)


def _fit_robust_andbox(
    H0: np.ndarray,
    H1: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    seed: int,
    max_active: int = 12,        # Restrict number of active dimensions
    robustness_ratio: float = 0.15, # Allow % of dimensions to violate threshold
    steps: int = 3000,
) -> RobustAndBoxModel:
    
    rng = np.random.default_rng(seed)
    N0, d = H0.shape
    max_fp = int(np.floor(alpha * N0))
    global_min = float(min(H0.min(), H1.min()) - 1.0)

    # 1. Select top active dimensions
    n_active = min(d, max_active)
    if d > n_active:
        active = np.argsort(weights)[-n_active:]
    else:
        active = np.arange(d)
    
    # 2. Determine allowed violations
    allowed_violations = int(np.floor(n_active * robustness_ratio))
    
    # 3. Initialize thresholds (conservative start)
    t = np.full(d, global_min, dtype=np.float32)
    H0_a = H0[:, active]
    # Start at 5th percentile
    t[active] = np.quantile(H0_a, 0.05, axis=0).astype(np.float32)

    qs = np.linspace(0.0, 0.99, 50)
    cand = np.quantile(H0, qs, axis=0).astype(np.float32)
    
    def get_metrics(model_t):
        tmp_model = RobustAndBoxModel(model_t, active, allowed_violations)
        
        # Binary decision: score >= 0 means positive
        scores_h0 = score_robust_box(tmp_model, H0)
        fp = np.sum(scores_h0 >= 0)
        
        if fp > max_fp:
            return -1, fp
        
        scores_h1 = score_robust_box(tmp_model, H1)
        tp = np.sum(scores_h1 >= 0)
        return tp, fp

    current_tp, current_fp = get_metrics(t)
    best_tp, best_fp = current_tp, current_fp
    best_t = t.copy()
    
    no_improve = 0
    
    for _ in range(steps):
        # Pick random dimension from active set
        idx_in_active = rng.integers(0, len(active))
        i = active[idx_in_active]
        
        col_cands = cand[:, i]
        curr_val = t[i]
        curr_idx = np.searchsorted(col_cands, curr_val)
        
        jump = int(rng.integers(-5, 6))
        new_idx = np.clip(curr_idx + jump, 0, len(col_cands) - 1)
        
        old_val = t[i]
        t[i] = col_cands[new_idx]
        
        new_tp, new_fp = get_metrics(t)
        
        if new_tp != -1:
            better = (new_tp > current_tp) or (new_tp == current_tp and new_fp < current_fp)
            if better:
                current_tp, current_fp = new_tp, new_fp
                no_improve = 0
                if (new_tp > best_tp) or (new_tp == best_tp and new_fp < best_fp):
                    best_tp, best_fp = new_tp, new_fp
                    best_t = t.copy()
            else:
                t[i] = old_val
                no_improve += 1
        else:
            t[i] = old_val
            no_improve += 1
            
        if no_improve > 600:
            break
            
    return RobustAndBoxModel(best_t.astype(np.float32), active.astype(np.int32), allowed_violations)


# =============================================================================
# 3. Method Classes
# =============================================================================

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
        alpha: float = 0.05
    ) -> "AndBoxHCMethod":
        if weights is None:
            raise ValueError("weights is required for AndBoxHCMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxHCMethod")

        self.model = _fit_andbox_core(
            H0_train, H1_train, weights, alpha, seed,
            weighted_pick=False, steps=4000
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit() must be called before score()")
        return score_andbox_margin(self.model, X)

    def run(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        H0_eval: np.ndarray,
        H1_eval: np.ndarray,
        alpha: float,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        tie_mode: str = "ge",
    ):
        if weights is None:
            raise ValueError("weights is required for AndBoxHCMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxHCMethod")

        self.model = _fit_andbox_core(
            H0_train, H1_train, weights, alpha, seed, weighted_pick=False, steps=4000
        )

        # Threshold from H0_train
        s0_tr = self.score(H0_train)
        thresh = float(np.quantile(s0_tr, 1.0 - alpha))

        s0_ev = self.score(H0_eval)
        s1_ev = self.score(H1_eval)

        if tie_mode == "gt":
            tpr = float(np.mean(s1_ev > thresh))
            fpr = float(np.mean(s0_ev > thresh))
        else:
            tpr = float(np.mean(s1_ev >= thresh))
            fpr = float(np.mean(s0_ev >= thresh))

        s1_tr = self.score(H1_train)
        if tie_mode == "gt":
            train_tpr = float(np.mean(s1_tr > thresh))
            train_fpr = float(np.mean(s0_tr > thresh))
        else:
            train_tpr = float(np.mean(s1_tr >= thresh))
            train_fpr = float(np.mean(s0_tr >= thresh))

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
        alpha: float = 0.05,
    ) -> "AndBoxWgtMethod":
        if weights is None:
            raise ValueError("weights is required for AndBoxWgtMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxWgtMethod")

        self._weights = weights
        self._seed = seed
        self.model = _fit_andbox_core(
            H0_train, H1_train, weights, alpha, seed,
            weighted_pick=True, steps=4000
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit() must be called before score()")
        return score_andbox_margin(self.model, X)

    def run(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        H0_eval: np.ndarray,
        H1_eval: np.ndarray,
        alpha: float,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        tie_mode: str = "ge",
    ):
        if weights is None:
            raise ValueError("weights is required for AndBoxWgtMethod")
        if seed is None:
            raise ValueError("seed is required for AndBoxWgtMethod")

        self.model = _fit_andbox_core(
            H0_train, H1_train, weights, alpha, seed, weighted_pick=True, steps=4000
        )

        # Threshold from H0_train
        s0_tr = self.score(H0_train)
        thresh = float(np.quantile(s0_tr, 1.0 - alpha))

        s0_ev = self.score(H0_eval)
        s1_ev = self.score(H1_eval)

        if tie_mode == "gt":
            tpr = float(np.mean(s1_ev > thresh))
            fpr = float(np.mean(s0_ev > thresh))
        else:
            tpr = float(np.mean(s1_ev >= thresh))
            fpr = float(np.mean(s0_ev >= thresh))

        s1_tr = self.score(H1_train)
        if tie_mode == "gt":
            train_tpr = float(np.mean(s1_tr > thresh))
            train_fpr = float(np.mean(s0_tr > thresh))
        else:
            train_tpr = float(np.mean(s1_tr >= thresh))
            train_fpr = float(np.mean(s0_tr >= thresh))

        t0 = time.perf_counter()
        _ = self.score(H1_eval)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return MethodResult(tpr=tpr, fpr=fpr, time_ms=float(dt_ms),
                            train_tpr=train_tpr, train_fpr=train_fpr)


class RobustAndBoxMethod(AndBoxWgtMethod):
    """
    Robust And-Box Method.
    Uses 'k-out-of-n' logic and reduced dimensionality to handle noise.
    """
    name = "RobustAndBox"
    needs_weights = True
    needs_seed = True

    def __init__(self):
        super().__init__()
        self.model = None

    def fit(self, H0_train, H1_train, *, weights=None, seed=None, alpha=0.05):
        if weights is None or seed is None:
            raise ValueError(f"{self.name} requires weights and seed")
            
        self.model = _fit_robust_andbox(
            H0_train, H1_train, weights, alpha, seed, 
            max_active=12, robustness_ratio=0.15, steps=3000
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("fit() must be called before score()")
        return score_robust_box(self.model, X)

    def run(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        H0_eval: np.ndarray,
        H1_eval: np.ndarray,
        alpha: float,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        tie_mode: str = "ge",
    ):
        if weights is None or seed is None:
            raise ValueError(f"{self.name} requires weights and seed")

        # Fit model using robust logic
        self.model = _fit_robust_andbox(
            H0_train, H1_train, weights, alpha, seed, 
            max_active=12, robustness_ratio=0.15, steps=3000
        )
        
        # Threshold from H0_train
        s0_tr = score_robust_box(self.model, H0_train)
        thresh = float(np.quantile(s0_tr, 1.0 - alpha))

        s0_ev = score_robust_box(self.model, H0_eval)
        s1_ev = score_robust_box(self.model, H1_eval)
        
        if tie_mode == "gt":
            tpr = float(np.mean(s1_ev > thresh))
            fpr = float(np.mean(s0_ev > thresh))
        else:
            tpr = float(np.mean(s1_ev >= thresh))
            fpr = float(np.mean(s0_ev >= thresh))
        
        s1_tr = score_robust_box(self.model, H1_train)
        if tie_mode == "gt":
            train_tpr = float(np.mean(s1_tr > thresh))
            train_fpr = float(np.mean(s0_tr > thresh))
        else:
            train_tpr = float(np.mean(s1_tr >= thresh))
            train_fpr = float(np.mean(s0_tr >= thresh))

        t0 = time.perf_counter()
        _ = score_robust_box(self.model, H1_eval)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return MethodResult(
            tpr=tpr, 
            fpr=fpr, 
            time_ms=float(dt_ms),
            train_tpr=train_tpr, 
            train_fpr=train_fpr
        )