from __future__ import annotations
import numpy as np
from .base import MethodResult
from ..utils.timing import time_ms


def _andbox_core(
    H0: np.ndarray,
    H1: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    seed: int,
    weighted_pick: bool,
    steps: int = 4000,
) -> MethodResult:
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

    def infer_h1():
        return np.all(H1 >= best_t, axis=1)

    _, dt = time_ms(infer_h1, reps=50, warmup=1)

    tpr = float(best_tp) / max(1, len(H1))
    fpr = float(best_fp) / max(1, len(H0))
    return MethodResult(tpr=tpr, fpr=fpr, time_ms=dt)


class AndBoxHCMethod:
    name = "AND-Box (HC)"
    needs_weights = True
    needs_seed = True

    def run(self, H0: np.ndarray, H1: np.ndarray, alpha: float, weights=None, seed=None) -> MethodResult:
        if weights is None:
            raise ValueError("AndBoxHCMethod requires weights")
        if seed is None:
            raise ValueError("AndBoxHCMethod requires seed")
        return _andbox_core(H0, H1, weights, alpha, seed, weighted_pick=False, steps=4000)


class AndBoxWgtMethod:
    name = "AND-Box (Wgt)"
    needs_weights = True
    needs_seed = True

    def run(self, H0: np.ndarray, H1: np.ndarray, alpha: float, weights=None, seed=None) -> MethodResult:
        if weights is None:
            raise ValueError("AndBoxWgtMethod requires weights")
        if seed is None:
            raise ValueError("AndBoxWgtMethod requires seed")
        return _andbox_core(H0, H1, weights, alpha, seed, weighted_pick=True, steps=4000)
