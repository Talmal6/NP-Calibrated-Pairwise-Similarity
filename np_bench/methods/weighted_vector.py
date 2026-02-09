from __future__ import annotations
import numpy as np
from .base import MethodResult
from ..utils.timing import time_ms
from ..utils.metrics import get_metrics_at_fpr

class VectorWeightedMethod:
    name = "Vec (Wgt)"
    needs_weights = True
    needs_seed = False

    def run(self, H0: np.ndarray, H1: np.ndarray, alpha: float, weights=None, seed=None) -> MethodResult:
        if weights is None:
            raise ValueError("weights is required")

        w = weights.astype(np.float32)
        w = w / (w.mean() + 1e-9)

        scores0 = H0 @ w

        def infer_h1():
            return H1 @ w

        scores1, dt = time_ms(infer_h1, reps=20, warmup=1)
        tpr, fpr = get_metrics_at_fpr(scores0, scores1, alpha, tie_mode="ge")
        return MethodResult(tpr=tpr, fpr=fpr, time_ms=dt)
