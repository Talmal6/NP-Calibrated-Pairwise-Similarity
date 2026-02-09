from __future__ import annotations
import numpy as np
from .base import MethodResult
from ..utils.timing import time_ms
from ..utils.metrics import get_metrics_at_fpr

class CosineMethod:
    name = "Cosine"
    needs_weights = False
    needs_seed = False

    def run(self, H0: np.ndarray, H1: np.ndarray, alpha: float, weights=None, seed=None) -> MethodResult:
        scores0 = H0.sum(axis=1)

        def infer_h1():
            return H1.sum(axis=1)

        scores1, dt = time_ms(infer_h1, reps=50, warmup=1)
        tpr, fpr = get_metrics_at_fpr(scores0, scores1, alpha, tie_mode="ge")
        return MethodResult(tpr=tpr, fpr=fpr, time_ms=dt)
