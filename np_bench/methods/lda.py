from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base import MethodResult
from ..utils.timing import time_ms
from ..utils.metrics import get_metrics_at_fpr


class LDAMethod:
    name = "LDA"
    needs_weights = False
    needs_seed = False

    def run(self, H0: np.ndarray, H1: np.ndarray, alpha: float, weights=None, seed=None) -> MethodResult:
        X_tr = np.vstack([H0, H1])
        y_tr = np.hstack([np.zeros(len(H0)), np.ones(len(H1))])

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr, y_tr)

        scores0 = clf.decision_function(H0)

        def infer_h1():
            return clf.decision_function(H1)

        scores1, dt = time_ms(infer_h1, reps=50, warmup=1)
        tpr, fpr = get_metrics_at_fpr(scores0, scores1, alpha, tie_mode="ge")
        return MethodResult(tpr=tpr, fpr=fpr, time_ms=dt)
