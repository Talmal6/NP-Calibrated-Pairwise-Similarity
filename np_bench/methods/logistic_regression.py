from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from .base import MethodResult
from ..utils.timing import time_ms
from ..utils.metrics import get_metrics_at_fpr


class LogisticRegressionMethod:
    name = "Log Reg"
    needs_weights = False
    needs_seed = False

    def run(self, H0: np.ndarray, H1: np.ndarray, alpha: float, weights=None, seed=None) -> MethodResult:
        X_tr = np.vstack([H0, H1])
        y_tr = np.hstack([np.zeros(len(H0)), np.ones(len(H1))])

        clf = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        scores0 = clf.predict_proba(H0)[:, 1]

        def infer_h1():
            return clf.predict_proba(H1)[:, 1]

        scores1, dt = time_ms(infer_h1, reps=10, warmup=1)
        tpr, fpr = get_metrics_at_fpr(scores0, scores1, alpha, tie_mode="ge")
        return MethodResult(tpr=tpr, fpr=fpr, time_ms=dt)
