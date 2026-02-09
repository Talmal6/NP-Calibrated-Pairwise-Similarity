from __future__ import annotations
import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base import MethodResult
from ..utils.timing import time_ms
from ..utils.metrics import get_metrics_at_fpr

class NaiveBayesMethod:
    name = "Naive Bayes"
    needs_weights = False
    needs_seed = False

    def run(self, H0: np.ndarray, H1: np.ndarray, alpha: float, weights=None, seed=None) -> MethodResult:
        std0 = H0.std(axis=0)
        std1 = H1.std(axis=0)
        good = (std0 > 1e-6) & (std1 > 1e-6)
        if not np.any(good):
            return MethodResult(tpr=0.0, fpr=0.0, time_ms=0.0)

        H0_c = H0[:, good]
        H1_c = H1[:, good]

        X_tr = np.vstack([H0_c, H1_c])
        y_tr = np.hstack([np.zeros(len(H0_c)), np.ones(len(H1_c))])

        clf = GaussianNB(var_smoothing=1e-3)
        clf.fit(X_tr, y_tr)

        jll0 = clf._joint_log_likelihood(H0_c)
        scores0 = jll0[:, 1] - jll0[:, 0]

        def infer_h1_llr():
            jll1 = clf._joint_log_likelihood(H1_c)
            return jll1[:, 1] - jll1[:, 0]

        scores1, dt = time_ms(infer_h1_llr, reps=50, warmup=1)
        tpr, fpr = get_metrics_at_fpr(scores0, scores1, alpha, tie_mode="ge")
        return MethodResult(tpr=tpr, fpr=fpr, time_ms=dt)
