from __future__ import annotations
import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base import BaseMethod
from typing import Optional


class NaiveBayesMethod(BaseMethod):
    name = "Naive Bayes"
    needs_weights = False
    needs_seed = False

    def __init__(self):
        self.clf = None
        self.good = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "NaiveBayesMethod":
        std0 = H0_train.std(axis=0)
        std1 = H1_train.std(axis=0)
        self.good = (std0 > 1e-6) & (std1 > 1e-6)
        
        if not np.any(self.good):
            # Create a dummy classifier that returns zeros
            self.clf = None
            return self

        H0_c = H0_train[:, self.good]
        H1_c = H1_train[:, self.good]

        X_tr = np.vstack([H0_c, H1_c])
        y_tr = np.hstack([np.zeros(len(H0_c)), np.ones(len(H1_c))])

        self.clf = GaussianNB(var_smoothing=1e-3)
        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None or self.good is None or not np.any(self.good):
            return np.zeros(X.shape[0], dtype=np.float32)
        
        X_c = X[:, self.good]
        jll = self.clf._joint_log_likelihood(X_c)
        return (jll[:, 1] - jll[:, 0]).astype(np.float32)
