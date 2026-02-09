from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from .base import BaseMethod
from typing import Optional


class LogisticRegressionMethod(BaseMethod):
    name = "Log Reg"
    needs_weights = False
    needs_seed = False

    def __init__(self):
        self.clf = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "LogisticRegressionMethod":
        X_tr = np.vstack([H0_train, H1_train])
        y_tr = np.hstack([np.zeros(len(H0_train)), np.ones(len(H1_train))])

        self.clf = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)
