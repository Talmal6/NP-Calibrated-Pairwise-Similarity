from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base import BaseMethod
from typing import Optional


class LDAMethod(BaseMethod):
    name = "LDA"
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
    ) -> "LDAMethod":
        X_tr = np.vstack([H0_train, H1_train])
        y_tr = np.hstack([np.zeros(len(H0_train)), np.ones(len(H1_train))])

        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(X).astype(np.float32)
