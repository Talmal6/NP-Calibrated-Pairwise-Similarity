from __future__ import annotations
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from .base import BaseMethod
from typing import Optional


class TinyMLPMethod(BaseMethod):
    name = "Tiny MLP"
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
    ) -> "TinyMLPMethod":
        X_tr = np.vstack([H0_train, H1_train])
        y_tr = np.hstack([np.zeros(len(H0_train)), np.ones(len(H1_train))])

        self.clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            max_iter=800,
            alpha=0.001,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)
