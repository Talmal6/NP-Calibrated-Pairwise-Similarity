from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base import BaseMethod
from typing import Optional


class LDAMethod(BaseMethod):
    name = "LDA"
    needs_weights = False
    needs_seed = False

    def __init__(
        self,
        *,
        solver: str = "lsqr",
        shrinkage: str | float | None = "auto",
        tol: float = 1e-4,
    ):
        self.clf = None
        if solver not in {"svd", "lsqr", "eigen"}:
            raise ValueError("LDAMethod solver must be one of {'svd', 'lsqr', 'eigen'}")
        self.solver = str(solver)
        if isinstance(shrinkage, str):
            shrinkage_norm = shrinkage.strip().lower()
            if shrinkage_norm in {"none", "null"}:
                self.shrinkage = None
            elif shrinkage_norm == "auto":
                self.shrinkage = "auto"
            else:
                self.shrinkage = float(shrinkage_norm)
        else:
            self.shrinkage = shrinkage
        self.tol = float(max(0.0, tol))

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

        shrinkage = None if self.solver == "svd" else self.shrinkage
        self.clf = LinearDiscriminantAnalysis(
            solver=self.solver,
            shrinkage=shrinkage,
            tol=self.tol,
        )
        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(X).astype(np.float32)
