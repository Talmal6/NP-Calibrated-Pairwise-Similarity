from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base import BaseMethod
from typing import Optional


BEST_LDA_PARAMS_FROM_AVAILABLE_OPTUNA_CONTEXT = {
    # WARNING:
    # These are the lda_* values found inside the best overall XGBoost Optuna trial.
    # They are not guaranteed to be a standalone LDA-best configuration unless
    # your Optuna study explicitly optimized/evaluated LDA with these params.
    "lda_solver": "lsqr",
    "lda_shrinkage": "auto",
    "lda_tol": 0.00824064280898226,
}


class LDAMethod(BaseMethod):
    name = "LDA"
    needs_weights = False
    needs_seed = False

    def __init__(
        self,
        *,
        # Optuna-style canonical names
        lda_solver: str = "lsqr",
        lda_shrinkage: str | float | None = "auto",
        lda_tol: float = 0.00824064280898226,

        # Backward-compatible aliases
        solver: Optional[str] = None,
        shrinkage: str | float | None = None,
        tol: Optional[float] = None,
    ):
        self.clf = None

        # Old names override Optuna-style defaults only when explicitly provided.
        if solver is not None:
            lda_solver = solver
        if shrinkage is not None:
            lda_shrinkage = shrinkage
        if tol is not None:
            lda_tol = tol

        if lda_solver not in {"svd", "lsqr", "eigen"}:
            raise ValueError("LDAMethod solver must be one of {'svd', 'lsqr', 'eigen'}")

        self.lda_solver = str(lda_solver)
        self.lda_shrinkage = self._normalize_shrinkage(lda_shrinkage)
        self.lda_tol = float(max(0.0, lda_tol))

        # Backward-compatible attributes
        self.solver = self.lda_solver
        self.shrinkage = self.lda_shrinkage
        self.tol = self.lda_tol

    @staticmethod
    def _normalize_shrinkage(
        shrinkage: str | float | None,
    ) -> str | float | None:
        if isinstance(shrinkage, str):
            shrinkage_norm = shrinkage.strip().lower()

            if shrinkage_norm in {"none", "null"}:
                return None

            if shrinkage_norm == "auto":
                return "auto"

            return float(shrinkage_norm)

        return shrinkage

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "LDAMethod":
        del weights, seed

        X_tr = np.vstack([H0_train, H1_train])
        y_tr = np.hstack([
            np.zeros(len(H0_train), dtype=np.int32),
            np.ones(len(H1_train), dtype=np.int32),
        ])

        # sklearn ignores shrinkage for svd; passing shrinkage with svd is invalid.
        shrinkage = None if self.lda_solver == "svd" else self.lda_shrinkage

        self.clf = LinearDiscriminantAnalysis(
            solver=self.lda_solver,
            shrinkage=shrinkage,
            tol=self.lda_tol,
        )

        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("LDAMethod.score() called before fit().")

        return self.clf.decision_function(X).astype(np.float32)

    def linear_form(self) -> tuple[str, np.ndarray, float]:
        if self.clf is None:
            raise RuntimeError("LDAMethod.linear_form() called before fit().")
        coef = np.asarray(getattr(self.clf, "coef_", None), dtype=np.float64)
        intercept = np.asarray(getattr(self.clf, "intercept_", [0.0]), dtype=np.float64).reshape(-1)
        if coef.ndim == 2:
            w = coef[0]
        else:
            w = coef.reshape(-1)
        b = float(intercept[0]) if intercept.size else 0.0
        return ("hadamard_linear", w, b)
