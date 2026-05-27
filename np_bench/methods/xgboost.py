from __future__ import annotations
import numpy as np
from .base import BaseMethod
from typing import Optional

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


class XGBoostLightMethod(BaseMethod):
    name = "XGBoost"
    needs_weights = False
    needs_seed = True

    def __init__(
        self,
        *,
        n_estimators: int = 30,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ):
        if not HAS_XGB:
            raise ImportError("xgboost is not available")
        self.clf = None
        self.n_estimators = int(max(1, n_estimators))
        self.max_depth = int(max(1, max_depth))
        self.learning_rate = float(max(1e-6, learning_rate))
        self.subsample = float(np.clip(subsample, 0.05, 1.0))
        self.colsample_bytree = float(np.clip(colsample_bytree, 0.05, 1.0))
        self.min_child_weight = float(max(0.0, min_child_weight))
        self.gamma = float(max(0.0, gamma))
        self.reg_alpha = float(max(0.0, reg_alpha))
        self.reg_lambda = float(max(0.0, reg_lambda))

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "XGBoostLightMethod":
        del weights
        X_tr = np.vstack([H0_train, H1_train])
        y_tr = np.hstack([np.zeros(len(H0_train)), np.ones(len(H1_train))])

        self.clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=int(seed if seed is not None else 42),
        )
        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)
