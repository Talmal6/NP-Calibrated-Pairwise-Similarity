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
        # Exact Optuna best-trial XGBoost params
        xgb_n_estimators: int = 238,
        xgb_max_depth: int = 4,
        xgb_learning_rate: float = 0.15,
        xgb_subsample: float = 0.85,
        xgb_colsample_bytree: float = 0.65,
        xgb_min_child_weight: float = 1.40,
        xgb_gamma: float = 2,
        xgb_reg_alpha: float = 0.03,
        xgb_reg_lambda: float = 0.01,

        # Backward-compatible aliases.
        # Keep these so old code using n_estimators=... still works.
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        gamma: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
    ):
        if not HAS_XGB:
            raise ImportError("xgboost is not available")

        self.clf = None

        # Old unprefixed args override defaults if explicitly provided.
        # This preserves old behavior while making xgb_* the canonical names.
        if n_estimators is not None:
            xgb_n_estimators = n_estimators
        if max_depth is not None:
            xgb_max_depth = max_depth
        if learning_rate is not None:
            xgb_learning_rate = learning_rate
        if subsample is not None:
            xgb_subsample = subsample
        if colsample_bytree is not None:
            xgb_colsample_bytree = colsample_bytree
        if min_child_weight is not None:
            xgb_min_child_weight = min_child_weight
        if gamma is not None:
            xgb_gamma = gamma
        if reg_alpha is not None:
            xgb_reg_alpha = reg_alpha
        if reg_lambda is not None:
            xgb_reg_lambda = reg_lambda

        self.xgb_n_estimators = int(max(1, xgb_n_estimators))
        self.xgb_max_depth = int(max(1, xgb_max_depth))
        self.xgb_learning_rate = float(max(1e-6, xgb_learning_rate))
        self.xgb_subsample = float(np.clip(xgb_subsample, 0.05, 1.0))
        self.xgb_colsample_bytree = float(np.clip(xgb_colsample_bytree, 0.05, 1.0))
        self.xgb_min_child_weight = float(max(0.0, xgb_min_child_weight))
        self.xgb_gamma = float(max(0.0, xgb_gamma))
        self.xgb_reg_alpha = float(max(0.0, xgb_reg_alpha))
        self.xgb_reg_lambda = float(max(0.0, xgb_reg_lambda))

        # Optional aliases, useful if other code reads method.n_estimators, etc.
        self.n_estimators = self.xgb_n_estimators
        self.max_depth = self.xgb_max_depth
        self.learning_rate = self.xgb_learning_rate
        self.subsample = self.xgb_subsample
        self.colsample_bytree = self.xgb_colsample_bytree
        self.min_child_weight = self.xgb_min_child_weight
        self.gamma = self.xgb_gamma
        self.reg_alpha = self.xgb_reg_alpha
        self.reg_lambda = self.xgb_reg_lambda

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
        y_tr = np.hstack([
            np.zeros(len(H0_train), dtype=np.int32),
            np.ones(len(H1_train), dtype=np.int32),
        ])

        self.clf = XGBClassifier(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            min_child_weight=self.xgb_min_child_weight,
            gamma=self.xgb_gamma,
            reg_alpha=self.xgb_reg_alpha,
            reg_lambda=self.xgb_reg_lambda,
            n_jobs=1,
            verbosity=0,
            eval_metric="logloss",
            random_state=int(seed if seed is not None else 42),
        )

        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("XGBoostLightMethod.score() called before fit().")

        return self.clf.predict_proba(X)[:, 1].astype(np.float32)