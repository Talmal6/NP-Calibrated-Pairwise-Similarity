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
    needs_seed = False

    def __init__(self):
        if not HAS_XGB:
            raise ImportError("xgboost is not available")
        self.clf = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "XGBoostLightMethod":
        X_tr = np.vstack([H0_train, H1_train])
        y_tr = np.hstack([np.zeros(len(H0_train)), np.ones(len(H1_train))])

        self.clf = XGBClassifier(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1,
            n_jobs=1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)
