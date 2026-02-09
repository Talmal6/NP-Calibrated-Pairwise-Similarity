from __future__ import annotations

from .cosine import CosineMethod
from .weighted_vector import VectorWeightedMethod
from .naive_bayes import NaiveBayesMethod
from .logistic_regression import LogisticRegressionMethod
from .lda import LDAMethod
from .tiny_mlp import TinyMLPMethod
from .andbox import AndBoxHCMethod, AndBoxWgtMethod


def _has_xgb() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def get_default_methods(has_xgb: bool | None = None):
    if has_xgb is None:
        has_xgb = _has_xgb()

    methods = [
        CosineMethod(),
        VectorWeightedMethod(),
        NaiveBayesMethod(),
        LogisticRegressionMethod(),
        LDAMethod(),
    ]

    if has_xgb:
        from .xgboost import XGBoostLightMethod
        methods.append(XGBoostLightMethod())

    methods += [
        TinyMLPMethod(),
        AndBoxHCMethod(),
        AndBoxWgtMethod(),
    ]
    return methods
