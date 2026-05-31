from __future__ import annotations

import importlib
from typing import Optional, Type

from .base import OnlineBaseMethod

# Core baselines
from .cosine import CosineMethod
from .weighted_vector import VectorWeightedMethod
from .logistic_regression import LogisticRegressionMethod
from .lda import LDAMethod
from .tiny_mlp import TinyMLPMethod
from .andbox import AndBoxHCMethod
from .separation import ProjectedSeparationScoreMethod, SeparationScoreMethod

# Main method
from .whitened_cosine import WhitenedCosineMethod

# Ensemble
from .weighted_ensemble import WeightedEnsembleMethod

# Hadamard / Fisher family
from .fisher_hadamard_methods import HadamardCosineMethod

# Ablations
from .abl import (
    AblationMode,
    CANONICAL_ABLATIONS,
    available_ablations,
    make_ablation_methods,
)


def _has_xgb() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _optional_class(module_name: str, class_name: str) -> Optional[Type[OnlineBaseMethod]]:
    """Import an optional method class without breaking the package."""
    try:
        module = importlib.import_module(f"{__package__}.{module_name}")
        cls = getattr(module, class_name)
        return cls
    except Exception:
        return None


def _dedupe_by_name(methods: list[OnlineBaseMethod]) -> list[OnlineBaseMethod]:
    seen: set[str] = set()
    out: list[OnlineBaseMethod] = []
    for method in methods:
        name = getattr(method, "name", method.__class__.__name__)
        if name in seen:
            continue
        seen.add(name)
        out.append(method)
    return out


def _make_main_whitened_method() -> OnlineBaseMethod:
    return WhitenedCosineMethod(
        name="WhitenedCosine",
        eps=1e-6,
        rel_eps=1e-6,
        max_rank=128,
        rank_mode="explained_variance",
        explained_variance=0.99,
    )


def _make_ensemble_judges(has_xgb: bool) -> list[OnlineBaseMethod]:
    judges: list[OnlineBaseMethod] = [CosineMethod()]
    if has_xgb:
        XGBoostLightMethod = _optional_class("xgboost", "XGBoostLightMethod")
        if XGBoostLightMethod is not None:
            judges.append(XGBoostLightMethod())
    judges.extend([WhitenedCosineMethod(), LDAMethod()])
    return judges


def _make_ablation_suite() -> list[OnlineBaseMethod]:
    methods = make_ablation_methods(
        ablations=(
            "raw_hadamard_linear",
            "center_only_hadamard_linear",
            "pca_whitened_hadamard_linear",
            "diag_whitened_hadamard_linear",
            "h0_only_pca_whitened_hadamard_linear",
            "h1_only_pca_whitened_hadamard_linear",
            "within_class_pca_whitened_hadamard_linear",
            "shuffle_labels_pca_whitened_hadamard_linear",
            "no_rank_truncation_pca_whitened_hadamard_linear",
        ),
        abs_eps=1e-6,
        rel_eps=1e-6,
        max_rank=128,
        rank_mode="explained_variance",
        explained_variance=0.99,
    )
    return list(methods.values())


def get_default_methods(
    has_xgb: bool | None = None,
    *,
    include_ablations: bool = True,
    include_optional: bool = True,
    include_ensemble: bool = True,
) -> list[OnlineBaseMethod]:
    """Return the default benchmark method list."""
    if has_xgb is None:
        has_xgb = _has_xgb()

    methods: list[OnlineBaseMethod] = [
        CosineMethod(),
        AndBoxHCMethod(),
        HadamardCosineMethod(),
        _make_main_whitened_method(),
    ]

    if include_optional:
        MahalanobisDeltaMethod = _optional_class("mahalanobis_delta", "MahalanobisDeltaMethod")
        if MahalanobisDeltaMethod is not None:
            methods.append(MahalanobisDeltaMethod())
        MultiPrototypeCosineMethod = _optional_class("multiprototype_cosine", "MultiPrototypeCosineMethod")
        if MultiPrototypeCosineMethod is not None:
            methods.append(MultiPrototypeCosineMethod(k=4))
        methods.extend(
            [
                SeparationScoreMethod(exact=False),
                SeparationScoreMethod(exact=True),
                ProjectedSeparationScoreMethod(projection="lda", exact=False),
                ProjectedSeparationScoreMethod(projection="lda", exact=True),
                ProjectedSeparationScoreMethod(projection="pca_whiten", exact=False, dim=64),
                ProjectedSeparationScoreMethod(projection="pca_whiten", exact=True, dim=64),
            ]
        )

    methods.extend([VectorWeightedMethod(), LogisticRegressionMethod(), LDAMethod(), TinyMLPMethod()])

    if include_ensemble:
        methods.append(WeightedEnsembleMethod(judges=_make_ensemble_judges(has_xgb)))

    if include_ablations:
        methods.extend(_make_ablation_suite())

    if has_xgb:
        XGBoostLightMethod = _optional_class("xgboost", "XGBoostLightMethod")
        if XGBoostLightMethod is not None:
            methods.append(XGBoostLightMethod())

    return _dedupe_by_name(methods)


__all__ = [
    "OnlineBaseMethod",
    "CosineMethod",
    "VectorWeightedMethod",
    "LogisticRegressionMethod",
    "LDAMethod",
    "WhitenedCosineMethod",
    "WeightedEnsembleMethod",
    "TinyMLPMethod",
    "AndBoxHCMethod",
    "HadamardCosineMethod",
    "SeparationScoreMethod",
    "ProjectedSeparationScoreMethod",
    "available_ablations",
    "make_ablation_methods",
    "CANONICAL_ABLATIONS",
    "AblationMode",
    "get_default_methods",
]
