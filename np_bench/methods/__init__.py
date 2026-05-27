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

# Main method
from .whitened_cosine import WhitenedCosineMethod

# Ensemble
from .weighted_ensemble import WeightedEnsembleMethod

# Hadamard / Fisher family
from .fisher_hadamard_methods import HadamardCosineMethod

# Ablations
from .abl import (
    available_ablations,
    make_ablation_methods,
    CANONICAL_ABLATIONS,
    AblationMode,
)


def _has_xgb() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _optional_class(module_name: str, class_name: str) -> Optional[Type[OnlineBaseMethod]]:
    """
    Import an optional method class without breaking the package if the file
    or dependency is unavailable.
    """
    try:
        module = importlib.import_module(f"{__package__}.{module_name}")
        cls = getattr(module, class_name)
        return cls
    except Exception:
        return None


def _dedupe_by_name(methods: list[OnlineBaseMethod]) -> list[OnlineBaseMethod]:
    """
    Keep the first method with each name.

    This prevents accidental duplicate entries when a method is imported through
    both the stable and optional paths.
    """
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
    """
    Main method.

    This should match the winning ablation:

        ablation:pca_whitened_hadamard_linear

    Expected scoring route under --hadamard_preprocess:

        X = emb(query) * emb(anchor)
        score(X) = X @ w + b
    """
    return WhitenedCosineMethod(
        name="WhitenedCosine",
        eps=1e-6,
        rel_eps=1e-6,
        max_rank=128,
        rank_mode="explained_variance",
        explained_variance=0.99,
    )


def _make_ensemble_judges(has_xgb: bool) -> list[OnlineBaseMethod]:
    """
    Build fresh judge instances for WeightedEnsemble.

    Do not reuse objects from base_methods, because each method stores fitted
    state after training.
    """
    judges: list[OnlineBaseMethod] = [
        CosineMethod(),
    ]

    if has_xgb:
        XGBoostLightMethod = _optional_class("xgboost", "XGBoostLightMethod")
        if XGBoostLightMethod is not None:
            judges.append(XGBoostLightMethod())

    judges.extend(
        [
            WhitenedCosineMethod(),
            LDAMethod(),
        ]
    )

    return judges


def _make_ablation_suite() -> list[OnlineBaseMethod]:
    """
    Default ablations for the current --hadamard_preprocess evaluation path.

    raw_cosine and current_whitened_cosine are intentionally excluded here
    because the current benchmark route gives precomputed Hadamard features,
    not raw embedding pairs. They remain available through make_ablation_methods.
    """
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
    """
    Return the default benchmark method list.

    The main method is WhitenedCosine, which should now be the production name
    for the pooled PCA-whitened Hadamard linear scorer.

    To run only core methods:

        get_default_methods(include_ablations=False)

    To run the full analysis suite:

        get_default_methods(include_ablations=True)
    """
    if has_xgb is None:
        has_xgb = _has_xgb()

    methods: list[OnlineBaseMethod] = []

    # ============================================================
    # Minimal geometric baselines
    # ============================================================
    methods.extend(
        [
            CosineMethod(),
            AndBoxHCMethod(),
            HadamardCosineMethod(),
        ]
    )

    # ============================================================
    # Main proposed method
    # ============================================================
    methods.append(_make_main_whitened_method())

    # ============================================================
    # Optional strong related methods
    # ============================================================
    if include_optional:
        MahalanobisDeltaMethod = _optional_class(
            "mahalanobis_delta",
            "MahalanobisDeltaMethod",
        )
        if MahalanobisDeltaMethod is not None:
            methods.append(MahalanobisDeltaMethod())

        MultiPrototypeCosineMethod = _optional_class(
            "multiprototype_cosine",
            "MultiPrototypeCosineMethod",
        )
        if MultiPrototypeCosineMethod is not None:
            methods.append(MultiPrototypeCosineMethod(k=4))

    # ============================================================
    # Linear / neural baselines
    # ============================================================
    methods.extend(
        [
            VectorWeightedMethod(),
            LogisticRegressionMethod(),
            LDAMethod(),
            TinyMLPMethod(),
        ]
    )

    # ============================================================
    # Weighted ensemble
    # ============================================================
    if include_ensemble:
        methods.append(
            WeightedEnsembleMethod(
                judges=_make_ensemble_judges(has_xgb),
            )
        )

    # ============================================================
    # Ablation suite
    # ============================================================
    if include_ablations:
        methods.extend(_make_ablation_suite())

    # ============================================================
    # XGBoost baseline
    # ============================================================
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
    "available_ablations",
    "make_ablation_methods",
    "CANONICAL_ABLATIONS",
    "AblationMode",
    "get_default_methods",
]