from __future__ import annotations

from .base import OnlineBaseMethod

# --- Strong baselines ---
from .cosine import CosineMethod
from .weighted_vector import VectorWeightedMethod
from .logistic_regression import LogisticRegressionMethod
from .lda import LDAMethod
from .whitened_cosine import WhitenedCosineMethod
from .naive_bayes import NaiveBayesMethod
from .auc_weighted_diff import AUCWeightedDiffMethod
from .pair_logreg import PairFeatureLogRegMethod
from .weighted_ensemble import WeightedEnsembleMethod
from .projector import ProjectedMethod
from .cosine_augmented import CosineAugmentedMethod


def _has_xgb() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _try_import_new_methods():
    """
    Import optional/new methods. If a file isn't present yet,
    we skip it without breaking the package.
    """
    out = {}

    # Whitening / Mahalanobis
    try:
        from .whitened_cosine import WhitenedCosineMethod  # type: ignore
        out["WhitenedCosineMethod"] = WhitenedCosineMethod
    except Exception:
        pass

    try:
        from .mahalanobis_delta import MahalanobisDeltaMethod  # type: ignore
        out["MahalanobisDeltaMethod"] = MahalanobisDeltaMethod
    except Exception:
        pass

    # Multi-prototype cosine
    try:
        from .multiprototype_cosine import MultiPrototypeCosineMethod  # type: ignore
        out["MultiPrototypeCosineMethod"] = MultiPrototypeCosineMethod
    except Exception:
        pass

    # Feature-rich linear model
    try:
        from .pair_logreg import PairFeatureLogRegMethod  # type: ignore
        out["PairFeatureLogRegMethod"] = PairFeatureLogRegMethod
    except Exception:
        pass

    # AUC-weighted diff (soft feature selection)
    try:
        from .auc_weighted_diff import AUCWeightedDiffMethod  # type: ignore
        out["AUCWeightedDiffMethod"] = AUCWeightedDiffMethod
    except Exception:
        pass

    # Calibration on cosine
    try:
        from .isotonic_cosine import IsotonicCalibratedCosineMethod  # type: ignore
        out["IsotonicCalibratedCosineMethod"] = IsotonicCalibratedCosineMethod
    except Exception:
        pass

    # Strong GBDT (sklearn)
    try:
        from .hist_gbdt import HistGBDTMethod  # type: ignore
        out["HistGBDTMethod"] = HistGBDTMethod
    except Exception:
        pass

    # Stabilized whitened cosine
    try:
        from .stabilized_whitened_cosine import StabilizedWhitenedCosineMethod  # type: ignore
        out["StabilizedWhitenedCosineMethod"] = StabilizedWhitenedCosineMethod
    except Exception:
        pass

    return out


def get_default_methods(has_xgb: bool | None = None):
    if has_xgb is None:
        has_xgb = _has_xgb()

    opt = _try_import_new_methods()

    # ============================================================
    # Base methods (keep only methods with sane inductive bias
    # for semantic embeddings; drop systematic losers)
    # ============================================================
    base_methods: list[OnlineBaseMethod] = [
        CosineMethod(),
    ]

    # Whitening / Mahalanobis family (NEW)
    if "WhitenedCosineMethod" in opt:
        base_methods.append(opt["WhitenedCosineMethod"]())  # WhitenedCosine
    if "MahalanobisDeltaMethod" in opt:
        base_methods.append(opt["MahalanobisDeltaMethod"]())  # -||W(x-y)||

    # Other strong contenders
    base_methods += [
        VectorWeightedMethod(),
        LogisticRegressionMethod(),
        LDAMethod(),
    ]

    # Multi-prototype cosine (NEW)
    if "MultiPrototypeCosineMethod" in opt:
        base_methods.append(opt["MultiPrototypeCosineMethod"](k=4))

    # ------------------------------------------------------------
    # NEW: Weighted Ensemble over strong "judges"
    # (create fresh instances; don't reuse those in base_methods)
    # ------------------------------------------------------------
    ensemble_judges: list[OnlineBaseMethod] = [
        CosineMethod(),
    ]

    if "WhitenedCosineMethod" in opt:
        ensemble_judges.append(opt["WhitenedCosineMethod"]())
    if "MahalanobisDeltaMethod" in opt:
        ensemble_judges.append(opt["MahalanobisDeltaMethod"]())
    
    ensemble_judges += [
        LDAMethod(),
        NaiveBayesMethod(),
    ]
    
    if "MultiPrototypeCosineMethod" in opt:
        ensemble_judges.append(opt["MultiPrototypeCosineMethod"](k=4))

    base_methods.append(
        WeightedEnsembleMethod(
            judges=ensemble_judges,
        )
    )

    # Pair-features LogReg (NEW) – requires pair-based data, skip in single-vector mode
    # if "PairFeatureLogRegMethod" in opt:
    #     base_methods.append(opt["PairFeatureLogRegMethod"]())

    # AUC-weighted diff (NEW) – requires pair-based data, skip in single-vector mode
    # if "AUCWeightedDiffMethod" in opt:
    #     base_methods.append(opt["AUCWeightedDiffMethod"](drop_frac=0.10))

    # Isotonic calibration (NEW) – requires pair-based data, skip in single-vector mode
    # if "IsotonicCalibratedCosineMethod" in opt:
    #     base_methods.append(opt["IsotonicCalibratedCosineMethod"]())

    # Strong tree baseline (NEW) – requires pair-based data, skip in single-vector mode
    # if "HistGBDTMethod" in opt:
    #     base_methods.append(opt["HistGBDTMethod"](max_depth=3))

    # Optional XGB
    if has_xgb:
        from .xgboost import XGBoostLightMethod
        base_methods.append(XGBoostLightMethod())

    # ============================================================
    # Projected methods (keep only ones that make geometric sense)
    # ============================================================
    projected_methods: list[OnlineBaseMethod] = [
        # LDA projections
        ProjectedMethod(
            name="LDA1+LogReg",
            base_method=LogisticRegressionMethod(),
            proj_kind="lda",
            proj_dim=1,
        ),
        ProjectedMethod(
            name="LDA1+Cosine",
            base_method=CosineMethod(),
            proj_kind="lda",
            proj_dim=1,
        ),

        # PCA only with strong linear-ish models
        ProjectedMethod(
            name="PCA16+LogReg",
            base_method=LogisticRegressionMethod(),
            proj_kind="pca",
            proj_dim=16,
        ),
        ProjectedMethod(
            name="PCA32+LogReg",
            base_method=LogisticRegressionMethod(),
            proj_kind="pca",
            proj_dim=32,
        ),
        ProjectedMethod(
            name="PCA16+VecWeighted",
            base_method=VectorWeightedMethod(),
            proj_kind="pca",
            proj_dim=16,
        ),
        ProjectedMethod(
            name="PCA32+VecWeighted",
            base_method=VectorWeightedMethod(),
            proj_kind="pca",
            proj_dim=32,
        ),
    ]

    # ============================================================
    # Cosine-augmented (keep the winners)
    # ============================================================
    # cosine_augmented_methods: list[OnlineBaseMethod] = [
    #     CosineAugmentedMethod(
    #         name="Cos+LDA1+LogReg",
    #         base_method=LogisticRegressionMethod(),
    #         proj_kind="lda",
    #         proj_dim=1,
    #     ),
    #     CosineAugmentedMethod(
    #         name="Cos+PCA8+LogReg",
    #         base_method=LogisticRegressionMethod(),
    #         proj_kind="pca",
    #         proj_dim=8,
    #     ),
    #     CosineAugmentedMethod(
    #         name="Cos+PCA16+LogReg",
    #         base_method=LogisticRegressionMethod(),
    #         proj_kind="pca",
    #         proj_dim=16,
    #     ),
    #     CosineAugmentedMethod(
    #         name="Cos+LDA1+VecWgt",
    #         base_method=VectorWeightedMethod(),
    #         proj_kind="lda",
    #         proj_dim=1,
    #     ),
    # ]

    return base_methods + projected_methods 