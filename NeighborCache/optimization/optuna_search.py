"""Optuna search utilities for NP-calibrated semantic-cache experiments.

The functions here deliberately optimize validation performance after the
normal TRAIN/CALIB protocol has already been created:

* TRAIN fits scorers.
* CALIB selects NP thresholds from H0 scores.
* VALIDATION is the only held-out split used by Optuna.
* TEST remains untouched for final reporting.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


CandidateEvaluator = Callable[[Any, Dict[str, Any], Any], Tuple[List[Dict[str, Any]], Dict[str, Any]]]


def _method_name_matches(candidate: str, target: str) -> bool:
    candidate_norm = str(candidate).strip().lower().replace(" ", "")
    target_norm = str(target).strip().lower().replace(" ", "")
    if candidate_norm == target_norm:
        return True
    aliases = {
        "pcawhitenedcosine": {"whitenedcosine"},
        "whitenedcosine": {"pcawhitenedcosine"},
    }
    return candidate_norm in aliases.get(target_norm, set())


@dataclass
class OptunaSearchResult:
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int
    trials: List[Dict[str, Any]]
    candidate_rows: List[Dict[str, Any]]


def _split_one_eval(
    idx: np.ndarray,
    *,
    val_frac: float,
    min_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(idx, dtype=np.int64).reshape(-1).copy()
    rng.shuffle(arr)
    n = int(arr.size)
    if n <= 1:
        return np.array([], dtype=np.int64), arr

    min_test = int(max(1, min_test))
    val_frac = float(np.clip(val_frac, 0.05, 0.95))
    n_val = int(round(n * val_frac))
    n_val = max(1, n_val)
    if n > min_test:
        n_val = min(n_val, n - min_test)
    else:
        n_val = 0

    return arr[:n_val].astype(np.int64, copy=False), arr[n_val:].astype(np.int64, copy=False)


def split_global_eval_for_validation(
    gs: Any,
    *,
    val_frac: float,
    seed: int,
    min_test_h0: int = 1,
    min_test_h1: int = 1,
) -> tuple[Any, Any, Dict[str, int]]:
    """Split only the eval partition into validation and final test views."""
    rng = np.random.default_rng(int(seed))
    h0_val, h0_test = _split_one_eval(
        gs.H0_eval,
        val_frac=val_frac,
        min_test=min_test_h0,
        rng=rng,
    )
    h1_val, h1_test = _split_one_eval(
        gs.H1_eval,
        val_frac=val_frac,
        min_test=min_test_h1,
        rng=rng,
    )
    val = type(gs)(
        H0_train=gs.H0_train,
        H1_train=gs.H1_train,
        H0_calib=gs.H0_calib,
        H1_calib=gs.H1_calib,
        H0_eval=h0_val,
        H1_eval=h1_val,
    )
    test = type(gs)(
        H0_train=gs.H0_train,
        H1_train=gs.H1_train,
        H0_calib=gs.H0_calib,
        H1_calib=gs.H1_calib,
        H0_eval=h0_test,
        H1_eval=h1_test,
    )
    stats = {
        "val_h0": int(h0_val.size),
        "val_h1": int(h1_val.size),
        "test_h0": int(h0_test.size),
        "test_h1": int(h1_test.size),
    }
    return val, test, stats


def split_region_eval_for_validation(
    splits: Iterable[Any],
    *,
    val_frac: float,
    seed: int,
    min_test_h0: int = 1,
    min_test_h1: int = 1,
) -> tuple[List[Any], List[Any], Dict[str, int]]:
    """Split each region's eval partition into validation and final test views.

    Regions without at least one H0 and one H1 validation and test example are
    dropped from both views, which keeps validation/test region comparability.
    """
    split_list = list(splits)
    rng = np.random.default_rng(int(seed))
    val_splits: List[Any] = []
    test_splits: List[Any] = []
    dropped = 0

    for s in split_list:
        h0_val, h0_test = _split_one_eval(
            s.H0_eval,
            val_frac=val_frac,
            min_test=min_test_h0,
            rng=rng,
        )
        h1_val, h1_test = _split_one_eval(
            s.H1_eval,
            val_frac=val_frac,
            min_test=min_test_h1,
            rng=rng,
        )
        if h0_val.size == 0 or h1_val.size == 0 or h0_test.size == 0 or h1_test.size == 0:
            dropped += 1
            continue

        common = {
            "rid": int(s.rid),
            "H0_train": s.H0_train,
            "H1_train": s.H1_train,
            "H0_calib": s.H0_calib,
            "H1_calib": s.H1_calib,
        }
        val_splits.append(type(s)(H0_eval=h0_val, H1_eval=h1_val, **common))
        test_splits.append(type(s)(H0_eval=h0_test, H1_eval=h1_test, **common))

    stats = {
        "input_regions": int(len(split_list)),
        "val_regions": int(len(val_splits)),
        "test_regions": int(len(test_splits)),
        "dropped_regions": int(dropped),
        "val_h0": int(sum(s.H0_eval.size for s in val_splits)),
        "val_h1": int(sum(s.H1_eval.size for s in val_splits)),
        "test_h0": int(sum(s.H0_eval.size for s in test_splits)),
        "test_h1": int(sum(s.H1_eval.size for s in test_splits)),
    }
    return val_splits, test_splits, stats


def _suggest_log_float(trial: Any, name: str, low: float, high: float) -> float:
    return float(trial.suggest_float(name, low, high, log=True))


def sample_search_params(
    trial: Any,
    args: Any,
    *,
    scope: str,
    has_xgb: bool,
    has_text_pairs: bool,
    has_x_cos: bool,
    include_ocats: bool,
    include_online: bool,
) -> Dict[str, Any]:
    """Sample scorer, calibration, reranker, ensemble, and online policy knobs."""
    del has_x_cos  # Kept in the signature for future scalar-score-only spaces.
    params: Dict[str, Any] = {}

    # PCAWhitenedCosine
    params["pca_whiten_rank_mode"] = trial.suggest_categorical(
        "pca_whiten_rank_mode",
        ["fixed", "explained_variance", "threshold"],
    )
    params["pca_whiten_max_rank"] = trial.suggest_categorical(
        "pca_whiten_max_rank",
        [8, 16, 32, 64, 128, 256, None],
    )
    params["pca_whiten_explained_variance"] = trial.suggest_float(
        "pca_whiten_explained_variance",
        0.85,
        0.999,
    )
    params["pca_whiten_abs_eps"] = _suggest_log_float(trial, "pca_whiten_abs_eps", 1e-9, 1e-3)
    params["pca_whiten_rel_eps"] = _suggest_log_float(trial, "pca_whiten_rel_eps", 1e-9, 1e-3)
    params["pca_whiten_norm_eps"] = _suggest_log_float(trial, "pca_whiten_norm_eps", 1e-12, 1e-6)

    if has_xgb:
        params["xgb_n_estimators"] = trial.suggest_int("xgb_n_estimators", 20, 250, log=True)
        params["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 2, 8)
        params["xgb_learning_rate"] = _suggest_log_float(trial, "xgb_learning_rate", 0.01, 0.3)
        params["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.55, 1.0)
        params["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.55, 1.0)
        params["xgb_min_child_weight"] = _suggest_log_float(trial, "xgb_min_child_weight", 0.1, 20.0)
        params["xgb_gamma"] = trial.suggest_float("xgb_gamma", 0.0, 5.0)
        params["xgb_reg_alpha"] = _suggest_log_float(trial, "xgb_reg_alpha", 1e-8, 10.0)
        params["xgb_reg_lambda"] = _suggest_log_float(trial, "xgb_reg_lambda", 1e-4, 50.0)

    params["tiny_mlp_hidden_dim"] = trial.suggest_categorical("tiny_mlp_hidden_dim", [8, 16, 32, 64, 128])
    params["tiny_mlp_n_layers"] = trial.suggest_int("tiny_mlp_n_layers", 1, 3)
    params["tiny_mlp_alpha"] = _suggest_log_float(trial, "tiny_mlp_alpha", 1e-6, 1e-1)
    params["tiny_mlp_learning_rate_init"] = _suggest_log_float(
        trial,
        "tiny_mlp_learning_rate_init",
        1e-4,
        1e-2,
    )
    params["tiny_mlp_max_iter"] = trial.suggest_categorical("tiny_mlp_max_iter", [200, 400, 800, 1200])
    params["tiny_mlp_batch_size"] = trial.suggest_categorical("tiny_mlp_batch_size", [32, 64, 128, 256, "auto"])
    params["tiny_mlp_activation"] = trial.suggest_categorical("tiny_mlp_activation", ["relu", "tanh"])
    params["tiny_mlp_early_stopping"] = trial.suggest_categorical("tiny_mlp_early_stopping", [False, True])
    params["tiny_mlp_validation_fraction"] = trial.suggest_float("tiny_mlp_validation_fraction", 0.05, 0.4)

    params["lda_solver"] = trial.suggest_categorical("lda_solver", ["lsqr", "eigen", "svd"])
    if params["lda_solver"] == "svd":
        params["lda_shrinkage"] = None
    else:
        shrink_kind = trial.suggest_categorical("lda_shrinkage_kind", ["auto", "float", "none"])
        if shrink_kind == "float":
            params["lda_shrinkage"] = trial.suggest_float("lda_shrinkage", 0.0, 1.0)
        elif shrink_kind == "none":
            params["lda_shrinkage"] = None
        else:
            params["lda_shrinkage"] = "auto"
    params["lda_tol"] = _suggest_log_float(trial, "lda_tol", 1e-6, 1e-2)

    # Weighted ensemble meta-optimization.
    params["ensemble_ridge"] = _suggest_log_float(trial, "ensemble_ridge", 1e-8, 1e-1)
    params["ensemble_standardize"] = trial.suggest_categorical("ensemble_standardize", [False, True])
    params["ensemble_nonneg_simplex"] = trial.suggest_categorical("ensemble_nonneg_simplex", [True, False])
    params["ensemble_meta_frac"] = trial.suggest_float("ensemble_meta_frac", 0.1, 0.5)
    params["ensemble_tpr_tie_tol"] = _suggest_log_float(trial, "ensemble_tpr_tie_tol", 1e-6, 1e-2)
    params["ensemble_tie_break_entropy"] = trial.suggest_categorical(
        "ensemble_tie_break_entropy",
        [True, False],
    )

    # Calibration policy. Threshold values are never tuned directly.
    if scope == "global":
        params["tau_mode"] = "global"
        params["tau_shrink"] = False
    else:
        if str(getattr(args, "local_fit_mode", "pooled")) == "per_region":
            params["tau_mode"] = "local"
            params["tau_shrink"] = False
        else:
            params["tau_mode"] = trial.suggest_categorical(
                "tau_mode",
                ["local", "shrink_local", "cluster_local"],
            )
            params["tau_shrink"] = bool(
                params["tau_mode"] == "cluster_local"
                and trial.suggest_categorical("tau_shrink", [False, True])
            )

    params["tau_guardrail"] = trial.suggest_categorical(
        "tau_guardrail",
        ["none", "clopper_pearson", "wilson", "beta_ucb"],
    )
    params["tau_guardrail_delta"] = _suggest_log_float(trial, "tau_guardrail_delta", 1e-4, 0.2)
    params["tau_shrink_m"] = _suggest_log_float(trial, "tau_shrink_m", 20.0, 5000.0)
    params["shrink_k"] = _suggest_log_float(trial, "shrink_k", 10.0, 5000.0)
    params["tau_cluster_k"] = trial.suggest_categorical("tau_cluster_k", [2, 4, 8, 16, 32, 64])

    params["swc_k"] = trial.suggest_categorical("swc_k", [8, 16, 32, 64, 128, 256])
    params["swc_shrinkage"] = trial.suggest_float("swc_shrinkage", 0.0, 0.8)
    params["swc_min_samples"] = trial.suggest_categorical("swc_min_samples", [20, 50, 100, 200, 400])
    params["swc_eps"] = _suggest_log_float(trial, "swc_eps", 1e-9, 1e-3)
    params["swc_fallback"] = trial.suggest_categorical("swc_fallback", [True, False])
    params["swc_mode"] = trial.suggest_categorical("swc_mode", ["global", "region", "cluster"])
    params["swc_cluster_n_clusters"] = trial.suggest_categorical(
        "swc_cluster_n_clusters",
        [4, 8, 16, 32, 64, 128],
    )

    params["cos_affine_grouping"] = trial.suggest_categorical("cos_affine_grouping", ["region", "cluster"])
    params["cos_affine_n_clusters"] = trial.suggest_categorical(
        "cos_affine_n_clusters",
        [4, 8, 16, 32, 64, 128],
    )
    params["rwe_k_shrink"] = _suggest_log_float(trial, "rwe_k_shrink", 10.0, 5000.0)
    params["rwe_min_region_h0"] = trial.suggest_categorical("rwe_min_region_h0", [5, 10, 20, 30, 50, 100])
    params["rwe_min_region_h1"] = trial.suggest_categorical("rwe_min_region_h1", [5, 10, 20, 30, 50, 100])

    if has_text_pairs and bool(getattr(args, "enable_bge_reranker", False)):
        params["bge_batch_size"] = trial.suggest_categorical("bge_batch_size", [8, 16, 32, 64])
        params["bge_max_length"] = trial.suggest_categorical("bge_max_length", [128, 256, 384, 512])
        params["bge_backend"] = trial.suggest_categorical("bge_backend", ["auto", "cross", "bi"])
        params["bge_normalize_scores"] = trial.suggest_categorical("bge_normalize_scores", [False, True])

    if include_ocats:
        params["cache_k"] = trial.suggest_categorical("cache_k", [1, 2, 4, 8, 16, 32])
        params["e_thresh"] = trial.suggest_float("e_thresh", 0.01, math.log(2.0))
        params["d_thresh"] = _suggest_log_float(trial, "d_thresh", 0.02, 5.0)
        params["knn_weight_power"] = trial.suggest_float("knn_weight_power", 0.5, 6.0)
        params["mlp_hidden_dim"] = trial.suggest_categorical("mlp_hidden_dim", [16, 32, 64, 128, 256])
        params["mlp_dropout"] = trial.suggest_float("mlp_dropout", 0.0, 0.5)
        params["mlp_lr"] = _suggest_log_float(trial, "mlp_lr", 1e-5, 1e-2)
        params["mlp_epochs"] = trial.suggest_categorical("mlp_epochs", [10, 20, 40, 80])
        params["mlp_batch_size"] = trial.suggest_categorical("mlp_batch_size", [32, 64, 128, 256])
        params["mlp_weight_decay"] = _suggest_log_float(trial, "mlp_weight_decay", 1e-8, 1e-2)
        params["online_retrain_interval"] = trial.suggest_categorical(
            "online_retrain_interval",
            [0, 25, 50, 100, 200],
        )
        params["online_retrain_last_p"] = trial.suggest_categorical(
            "online_retrain_last_p",
            [32, 64, 128, 256, 512],
        )

    if include_online:
        params["online_batch_size"] = trial.suggest_categorical("online_batch_size", [16, 32, 64, 128, 256])
        params["online_mem_cap"] = trial.suggest_categorical("online_mem_cap", [128, 256, 512, 1000, 2000, 5000])
        params["online_update_mode"] = trial.suggest_categorical("online_update_mode", ["refit", "hill_climb"])
        params["online_hill_lr"] = trial.suggest_float("online_hill_lr", 0.01, 0.5)
        params["online_init_h0"] = trial.suggest_categorical("online_init_h0", [10, 25, 50, 100, 200])
        params["online_init_h1"] = trial.suggest_categorical("online_init_h1", [10, 25, 50, 100, 200])
        params["stop_check_every"] = trial.suggest_categorical("stop_check_every", [1, 2, 5, 10, 20])
        params["stop_window"] = trial.suggest_categorical("stop_window", [3, 5, 8, 10])
        params["stop_patience"] = trial.suggest_categorical("stop_patience", [1, 2, 3, 5])
        params["stop_eps_tpr"] = _suggest_log_float(trial, "stop_eps_tpr", 1e-5, 1e-2)
        params["stop_eps_fpr"] = _suggest_log_float(trial, "stop_eps_fpr", 1e-5, 1e-2)
        params["stop_eps_tau"] = _suggest_log_float(trial, "stop_eps_tau", 1e-6, 1e-2)
        params["stop_fpr_margin"] = trial.suggest_float("stop_fpr_margin", 0.0, 0.05)

    return params


def apply_params_to_namespace(args: Any, params: Dict[str, Any]) -> Any:
    """Return a shallow copy of argparse.Namespace with Optuna params applied."""
    out = copy.copy(args)
    for key, value in params.items():
        setattr(out, key, value)
    return out


def score_candidate_rows(
    rows: List[Dict[str, Any]],
    *,
    alpha: float,
    target_method: str,
    metric: str,
    fpr_penalty: float,
) -> tuple[float, Dict[str, Any]]:
    """Score validation rows without ever looking at test rows."""
    if not rows:
        return -1e9, {}

    if target_method not in {"", "all", "best", "best_feasible"}:
        filtered = [r for r in rows if _method_name_matches(str(r.get("method", "")), target_method)]
    else:
        filtered = rows
    if not filtered:
        return -1e9, {}

    def _metric_value(r: Dict[str, Any]) -> float:
        if metric == "macro_tpr":
            return float(r.get("macro_tpr", r.get("tpr", float("nan"))))
        if metric == "utility":
            return float(r.get("discounted_score", r.get("micro_tpr", r.get("tpr", float("nan")))))
        return float(r.get("micro_tpr", r.get("tpr", float("nan"))))

    # Enforce FPR constraint strictly: consider only rows with micro_fpr <= alpha.
    feasible: List[tuple[float, Dict[str, Any]]] = []
    for r in filtered:
        val = _metric_value(r)
        fpr = float(r.get("micro_fpr", r.get("fpr", float("inf"))))
        if not np.isfinite(val):
            continue
        if not np.isfinite(fpr):
            continue
        if fpr <= float(alpha) + 1e-12:
            feasible.append((float(val), r))

    if not feasible:
        # No feasible candidate found under the FPR constraint.
        return -1e9, {}

    # Pick the candidate with highest metric (TPR/utility). Tie-break by lower FPR.
    feasible.sort(key=lambda item: (item[0], -float(item[1].get("micro_fpr", item[1].get("fpr", float("inf"))))), reverse=True)
    best_value, best_row = feasible[0]
    return float(best_value), best_row


def run_optuna_search(
    *,
    args: Any,
    scope: str,
    evaluate_candidate: CandidateEvaluator,
    has_xgb: bool,
    has_text_pairs: bool,
    has_x_cos: bool,
    include_ocats: bool,
    include_online: bool,
    n_trials: int,
    timeout: Optional[float],
    seed: int,
    target_method: str,
    metric: str,
    fpr_penalty: float,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
) -> OptunaSearchResult:
    """Run an Optuna study using a project-specific validation callback."""
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - exercised only when dependency missing
        raise ImportError(
            "Optuna search requested but optuna is not installed. "
            "Install requirements.txt or `pip install optuna`."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, min(10, int(n_trials) // 4)))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage and study_name),
    )

    trial_records: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = sample_search_params(
            trial,
            args,
            scope=scope,
            has_xgb=has_xgb,
            has_text_pairs=has_text_pairs,
            has_x_cos=has_x_cos,
            include_ocats=include_ocats,
            include_online=include_online,
        )
        candidate_args = apply_params_to_namespace(args, params)
        rows: List[Dict[str, Any]] = []
        extra: Dict[str, Any] = {}
        try:
            rows, extra = evaluate_candidate(candidate_args, params, trial)
            value, selected = score_candidate_rows(
                rows,
                alpha=float(getattr(args, "alpha")),
                target_method=target_method,
                metric=metric,
                fpr_penalty=fpr_penalty,
            )
        except Exception as exc:
            value = -1e9
            selected = {}
            extra = {"exception": str(exc)}

        for row in rows:
            row_out = dict(row)
            row_out["optuna_trial"] = int(trial.number)
            candidate_rows.append(row_out)

        selected_method = str(selected.get("method", ""))
        selected_fpr = float(selected.get("micro_fpr", selected.get("fpr", float("nan")))) if selected else float("nan")
        selected_tpr = float(selected.get("micro_tpr", selected.get("tpr", float("nan")))) if selected else float("nan")
        record = {
            "optuna_trial": int(trial.number),
            "value": float(value),
            "selected_method": selected_method,
            "selected_tpr": selected_tpr,
            "selected_fpr": selected_fpr,
            "n_rows": int(len(rows)),
            "failed": bool(value <= -1e8),
            "params": dict(params),
            "extra": dict(extra),
        }
        trial_records.append(record)
        trial.set_user_attr("sampled_params", dict(params))
        trial.set_user_attr("selected_method", selected_method)
        trial.set_user_attr("selected_tpr", selected_tpr)
        trial.set_user_attr("selected_fpr", selected_fpr)
        if extra:
            trial.set_user_attr("extra", dict(extra))
        return float(value)

    study.optimize(objective, n_trials=int(n_trials), timeout=timeout, gc_after_trial=True)

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs.get("sampled_params", best_trial.params))
    return OptunaSearchResult(
        best_params=best_params,
        best_value=float(best_trial.value),
        best_trial_number=int(best_trial.number),
        trials=trial_records,
        candidate_rows=candidate_rows,
    )
