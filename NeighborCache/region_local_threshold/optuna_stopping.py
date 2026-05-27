"""Optuna tuner for the existing NP-Calib online stopping mechanism."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .evaluation import apply_threshold
from .io_helpers import load_npz, resolve_features, resolve_npz_path
from .online_stopping_eval import (
    run_online_stopping,
    sample_monitor_from_global_eval,
    sample_monitor_from_region_splits,
)
from .preprocessing import (
    _build_hadamard_features,
    _build_semantic_buckets_from_regions,
    _l2_normalize_rows,
)
from .splits import (
    filter_global_split_by_score_range,
    filter_region_splits_by_score_range_detailed,
    split_global,
    split_indices_per_region_detailed,
)
from .stopping_mechanism import StopConfig


REGION_KEY_ALIASES: Dict[str, str] = {}
STOP_CONFIG_FIELD_NAMES = list(StopConfig.__dataclass_fields__.keys())

STOPPING_PARAM_NAMES = [
    "stop_check_every",
    "stop_window",
    "stop_patience",
    "stop_eps_tpr",
    "stop_eps_fpr",
    "stop_eps_tau",
    "stop_fpr_margin",
    "n_monitor_h0",
    "n_monitor_h1",
    "online_init_h0",
    "online_init_h1",
    "online_batch_size",
    "online_mem_cap",
    "online_update_mode",
    "online_hill_lr",
]


def _concat_indices(parts: List[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(p, dtype=np.int64) for p in parts if p is not None and p.size > 0]
    if not valid:
        return np.array([], dtype=np.int64)
    return np.concatenate(valid).astype(np.int64, copy=False)

TRIAL_METRIC_NAMES = [
    "selection_mode",
    "raw_alpha_constraint_passed",
    "relaxed_margin_constraint_passed",
    "final_constraint_used",
    "alpha",
    "alpha_plus_margin",
    "mean_tpr",
    "mean_fpr",
    "max_fpr",
    "p95_fpr",
    "fpr_violation_rate",
    "mean_steps_until_stop",
    "mean_samples_total_used",
    "mean_samples_streamed",
    "stop_success_rate",
    "tau_mean",
    "tau_std",
    "tau_p95_abs_slope",
    "constraint_passed",
    "strict_objective_value",
    "relaxed_objective_value",
]


def sample_stopping_params(
    trial: Any,
    *,
    tune_stop_fpr_margin: bool = True,
    fixed_stop_fpr_margin: float = 0.0,
) -> Dict[str, Any]:
    """Sample only online stopping and online update hyperparameters."""
    params = {
        "stop_check_every": trial.suggest_categorical("stop_check_every", [10, 25, 50, 100, 200]),
        "stop_window": trial.suggest_categorical("stop_window", [3, 5, 10, 20]),
        "stop_patience": trial.suggest_int("stop_patience", 2, 10),
        "stop_eps_tpr": trial.suggest_float("stop_eps_tpr", 1e-4, 5e-2, log=True),
        "stop_eps_fpr": trial.suggest_float("stop_eps_fpr", 1e-4, 2e-2, log=True),
        "stop_eps_tau": trial.suggest_float("stop_eps_tau", 1e-5, 5e-2, log=True),
        "n_monitor_h0": trial.suggest_categorical("n_monitor_h0", [100, 250, 500, 1000]),
        "n_monitor_h1": trial.suggest_categorical("n_monitor_h1", [100, 250, 500, 1000]),
        "online_init_h0": trial.suggest_categorical("online_init_h0", [50, 100, 250, 500]),
        "online_init_h1": trial.suggest_categorical("online_init_h1", [50, 100, 250, 500]),
        "online_batch_size": trial.suggest_categorical("online_batch_size", [8, 16, 32, 64, 128]),
        "online_mem_cap": trial.suggest_categorical("online_mem_cap", [500, 1000, 2000, 5000]),
        "online_update_mode": trial.suggest_categorical("online_update_mode", ["refit", "hill_climb"]),
        "online_hill_lr": trial.suggest_float("online_hill_lr", 1e-3, 0.5, log=True),
    }
    if tune_stop_fpr_margin:
        params["stop_fpr_margin"] = trial.suggest_float("stop_fpr_margin", 0.0, 0.02)
    else:
        params["stop_fpr_margin"] = float(fixed_stop_fpr_margin)
    return params


def build_args_with_trial_params(base_args: argparse.Namespace, params: Dict[str, Any]) -> argparse.Namespace:
    """Return a shallow namespace copy with one trial's sampled parameters applied."""
    out = copy.copy(base_args)
    for key, value in params.items():
        setattr(out, key, value)
    setattr(out, "enable_online_stopping", True)
    return out


def _parse_eval_seeds(raw: Optional[str], fallback_seed: int) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return [int(fallback_seed)]
    out: List[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            out.append(int(text))
    return out or [int(fallback_seed)]


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if np.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=np.float64)


def _mean_or(values: Iterable[Any], default: float) -> float:
    arr = _finite_values(values)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    return obj


def _ensure_prepared_data(base_args: argparse.Namespace) -> argparse.Namespace:
    if hasattr(base_args, "_X_main") and hasattr(base_args, "_y") and hasattr(base_args, "_region_id"):
        return base_args

    npz_path = resolve_npz_path(str(base_args.data))
    ds = load_npz(npz_path)
    if "label" not in ds:
        raise ValueError(f"{npz_path} missing required array 'label'")

    y = ds["label"].astype(np.int32, copy=False)
    feat_key, X_main, X_cos = resolve_features(ds)
    X_anchor_source = np.asarray(X_main, dtype=np.float32)

    region_key = str(base_args.region_key)
    sem_bucket_source_key_used: Optional[str] = None
    if region_key == "sem_bucket" and region_key not in ds:
        src_key = str(getattr(base_args, "sem_bucket_source_key", "global_cluster"))
        if src_key not in ds:
            raise ValueError(
                "On-the-fly sem_bucket generation requires a valid source region key; "
                f"missing '{src_key}' in {npz_path}. Available keys={sorted(ds.keys())}"
            )
        source_region_id = ds[src_key].astype(np.int64, copy=False)
        region_id, _ = _build_semantic_buckets_from_regions(
            X_anchor_source,
            source_region_id,
            n_buckets=max(1, int(getattr(base_args, "sem_bucket_k", 64))),
            seed=int(base_args.seed),
            anchor_strategy=str(base_args.hadamard_anchor_strategy),
        )
        sem_bucket_source_key_used = src_key
        region_key = "__computed_sem_bucket"
    elif region_key not in ds:
        alias_key = REGION_KEY_ALIASES.get(region_key)
        if alias_key in ds:
            region_key = alias_key

    if region_key in ds:
        region_id = ds[region_key].astype(np.int64, copy=False)
    elif region_key == "__computed_sem_bucket":
        pass
    elif str(base_args.tau_mode) == "global":
        region_key = "__synthetic_global_region"
        region_id = np.zeros(y.shape[0], dtype=np.int64)
    else:
        raise ValueError(
            f"{npz_path} missing required region array '{base_args.region_key}'; "
            f"have={sorted(ds.keys())}"
        )

    abs_diff_only = bool(base_args.use_abs_diff and not base_args.hadamard_preprocess)
    if base_args.use_delta_vec and not base_args.hadamard_preprocess:
        pass
    if base_args.use_delta_vec and base_args.use_abs_diff:
        raise ValueError("--use_delta_vec and --use_abs_diff are mutually exclusive")

    if base_args.hadamard_preprocess or abs_diff_only:
        if X_main.ndim != 2 or X_main.shape[1] <= 1:
            raise ValueError(
                f"--hadamard_preprocess requires embedding-like X_main with shape (N,D), D>1; got {X_main.shape}"
            )
        X_main, X_cos_had = _build_hadamard_features(
            X_main,
            region_id,
            strategy=str(base_args.hadamard_anchor_strategy),
            seed=int(base_args.seed),
            use_delta_vec=bool(base_args.use_delta_vec),
            use_abs_diff=bool(base_args.use_abs_diff),
            abs_diff_only=abs_diff_only,
        )
        X_cos = X_cos_had
        if abs_diff_only:
            feat_key = f"{feat_key}+absdiff_only"
        else:
            suffix = ["hadamard"]
            if base_args.use_delta_vec:
                suffix.append("delta")
            elif base_args.use_abs_diff:
                suffix.append("absdiff")
            feat_key = f"{feat_key}+{'_'.join(suffix)}"

    if base_args.normalize_data:
        X_main = _l2_normalize_rows(X_main)

    if X_main.shape[0] != y.shape[0] or region_id.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row mismatch: X={X_main.shape[0]} y={y.shape[0]} region_id={region_id.shape[0]}"
        )
    if X_cos is not None and X_cos.shape[0] != X_main.shape[0]:
        raise ValueError(f"Row mismatch: X_cos={X_cos.shape[0]} X_main={X_main.shape[0]}")

    setattr(base_args, "_npz_path", str(npz_path))
    setattr(base_args, "_feat_key", str(feat_key))
    setattr(base_args, "_X_main", np.asarray(X_main, dtype=np.float32))
    setattr(base_args, "_X_cos", X_cos)
    setattr(base_args, "_y", y)
    setattr(base_args, "_region_id", region_id)
    setattr(base_args, "_region_key_resolved", str(region_key))
    setattr(base_args, "_sem_bucket_source_key_used", sem_bucket_source_key_used)
    return base_args


def _apply_filter_policy_global(base_args: argparse.Namespace, gs: Any) -> tuple[Any, Dict[str, int]]:
    if str(base_args.filter_policy) != "ambiguous_only":
        return gs, {}
    X_cos = getattr(base_args, "_X_cos", None)
    if X_cos is None:
        raise RuntimeError("filter_policy='ambiguous_only' requires cosine_to_anchor")
    return filter_global_split_by_score_range(
        gs,
        score=X_cos[:, 0],
        score_min=float(base_args.ambiguous_cos_min),
        score_max=float(base_args.ambiguous_cos_max),
    )


def _apply_filter_policy_regions(base_args: argparse.Namespace, splits: List[Any]) -> tuple[List[Any], Dict[str, int]]:
    if str(base_args.filter_policy) != "ambiguous_only":
        return splits, {}
    X_cos = getattr(base_args, "_X_cos", None)
    if X_cos is None:
        raise RuntimeError("filter_policy='ambiguous_only' requires cosine_to_anchor")
    out, stats, _ = filter_region_splits_by_score_range_detailed(
        splits,
        score=X_cos[:, 0],
        score_min=float(base_args.ambiguous_cos_min),
        score_max=float(base_args.ambiguous_cos_max),
        min_h0_eval=int(base_args.min_h0_eval),
        min_h1_eval=int(base_args.min_h1_eval),
    )
    return out, stats


def _overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    if left.size == 0 or right.size == 0:
        return 0
    return int(np.intersect1d(left.astype(np.int64), right.astype(np.int64)).size)


def _monitor_count_with_holdout(requested: int, available: int, *, min_holdout: int = 1) -> int:
    requested = int(max(0, requested))
    available = int(max(0, available))
    min_holdout = int(max(0, min_holdout))
    return int(min(requested, max(0, available - min_holdout)))


def _evaluate_frozen_policy(
    online: Any,
    X_main: np.ndarray,
    *,
    h0_eval_idx: np.ndarray,
    h1_eval_idx: np.ndarray,
    tau: float,
    tie_mode: str,
) -> tuple[float, float, int, int]:
    if online is None or h0_eval_idx.size == 0 or h1_eval_idx.size == 0:
        return float("nan"), float("nan"), int(h0_eval_idx.size), int(h1_eval_idx.size)
    sc0 = np.asarray(online.score(X_main[h0_eval_idx]), dtype=np.float32).reshape(-1)
    sc1 = np.asarray(online.score(X_main[h1_eval_idx]), dtype=np.float32).reshape(-1)
    p0 = apply_threshold(sc0, float(tau), tie_mode)
    p1 = apply_threshold(sc1, float(tau), tie_mode)
    fpr = float(np.mean(p0 == 1)) if p0.size else float("nan")
    tpr = float(np.mean(p1 == 1)) if p1.size else float("nan")
    return tpr, fpr, int(p0.size), int(p1.size)


def run_single_seed_eval(base_args: argparse.Namespace, params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Run one sampled stopping configuration on one data split seed."""
    base_args = _ensure_prepared_data(base_args)
    args = build_args_with_trial_params(base_args, params)

    X_main = getattr(base_args, "_X_main")
    y = getattr(base_args, "_y")
    region_id = getattr(base_args, "_region_id")

    h0_monitor_idx = np.array([], dtype=np.int64)
    h1_monitor_idx = np.array([], dtype=np.int64)
    split_stats: Dict[str, Any] = {}
    monitor_stats: Dict[str, Any] = {}
    filter_stats: Dict[str, Any] = {}

    if str(base_args.tau_mode) == "global":
        gs, split_stats = split_global(
            y=y,
            n_train_cap=int(base_args.n_train),
            n_calib_cap=int(base_args.n_calib),
            n_eval_cap=int(base_args.n_eval),
            seed=int(seed),
        )
        gs, filter_stats = _apply_filter_policy_global(base_args, gs)
        n_monitor_h0 = _monitor_count_with_holdout(
            int(args.n_monitor_h0),
            int(gs.H0_eval.size),
            min_holdout=1,
        )
        n_monitor_h1 = _monitor_count_with_holdout(
            int(args.n_monitor_h1),
            int(gs.H1_eval.size),
            min_holdout=1,
        )
        gs, h0_monitor_idx, h1_monitor_idx, monitor_stats = sample_monitor_from_global_eval(
            gs,
            n_monitor_h0=n_monitor_h0,
            n_monitor_h1=n_monitor_h1,
            seed=int(seed),
        )
        h0_train_idx = np.asarray(gs.H0_train, dtype=np.int64)
        h1_train_idx = np.asarray(gs.H1_train, dtype=np.int64)
        h0_calib_idx = np.asarray(gs.H0_calib, dtype=np.int64)
        h1_calib_idx = np.asarray(gs.H1_calib, dtype=np.int64)
        h0_eval_idx = np.asarray(gs.H0_eval, dtype=np.int64)
        h1_eval_idx = np.asarray(gs.H1_eval, dtype=np.int64)
    else:
        splits, split_stats, _ = split_indices_per_region_detailed(
            region_id=region_id,
            y=y,
            n_train_cap=int(base_args.n_train),
            n_calib_cap=int(base_args.n_calib),
            n_eval_cap=int(base_args.n_eval),
            seed=int(seed),
            min_h0_eval=int(base_args.min_h0_eval),
            min_h1_eval=int(base_args.min_h1_eval),
        )
        splits, filter_stats = _apply_filter_policy_regions(base_args, splits)
        if not splits:
            return {
                "seed": int(seed),
                "success": False,
                "reason": "empty_region_splits",
                "tpr": float("nan"),
                "fpr": 1.0,
                "stopped": False,
                "steps_until_stop": 0,
                "samples_total_used": 0,
                "samples_streamed": 0,
                "tau": float("nan"),
                "tau_p95_abs_slope": float("nan"),
                "monitor_final_eval_overlap": 0,
            }
        splits, h0_monitor_idx, h1_monitor_idx, monitor_stats = sample_monitor_from_region_splits(
            splits,
            n_monitor_h0=int(args.n_monitor_h0),
            n_monitor_h1=int(args.n_monitor_h1),
            min_h0_eval=int(base_args.min_h0_eval),
            min_h1_eval=int(base_args.min_h1_eval),
            seed=int(seed),
        )
        h0_train_idx = _concat_indices([s.H0_train for s in splits])
        h1_train_idx = _concat_indices([s.H1_train for s in splits])
        h0_calib_idx = _concat_indices([s.H0_calib for s in splits])
        h1_calib_idx = _concat_indices([s.H1_calib for s in splits])
        h0_eval_idx = _concat_indices([s.H0_eval for s in splits])
        h1_eval_idx = _concat_indices([s.H1_eval for s in splits])

    online, history_rows, online_summary, _, _ = run_online_stopping(
        X_main,
        y,
        h0_train_idx=h0_train_idx,
        h1_train_idx=h1_train_idx,
        h0_calib_idx=h0_calib_idx,
        h0_monitor_idx=h0_monitor_idx,
        h1_monitor_idx=h1_monitor_idx,
        alpha=float(base_args.alpha),
        tie_mode=str(base_args.tie_mode),
        tau_guardrail=str(base_args.tau_guardrail),
        tau_guardrail_delta=float(base_args.tau_guardrail_delta),
        seed=int(seed),
        args=args,
    )

    tau = float(online_summary.get("final_tau", float("nan")))
    tpr, fpr, n0_eval, n1_eval = _evaluate_frozen_policy(
        online,
        X_main,
        h0_eval_idx=h0_eval_idx,
        h1_eval_idx=h1_eval_idx,
        tau=tau,
        tie_mode=str(base_args.tie_mode),
    )

    final_eval_idx = _concat_indices([h0_eval_idx, h1_eval_idx])
    monitor_idx = _concat_indices([h0_monitor_idx, h1_monitor_idx])
    train_idx = _concat_indices([h0_train_idx, h1_train_idx])
    calib_idx = _concat_indices([h0_calib_idx, h1_calib_idx])

    tau_slopes = _finite_values([abs(float(r.get("slope_tau", float("nan")))) for r in history_rows])
    tau_p95_abs_slope = float(np.percentile(tau_slopes, 95)) if tau_slopes.size else float("nan")
    samples_stream_available = int(online_summary.get("samples_stream_available", 0))
    samples_streamed = int(online_summary.get("samples_streamed", 0))
    normalized_steps = (
        float(samples_streamed / max(1, samples_stream_available))
        if samples_stream_available > 0
        else 0.0
    )

    success = bool(
        online is not None
        and n0_eval > 0
        and n1_eval > 0
        and np.isfinite(float(tpr))
        and np.isfinite(float(fpr))
    )

    return {
        "seed": int(seed),
        "alpha": float(base_args.alpha),
        "success": bool(success),
        "reason": str(online_summary.get("reason", "ok" if success else "unknown")),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "stopped": bool(online_summary.get("stopped", False)),
        "steps_until_stop": int(online_summary.get("updates", 0)),
        "normalized_steps_until_stop": float(np.clip(normalized_steps, 0.0, 1.0)),
        "samples_total_used": int(online_summary.get("samples_total_used", 0)),
        "samples_streamed": int(samples_streamed),
        "samples_init": int(online_summary.get("samples_init", 0)),
        "samples_stream_available": int(samples_stream_available),
        "tau": float(tau),
        "tau_p95_abs_slope": float(tau_p95_abs_slope),
        "history_len": int(online_summary.get("history_len", 0)),
        "n_eval_h0": int(n0_eval),
        "n_eval_h1": int(n1_eval),
        "n_monitor_h0_actual": int(h0_monitor_idx.size),
        "n_monitor_h1_actual": int(h1_monitor_idx.size),
        "monitor_final_eval_overlap": _overlap_count(final_eval_idx, monitor_idx),
        "train_final_eval_overlap": _overlap_count(final_eval_idx, train_idx),
        "calib_final_eval_overlap": _overlap_count(final_eval_idx, calib_idx),
        "split_stats": dict(split_stats),
        "filter_stats": dict(filter_stats),
        "monitor_stats": dict(monitor_stats),
    }


def aggregate_seed_metrics(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate trial metrics across split/evaluation seeds."""
    if not seed_results:
        return {
            "mean_tpr": 0.0,
            "mean_fpr": 1.0,
            "max_fpr": 1.0,
            "p95_fpr": 1.0,
            "fpr_violation_rate": 1.0,
            "mean_steps_until_stop": 0.0,
            "mean_samples_total_used": 0.0,
            "mean_samples_streamed": 0.0,
            "stop_success_rate": 0.0,
            "tau_mean": float("nan"),
            "tau_std": float("nan"),
            "tau_p95_abs_slope": float("nan"),
            "mean_normalized_steps_until_stop": 1.0,
            "monitor_final_eval_overlap": 0,
            "train_final_eval_overlap": 0,
            "calib_final_eval_overlap": 0,
            "n_seeds": 0,
        }

    fprs = _finite_values(r.get("fpr", float("nan")) for r in seed_results)
    tprs = _finite_values(r.get("tpr", float("nan")) for r in seed_results)
    taus = _finite_values(r.get("tau", float("nan")) for r in seed_results)
    tau_slope_parts = _finite_values(r.get("tau_p95_abs_slope", float("nan")) for r in seed_results)

    if fprs.size == 0:
        fprs = np.asarray([1.0], dtype=np.float64)
    if tprs.size == 0:
        tprs = np.asarray([0.0], dtype=np.float64)

    alpha = float(seed_results[0].get("alpha", float("nan")))
    if not np.isfinite(alpha):
        alpha = float("nan")

    return {
        "mean_tpr": float(np.mean(tprs)),
        "mean_fpr": float(np.mean(fprs)),
        "max_fpr": float(np.max(fprs)),
        "p95_fpr": float(np.percentile(fprs, 95)),
        "fpr_violation_rate": float(np.mean(fprs > alpha)) if np.isfinite(alpha) else float("nan"),
        "mean_steps_until_stop": _mean_or((r.get("steps_until_stop", 0) for r in seed_results), 0.0),
        "mean_samples_total_used": _mean_or((r.get("samples_total_used", 0) for r in seed_results), 0.0),
        "mean_samples_streamed": _mean_or((r.get("samples_streamed", 0) for r in seed_results), 0.0),
        "stop_success_rate": float(np.mean([bool(r.get("stopped", False)) for r in seed_results])),
        "tau_mean": float(np.mean(taus)) if taus.size else float("nan"),
        "tau_std": float(np.std(taus)) if taus.size else float("nan"),
        "tau_p95_abs_slope": float(np.percentile(tau_slope_parts, 95)) if tau_slope_parts.size else 0.0,
        "mean_normalized_steps_until_stop": _mean_or(
            (r.get("normalized_steps_until_stop", 1.0) for r in seed_results),
            1.0,
        ),
        "monitor_final_eval_overlap": int(sum(int(r.get("monitor_final_eval_overlap", 0)) for r in seed_results)),
        "train_final_eval_overlap": int(sum(int(r.get("train_final_eval_overlap", 0)) for r in seed_results)),
        "calib_final_eval_overlap": int(sum(int(r.get("calib_final_eval_overlap", 0)) for r in seed_results)),
        "n_seeds": int(len(seed_results)),
        "successful_seed_count": int(sum(bool(r.get("success", False)) for r in seed_results)),
    }


def score_aggregate_metrics(
    base_args: argparse.Namespace,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
) -> float:
    """Compute strict and relaxed NP-safety-first objectives from aggregate metrics."""
    alpha = float(base_args.alpha)
    stop_fpr_margin = float(params.get("stop_fpr_margin", 0.0))
    alpha_plus_margin = alpha + stop_fpr_margin
    p95_fpr = float(metrics.get("p95_fpr", float("inf")))
    if not np.isfinite(p95_fpr):
        p95_fpr = float("inf")

    raw_alpha_passed = bool(p95_fpr <= alpha)
    relaxed_margin_passed = bool(p95_fpr <= alpha_plus_margin)

    mean_tpr = float(metrics.get("mean_tpr", 0.0))
    normalized_steps = float(metrics.get("mean_normalized_steps_until_stop", 1.0))
    tau_instability = float(metrics.get("tau_p95_abs_slope", 0.0))
    stop_failure_rate = 1.0 - float(metrics.get("stop_success_rate", 0.0))

    base_score = (
        mean_tpr
        - float(base_args.lambda_cost) * normalized_steps
        - float(base_args.lambda_tau) * tau_instability
        - float(base_args.lambda_failure) * stop_failure_rate
    )
    strict_score = float(base_score) if raw_alpha_passed else -1e9
    relaxed_score = (
        float(base_score) - float(base_args.lambda_margin) * stop_fpr_margin
        if relaxed_margin_passed
        else -1e9
    )

    selection_mode = str(getattr(base_args, "selection_mode", "strict_np"))
    if selection_mode == "relaxed_margin":
        objective = relaxed_score
        final_constraint = "p95_fpr <= alpha + stop_fpr_margin"
        final_passed = relaxed_margin_passed
    else:
        objective = strict_score
        final_constraint = "p95_fpr <= alpha"
        final_passed = raw_alpha_passed

    metrics["selection_mode"] = selection_mode
    metrics["alpha"] = float(alpha)
    metrics["stop_fpr_margin"] = float(stop_fpr_margin)
    metrics["alpha_plus_margin"] = float(alpha_plus_margin)
    metrics["raw_alpha_constraint_passed"] = bool(raw_alpha_passed)
    metrics["relaxed_margin_constraint_passed"] = bool(relaxed_margin_passed)
    metrics["final_constraint_used"] = final_constraint
    metrics["constraint_passed"] = bool(final_passed)
    metrics["constraint_metric_value"] = float(p95_fpr)
    metrics["constraint_threshold"] = float(alpha if selection_mode == "strict_np" else alpha_plus_margin)
    metrics["strict_objective_value"] = float(strict_score)
    metrics["relaxed_objective_value"] = float(relaxed_score)
    return float(objective)


def objective_factory(base_args: argparse.Namespace) -> Callable[[Any], float]:
    """Build an Optuna objective that evaluates every trial across configured seeds."""
    base_args = _ensure_prepared_data(base_args)
    seeds = _parse_eval_seeds(getattr(base_args, "eval_seeds", None), int(base_args.seed))

    def objective(trial: Any) -> float:
        selection_mode = str(getattr(base_args, "selection_mode", "strict_np"))
        params = sample_stopping_params(
            trial,
            tune_stop_fpr_margin=(selection_mode == "relaxed_margin"),
            fixed_stop_fpr_margin=float(getattr(base_args, "fixed_stop_fpr_margin", 0.0)),
        )
        seed_results: List[Dict[str, Any]] = []
        try:
            for seed in seeds:
                result = run_single_seed_eval(base_args, params, int(seed))
                result["alpha"] = float(base_args.alpha)
                seed_results.append(result)
            metrics = aggregate_seed_metrics(seed_results)
            value = score_aggregate_metrics(base_args, params, metrics)
            metrics["objective_value"] = float(value)
        except Exception as exc:
            metrics = aggregate_seed_metrics(seed_results)
            metrics["objective_value"] = -1e9
            metrics["selection_mode"] = selection_mode
            metrics["alpha"] = float(base_args.alpha)
            metrics["stop_fpr_margin"] = float(params.get("stop_fpr_margin", 0.0))
            metrics["alpha_plus_margin"] = float(base_args.alpha) + float(params.get("stop_fpr_margin", 0.0))
            metrics["raw_alpha_constraint_passed"] = False
            metrics["relaxed_margin_constraint_passed"] = False
            metrics["final_constraint_used"] = (
                "p95_fpr <= alpha + stop_fpr_margin"
                if selection_mode == "relaxed_margin"
                else "p95_fpr <= alpha"
            )
            metrics["constraint_passed"] = False
            metrics["strict_objective_value"] = -1e9
            metrics["relaxed_objective_value"] = -1e9
            metrics["exception"] = str(exc)
            value = -1e9

        trial.set_user_attr("sampled_params", dict(params))
        trial.set_user_attr("metrics", _jsonable(metrics))
        trial.set_user_attr("seed_results", _jsonable(seed_results))
        return float(value)

    return objective


def _prepare_sqlite_storage_path(storage: Optional[str]) -> None:
    if not storage:
        return
    prefix = "sqlite:///"
    if not str(storage).startswith(prefix):
        return
    path = Path(str(storage)[len(prefix):])
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def _summary_lambda(summary: Dict[str, Any], name: str, default: float) -> float:
    values = summary.get("lambda_values", {})
    if isinstance(values, dict) and name in values:
        try:
            return float(values[name])
        except Exception:
            pass
    return float(default)


def _metric_float(metrics: Dict[str, Any], name: str, default: float) -> float:
    try:
        value = float(metrics.get(name, default))
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def _normalized_trial_metrics(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    summary: Dict[str, Any],
    *,
    selection_mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Fill strict/relaxed fields for new and older persisted trial metrics."""
    out = dict(metrics)
    alpha = _metric_float(out, "alpha", float(summary.get("alpha", 0.05)))
    stop_fpr_margin = _metric_float(
        out,
        "stop_fpr_margin",
        float(params.get("stop_fpr_margin", summary.get("fixed_stop_fpr_margin", 0.0))),
    )
    alpha_plus_margin = float(alpha + stop_fpr_margin)
    p95_fpr = _metric_float(out, "p95_fpr", float("inf"))
    raw_alpha_passed = bool(p95_fpr <= alpha)
    relaxed_margin_passed = bool(p95_fpr <= alpha_plus_margin)

    mean_tpr = _metric_float(out, "mean_tpr", 0.0)
    normalized_steps = _metric_float(out, "mean_normalized_steps_until_stop", 1.0)
    tau_instability = _metric_float(out, "tau_p95_abs_slope", 0.0)
    stop_failure_rate = 1.0 - _metric_float(out, "stop_success_rate", 0.0)
    base_score = (
        mean_tpr
        - _summary_lambda(summary, "lambda_cost", 0.05) * normalized_steps
        - _summary_lambda(summary, "lambda_tau", 0.1) * tau_instability
        - _summary_lambda(summary, "lambda_failure", 0.25) * stop_failure_rate
    )
    strict_score = float(base_score) if raw_alpha_passed else -1e9
    relaxed_score = (
        float(base_score) - _summary_lambda(summary, "lambda_margin", 1.0) * stop_fpr_margin
        if relaxed_margin_passed
        else -1e9
    )

    selection_mode = str(
        selection_mode_override
        if selection_mode_override is not None
        else summary.get("selection_mode", out.get("selection_mode", "strict_np"))
    )
    if selection_mode == "relaxed_margin":
        objective = relaxed_score
        final_constraint = "p95_fpr <= alpha + stop_fpr_margin"
        final_passed = relaxed_margin_passed
        threshold = alpha_plus_margin
    else:
        objective = strict_score
        final_constraint = "p95_fpr <= alpha"
        final_passed = raw_alpha_passed
        threshold = alpha

    out["selection_mode"] = selection_mode
    out["alpha"] = float(alpha)
    out["stop_fpr_margin"] = float(stop_fpr_margin)
    out["alpha_plus_margin"] = float(alpha_plus_margin)
    out["raw_alpha_constraint_passed"] = bool(raw_alpha_passed)
    out["relaxed_margin_constraint_passed"] = bool(relaxed_margin_passed)
    out["final_constraint_used"] = final_constraint
    out["constraint_passed"] = bool(final_passed)
    out["constraint_metric_value"] = float(p95_fpr)
    out["constraint_threshold"] = float(threshold)
    out["strict_objective_value"] = float(strict_score)
    out["relaxed_objective_value"] = float(relaxed_score)
    out["objective_value"] = float(objective)
    return out


def _trial_rows_from_study(study: Any, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        params = dict(trial.user_attrs.get("sampled_params", {}))
        metrics = _normalized_trial_metrics(
            params,
            dict(trial.user_attrs.get("metrics", {})),
            summary,
        )
        objective_value = metrics.get("objective_value", float("nan"))
        row: Dict[str, Any] = {
            "trial_number": int(trial.number),
            "objective_value": float(objective_value) if objective_value is not None else float("nan"),
        }
        for name in STOPPING_PARAM_NAMES:
            row[name] = params.get(name, "")
        for name in TRIAL_METRIC_NAMES:
            row[name] = metrics.get(name, "")
        rows.append(row)
    return rows


def _params_with_alpha(params: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if "alpha" in summary:
        out["alpha"] = float(summary["alpha"])
    for field in STOP_CONFIG_FIELD_NAMES:
        if field == "alpha":
            continue
        if field in params:
            out.setdefault(field, params[field])
    return out


def _empty_best_metrics() -> Dict[str, Any]:
    return {
        "constraint_passed": False,
        "raw_alpha_constraint_passed": False,
        "relaxed_margin_constraint_passed": False,
    }


def _best_trial_for(
    study: Any,
    summary: Dict[str, Any],
    *,
    pass_key: str,
    objective_key: str,
) -> Optional[Any]:
    candidates: List[tuple[float, int, Any]] = []
    for trial in study.trials:
        params = dict(trial.user_attrs.get("sampled_params", trial.params))
        metrics = _normalized_trial_metrics(
            params,
            dict(trial.user_attrs.get("metrics", {})),
            summary,
        )
        if not bool(metrics.get(pass_key, False)):
            continue
        try:
            value = float(metrics.get(objective_key, float("-inf")))
        except Exception:
            value = float("-inf")
        if not np.isfinite(value) or value <= -1e8:
            continue
        candidates.append((value, int(trial.number), trial))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return candidates[0][2]


def _trial_params_and_metrics(
    trial: Optional[Any],
    summary: Dict[str, Any],
    *,
    selection_mode_override: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if trial is None:
        return {}, _empty_best_metrics()
    params = dict(trial.user_attrs.get("sampled_params", trial.params))
    metrics = _normalized_trial_metrics(
        params,
        dict(trial.user_attrs.get("metrics", {})),
        summary,
        selection_mode_override=selection_mode_override,
    )
    metrics.setdefault("trial_number", int(trial.number))
    return _params_with_alpha(params, summary), metrics


def save_study_outputs(study: Any, out_dir: str | Path, summary: Dict[str, Any]) -> None:
    """Persist the required Optuna stopping-tuner artifacts."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    best_strict_trial = _best_trial_for(
        study,
        summary,
        pass_key="raw_alpha_constraint_passed",
        objective_key="strict_objective_value",
    )
    best_relaxed_trial = _best_trial_for(
        study,
        summary,
        pass_key="relaxed_margin_constraint_passed",
        objective_key="relaxed_objective_value",
    )
    best_strict_params, best_strict_metrics = _trial_params_and_metrics(
        best_strict_trial,
        summary,
        selection_mode_override="strict_np",
    )
    best_relaxed_params, best_relaxed_metrics = _trial_params_and_metrics(
        best_relaxed_trial,
        summary,
        selection_mode_override="relaxed_margin",
    )

    _save_json(out_path / "best_strict_params.json", best_strict_params)
    _save_json(out_path / "best_strict_trial_metrics.json", best_strict_metrics)
    _save_json(out_path / "best_relaxed_params.json", best_relaxed_params)
    _save_json(out_path / "best_relaxed_trial_metrics.json", best_relaxed_metrics)

    selection_mode = str(summary.get("selection_mode", "strict_np"))
    if selection_mode == "relaxed_margin" and best_relaxed_trial is not None:
        primary_trial = best_relaxed_trial
        primary_params = best_relaxed_params
        primary_metrics = best_relaxed_metrics
    elif selection_mode == "strict_np" and best_strict_trial is not None:
        primary_trial = best_strict_trial
        primary_params = best_strict_params
        primary_metrics = best_strict_metrics
    else:
        primary_trial = None
        primary_params = {}
        primary_metrics = _empty_best_metrics()

    _save_json(out_path / "best_params.json", primary_params)
    _save_json(out_path / "best_trial_metrics.json", primary_metrics)

    rows = _trial_rows_from_study(study, summary)
    fieldnames = ["trial_number", "objective_value"] + STOPPING_PARAM_NAMES + TRIAL_METRIC_NAMES
    with (out_path / "optuna_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k, "")) for k in fieldnames})

    summary_out = dict(summary)
    summary_out["best_params"] = primary_params
    summary_out["best_value"] = (
        float(primary_trial.value)
        if primary_trial is not None and primary_trial.value is not None
        else float(primary_metrics.get("objective_value", float("nan")))
    )
    summary_out["best_trial_metrics"] = primary_metrics
    summary_out["best_strict_trial_number"] = (
        int(best_strict_trial.number) if best_strict_trial is not None else None
    )
    summary_out["best_strict_tpr"] = best_strict_metrics.get("mean_tpr")
    summary_out["best_strict_p95_fpr"] = best_strict_metrics.get("p95_fpr")
    summary_out["best_strict_constraint_passed"] = bool(
        best_strict_metrics.get("raw_alpha_constraint_passed", False)
    )
    summary_out["best_relaxed_trial_number"] = (
        int(best_relaxed_trial.number) if best_relaxed_trial is not None else None
    )
    summary_out["best_relaxed_tpr"] = best_relaxed_metrics.get("mean_tpr")
    summary_out["best_relaxed_p95_fpr"] = best_relaxed_metrics.get("p95_fpr")
    summary_out["best_relaxed_stop_fpr_margin"] = best_relaxed_params.get("stop_fpr_margin")
    summary_out["best_relaxed_constraint_passed"] = bool(
        best_relaxed_metrics.get("relaxed_margin_constraint_passed", False)
    )
    _save_json(out_path / "study_summary.json", summary_out)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Offline Optuna tuner for NP-Calib OnlineStopper hyperparameters."
    )
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--region_key", type=str, required=True)
    ap.add_argument("--sem_bucket_k", type=int, default=64)
    ap.add_argument("--sem_bucket_source_key", type=str, default="global_cluster")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau_mode", type=str, default="global", choices=["global", "local", "shrink_local", "cluster_local"])
    ap.add_argument("--tie_mode", type=str, default="ge", choices=["ge", "gt"])
    ap.add_argument("--tau_guardrail", type=str, default="none", choices=["none", "clopper_pearson", "wilson", "beta_ucb"])
    ap.add_argument("--tau_guardrail_delta", type=float, default=0.01)
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_calib", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--min_h0_eval", type=int, default=20)
    ap.add_argument("--min_h1_eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_seeds", type=str, default=None)
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--out_dir", type=str, default="NeighborCache/outputs/optuna_stopping")
    ap.add_argument("--study_name", type=str, default="np_stopping")
    ap.add_argument("--storage", type=str, default=None)

    ap.add_argument("--normalize_data", action="store_true", default=False)
    ap.add_argument("--hadamard_preprocess", action="store_true", default=False)
    ap.add_argument("--use_abs_diff", action="store_true", default=False)
    ap.add_argument("--use_delta_vec", action="store_true", default=False)
    ap.add_argument(
        "--hadamard_anchor_strategy",
        type=str,
        default="centroid_nearest",
        choices=["centroid_nearest", "random"],
    )
    ap.add_argument("--filter_policy", type=str, default="none", choices=["none", "ambiguous_only"])
    ap.add_argument("--ambiguous_cos_min", type=float, default=0.7)
    ap.add_argument("--ambiguous_cos_max", type=float, default=0.9)

    ap.add_argument("--constraint_metric", type=str, default="p95_fpr", choices=["mean_fpr", "max_fpr", "p95_fpr"])
    ap.add_argument(
        "--selection_mode",
        type=str,
        default="strict_np",
        choices=["strict_np", "relaxed_margin"],
        help="strict_np selects only p95_fpr <= alpha trials; relaxed_margin is diagnostic.",
    )
    ap.add_argument(
        "--fixed_stop_fpr_margin",
        type=float,
        default=0.0,
        help="Fixed internal OnlineStopper FPR margin used when selection_mode=strict_np.",
    )
    ap.add_argument("--soft_penalty", action="store_true", default=False)
    ap.add_argument("--lambda_cost", type=float, default=0.05)
    ap.add_argument("--lambda_tau", type=float, default=0.1)
    ap.add_argument("--lambda_failure", type=float, default=0.25)
    ap.add_argument("--lambda_fpr", type=float, default=100.0)
    ap.add_argument("--lambda_margin", type=float, default=1.0)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if int(args.n_trials) <= 0:
        raise ValueError("--n_trials must be > 0")
    if float(args.fixed_stop_fpr_margin) < 0.0:
        raise ValueError("--fixed_stop_fpr_margin must be >= 0")
    if float(args.lambda_margin) < 0.0:
        raise ValueError("--lambda_margin must be >= 0")
    args.hard_constraint = True

    try:
        import optuna
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError(
            "Optuna stopping tuner requested but optuna is not installed. "
            "Install requirements.txt or `pip install optuna`."
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _prepare_sqlite_storage_path(args.storage)
    _ensure_prepared_data(args)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=bool(args.storage and args.study_name),
    )
    study.optimize(
        objective_factory(args),
        n_trials=int(args.n_trials),
        timeout=args.timeout,
        gc_after_trial=True,
    )

    seeds = _parse_eval_seeds(args.eval_seeds, int(args.seed))
    summary = {
        "alpha": float(args.alpha),
        "study_name": args.study_name,
        "storage": args.storage,
        "n_trials": int(args.n_trials),
        "seeds": seeds,
        "region_key": str(args.region_key),
        "region_key_resolved": str(getattr(args, "_region_key_resolved", args.region_key)),
        "tau_mode": str(args.tau_mode),
        "data_path": str(getattr(args, "_npz_path", args.data)),
        "features": str(getattr(args, "_feat_key", "")),
        "constraint_metric": str(args.constraint_metric),
        "selection_mode": str(args.selection_mode),
        "primary_constraint": (
            "p95_fpr <= alpha"
            if str(args.selection_mode) == "strict_np"
            else "p95_fpr <= alpha + stop_fpr_margin"
        ),
        "hard_constraint": bool(args.hard_constraint),
        "fixed_stop_fpr_margin": float(args.fixed_stop_fpr_margin),
        "lambda_values": {
            "lambda_cost": float(args.lambda_cost),
            "lambda_tau": float(args.lambda_tau),
            "lambda_failure": float(args.lambda_failure),
            "lambda_fpr": float(args.lambda_fpr),
            "lambda_margin": float(args.lambda_margin),
        },
    }
    save_study_outputs(study, out_dir, summary)
    print(f"[Done] Optuna stopping outputs at: {out_dir}")


if __name__ == "__main__":
    main()
