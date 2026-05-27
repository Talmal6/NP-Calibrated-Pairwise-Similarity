"""Grid sweep for online early stopping hyperparameters.

The sweep ranks configurations by held-out eval TPR/FPR after the online policy
has frozen. Monitor TPR/FPR are used only by the stopping mechanism.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .online_stopping_eval import (
    run_online_stopping,
    sample_monitor_from_global_eval,
    sample_monitor_from_region_splits,
)
from .optuna_stopping import (
    _apply_filter_policy_global,
    _apply_filter_policy_regions,
    _concat_indices,
    _ensure_prepared_data,
    _evaluate_frozen_policy,
    _jsonable,
    _monitor_count_with_holdout,
)
from .splits import split_global, split_indices_per_region_detailed


SWEEP_PARAM_NAMES = [
    "online_init_h0",
    "online_init_h1",
    "online_batch_size",
    "online_mem_cap",
    "online_update_mode",
    "online_hill_lr",
    "n_monitor_h0",
    "n_monitor_h1",
    "stop_check_every",
    "stop_window",
    "stop_patience",
    "stop_eps_tpr",
    "stop_eps_fpr",
    "stop_eps_tau",
    "stop_fpr_margin",
]

SUMMARY_FIELDNAMES = [
    "config_id",
    "phase",
    "n_trials",
    *SWEEP_PARAM_NAMES,
    "n_success",
    "n_errors",
    "mean_eval_tpr",
    "std_eval_tpr",
    "mean_eval_fpr",
    "std_eval_fpr",
    "mean_samples_total_used",
    "std_samples_total_used",
    "mean_samples_total_used_fraction",
    "mean_samples_streamed",
    "mean_updates",
    "stopped_rate",
    "valid_rate",
    "constrained_ok",
    "selection_score",
    "mean_samples_total_possible",
    "max_possible_training_samples",
    "most_common_reason",
    "error",
]

PER_TRIAL_FIELDNAMES = [
    "config_id",
    "phase",
    "trial_index",
    "seed",
    *SWEEP_PARAM_NAMES,
    "success",
    "error",
    "reason",
    "stopped",
    "eval_tpr",
    "eval_fpr",
    "valid",
    "updates",
    "final_tau",
    "history_len",
    "samples_init",
    "samples_streamed",
    "samples_stream_available",
    "samples_total_used",
    "samples_total_possible",
    "samples_total_used_fraction",
    "n_eval_h0",
    "n_eval_h1",
    "n_monitor_h0_actual",
    "n_monitor_h1_actual",
]

HISTORY_FIELDNAMES = [
    "config_id",
    "phase",
    "trial_index",
    "seed",
    "checkpoint",
    "tpr_monitor",
    "fpr_monitor",
    "tau",
    "slope_tpr",
    "slope_fpr",
    "slope_tau",
    "condition_passed",
    "stop_streak",
    "should_stop",
]

DEFAULT_SWEEP_CONFIG: Dict[str, Any] = {
    "online_init_h0": 100,
    "online_init_h1": 100,
    "online_batch_size": 50,
    "online_mem_cap": 1200,
    "online_update_mode": "reservoir",
    "online_hill_lr": 0.05,
    "n_monitor_h0": 300,
    "n_monitor_h1": 300,
    "stop_check_every": 2,
    "stop_window": 4,
    "stop_patience": 2,
    "stop_eps_tpr": 0.005,
    "stop_eps_fpr": 0.003,
    "stop_eps_tau": 0.01,
    "stop_fpr_margin": 0.005,
}

COARSE_GRID = {
    "online_init": [50, 100, 200],
    "online_batch_size": [25, 50, 100],
    "stop_check_every": [1, 2, 4],
    "stop_window": [3, 4, 6],
    "stop_patience": [1, 2, 3],
}

REFINE_GRID = {
    "stop_eps_tpr": [0.003, 0.005, 0.01],
    "stop_eps_fpr": [0.002, 0.003, 0.005],
    "stop_fpr_margin": [0.003, 0.005, 0.01],
    "stop_eps_tau": [0.005, 0.01, 0.02],
}


def _ordered(values: Iterable[Any], preferred: Any) -> List[Any]:
    vals = list(values)
    out = [v for v in vals if v == preferred]
    out.extend(v for v in vals if v != preferred)
    return out


def _config_key(config: Dict[str, Any]) -> str:
    return json.dumps({k: config.get(k) for k in SWEEP_PARAM_NAMES}, sort_keys=True, default=str)


def _dedupe_and_id(configs: Iterable[Dict[str, Any]], phase: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for config in configs:
        c = {k: config[k] for k in SWEEP_PARAM_NAMES if k in config}
        key = _config_key(c)
        if key in seen:
            continue
        seen.add(key)
        c["config_id"] = f"{phase}_{len(out):04d}"
        c["phase"] = phase
        out.append(c)
    return out


def _base_fixed_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    fixed = dict(DEFAULT_SWEEP_CONFIG)
    fixed.update(
        {
            "online_mem_cap": int(args.online_mem_cap),
            "online_update_mode": str(args.online_update_mode),
            "online_hill_lr": float(args.online_hill_lr),
            "n_monitor_h0": int(args.n_monitor_h0),
            "n_monitor_h1": int(args.n_monitor_h1),
            "stop_eps_tpr": float(args.stop_eps_tpr),
            "stop_eps_fpr": float(args.stop_eps_fpr),
            "stop_eps_tau": float(args.stop_eps_tau),
            "stop_fpr_margin": float(args.stop_fpr_margin),
        }
    )
    return fixed


def build_smoke_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    default = _base_fixed_from_args(args)
    configs = [
        dict(default),
        {
            **default,
            "stop_check_every": 1,
            "stop_window": 3,
            "stop_patience": 1,
            "online_batch_size": 25,
        },
        {
            **default,
            "online_init_h0": 50,
            "online_init_h1": 50,
            "online_batch_size": 100,
            "stop_check_every": 4,
            "stop_window": 6,
            "stop_patience": 3,
        },
    ]
    return _dedupe_and_id(configs, "smoke")


def build_coarse_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    fixed = _base_fixed_from_args(args)
    init_values = _ordered(COARSE_GRID["online_init"], fixed["online_init_h0"])
    batch_values = _ordered(COARSE_GRID["online_batch_size"], fixed["online_batch_size"])
    check_values = _ordered(COARSE_GRID["stop_check_every"], fixed["stop_check_every"])
    window_values = _ordered(COARSE_GRID["stop_window"], fixed["stop_window"])
    patience_values = _ordered(COARSE_GRID["stop_patience"], fixed["stop_patience"])

    if args.decouple_init_grid:
        init_pairs = list(itertools.product(init_values, init_values))
    else:
        init_pairs = [(v, v) for v in init_values]

    configs: List[Dict[str, Any]] = []
    for (init_h0, init_h1), batch_size, check_every, window, patience in itertools.product(
        init_pairs,
        batch_values,
        check_values,
        window_values,
        patience_values,
    ):
        config = dict(fixed)
        config.update(
            {
                "online_init_h0": int(init_h0),
                "online_init_h1": int(init_h1),
                "online_batch_size": int(batch_size),
                "stop_check_every": int(check_every),
                "stop_window": int(window),
                "stop_patience": int(patience),
            }
        )
        configs.append(config)
    return _dedupe_and_id(configs, "coarse")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _coerce_config(row: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(DEFAULT_SWEEP_CONFIG)
    int_fields = {
        "online_init_h0",
        "online_init_h1",
        "online_batch_size",
        "online_mem_cap",
        "n_monitor_h0",
        "n_monitor_h1",
        "stop_check_every",
        "stop_window",
        "stop_patience",
    }
    float_fields = {
        "online_hill_lr",
        "stop_eps_tpr",
        "stop_eps_fpr",
        "stop_eps_tau",
        "stop_fpr_margin",
    }
    for name in int_fields:
        if name in row and str(row[name]).strip() != "":
            config[name] = _as_int(row[name], int(config[name]))
    for name in float_fields:
        if name in row and str(row[name]).strip() != "":
            config[name] = _as_float(row[name], float(config[name]))
    if row.get("online_update_mode"):
        config["online_update_mode"] = str(row["online_update_mode"])
    return config


def _apply_loaded_config_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(config)
    out.update(
        {
            "online_mem_cap": int(args.online_mem_cap),
            "online_update_mode": str(args.online_update_mode),
            "online_hill_lr": float(args.online_hill_lr),
            "n_monitor_h0": int(args.n_monitor_h0),
            "n_monitor_h1": int(args.n_monitor_h1),
        }
    )
    return out


def _summary_sort_key(row: Dict[str, Any]) -> tuple[int, float, float]:
    constrained = 1 if _as_bool(row.get("constrained_ok", False)) else 0
    tpr = _as_float(row.get("mean_eval_tpr", float("-inf")), float("-inf"))
    samples = _as_float(row.get("mean_samples_total_used", float("inf")), float("inf"))
    return (-constrained, -tpr, samples)


def _resolve_summary_csv(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        best = p / "best_configs.csv"
        if best.exists():
            return best
        return p / "sweep_summary.csv"
    return p


def _load_ranked_summary_rows(path: str | Path) -> List[Dict[str, Any]]:
    csv_path = _resolve_summary_csv(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=_summary_sort_key)
    return rows


def build_refine_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.refine_from:
        raise ValueError("--refine_from is required for --phase refine")
    rows = _load_ranked_summary_rows(args.refine_from)
    top_rows = rows[: max(1, int(args.top_k))]
    configs: List[Dict[str, Any]] = []
    for row in top_rows:
        base = _apply_loaded_config_overrides(_coerce_config(row), args)
        tau_values = (
            REFINE_GRID["stop_eps_tau"]
            if bool(args.refine_stop_eps_tau)
            else [float(base["stop_eps_tau"])]
        )
        for eps_tpr, eps_fpr, fpr_margin, eps_tau in itertools.product(
            REFINE_GRID["stop_eps_tpr"],
            REFINE_GRID["stop_eps_fpr"],
            REFINE_GRID["stop_fpr_margin"],
            tau_values,
        ):
            config = dict(base)
            config.update(
                {
                    "stop_eps_tpr": float(eps_tpr),
                    "stop_eps_fpr": float(eps_fpr),
                    "stop_fpr_margin": float(fpr_margin),
                    "stop_eps_tau": float(eps_tau),
                }
            )
            configs.append(config)
    return _dedupe_and_id(configs, "refine")


def build_final_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.refine_from:
        raise ValueError("--refine_from is required for --phase final")
    rows = _load_ranked_summary_rows(args.refine_from)
    configs = [
        _apply_loaded_config_overrides(_coerce_config(row), args)
        for row in rows[: max(1, int(args.top_k))]
    ]
    return _dedupe_and_id(configs, "final")


def build_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.phase == "smoke":
        configs = build_smoke_configs(args)
    elif args.phase == "coarse":
        configs = build_coarse_configs(args)
    elif args.phase == "refine":
        configs = build_refine_configs(args)
    elif args.phase == "final":
        configs = build_final_configs(args)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

    if args.max_configs is not None:
        configs = configs[: max(0, int(args.max_configs))]
    return configs


def build_args_with_config(base_args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    args = copy.copy(base_args)
    for key in SWEEP_PARAM_NAMES:
        setattr(args, key, config[key])
    setattr(args, "enable_online_stopping", True)
    return args


def trial_seeds(args: argparse.Namespace) -> List[int]:
    if args.eval_seeds:
        seeds: List[int] = []
        for part in str(args.eval_seeds).split(","):
            text = part.strip()
            if text:
                seeds.append(int(text))
        return seeds
    return [int(args.seed) + int(i) * int(args.seed_stride) for i in range(int(args.n_trials))]


def _reason_mode(rows: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", ""))
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def run_single_trial_with_history(
    base_args: argparse.Namespace,
    config: Dict[str, Any],
    *,
    seed: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base_args = _ensure_prepared_data(base_args)
    args = build_args_with_config(base_args, config)

    X_main = getattr(base_args, "_X_main")
    y = getattr(base_args, "_y")
    region_id = getattr(base_args, "_region_id")

    if str(base_args.tau_mode) == "global":
        gs, _split_stats = split_global(
            y=y,
            n_train_cap=int(base_args.n_train),
            n_calib_cap=int(base_args.n_calib),
            n_eval_cap=int(base_args.n_eval),
            seed=int(seed),
        )
        gs, _filter_stats = _apply_filter_policy_global(base_args, gs)
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
        gs, h0_monitor_idx, h1_monitor_idx, _monitor_stats = sample_monitor_from_global_eval(
            gs,
            n_monitor_h0=n_monitor_h0,
            n_monitor_h1=n_monitor_h1,
            seed=int(seed),
        )
        h0_train_idx = np.asarray(gs.H0_train, dtype=np.int64)
        h1_train_idx = np.asarray(gs.H1_train, dtype=np.int64)
        h0_calib_idx = np.asarray(gs.H0_calib, dtype=np.int64)
        h0_eval_idx = np.asarray(gs.H0_eval, dtype=np.int64)
        h1_eval_idx = np.asarray(gs.H1_eval, dtype=np.int64)
    else:
        splits, _split_stats, _region_status = split_indices_per_region_detailed(
            region_id=region_id,
            y=y,
            n_train_cap=int(base_args.n_train),
            n_calib_cap=int(base_args.n_calib),
            n_eval_cap=int(base_args.n_eval),
            seed=int(seed),
            min_h0_eval=int(base_args.min_h0_eval),
            min_h1_eval=int(base_args.min_h1_eval),
        )
        splits, _filter_stats = _apply_filter_policy_regions(base_args, splits)
        if not splits:
            return {
                "success": False,
                "error": "",
                "reason": "empty_region_splits",
                "stopped": False,
                "eval_tpr": float("nan"),
                "eval_fpr": float("nan"),
                "valid": False,
                "updates": 0,
                "final_tau": float("nan"),
                "history_len": 0,
                "samples_init": 0,
                "samples_streamed": 0,
                "samples_stream_available": 0,
                "samples_total_used": 0,
                "samples_total_possible": 0,
                "samples_total_used_fraction": float("nan"),
                "n_eval_h0": 0,
                "n_eval_h1": 0,
                "n_monitor_h0_actual": 0,
                "n_monitor_h1_actual": 0,
            }, []

        splits, h0_monitor_idx, h1_monitor_idx, _monitor_stats = sample_monitor_from_region_splits(
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
    eval_tpr, eval_fpr, n0_eval, n1_eval = _evaluate_frozen_policy(
        online,
        X_main,
        h0_eval_idx=h0_eval_idx,
        h1_eval_idx=h1_eval_idx,
        tau=tau,
        tie_mode=str(base_args.tie_mode),
    )
    samples_init = int(online_summary.get("samples_init", 0))
    samples_stream_available = int(online_summary.get("samples_stream_available", 0))
    samples_total_possible = int(samples_init + samples_stream_available)
    samples_total_used = int(online_summary.get("samples_total_used", 0))
    used_fraction = (
        float(samples_total_used / samples_total_possible)
        if samples_total_possible > 0
        else float("nan")
    )
    success = bool(
        online is not None
        and n0_eval > 0
        and n1_eval > 0
        and np.isfinite(float(eval_tpr))
        and np.isfinite(float(eval_fpr))
    )
    valid = bool(success and float(eval_fpr) <= float(base_args.alpha))

    return {
        "success": bool(success),
        "error": "",
        "reason": str(online_summary.get("reason", "ok" if success else "unknown")),
        "stopped": bool(online_summary.get("stopped", False)),
        "eval_tpr": float(eval_tpr),
        "eval_fpr": float(eval_fpr),
        "valid": bool(valid),
        "updates": int(online_summary.get("updates", 0)),
        "final_tau": float(tau),
        "history_len": int(online_summary.get("history_len", 0)),
        "samples_init": int(samples_init),
        "samples_streamed": int(online_summary.get("samples_streamed", 0)),
        "samples_stream_available": int(samples_stream_available),
        "samples_total_used": int(samples_total_used),
        "samples_total_possible": int(samples_total_possible),
        "samples_total_used_fraction": float(used_fraction),
        "n_eval_h0": int(n0_eval),
        "n_eval_h1": int(n1_eval),
        "n_monitor_h0_actual": int(h0_monitor_idx.size),
        "n_monitor_h1_actual": int(h1_monitor_idx.size),
    }, history_rows


def _finite_values(rows: Iterable[Dict[str, Any]], key: str) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = _finite_values(rows, key)
    return float(np.mean(vals)) if vals.size else float("nan")


def _std(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = _finite_values(rows, key)
    if vals.size <= 1:
        return 0.0 if vals.size == 1 else float("nan")
    return float(np.std(vals, ddof=1))


def aggregate_config(
    config: Dict[str, Any],
    trial_rows: List[Dict[str, Any]],
    *,
    alpha: float,
) -> Dict[str, Any]:
    n_trials = int(len(trial_rows))
    n_success = int(sum(bool(r.get("success", False)) for r in trial_rows))
    n_errors = int(sum(bool(r.get("error", "")) for r in trial_rows))
    mean_eval_tpr = _mean(trial_rows, "eval_tpr")
    mean_eval_fpr = _mean(trial_rows, "eval_fpr")
    mean_samples = _mean(trial_rows, "samples_total_used")
    mean_possible = _mean(trial_rows, "samples_total_possible")
    max_possible_values = _finite_values(trial_rows, "samples_total_possible")
    max_possible = float(np.max(max_possible_values)) if max_possible_values.size else float("nan")
    denom = max_possible if np.isfinite(max_possible) and max_possible > 0 else mean_possible
    mean_fraction = (
        float(mean_samples / denom)
        if np.isfinite(mean_samples) and np.isfinite(denom) and denom > 0
        else float("nan")
    )

    stopped_rate = (
        float(np.mean([bool(r.get("stopped", False)) for r in trial_rows]))
        if trial_rows
        else float("nan")
    )
    valid_rate = (
        float(np.mean([bool(r.get("valid", False)) for r in trial_rows]))
        if trial_rows
        else 0.0
    )
    constrained_ok = bool(np.isfinite(mean_eval_fpr) and mean_eval_fpr <= float(alpha))

    if np.isfinite(mean_eval_tpr) and np.isfinite(mean_eval_fpr):
        selection_score = (
            float(mean_eval_tpr)
            - 10.0 * max(0.0, float(mean_eval_fpr) - float(alpha))
            - 0.05 * (float(mean_fraction) if np.isfinite(mean_fraction) else 1.0)
        )
    else:
        selection_score = -1e9

    error_text = ""
    for row in trial_rows:
        if row.get("error"):
            error_text = str(row.get("error"))
            break

    out: Dict[str, Any] = {
        "config_id": config["config_id"],
        "phase": config["phase"],
        "n_trials": n_trials,
        **{k: config[k] for k in SWEEP_PARAM_NAMES},
        "n_success": n_success,
        "n_errors": n_errors,
        "mean_eval_tpr": float(mean_eval_tpr),
        "std_eval_tpr": float(_std(trial_rows, "eval_tpr")),
        "mean_eval_fpr": float(mean_eval_fpr),
        "std_eval_fpr": float(_std(trial_rows, "eval_fpr")),
        "mean_samples_total_used": float(mean_samples),
        "std_samples_total_used": float(_std(trial_rows, "samples_total_used")),
        "mean_samples_total_used_fraction": float(mean_fraction),
        "mean_samples_streamed": float(_mean(trial_rows, "samples_streamed")),
        "mean_updates": float(_mean(trial_rows, "updates")),
        "stopped_rate": float(stopped_rate),
        "valid_rate": float(valid_rate),
        "constrained_ok": bool(constrained_ok),
        "selection_score": float(selection_score),
        "mean_samples_total_possible": float(mean_possible),
        "max_possible_training_samples": float(max_possible),
        "most_common_reason": _reason_mode(trial_rows),
        "error": error_text,
    }
    return out


def _csvable(value: Any) -> Any:
    value = _jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csvable(row.get(name, "")) for name in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def sort_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=_summary_sort_key)


def write_outputs(
    out_dir: Path,
    *,
    summary_rows: List[Dict[str, Any]],
    per_trial_rows: List[Dict[str, Any]],
    history_rows: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    write_history: bool,
) -> None:
    sorted_rows = sort_summary_rows(summary_rows)
    write_csv_rows(out_dir / "sweep_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    write_json(out_dir / "sweep_summary.json", {"metadata": metadata, "rows": summary_rows})
    write_csv_rows(out_dir / "best_configs.csv", sorted_rows, SUMMARY_FIELDNAMES)
    write_csv_rows(out_dir / "per_trial_results.csv", per_trial_rows, PER_TRIAL_FIELDNAMES)
    if write_history:
        write_csv_rows(out_dir / "per_checkpoint_history.csv", history_rows, HISTORY_FIELDNAMES)


def print_config_start(index: int, total: int, config: Dict[str, Any], n_trials: int) -> None:
    hp = " ".join(f"{k}={config[k]}" for k in SWEEP_PARAM_NAMES)
    print(
        f"[config {index}/{total}] id={config['config_id']} n_trials={n_trials} {hp}",
        flush=True,
    )


def print_trial_result(row: Dict[str, Any]) -> None:
    print(
        "[trial] "
        f"config={row['config_id']} seed={row['seed']} "
        f"eval_tpr={_as_float(row.get('eval_tpr'), float('nan')):.6f} "
        f"eval_fpr={_as_float(row.get('eval_fpr'), float('nan')):.6f} "
        f"valid={row.get('valid')} stopped={row.get('stopped')} "
        f"reason={row.get('reason')} samples={row.get('samples_total_used')} "
        f"updates={row.get('updates')} tau={_as_float(row.get('final_tau'), float('nan')):.6g}",
        flush=True,
    )


def run_config(
    base_args: argparse.Namespace,
    config: Dict[str, Any],
    seeds: List[int],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_trial_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    for trial_index, seed in enumerate(seeds):
        trial_prefix = {
            "config_id": config["config_id"],
            "phase": config["phase"],
            "trial_index": int(trial_index),
            "seed": int(seed),
            **{k: config[k] for k in SWEEP_PARAM_NAMES},
        }
        try:
            result, history = run_single_trial_with_history(base_args, config, seed=int(seed))
            row = {**trial_prefix, **result}
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            row = {
                **trial_prefix,
                "success": False,
                "error": err,
                "reason": "exception",
                "stopped": False,
                "eval_tpr": float("nan"),
                "eval_fpr": float("nan"),
                "valid": False,
                "updates": 0,
                "final_tau": float("nan"),
                "history_len": 0,
                "samples_init": 0,
                "samples_streamed": 0,
                "samples_stream_available": 0,
                "samples_total_used": 0,
                "samples_total_possible": 0,
                "samples_total_used_fraction": float("nan"),
                "n_eval_h0": 0,
                "n_eval_h1": 0,
                "n_monitor_h0_actual": 0,
                "n_monitor_h1_actual": 0,
            }
            history = []
            print(f"[error] config={config['config_id']} seed={seed} {err}", flush=True)
            print(traceback.format_exc(limit=5), flush=True)
        per_trial_rows.append(row)
        for h in history:
            checkpoint_rows.append(
                {
                    "config_id": config["config_id"],
                    "phase": config["phase"],
                    "trial_index": int(trial_index),
                    "seed": int(seed),
                    **h,
                }
            )
        print_trial_result(row)

    summary = aggregate_config(config, per_trial_rows, alpha=float(base_args.alpha))
    print(
        "[summary] "
        f"config={config['config_id']} constrained_ok={summary['constrained_ok']} "
        f"valid_rate={summary['valid_rate']:.3f} "
        f"mean_tpr={summary['mean_eval_tpr']:.6f} "
        f"mean_fpr={summary['mean_eval_fpr']:.6f} "
        f"mean_samples={summary['mean_samples_total_used']:.1f} "
        f"score={summary['selection_score']:.6f}",
        flush=True,
    )
    return summary, per_trial_rows, checkpoint_rows


def make_out_dir(args: argparse.Namespace) -> Path:
    base = Path(args.out_dir)
    if args.no_timestamp:
        out_dir = base
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base / f"{stamp}_{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def report_final(rows: List[Dict[str, Any]], *, alpha: float, tpr_tolerance: float) -> None:
    if not rows:
        print("[report] no completed configurations", flush=True)
        return
    ranked = sort_summary_rows(rows)
    constrained = [r for r in ranked if _as_bool(r.get("constrained_ok", False))]
    if not constrained:
        print(f"[report] no constrained configs with mean_eval_fpr <= alpha ({alpha})", flush=True)
        return

    best = constrained[0]
    best_tpr = _as_float(best.get("mean_eval_tpr"), 0.0)
    cutoff = best_tpr - float(tpr_tolerance)
    efficient_candidates = [
        r for r in constrained if _as_float(r.get("mean_eval_tpr"), -1.0) >= cutoff
    ]
    efficient = min(
        efficient_candidates,
        key=lambda r: _as_float(r.get("mean_samples_total_used"), float("inf")),
    )
    violators = [
        r
        for r in rows
        if np.isfinite(_as_float(r.get("mean_eval_fpr"), float("nan")))
        and _as_float(r.get("mean_eval_fpr"), 0.0) > float(alpha)
    ]
    violators.sort(
        key=lambda r: (
            _as_float(r.get("mean_samples_total_used"), float("inf")),
            -_as_float(r.get("mean_eval_fpr"), 0.0),
        )
    )

    full = _as_float(best.get("max_possible_training_samples"), float("nan"))
    used = _as_float(best.get("mean_samples_total_used"), float("nan"))
    saving = 1.0 - used / full if np.isfinite(full) and full > 0 and np.isfinite(used) else float("nan")

    print(
        "[report] best_constrained "
        f"id={best['config_id']} tpr={best['mean_eval_tpr']:.6f} "
        f"fpr={best['mean_eval_fpr']:.6f} valid_rate={best['valid_rate']:.3f} "
        f"samples={best['mean_samples_total_used']:.1f}",
        flush=True,
    )
    print(
        "[report] sample_efficient_within_tpr_tol "
        f"id={efficient['config_id']} tpr={efficient['mean_eval_tpr']:.6f} "
        f"fpr={efficient['mean_eval_fpr']:.6f} samples={efficient['mean_samples_total_used']:.1f}",
        flush=True,
    )
    print(
        "[report] early_stop_savings "
        f"best_used={used:.1f} full_stream={full:.1f} saving_fraction={saving:.3f}",
        flush=True,
    )
    if violators:
        preview = ", ".join(
            f"{r['config_id']}:fpr={_as_float(r.get('mean_eval_fpr'), 0.0):.4f},"
            f"samples={_as_float(r.get('mean_samples_total_used'), 0.0):.1f},"
            f"patience={r.get('stop_patience')}"
            for r in violators[:5]
        )
        print(f"[report] fpr_violating_configs count={len(violators)} {preview}", flush=True)
    else:
        print("[report] no configs violated mean eval FPR", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Systematic grid sweep for NeighborCache online early stopping."
    )
    ap.add_argument("--phase", choices=["smoke", "coarse", "refine", "final"], default="coarse")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--region_key", type=str, required=True)
    ap.add_argument("--sem_bucket_k", type=int, default=64)
    ap.add_argument("--sem_bucket_source_key", type=str, default="global_cluster")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau_mode", type=str, default="global", choices=["global", "local", "shrink_local", "cluster_local"])
    ap.add_argument("--tie_mode", type=str, default="ge", choices=["ge", "gt"])
    ap.add_argument("--tau_guardrail", type=str, default="none", choices=["none", "clopper_pearson", "wilson", "beta_ucb"])
    ap.add_argument("--tau_guardrail_delta", type=float, default=0.01)
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_calib", type=int, default=1200)
    ap.add_argument("--n_eval", type=int, default=1200)
    ap.add_argument("--min_h0_eval", type=int, default=20)
    ap.add_argument("--min_h1_eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_stride", type=int, default=1)
    ap.add_argument("--eval_seeds", type=str, default=None)
    ap.add_argument("--n_trials", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="NeighborCache/outputs/online_stop_sweep")
    ap.add_argument("--no_timestamp", action="store_true", default=False)
    ap.add_argument("--max_configs", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--refine_from", type=str, default=None)
    ap.add_argument("--refine_stop_eps_tau", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decouple_init_grid", action="store_true", default=False)
    ap.add_argument("--no_history", action="store_true", default=False)
    ap.add_argument("--tpr_tolerance", type=float, default=0.01)

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

    ap.add_argument("--online_mem_cap", type=int, default=1200)
    ap.add_argument("--online_update_mode", type=str, default="reservoir", choices=["reservoir", "refit", "hill_climb"])
    ap.add_argument("--online_hill_lr", type=float, default=0.05)
    ap.add_argument("--n_monitor_h0", type=int, default=300)
    ap.add_argument("--n_monitor_h1", type=int, default=300)
    ap.add_argument("--stop_eps_tpr", type=float, default=0.005)
    ap.add_argument("--stop_eps_fpr", type=float, default=0.003)
    ap.add_argument("--stop_eps_tau", type=float, default=0.01)
    ap.add_argument("--stop_fpr_margin", type=float, default=0.005)
    return ap.parse_args(argv)


def _phase_default_trials(phase: str) -> int:
    if phase == "smoke":
        return 1
    if phase == "coarse":
        return 3
    if phase == "refine":
        return 5
    if phase == "final":
        return 10
    return 3


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.n_trials is None:
        args.n_trials = _phase_default_trials(str(args.phase))
    if int(args.n_trials) <= 0:
        raise ValueError("--n_trials must be > 0")
    if int(args.seed_stride) <= 0:
        raise ValueError("--seed_stride must be > 0")

    out_dir = make_out_dir(args)
    seeds = trial_seeds(args)
    args.n_trials = len(seeds)
    configs = build_configs(args)
    if not configs:
        raise ValueError("No sweep configurations were generated")

    print(f"[start] phase={args.phase} configs={len(configs)} seeds={seeds} out_dir={out_dir}", flush=True)
    if args.phase == "coarse" and not args.decouple_init_grid:
        print("[info] online_init_h0 and online_init_h1 are coupled by default; use --decouple_init_grid for full cross-product.", flush=True)

    _ensure_prepared_data(args)
    metadata = {
        "phase": str(args.phase),
        "alpha": float(args.alpha),
        "n_trials": int(args.n_trials),
        "seeds": seeds,
        "data_path": str(getattr(args, "_npz_path", args.data)),
        "region_key": str(args.region_key),
        "region_key_resolved": str(getattr(args, "_region_key_resolved", args.region_key)),
        "tau_mode": str(args.tau_mode),
        "features": str(getattr(args, "_feat_key", "")),
        "selection_rule": "constrained_ok desc, mean_eval_tpr desc, mean_samples_total_used asc",
        "selection_score": "mean_eval_tpr - 10.0 * max(0, mean_eval_fpr - alpha) - 0.05 * mean_samples_total_used_fraction",
        "monitor_note": "monitor TPR/FPR/tau are used only for stopping; ranking uses held-out eval TPR/FPR",
        "config_count": int(len(configs)),
        "decouple_init_grid": bool(args.decouple_init_grid),
    }
    write_json(out_dir / "run_metadata.json", metadata)

    summary_rows: List[Dict[str, Any]] = []
    per_trial_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    for idx, config in enumerate(configs, start=1):
        print_config_start(idx, len(configs), config, len(seeds))
        summary, trials, histories = run_config(args, config, seeds)
        summary_rows.append(summary)
        per_trial_rows.extend(trials)
        history_rows.extend(histories)
        write_outputs(
            out_dir,
            summary_rows=summary_rows,
            per_trial_rows=per_trial_rows,
            history_rows=history_rows,
            metadata=metadata,
            write_history=not bool(args.no_history),
        )

    report_final(summary_rows, alpha=float(args.alpha), tpr_tolerance=float(args.tpr_tolerance))
    print(f"[done] outputs at: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
