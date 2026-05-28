from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from .online_mined_ours import accept_scores, load_pairs_dir, select_tau
from .risk_aware_stopping import (
    _counts,
    _dataset_from_config,
    _float,
    _fmt,
    _fmt_int,
    _seed_from_config,
    _write_csv,
)
from .threshold_refresh import _score_bundle


DEFAULT_DATASETS = ["SemCacheLMArena", "SemCacheSearchQueries"]
DEFAULT_METHODS = ["ours_weighted_ensemble", "ours_whitened_hadamard"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_NUM_BLOCKS = [5, 10, 20, 30]
DEFAULT_ROLLING_K_H0 = [1000, 2000, 5000, 10000]
DEFAULT_ALPHA = 0.05


BLOCK_FIELDS = [
    "iteration",
    "repetition_mode",
    "seed",
    "dataset",
    "method",
    "policy",
    "alpha",
    "alpha_cal_multiplier",
    "alpha_cal",
    "num_blocks",
    "block_index",
    "block_start",
    "block_end",
    "rolling_K_H0",
    "tau",
    "calib_memory_size_total",
    "calib_memory_H0",
    "calib_memory_H1",
    "block_n",
    "block_H0",
    "block_H1",
    "block_FP",
    "block_TN",
    "block_FPR",
    "block_TP",
    "block_FN",
    "block_TPR",
    "block_hit",
    "block_precision",
    "block_FP_over_n",
    "cumulative_FP",
    "cumulative_TN",
    "cumulative_FPR",
    "cumulative_TP",
    "cumulative_FN",
    "cumulative_TPR",
    "cumulative_hit",
    "cumulative_precision",
    "cumulative_FP_over_n",
    "safe_block",
    "safe_cumulative_so_far",
    "calib_H0_mean",
    "calib_H0_p95",
    "block_H0_mean",
    "block_H0_p95",
    "block_minus_calib_H0_p95",
]

ITERATION_FIELDS = [
    "iteration",
    "repetition_mode",
    "seed",
    "dataset",
    "method",
    "policy",
    "alpha",
    "alpha_cal_multiplier",
    "alpha_cal",
    "num_blocks",
    "rolling_K_H0",
    "final_cumulative_FPR",
    "final_cumulative_TPR",
    "final_cumulative_hit",
    "final_cumulative_precision",
    "final_cumulative_FP_over_n",
    "mean_block_FPR",
    "max_block_FPR",
    "p95_block_FPR",
    "std_block_FPR",
    "fraction_safe_blocks",
    "all_blocks_safe",
    "final_safe",
    "first_unsafe_block",
    "unsafe_blocks",
    "final_tau",
    "tau_min",
    "tau_max",
    "tau_drift",
    "final_calib_memory_H0",
    "final_calib_memory_H1",
]

AGGREGATE_FIELDS = [
    "repetition_mode",
    "dataset",
    "method",
    "policy",
    "alpha",
    "alpha_cal_multiplier",
    "num_blocks",
    "rolling_K_H0",
    "n_iterations",
    "mean_final_FPR",
    "std_final_FPR",
    "median_final_FPR",
    "p05_final_FPR",
    "p95_final_FPR",
    "p99_final_FPR",
    "max_final_FPR",
    "mean_final_hit",
    "std_final_hit",
    "median_final_hit",
    "p05_final_hit",
    "p95_final_hit",
    "mean_final_TPR",
    "std_final_TPR",
    "mean_precision",
    "std_precision",
    "final_safe_rate",
    "all_blocks_safe_rate",
    "mean_fraction_safe_blocks",
    "mean_max_block_FPR",
    "p95_max_block_FPR",
    "max_observed_block_FPR",
    "mean_tau_drift",
    "p95_tau_drift",
    "mean_final_calib_H0",
    "mean_final_calib_H1",
]

SUCCESS_FIELDS = [
    "dataset",
    "method",
    "policy",
    "alpha_cal_multiplier",
    "num_blocks",
    "rolling_K_H0",
    "mean_final_FPR",
    "final_safe_rate",
    "all_blocks_safe_rate",
    "mean_final_hit",
    "reference_hit",
    "reference_is_safe",
    "relative_mean_hit_vs_reference",
    "avg_success_90",
    "avg_success_95",
    "high_probability_final_safety",
    "strict_block_success",
]

CI_FIELDS = [
    "dataset",
    "method",
    "policy",
    "alpha",
    "alpha_cal_multiplier",
    "num_blocks",
    "rolling_K_H0",
    "n_iterations",
    "mean_final_FPR",
    "ci95_low_final_FPR",
    "ci95_high_final_FPR",
    "mean_final_hit",
    "ci95_low_final_hit",
    "ci95_high_final_hit",
    "one_sided_p_value_H1_mean_FPR_lt_alpha",
    "ci_method",
    "ci_upper_below_alpha",
    "evidence_note",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Repeated freeze-scorer / refresh-threshold evaluation.")
    ap.add_argument("--mode", choices=["multi_seed", "chronological_partition", "block_bootstrap"], default="chronological_partition")
    ap.add_argument("--n_iterations", type=int, default=1000)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["ours_weighted_ensemble", "ours_whitened_hadamard", "cosine"])
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--num_blocks", nargs="+", type=int, default=DEFAULT_NUM_BLOCKS)
    ap.add_argument("--rolling_K_H0", nargs="+", type=int, default=DEFAULT_ROLLING_K_H0)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--pairs_root", default="results/online_candidate_pairs")
    ap.add_argument("--output_dir", default="results/repeated_threshold_refresh")
    ap.add_argument("--random_seed", type=int, default=12345)
    ap.add_argument("--jitter_fraction", type=float, default=0.20)
    ap.add_argument("--block_count_strategy", choices=["cycle", "random", "all"], default="cycle")
    ap.add_argument("--skip_plots", action="store_true")
    return ap.parse_args(argv)


def _policy_specs(rolling_values: Sequence[int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {"policy": "one_shot_naive", "alpha_cal_multiplier": 1.0, "rolling_K_H0": "", "refresh": "none"},
    ]
    for mult in [0.7, 0.5]:
        specs.append({"policy": "one_shot_conservative", "alpha_cal_multiplier": mult, "rolling_K_H0": "", "refresh": "none"})
    for mult in [1.0, 0.7, 0.5]:
        specs.append({"policy": "cumulative_refresh", "alpha_cal_multiplier": mult, "rolling_K_H0": "", "refresh": "cumulative"})
    for mult in [1.0, 0.7, 0.5]:
        for k in rolling_values:
            specs.append({"policy": "rolling_refresh", "alpha_cal_multiplier": mult, "rolling_K_H0": int(k), "refresh": "rolling"})
    for mult in [1.0, 0.7, 0.5]:
        for k in rolling_values:
            specs.append({"policy": "conservative_rolling_refresh", "alpha_cal_multiplier": mult, "rolling_K_H0": int(k), "refresh": "rolling"})
    return specs


def _discover_pair_dirs(root: Path, datasets: Sequence[str], seeds: Sequence[int]) -> Dict[Tuple[str, int], Path]:
    out: Dict[Tuple[str, int], Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Missing online candidate-pair root: {root}")
    for config_path in sorted(root.glob("*/config.json")):
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        dataset = _dataset_from_config(config)
        seed = _seed_from_config(config)
        if dataset in set(datasets) and seed in set(int(s) for s in seeds):
            out[(dataset, seed)] = config_path.parent
    return out


def _choose_block_counts(iteration: int, values: Sequence[int], strategy: str, rng: np.random.Generator) -> List[int]:
    vals = [int(v) for v in values]
    if strategy == "all":
        return vals
    if strategy == "random":
        return [int(rng.choice(vals))]
    return [vals[int(iteration) % len(vals)]]


def _balanced_sizes(n: int, num_blocks: int) -> np.ndarray:
    base = np.full(int(num_blocks), n // int(num_blocks), dtype=np.int64)
    base[: n % int(num_blocks)] += 1
    return base


def _jittered_blocks(n: int, num_blocks: int, rng: np.random.Generator, jitter_fraction: float) -> List[Tuple[int, int, np.ndarray]]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > n:
        num_blocks = n
    base = _balanced_sizes(n, num_blocks)
    if jitter_fraction <= 0.0 or num_blocks == 1:
        sizes = base
    else:
        weights = rng.uniform(max(0.05, 1.0 - jitter_fraction), 1.0 + jitter_fraction, size=int(num_blocks))
        raw = weights * base.astype(float)
        sizes = np.maximum(1, np.floor(raw / raw.sum() * n).astype(np.int64))
        while int(sizes.sum()) < n:
            sizes[int(rng.integers(0, num_blocks))] += 1
        while int(sizes.sum()) > n:
            candidates = np.where(sizes > 1)[0]
            sizes[int(rng.choice(candidates))] -= 1
    blocks: List[Tuple[int, int, np.ndarray]] = []
    start = 0
    for size in sizes:
        end = start + int(size)
        idx = np.arange(start, end, dtype=np.int64)
        blocks.append((int(start), int(end - 1), idx))
        start = end
    if start != n:
        raise AssertionError(f"Partition ended at {start}, expected {n}")
    return blocks


def _bootstrap_blocks(n: int, num_blocks: int, rng: np.random.Generator) -> List[Tuple[int, int, np.ndarray]]:
    base_blocks = _jittered_blocks(n, num_blocks, rng, 0.0)
    sampled = rng.integers(0, len(base_blocks), size=len(base_blocks))
    return [base_blocks[int(i)] for i in sampled]


def _tau_h0_scores(memory_scores: np.ndarray, memory_y: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    yy = np.asarray(memory_y).reshape(-1).astype(np.int32)
    h0 = np.asarray(memory_scores, dtype=np.float64).reshape(-1)[yy == 0]
    if spec.get("refresh") == "rolling" and spec.get("rolling_K_H0") not in {"", None}:
        k = int(spec["rolling_K_H0"])
        if h0.size > k:
            return h0[-k:]
    return h0


def _eval_counts(y: np.ndarray, scores: np.ndarray, tau: float, orientation: str) -> Tuple[int, int, int, int]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    aa = accept_scores(np.asarray(scores, dtype=np.float64).reshape(-1), float(tau), orientation)
    tp = int(np.sum(aa & (yy == 1)))
    fp = int(np.sum(aa & (yy == 0)))
    tn = int(np.sum((~aa) & (yy == 0)))
    fn = int(np.sum((~aa) & (yy == 1)))
    return tp, fp, tn, fn


def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int, prefix: str) -> Dict[str, Any]:
    n = int(tp + fp + tn + fn)
    return {
        f"{prefix}_FP": int(fp),
        f"{prefix}_TN": int(tn),
        f"{prefix}_FPR": fp / float(fp + tn) if fp + tn else None,
        f"{prefix}_TP": int(tp),
        f"{prefix}_FN": int(fn),
        f"{prefix}_TPR": tp / float(tp + fn) if tp + fn else None,
        f"{prefix}_hit": (tp + fp) / float(n) if n else None,
        f"{prefix}_precision": tp / float(tp + fp) if tp + fp else None,
        f"{prefix}_FP_over_n": fp / float(n) if n else None,
    }


def _h0_mean_p95(y: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    ss = np.asarray(scores, dtype=np.float64).reshape(-1)
    h0 = ss[yy == 0]
    if h0.size == 0:
        return None, None
    return float(np.mean(h0)), float(np.quantile(h0, 0.95))


def _safe_float_array(values: Iterable[Any]) -> np.ndarray:
    vals = [_float(v) for v in values]
    return np.asarray([v for v in vals if v is not None and np.isfinite(float(v))], dtype=float)


def _std(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    if values.size == 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _policy_label(row: Dict[str, Any]) -> str:
    label = f"{row.get('policy')}@alpha_cal_x={_fmt(row.get('alpha_cal_multiplier'))},blocks={row.get('num_blocks')}"
    if str(row.get("rolling_K_H0", "")).strip():
        label += f",K_H0={row.get('rolling_K_H0')}"
    return label


def _simulate_one(
    *,
    block_writer: csv.DictWriter,
    block_plot_acc: DefaultDict[Tuple[Any, ...], Dict[str, List[float]]],
    iteration: int,
    repetition_mode: str,
    seed: int,
    dataset: str,
    method: str,
    spec: Dict[str, Any],
    alpha: float,
    num_blocks: int,
    blocks: Sequence[Tuple[int, int, np.ndarray]],
    calib_scores: np.ndarray,
    calib_y: np.ndarray,
    eval_scores: np.ndarray,
    eval_y: np.ndarray,
    orientation: str,
) -> Dict[str, Any]:
    memory_scores = np.asarray(calib_scores, dtype=np.float64).reshape(-1).copy()
    memory_y = np.asarray(calib_y, dtype=np.int32).reshape(-1).copy()
    alpha_cal_multiplier = float(spec["alpha_cal_multiplier"])
    alpha_cal = float(alpha) * alpha_cal_multiplier
    cumulative_tp = cumulative_fp = cumulative_tn = cumulative_fn = 0
    block_fprs: List[float] = []
    block_hits: List[float] = []
    taus: List[float] = []
    unsafe_blocks: List[int] = []
    last_row: Dict[str, Any] = {}

    fixed_tau: Optional[float] = None
    if spec["refresh"] == "none":
        h0_for_tau = _tau_h0_scores(memory_scores, memory_y, spec)
        fixed_tau = select_tau(h0_for_tau, alpha_cal, orientation)

    for block_index, (block_start, block_end, idx) in enumerate(blocks):
        block_scores = np.asarray(eval_scores, dtype=np.float64).reshape(-1)[idx]
        block_y = np.asarray(eval_y, dtype=np.int32).reshape(-1)[idx]
        h0_for_tau = _tau_h0_scores(memory_scores, memory_y, spec)
        tau = fixed_tau if fixed_tau is not None else select_tau(h0_for_tau, alpha_cal, orientation)
        tp, fp, tn, fn = _eval_counts(block_y, block_scores, tau, orientation)
        cumulative_tp += tp
        cumulative_fp += fp
        cumulative_tn += tn
        cumulative_fn += fn
        block_h1, block_h0 = _counts(block_y)
        calib_h1 = int(np.sum(memory_y == 1))
        calib_h0 = int(h0_for_tau.size)
        calib_h0_mean = float(np.mean(h0_for_tau)) if h0_for_tau.size else None
        calib_h0_p95 = float(np.quantile(h0_for_tau, 0.95)) if h0_for_tau.size else None
        block_h0_mean, block_h0_p95 = _h0_mean_p95(block_y, block_scores)
        block_drift = None if calib_h0_p95 is None or block_h0_p95 is None else block_h0_p95 - calib_h0_p95

        block_metrics = _metrics_from_counts(tp, fp, tn, fn, "block")
        cumulative_metrics = _metrics_from_counts(cumulative_tp, cumulative_fp, cumulative_tn, cumulative_fn, "cumulative")
        block_fpr = _float(block_metrics.get("block_FPR"))
        block_hit = _float(block_metrics.get("block_hit"))
        if block_fpr is not None:
            block_fprs.append(block_fpr)
            if block_fpr > float(alpha):
                unsafe_blocks.append(int(block_index))
        if block_hit is not None:
            block_hits.append(block_hit)
        if np.isfinite(float(tau)):
            taus.append(float(tau))

        row: Dict[str, Any] = {
            "iteration": int(iteration),
            "repetition_mode": repetition_mode,
            "seed": int(seed),
            "dataset": dataset,
            "method": method,
            "policy": spec["policy"],
            "alpha": float(alpha),
            "alpha_cal_multiplier": alpha_cal_multiplier,
            "alpha_cal": alpha_cal,
            "num_blocks": int(num_blocks),
            "block_index": int(block_index),
            "block_start": int(block_start),
            "block_end": int(block_end),
            "rolling_K_H0": spec.get("rolling_K_H0", ""),
            "tau": float(tau),
            "calib_memory_size_total": int(calib_h0 + calib_h1),
            "calib_memory_H0": int(calib_h0),
            "calib_memory_H1": int(calib_h1),
            "block_n": int(block_y.size),
            "block_H0": int(block_h0),
            "block_H1": int(block_h1),
            "calib_H0_mean": calib_h0_mean,
            "calib_H0_p95": calib_h0_p95,
            "block_H0_mean": block_h0_mean,
            "block_H0_p95": block_h0_p95,
            "block_minus_calib_H0_p95": block_drift,
        }
        row.update(block_metrics)
        row.update(cumulative_metrics)
        row["safe_block"] = bool(block_fpr is not None and block_fpr <= float(alpha))
        row["safe_cumulative_so_far"] = bool(
            (_float(row.get("cumulative_FPR")) is not None) and float(row["cumulative_FPR"]) <= float(alpha)
        )
        block_writer.writerow({field: row.get(field, "") for field in BLOCK_FIELDS})
        last_row = row

        acc_key = (
            repetition_mode,
            dataset,
            method,
            spec["policy"],
            float(alpha),
            alpha_cal_multiplier,
            int(num_blocks),
            spec.get("rolling_K_H0", ""),
            int(block_index),
        )
        block_plot_acc[acc_key]["block_FPR"].append(block_fpr if block_fpr is not None else np.nan)
        cumulative_fpr = _float(row.get("cumulative_FPR"))
        block_plot_acc[acc_key]["cumulative_FPR"].append(cumulative_fpr if cumulative_fpr is not None else np.nan)

        if spec["refresh"] != "none":
            memory_scores = np.concatenate([memory_scores, block_scores])
            memory_y = np.concatenate([memory_y, block_y])

    final_h0_for_tau = _tau_h0_scores(memory_scores, memory_y, spec)
    final_calib_h0 = int(final_h0_for_tau.size)
    final_calib_h1 = int(np.sum(memory_y == 1))
    block_fpr_arr = np.asarray(block_fprs, dtype=float)
    tau_arr = np.asarray(taus, dtype=float)
    iteration_row = {
        "iteration": int(iteration),
        "repetition_mode": repetition_mode,
        "seed": int(seed),
        "dataset": dataset,
        "method": method,
        "policy": spec["policy"],
        "alpha": float(alpha),
        "alpha_cal_multiplier": alpha_cal_multiplier,
        "alpha_cal": alpha_cal,
        "num_blocks": int(num_blocks),
        "rolling_K_H0": spec.get("rolling_K_H0", ""),
        "final_cumulative_FPR": last_row.get("cumulative_FPR"),
        "final_cumulative_TPR": last_row.get("cumulative_TPR"),
        "final_cumulative_hit": last_row.get("cumulative_hit"),
        "final_cumulative_precision": last_row.get("cumulative_precision"),
        "final_cumulative_FP_over_n": last_row.get("cumulative_FP_over_n"),
        "mean_block_FPR": float(np.nanmean(block_fpr_arr)) if block_fpr_arr.size else None,
        "max_block_FPR": float(np.nanmax(block_fpr_arr)) if block_fpr_arr.size else None,
        "p95_block_FPR": float(np.nanquantile(block_fpr_arr, 0.95)) if block_fpr_arr.size else None,
        "std_block_FPR": _std(block_fpr_arr[~np.isnan(block_fpr_arr)]) if block_fpr_arr.size else None,
        "fraction_safe_blocks": float(np.mean(block_fpr_arr <= float(alpha))) if block_fpr_arr.size else None,
        "all_blocks_safe": bool(block_fpr_arr.size > 0 and np.nanmax(block_fpr_arr) <= float(alpha)),
        "final_safe": bool((_float(last_row.get("cumulative_FPR")) is not None) and float(last_row["cumulative_FPR"]) <= float(alpha)),
        "first_unsafe_block": int(unsafe_blocks[0]) if unsafe_blocks else "",
        "unsafe_blocks": int(len(unsafe_blocks)),
        "final_tau": float(tau_arr[-1]) if tau_arr.size else None,
        "tau_min": float(np.min(tau_arr)) if tau_arr.size else None,
        "tau_max": float(np.max(tau_arr)) if tau_arr.size else None,
        "tau_drift": float(np.max(tau_arr) - np.min(tau_arr)) if tau_arr.size else None,
        "final_calib_memory_H0": final_calib_h0,
        "final_calib_memory_H1": final_calib_h1,
    }
    return iteration_row


def _aggregate_iteration_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: DefaultDict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("repetition_mode"),
            row.get("dataset"),
            row.get("method"),
            row.get("policy"),
            row.get("alpha"),
            row.get("alpha_cal_multiplier"),
            row.get("num_blocks"),
            row.get("rolling_K_H0"),
        )
        groups[key].append(row)
    out: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        fprs = _safe_float_array(r.get("final_cumulative_FPR") for r in group)
        hits = _safe_float_array(r.get("final_cumulative_hit") for r in group)
        tprs = _safe_float_array(r.get("final_cumulative_TPR") for r in group)
        precisions = _safe_float_array(r.get("final_cumulative_precision") for r in group)
        safe = np.asarray([str(r.get("final_safe")).lower() == "true" or r.get("final_safe") is True for r in group], dtype=float)
        all_block = np.asarray([str(r.get("all_blocks_safe")).lower() == "true" or r.get("all_blocks_safe") is True for r in group], dtype=float)
        frac_safe_blocks = _safe_float_array(r.get("fraction_safe_blocks") for r in group)
        max_block_fprs = _safe_float_array(r.get("max_block_FPR") for r in group)
        tau_drifts = _safe_float_array(r.get("tau_drift") for r in group)
        final_h0 = _safe_float_array(r.get("final_calib_memory_H0") for r in group)
        final_h1 = _safe_float_array(r.get("final_calib_memory_H1") for r in group)
        repetition_mode, dataset, method, policy, alpha, mult, num_blocks, rolling_k = key
        out.append(
            {
                "repetition_mode": repetition_mode,
                "dataset": dataset,
                "method": method,
                "policy": policy,
                "alpha": alpha,
                "alpha_cal_multiplier": mult,
                "num_blocks": num_blocks,
                "rolling_K_H0": rolling_k,
                "n_iterations": int(len(group)),
                "mean_final_FPR": float(np.mean(fprs)) if fprs.size else None,
                "std_final_FPR": _std(fprs),
                "median_final_FPR": float(np.median(fprs)) if fprs.size else None,
                "p05_final_FPR": float(np.quantile(fprs, 0.05)) if fprs.size else None,
                "p95_final_FPR": float(np.quantile(fprs, 0.95)) if fprs.size else None,
                "p99_final_FPR": float(np.quantile(fprs, 0.99)) if fprs.size else None,
                "max_final_FPR": float(np.max(fprs)) if fprs.size else None,
                "mean_final_hit": float(np.mean(hits)) if hits.size else None,
                "std_final_hit": _std(hits),
                "median_final_hit": float(np.median(hits)) if hits.size else None,
                "p05_final_hit": float(np.quantile(hits, 0.05)) if hits.size else None,
                "p95_final_hit": float(np.quantile(hits, 0.95)) if hits.size else None,
                "mean_final_TPR": float(np.mean(tprs)) if tprs.size else None,
                "std_final_TPR": _std(tprs),
                "mean_precision": float(np.mean(precisions)) if precisions.size else None,
                "std_precision": _std(precisions),
                "final_safe_rate": float(np.mean(safe)) if safe.size else None,
                "all_blocks_safe_rate": float(np.mean(all_block)) if all_block.size else None,
                "mean_fraction_safe_blocks": float(np.mean(frac_safe_blocks)) if frac_safe_blocks.size else None,
                "mean_max_block_FPR": float(np.mean(max_block_fprs)) if max_block_fprs.size else None,
                "p95_max_block_FPR": float(np.quantile(max_block_fprs, 0.95)) if max_block_fprs.size else None,
                "max_observed_block_FPR": float(np.max(max_block_fprs)) if max_block_fprs.size else None,
                "mean_tau_drift": float(np.mean(tau_drifts)) if tau_drifts.size else None,
                "p95_tau_drift": float(np.quantile(tau_drifts, 0.95)) if tau_drifts.size else None,
                "mean_final_calib_H0": float(np.mean(final_h0)) if final_h0.size else None,
                "mean_final_calib_H1": float(np.mean(final_h1)) if final_h1.size else None,
            }
        )
    return out


def _build_success(aggregate_rows: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted({(r.get("dataset"), r.get("method")) for r in aggregate_rows})
    for dataset, method in keys:
        one_shot = [
            r
            for r in aggregate_rows
            if r.get("dataset") == dataset
            and r.get("method") == method
            and str(r.get("policy", "")).startswith("one_shot")
        ]
        safe_one_shot = [r for r in one_shot if (_float(r.get("mean_final_FPR")) is not None and float(r["mean_final_FPR"]) <= float(alpha))]
        reference_is_safe = bool(safe_one_shot)
        if safe_one_shot:
            ref = max(safe_one_shot, key=lambda r: _float(r.get("mean_final_hit")) or -1.0)
        elif one_shot:
            ref = max(one_shot, key=lambda r: _float(r.get("mean_final_hit")) or -1.0)
        else:
            continue
        reference_hit = _float(ref.get("mean_final_hit"))
        for row in [r for r in aggregate_rows if r.get("dataset") == dataset and r.get("method") == method]:
            mean_fpr = _float(row.get("mean_final_FPR"))
            mean_hit = _float(row.get("mean_final_hit"))
            rel_hit = mean_hit / reference_hit if mean_hit is not None and reference_hit and reference_hit > 0 else None
            final_safe_rate = _float(row.get("final_safe_rate"))
            all_blocks_safe_rate = _float(row.get("all_blocks_safe_rate"))
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "policy": row.get("policy"),
                    "alpha_cal_multiplier": row.get("alpha_cal_multiplier"),
                    "num_blocks": row.get("num_blocks"),
                    "rolling_K_H0": row.get("rolling_K_H0"),
                    "mean_final_FPR": mean_fpr,
                    "final_safe_rate": final_safe_rate,
                    "all_blocks_safe_rate": all_blocks_safe_rate,
                    "mean_final_hit": mean_hit,
                    "reference_hit": reference_hit,
                    "reference_is_safe": reference_is_safe,
                    "relative_mean_hit_vs_reference": rel_hit,
                    "avg_success_90": bool(mean_fpr is not None and mean_fpr <= float(alpha) and rel_hit is not None and rel_hit >= 0.90),
                    "avg_success_95": bool(mean_fpr is not None and mean_fpr <= float(alpha) and rel_hit is not None and rel_hit >= 0.95),
                    "high_probability_final_safety": bool(final_safe_rate is not None and final_safe_rate >= 0.95),
                    "strict_block_success": bool(all_blocks_safe_rate is not None and all_blocks_safe_rate >= 0.95),
                }
            )
    return out


def _normal_ci(values: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, None, None
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean, 0.0
    se = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se, se


def _one_sided_p_value(mean: Optional[float], se: Optional[float], alpha: float) -> Optional[float]:
    if mean is None or se is None:
        return None
    if se == 0.0:
        return 0.0 if mean < alpha else 1.0
    z = (mean - alpha) / se
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _build_main_ci(iteration_rows: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    target = {
        "dataset": "SemCacheLMArena",
        "method": "ours_weighted_ensemble",
        "policy": "conservative_rolling_refresh",
        "alpha_cal_multiplier": 0.7,
        "rolling_K_H0": "10000",
        "num_blocks": "5",
    }
    group = [
        r
        for r in iteration_rows
        if r.get("dataset") == target["dataset"]
        and r.get("method") == target["method"]
        and r.get("policy") == target["policy"]
        and abs((_float(r.get("alpha_cal_multiplier")) or -999.0) - target["alpha_cal_multiplier"]) < 1e-12
        and str(r.get("rolling_K_H0")) == target["rolling_K_H0"]
        and str(r.get("num_blocks")) == target["num_blocks"]
    ]
    fprs = _safe_float_array(r.get("final_cumulative_FPR") for r in group)
    hits = _safe_float_array(r.get("final_cumulative_hit") for r in group)
    mean_fpr, low_fpr, high_fpr, se_fpr = _normal_ci(fprs)
    mean_hit, low_hit, high_hit, _se_hit = _normal_ci(hits)
    p_value = _one_sided_p_value(mean_fpr, se_fpr, alpha)
    note = ""
    if mean_fpr is None:
        note = "No matching iterations were found for the main policy."
    elif high_fpr is not None and high_fpr <= float(alpha):
        note = "The normal-approximation 95% CI upper bound is below alpha."
    elif mean_fpr <= float(alpha):
        note = "Mean FPR is below alpha, but the 95% CI upper bound exceeds alpha; evidence is suggestive, not conclusive."
    else:
        note = "Mean FPR is not below alpha in this repeated setting."
    return [
        {
            "dataset": target["dataset"],
            "method": target["method"],
            "policy": target["policy"],
            "alpha": float(alpha),
            "alpha_cal_multiplier": target["alpha_cal_multiplier"],
            "num_blocks": int(target["num_blocks"]),
            "rolling_K_H0": int(target["rolling_K_H0"]),
            "n_iterations": int(len(group)),
            "mean_final_FPR": mean_fpr,
            "ci95_low_final_FPR": low_fpr,
            "ci95_high_final_FPR": high_fpr,
            "mean_final_hit": mean_hit,
            "ci95_low_final_hit": low_hit,
            "ci95_high_final_hit": high_hit,
            "one_sided_p_value_H1_mean_FPR_lt_alpha": p_value,
            "ci_method": "normal_approximation_over_iterations",
            "ci_upper_below_alpha": bool(high_fpr is not None and high_fpr <= float(alpha)),
            "evidence_note": note,
        }
    ]


def _block_plot_rows(block_plot_acc: Dict[Tuple[Any, ...], Dict[str, List[float]]], metric: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key, values in sorted(block_plot_acc.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = np.asarray(values[metric], dtype=float)
        vals = vals[np.isfinite(vals)]
        repetition_mode, dataset, method, policy, alpha, mult, num_blocks, rolling_k, block_index = key
        out.append(
            {
                "repetition_mode": repetition_mode,
                "dataset": dataset,
                "method": method,
                "policy": policy,
                "alpha": alpha,
                "alpha_cal_multiplier": mult,
                "num_blocks": num_blocks,
                "rolling_K_H0": rolling_k,
                "block_index": block_index,
                f"mean_{metric}": float(np.mean(vals)) if vals.size else None,
                f"p05_{metric}": float(np.quantile(vals, 0.05)) if vals.size else None,
                f"p95_{metric}": float(np.quantile(vals, 0.95)) if vals.size else None,
                "n": int(vals.size),
            }
        )
    return out


def _write_plot_data(
    out_dir: Path,
    iteration_rows: Sequence[Dict[str, Any]],
    aggregate_rows: Sequence[Dict[str, Any]],
    block_plot_acc: Dict[Tuple[Any, ...], Dict[str, List[float]]],
) -> List[Path]:
    paths: List[Path] = []
    final_dist_fields = [
        "iteration",
        "repetition_mode",
        "seed",
        "dataset",
        "method",
        "policy",
        "alpha",
        "alpha_cal_multiplier",
        "num_blocks",
        "rolling_K_H0",
        "final_cumulative_FPR",
        "final_cumulative_hit",
        "final_safe",
        "all_blocks_safe",
    ]
    path = out_dir / "plot_final_fpr_distribution.csv"
    _write_csv(path, iteration_rows, final_dist_fields)
    paths.append(path)

    path = out_dir / "plot_fpr_vs_hit_scatter.csv"
    _write_csv(path, iteration_rows, final_dist_fields)
    paths.append(path)

    block_rows = _block_plot_rows(block_plot_acc, "block_FPR")
    path = out_dir / "plot_mean_block_fpr.csv"
    _write_csv(
        path,
        block_rows,
        [
            "repetition_mode",
            "dataset",
            "method",
            "policy",
            "alpha",
            "alpha_cal_multiplier",
            "num_blocks",
            "rolling_K_H0",
            "block_index",
            "mean_block_FPR",
            "p05_block_FPR",
            "p95_block_FPR",
            "n",
        ],
    )
    paths.append(path)

    cumulative_rows = _block_plot_rows(block_plot_acc, "cumulative_FPR")
    path = out_dir / "plot_cumulative_fpr_over_time.csv"
    _write_csv(
        path,
        cumulative_rows,
        [
            "repetition_mode",
            "dataset",
            "method",
            "policy",
            "alpha",
            "alpha_cal_multiplier",
            "num_blocks",
            "rolling_K_H0",
            "block_index",
            "mean_cumulative_FPR",
            "p05_cumulative_FPR",
            "p95_cumulative_FPR",
            "n",
        ],
    )
    paths.append(path)

    policy_rows = []
    for r in aggregate_rows:
        n = _float(r.get("n_iterations")) or 0.0
        std_fpr = _float(r.get("std_final_FPR")) or 0.0
        std_hit = _float(r.get("std_final_hit")) or 0.0
        policy_rows.append(
            {
                **r,
                "ci95_low_mean_final_FPR": (_float(r.get("mean_final_FPR")) or 0.0) - 1.96 * std_fpr / math.sqrt(n) if n else None,
                "ci95_high_mean_final_FPR": (_float(r.get("mean_final_FPR")) or 0.0) + 1.96 * std_fpr / math.sqrt(n) if n else None,
                "ci95_low_mean_final_hit": (_float(r.get("mean_final_hit")) or 0.0) - 1.96 * std_hit / math.sqrt(n) if n else None,
                "ci95_high_mean_final_hit": (_float(r.get("mean_final_hit")) or 0.0) + 1.96 * std_hit / math.sqrt(n) if n else None,
            }
        )
    path = out_dir / "plot_policy_comparison.csv"
    _write_csv(
        path,
        policy_rows,
        [
            *AGGREGATE_FIELDS,
            "ci95_low_mean_final_FPR",
            "ci95_high_mean_final_FPR",
            "ci95_low_mean_final_hit",
            "ci95_high_mean_final_hit",
        ],
    )
    paths.append(path)
    return paths


def _maybe_write_pngs(out_dir: Path, alpha: float, skip_plots: bool) -> List[Path]:
    if skip_plots:
        return []
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping PNG plots because matplotlib is unavailable: {exc}")
        return []
    paths: List[Path] = []

    def read_rows(name: str) -> List[Dict[str, str]]:
        with (out_dir / name).open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    rows = [
        r
        for r in read_rows("plot_final_fpr_distribution.csv")
        if r["dataset"] == "SemCacheLMArena"
        and r["method"] == "ours_weighted_ensemble"
        and r["policy"] in {"one_shot_conservative", "conservative_rolling_refresh", "rolling_refresh"}
        and r["num_blocks"] == "5"
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy in sorted({r["policy"] for r in rows}):
        vals = [float(r["final_cumulative_FPR"]) for r in rows if r["policy"] == policy and r["final_cumulative_FPR"]]
        if vals:
            ax.hist(vals, bins=25, alpha=0.45, label=policy)
    ax.axvline(alpha, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Final cumulative FPR")
    ax.set_ylabel("Iterations")
    ax.set_title("LMArena / WeightedEnsemble final FPR distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "plot_final_fpr_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    rows = [
        r
        for r in read_rows("plot_fpr_vs_hit_scatter.csv")
        if r["dataset"] == "SemCacheLMArena"
        and r["method"] == "ours_weighted_ensemble"
        and r["num_blocks"] == "5"
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for policy in sorted({r["policy"] for r in rows}):
        vals = [r for r in rows if r["policy"] == policy]
        xs = [float(r["final_cumulative_FPR"]) for r in vals if r["final_cumulative_FPR"]]
        ys = [float(r["final_cumulative_hit"]) for r in vals if r["final_cumulative_hit"]]
        if xs and ys:
            ax.scatter(xs, ys, s=8, alpha=0.35, label=policy)
    ax.axvline(alpha, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Final cumulative FPR")
    ax.set_ylabel("Final hit")
    ax.set_title("LMArena / WeightedEnsemble FPR vs hit")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "plot_fpr_vs_hit_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    for csv_name, out_name, y_name, title in [
        ("plot_mean_block_fpr.csv", "plot_mean_block_fpr.png", "mean_block_FPR", "Mean block FPR"),
        ("plot_cumulative_fpr_over_time.csv", "plot_cumulative_fpr_over_time.png", "mean_cumulative_FPR", "Mean cumulative FPR"),
    ]:
        rows = [
            r
            for r in read_rows(csv_name)
            if r["dataset"] == "SemCacheLMArena"
            and r["method"] == "ours_weighted_ensemble"
            and r["num_blocks"] == "5"
            and (
                (r["policy"] == "conservative_rolling_refresh" and r["alpha_cal_multiplier"] == "0.7" and r["rolling_K_H0"] == "10000")
                or (r["policy"] == "one_shot_conservative" and r["alpha_cal_multiplier"] == "0.7")
                or (r["policy"] == "rolling_refresh" and r["alpha_cal_multiplier"] == "1.0" and r["rolling_K_H0"] == "1000")
            )
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key in sorted({(r["policy"], r["alpha_cal_multiplier"], r["rolling_K_H0"]) for r in rows}):
            vals = [r for r in rows if (r["policy"], r["alpha_cal_multiplier"], r["rolling_K_H0"]) == key]
            vals.sort(key=lambda r: int(r["block_index"]))
            x = [int(r["block_index"]) for r in vals]
            y = [float(r[y_name]) for r in vals]
            ax.plot(x, y, marker="o", linewidth=1, label=f"{key[0]} x={key[1]} K={key[2]}")
        ax.axhline(alpha, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Block index")
        ax.set_ylabel(y_name)
        ax.set_title(f"LMArena / WeightedEnsemble {title}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / out_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    rows = [
        r
        for r in read_rows("plot_policy_comparison.csv")
        if r["dataset"] == "SemCacheLMArena" and r["method"] == "ours_weighted_ensemble" and r["num_blocks"] == "5"
    ]
    rows.sort(key=lambda r: float(r["mean_final_FPR"]) if r["mean_final_FPR"] else 1e9)
    rows = rows[:12]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [_policy_label(r) for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x, [float(r["mean_final_FPR"]) for r in rows])
    axes[0].axhline(alpha, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Mean final FPR")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[1].bar(x, [float(r["mean_final_hit"]) for r in rows])
    axes[1].set_title("Mean final hit")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout()
    path = out_dir / "plot_policy_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows:
        lines.append("| " + " | ".join("" for _ in headers) + " |")
    else:
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _top_aggregate_table(rows: Sequence[Dict[str, Any]], dataset: str, method: str, limit: int = 10) -> List[str]:
    group = [r for r in rows if r.get("dataset") == dataset and r.get("method") == method]
    group.sort(
        key=lambda r: (
            0 if (_float(r.get("mean_final_FPR")) is not None and float(r["mean_final_FPR"]) <= float(r["alpha"])) else 1,
            -(_float(r.get("mean_final_hit")) or -1.0),
            -(_float(r.get("final_safe_rate")) or -1.0),
        )
    )
    body = [
        [
            _policy_label(r),
            str(r.get("n_iterations")),
            _fmt(r.get("mean_final_FPR")),
            _fmt(r.get("p95_final_FPR")),
            _fmt(r.get("final_safe_rate")),
            _fmt(r.get("all_blocks_safe_rate")),
            _fmt(r.get("mean_final_hit")),
            _fmt(r.get("mean_max_block_FPR")),
        ]
        for r in group[:limit]
    ]
    return _table(["Policy", "N", "Mean FPR", "p95 FPR", "Final-safe rate", "All-block-safe rate", "Mean hit", "Mean max block FPR"], body)


def _find_row(rows: Sequence[Dict[str, Any]], **criteria: Any) -> Optional[Dict[str, Any]]:
    for row in rows:
        ok = True
        for key, value in criteria.items():
            if key in {"alpha_cal_multiplier", "alpha"}:
                ok = ok and abs((_float(row.get(key)) or -999.0) - float(value)) < 1e-12
            else:
                ok = ok and str(row.get(key)) == str(value)
        if ok:
            return row
    return None


def _best_family(rows: Sequence[Dict[str, Any]], dataset: str, method: str, family: str) -> Optional[Dict[str, Any]]:
    group = [r for r in rows if r.get("dataset") == dataset and r.get("method") == method and r.get("policy") == family]
    if not group:
        return None
    safe = [r for r in group if _float(r.get("mean_final_FPR")) is not None and float(r["mean_final_FPR"]) <= float(r["alpha"])]
    pool = safe if safe else group
    return max(pool, key=lambda r: (_float(r.get("mean_final_hit")) or -1.0, _float(r.get("final_safe_rate")) or -1.0))


def write_markdown(
    path: Path,
    *,
    mode: str,
    block_count_strategy: str,
    aggregate_rows: Sequence[Dict[str, Any]],
    success_rows: Sequence[Dict[str, Any]],
    ci_rows: Sequence[Dict[str, Any]],
    available_seeds: Sequence[int],
    requested_iterations: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    main = _find_row(
        aggregate_rows,
        dataset="SemCacheLMArena",
        method="ours_weighted_ensemble",
        policy="conservative_rolling_refresh",
        alpha_cal_multiplier=0.7,
        rolling_K_H0=10000,
        num_blocks=5,
    )
    main_ci = ci_rows[0] if ci_rows else {}
    lma_roll = _best_family(aggregate_rows, "SemCacheLMArena", "ours_weighted_ensemble", "rolling_refresh")
    lma_cum = _best_family(aggregate_rows, "SemCacheLMArena", "ours_weighted_ensemble", "cumulative_refresh")
    sq_roll = _best_family(aggregate_rows, "SemCacheSearchQueries", "ours_weighted_ensemble", "rolling_refresh")
    sq_cum = _best_family(aggregate_rows, "SemCacheSearchQueries", "ours_weighted_ensemble", "cumulative_refresh")
    main_success = _find_row(
        success_rows,
        dataset="SemCacheLMArena",
        method="ours_weighted_ensemble",
        policy="conservative_rolling_refresh",
        alpha_cal_multiplier=0.7,
        rolling_K_H0=10000,
        num_blocks=5,
    )

    lines: List[str] = [
        "# Repeated Threshold Refresh",
        "",
        "## 1. Executive Summary",
        "",
        "This experiment repeats freeze-scorer / refresh-threshold evaluation to estimate average cumulative FPR, variance, and tail risk. The deployment target is candidate-level `FPR = FP / (FP + TN) <= 0.05`.",
        "",
    ]
    if mode == "multi_seed":
        lines.append(f"This report is multi-seed over the available requested seeds: `{', '.join(str(s) for s in available_seeds)}`.")
    else:
        lines.append("This is repeated chronological/block-partition evidence on one seed, not multi-seed robustness.")
    if main is not None:
        lines.append(
            "For the primary LMArena / WeightedEnsemble policy, "
            f"`{_policy_label(main)}`, mean cumulative FPR is `{_fmt(main.get('mean_final_FPR'))}`, "
            f"final-safe rate is `{_fmt(main.get('final_safe_rate'))}`, all-block-safe rate is `{_fmt(main.get('all_blocks_safe_rate'))}`, "
            f"and mean hit is `{_fmt(main.get('mean_final_hit'))}`."
        )
    lines.extend(
        [
            "",
            "The experiment does not test deterministic per-block FPR control, and unsafe individual iterations or unsafe blocks are retained in the output tables.",
            "",
            "## 2. Why Repeated Iterations Are Needed",
            "",
            "A single chronological partition can make cumulative FPR look safe while hiding sensitivity to block boundaries. Repeated trials estimate average behavior and tail risk without claiming guaranteed control.",
            "",
            "## 3. Protocol",
            "",
            "- Train each scorer once on `online_train` for the relevant seed.",
            "- Initialize threshold calibration memory from `online_calib`.",
            "- Split `online_eval` into chronological blocks.",
            "- Select each threshold from currently available H0 calibration scores only.",
            "- Evaluate the current block before adding that block's labels to calibration memory.",
            "- Do not update model weights after initial training.",
            "",
            "## 4. Repetition Modes",
            "",
            f"Requested mode: `{mode}`. Requested iterations: `{requested_iterations}`. Block-count strategy: `{block_count_strategy}`.",
            "",
            "- `multi_seed`: uses separately mined candidate pairs for each seed when available.",
            "- `chronological_partition`: preserves chronological order and varies contiguous block boundaries.",
            "- `block_bootstrap`: samples already-formed chronological blocks with replacement as a diagnostic only.",
            "",
            "## 5. Main LMArena Results",
            "",
        ]
    )
    lines.extend(_top_aggregate_table(aggregate_rows, "SemCacheLMArena", "ours_weighted_ensemble"))
    lines.extend(["", "## 6. Main SearchQueries Results", ""])
    lines.extend(_top_aggregate_table(aggregate_rows, "SemCacheSearchQueries", "ours_weighted_ensemble"))
    lines.extend(
        [
            "",
            "## 7. Average Cumulative FPR Results",
            "",
            "Average-FPR success is defined as `mean_final_FPR <= alpha`. See `repeated_threshold_refresh_aggregate.csv` for all policies and methods.",
            "",
            "## 8. Final Safety Rate",
            "",
            "High-probability final safety is defined as `final_safe_rate >= 0.95`. This is stricter than mean FPR being below alpha.",
            "",
            "## 9. Block-Level Safety Rate",
            "",
            "Strict block-level success is defined as `all_blocks_safe_rate >= 0.95`. Low all-block-safe rates mean cumulative safety should not be described as per-block control.",
            "",
            "## 10. Utility Tradeoff",
            "",
            "`repeated_threshold_refresh_success.csv` reports relative mean hit against the best one-shot reference for each dataset/method.",
            "",
            "## 11. Comparison To One-Shot And Previous Threshold Refresh",
            "",
        ]
    )
    if main_success:
        lines.append(
            f"The primary policy has relative mean hit `{_fmt(main_success.get('relative_mean_hit_vs_reference'))}` versus the selected one-shot reference. "
            f"The reference is marked safe on average: `{main_success.get('reference_is_safe')}`."
        )
    if lma_roll and lma_cum:
        better = "rolling_refresh" if (_float(lma_roll.get("mean_final_hit")) or -1) > (_float(lma_cum.get("mean_final_hit")) or -1) else "cumulative_refresh"
        lines.append(f"For LMArena / WeightedEnsemble, the best selected rolling-vs-cumulative utility row favors `{better}` among average-safe candidates.")
    if sq_roll and sq_cum:
        better = "rolling_refresh" if (_float(sq_roll.get("mean_final_hit")) or -1) > (_float(sq_cum.get("mean_final_hit")) or -1) else "cumulative_refresh"
        lines.append(f"For SearchQueries / WeightedEnsemble, the best selected rolling-vs-cumulative utility row favors `{better}` among average-safe candidates.")
    lines.extend(
        [
            "",
            "## 12. Statistical Confidence Intervals",
            "",
        ]
    )
    ci_table = [
        [
            str(r.get("n_iterations")),
            _fmt(r.get("mean_final_FPR")),
            f"[{_fmt(r.get('ci95_low_final_FPR'))}, {_fmt(r.get('ci95_high_final_FPR'))}]",
            _fmt(r.get("mean_final_hit")),
            f"[{_fmt(r.get('ci95_low_final_hit'))}, {_fmt(r.get('ci95_high_final_hit'))}]",
            _fmt(r.get("one_sided_p_value_H1_mean_FPR_lt_alpha"), 6),
            str(r.get("evidence_note")),
        ]
        for r in ci_rows
    ]
    lines.extend(_table(["N", "Mean FPR", "95% CI FPR", "Mean hit", "95% CI hit", "One-sided p", "Note"], ci_table))
    lines.extend(
        [
            "",
            "## 13. Failure Modes",
            "",
            "- Mean cumulative FPR can be below alpha while individual iterations exceed alpha.",
            "- Cumulative FPR can be safe while individual blocks are unsafe.",
            "- Repeated partitions of seed 42 are not a substitute for independently mined multi-seed streams.",
            "- The block bootstrap mode is a stress diagnostic, not true chronological deployment.",
            "",
            "## 14. Supported Claims",
            "",
            "- The experiment estimates average cumulative FPR and tail risk across repeated trials.",
            "- Thresholds are selected from past H0 calibration evidence only.",
            "- Claims should be limited to the repetition mode and seeds reported here.",
            "",
            "## 15. Unsupported Claims",
            "",
            "- Do not write that the method guarantees FPR control.",
            "- Do not write that the method is robust across seeds unless `multi_seed` mode was run with multiple seeds.",
            "- Do not hide unsafe runs or unsafe blocks; they are reported explicitly.",
            "",
            "## 16. Next Experiments",
            "",
            "- Generate online-mined candidate pairs for seeds 43-46 and rerun `--mode multi_seed`.",
            "- Add confidence-bound calibration on rolling H0 windows.",
            "- Test adaptive block sizes and drift-triggered recalibration.",
            "",
            "## Explicit Answers",
            "",
            "A. Is mean cumulative FPR below alpha? "
            + (
                f"For the primary policy, `{_fmt(main.get('mean_final_FPR'))}` {'is' if (_float(main.get('mean_final_FPR')) or 1) <= 0.05 else 'is not'} below 0.05."
                if main
                else "No primary-policy aggregate row was available."
            ),
            "B. 95% CI for mean cumulative FPR: "
            + (
                f"`[{_fmt(main_ci.get('ci95_low_final_FPR'))}, {_fmt(main_ci.get('ci95_high_final_FPR'))}]`."
                if main_ci
                else "Unavailable."
            ),
            "C. Fraction of iterations final-safe: " + (_fmt(main.get("final_safe_rate")) if main else "Unavailable."),
            "D. Fraction of iterations all-block-safe: " + (_fmt(main.get("all_blocks_safe_rate")) if main else "Unavailable."),
            "E. Mean hit rate: " + (_fmt(main.get("mean_final_hit")) if main else "Unavailable."),
            "F. Hit relative to one-shot: "
            + (_fmt(main_success.get("relative_mean_hit_vs_reference")) if main_success else "Unavailable."),
            "G. Rolling vs cumulative: see Section 11; the answer is dataset-specific.",
            "H. Best K_H0: choose the average-safe row with the highest mean hit in `repeated_threshold_refresh_aggregate.csv`; no universal K is claimed.",
            "I. Both datasets are reported when their pair dirs are available; do not generalize across missing methods or seeds.",
            "J. Repetition source: "
            + ("multi-seed." if mode == "multi_seed" else "repeated partitions/bootstrap of seed 42, not multi-seed robustness."),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> List[Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_dirs_by_key = _discover_pair_dirs(Path(args.pairs_root), args.datasets, args.seeds)
    if not pair_dirs_by_key:
        raise FileNotFoundError(
            f"No online-mined pair directories found under {args.pairs_root} for datasets={args.datasets} seeds={args.seeds}"
        )
    if args.mode == "multi_seed":
        missing = [(d, s) for d in args.datasets for s in args.seeds if (d, int(s)) not in pair_dirs_by_key]
        if missing:
            raise FileNotFoundError(
                "Missing online-mined candidate pairs for multi_seed mode: "
                + ", ".join(f"{d}/seed{s}" for d, s in missing)
                + ". Generate these pair dirs first or use --mode chronological_partition."
            )
    else:
        missing_seed42 = [d for d in args.datasets if (d, 42) not in pair_dirs_by_key]
        if missing_seed42:
            raise FileNotFoundError(
                "chronological_partition and block_bootstrap modes require seed 42 pair dirs for: "
                + ", ".join(missing_seed42)
            )

    rng = np.random.default_rng(int(args.random_seed))
    specs = _policy_specs(args.rolling_K_H0)
    outputs: List[Path] = []
    iteration_rows: List[Dict[str, Any]] = []
    block_plot_acc: DefaultDict[Tuple[Any, ...], Dict[str, List[float]]] = defaultdict(lambda: {"block_FPR": [], "cumulative_FPR": []})

    block_path = out_dir / "repeated_threshold_refresh_blocks.csv"
    with block_path.open("w", encoding="utf-8", newline="") as f:
        block_writer = csv.DictWriter(f, fieldnames=BLOCK_FIELDS, extrasaction="ignore")
        block_writer.writeheader()

        work_items: List[Tuple[int, str, int, Path, Optional[int]]] = []
        if args.mode == "multi_seed":
            iteration = 0
            for dataset in args.datasets:
                for seed in args.seeds:
                    pair_dir = pair_dirs_by_key[(dataset, int(seed))]
                    for num_blocks in args.num_blocks:
                        work_items.append((iteration, dataset, int(seed), pair_dir, int(num_blocks)))
                        iteration += 1
        else:
            for dataset in args.datasets:
                pair_dir = pair_dirs_by_key[(dataset, 42)]
                for iteration in range(int(args.n_iterations)):
                    work_items.append((iteration, dataset, 42, pair_dir, None))

        score_cache: Dict[Tuple[str, int, str], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = {}
        last_pair_key: Optional[Tuple[str, int]] = None
        train = calib = eval_pairs = config = None

        for item_pos, (iteration, dataset, seed, pair_dir, fixed_num_blocks) in enumerate(work_items, start=1):
            pair_key = (dataset, seed)
            if pair_key != last_pair_key:
                train, calib, eval_pairs, config = load_pairs_dir(pair_dir)
                last_pair_key = pair_key
                print(f"Loaded {dataset} seed {seed}: train={train.y.size}, calib={calib.y.size}, eval={eval_pairs.y.size}")
            assert train is not None and calib is not None and eval_pairs is not None

            if args.mode == "multi_seed":
                if fixed_num_blocks is None:
                    raise AssertionError("multi_seed work item must carry a fixed num_blocks value")
                num_blocks_values = [int(fixed_num_blocks)]
            else:
                num_blocks_values = _choose_block_counts(iteration, args.num_blocks, args.block_count_strategy, rng)

            for method in args.methods:
                cache_key = (dataset, seed, method)
                if cache_key not in score_cache:
                    print(f"Scoring {dataset} seed {seed} / {method}")
                    calib_scores, eval_scores, orientation = _score_bundle(
                        method,
                        train,
                        calib,
                        eval_pairs,
                        alpha=float(args.alpha),
                        seed=int(seed),
                    )
                    score_cache[cache_key] = (calib_scores, calib.y.copy(), eval_scores, eval_pairs.y.copy(), orientation)
                calib_scores, calib_y, eval_scores, eval_y, orientation = score_cache[cache_key]

                for num_blocks in num_blocks_values:
                    if args.mode == "block_bootstrap":
                        blocks = _bootstrap_blocks(int(eval_y.size), int(num_blocks), rng)
                    elif args.mode == "multi_seed":
                        blocks = _jittered_blocks(int(eval_y.size), int(num_blocks), rng, 0.0)
                    else:
                        blocks = _jittered_blocks(int(eval_y.size), int(num_blocks), rng, float(args.jitter_fraction))
                    for spec in specs:
                        row = _simulate_one(
                            block_writer=block_writer,
                            block_plot_acc=block_plot_acc,
                            iteration=int(iteration),
                            repetition_mode=args.mode,
                            seed=int(seed),
                            dataset=dataset,
                            method=method,
                            spec=spec,
                            alpha=float(args.alpha),
                            num_blocks=int(num_blocks),
                            blocks=blocks,
                            calib_scores=calib_scores,
                            calib_y=calib_y,
                            eval_scores=eval_scores,
                            eval_y=eval_y,
                            orientation=orientation,
                        )
                        iteration_rows.append(row)
    outputs.append(block_path)

    path = out_dir / "repeated_threshold_refresh_iterations.csv"
    _write_csv(path, iteration_rows, ITERATION_FIELDS)
    outputs.append(path)

    aggregate_rows = _aggregate_iteration_rows(iteration_rows)
    path = out_dir / "repeated_threshold_refresh_aggregate.csv"
    _write_csv(path, aggregate_rows, AGGREGATE_FIELDS)
    outputs.append(path)

    success_rows = _build_success(aggregate_rows, float(args.alpha))
    path = out_dir / "repeated_threshold_refresh_success.csv"
    _write_csv(path, success_rows, SUCCESS_FIELDS)
    outputs.append(path)

    ci_rows = _build_main_ci(iteration_rows, float(args.alpha))
    path = out_dir / "main_policy_ci.csv"
    _write_csv(path, ci_rows, CI_FIELDS)
    outputs.append(path)

    outputs.extend(_write_plot_data(out_dir, iteration_rows, aggregate_rows, block_plot_acc))
    outputs.extend(_maybe_write_pngs(out_dir, float(args.alpha), bool(args.skip_plots)))

    available_seeds = sorted({seed for (dataset, seed) in pair_dirs_by_key if dataset in set(args.datasets)})
    path = out_dir / "repeated_threshold_refresh_report.md"
    write_markdown(
        path,
        mode=args.mode,
        block_count_strategy=args.block_count_strategy,
        aggregate_rows=aggregate_rows,
        success_rows=success_rows,
        ci_rows=ci_rows,
        available_seeds=available_seeds,
        requested_iterations=int(args.n_iterations),
    )
    outputs.append(path)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    outputs = run(args)
    print("Wrote repeated threshold-refresh outputs:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
