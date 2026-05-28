from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .online_mined_ours import (
    OnlinePairs,
    accept_scores,
    infer_orientation,
    load_pairs_dir,
    select_tau,
)
from .risk_aware_stopping import (
    DEFAULT_PAIR_DIRS,
    _counts,
    _dataset_from_config,
    _fit_learned,
    _float,
    _fmt,
    _fmt_int,
    _score_cosine,
    _score_learned,
    _write_csv,
    _seed_from_config,
)


DEFAULT_METHODS = ["ours_weighted_ensemble", "ours_whitened_hadamard", "cosine"]
DEFAULT_ALPHA = 0.05
DEFAULT_NUM_BLOCKS = [5, 10, 20]
ROLLING_K_H0_VALUES = [1000, 2000, 5000, 10000]


BLOCK_METADATA_FIELDS = [
    "dataset",
    "seed",
    "num_blocks",
    "block_index",
    "block_start_index",
    "block_end_index",
    "block_n",
    "block_H0",
    "block_H1",
]

BLOCK_FIELDS = [
    "dataset",
    "method",
    "policy",
    "alpha",
    "alpha_cal_multiplier",
    "num_blocks",
    "rolling_K_H0",
    "block_index",
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
    "calib_H0_mean",
    "calib_H0_p95",
    "block_H0_mean",
    "block_H0_p95",
    "block_minus_calib_H0_p95",
    "safe_block",
    "safe_cumulative",
    "invalid_reason",
]

SUMMARY_FIELDS = [
    "dataset",
    "method",
    "policy",
    "alpha",
    "alpha_cal_multiplier",
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
    "fraction_safe_blocks",
    "mean_block_hit",
    "min_block_hit",
    "max_block_hit",
    "total_blocks",
    "unsafe_blocks",
    "first_unsafe_block",
    "final_tau",
    "tau_min",
    "tau_max",
    "tau_drift",
    "final_safe",
    "all_blocks_safe",
]

FIXED_VS_REFRESH_FIELDS = [
    "dataset",
    "method",
    "baseline_policy",
    "refresh_policy",
    "baseline_final_FPR",
    "refresh_final_FPR",
    "baseline_hit",
    "refresh_hit",
    "baseline_safe",
    "refresh_safe",
    "delta_FPR",
    "delta_hit",
]

SUCCESS_FIELDS = [
    "dataset",
    "method",
    "policy",
    "alpha_cal_multiplier",
    "num_blocks",
    "rolling_K_H0",
    "final_cumulative_FPR",
    "final_cumulative_hit",
    "reference_hit",
    "reference_is_safe",
    "relative_hit_vs_reference",
    "fraction_safe_blocks",
    "max_block_FPR",
    "success_90",
    "success_95",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate freeze-scorer / refresh-threshold policies on online-mined pairs.")
    ap.add_argument("--pair_dirs", nargs="+", default=DEFAULT_PAIR_DIRS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--num_blocks", nargs="+", type=int, default=DEFAULT_NUM_BLOCKS)
    ap.add_argument("--rolling_K_H0", nargs="+", type=int, default=ROLLING_K_H0_VALUES)
    ap.add_argument("--output_dir", default="results/threshold_refresh")
    return ap.parse_args(argv)


def _split_indices(n: int, num_blocks: int) -> List[np.ndarray]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    return [np.asarray(idx, dtype=np.int64) for idx in np.array_split(np.arange(n, dtype=np.int64), int(num_blocks))]


def _subset_pairs(pairs: OnlinePairs, idx: np.ndarray) -> OnlinePairs:
    return OnlinePairs(
        X=pairs.X[idx],
        y=pairs.y[idx],
        cosine=pairs.cosine[idx],
        query_id=pairs.query_id[idx],
        candidate_id=pairs.candidate_id[idx],
        query_emb=pairs.query_emb[idx] if pairs.query_emb is not None else None,
        candidate_emb=pairs.candidate_emb[idx] if pairs.candidate_emb is not None else None,
    )


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


def _eval_counts(y: np.ndarray, scores: np.ndarray, tau: float, orientation: str) -> Tuple[int, int, int, int]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    accepted = accept_scores(np.asarray(scores, dtype=np.float64).reshape(-1), float(tau), orientation)
    tp = int(np.sum(accepted & (yy == 1)))
    fp = int(np.sum(accepted & (yy == 0)))
    tn = int(np.sum((~accepted) & (yy == 0)))
    fn = int(np.sum((~accepted) & (yy == 1)))
    return tp, fp, tn, fn


def _h0_mean_p95(y: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    ss = np.asarray(scores, dtype=np.float64).reshape(-1)
    h0 = ss[yy == 0]
    if h0.size == 0:
        return None, None
    return float(np.mean(h0)), float(np.quantile(h0, 0.95))


def _score_bundle(method_key: str, train: OnlinePairs, calib: OnlinePairs, eval_pairs: OnlinePairs, *, alpha: float, seed: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if method_key == "cosine":
        return _score_cosine(calib), _score_cosine(eval_pairs), "higher"
    fitted, uses_alt, _method_name = _fit_learned(method_key, train, alpha=alpha, seed=seed)
    calib_scores = _score_learned(fitted, uses_alt, calib)
    eval_scores = _score_learned(fitted, uses_alt, eval_pairs)
    s0 = calib_scores[calib.y == 0]
    s1 = calib_scores[calib.y == 1]
    orientation, warning = infer_orientation(s0, s1)
    if orientation == "ambiguous":
        orientation = "higher"
        if warning:
            print(f"Warning: {method_key} orientation ambiguous on calibration split; defaulting to higher. {warning}")
    return calib_scores, eval_scores, orientation


def _policy_specs(rolling_values: Sequence[int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {"policy": "one_shot_naive", "alpha_cal_multiplier": 1.0, "rolling_K_H0": "", "refresh": "none"},
        {"policy": "one_shot_conservative", "alpha_cal_multiplier": 0.7, "rolling_K_H0": "", "refresh": "none"},
        {"policy": "one_shot_conservative", "alpha_cal_multiplier": 0.5, "rolling_K_H0": "", "refresh": "none"},
        {"policy": "cumulative_refresh", "alpha_cal_multiplier": 1.0, "rolling_K_H0": "", "refresh": "cumulative"},
    ]
    for k in rolling_values:
        specs.append({"policy": "rolling_refresh", "alpha_cal_multiplier": 1.0, "rolling_K_H0": int(k), "refresh": "rolling"})
    for mult in [0.7, 0.5]:
        for k in rolling_values:
            specs.append(
                {
                    "policy": "conservative_rolling_refresh",
                    "alpha_cal_multiplier": float(mult),
                    "rolling_K_H0": int(k),
                    "refresh": "rolling",
                }
            )
    return specs


def _tau_h0_scores(memory_scores: np.ndarray, memory_y: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    h0 = np.asarray(memory_scores, dtype=np.float64).reshape(-1)[np.asarray(memory_y).reshape(-1).astype(np.int32) == 0]
    if spec["refresh"] == "rolling" and spec.get("rolling_K_H0") not in {"", None}:
        k = int(spec["rolling_K_H0"])
        if h0.size > k:
            return h0[-k:]
    return h0


def _simulate_policy(
    *,
    dataset: str,
    method: str,
    alpha: float,
    num_blocks: int,
    spec: Dict[str, Any],
    calib_scores: np.ndarray,
    calib_y: np.ndarray,
    eval_scores: np.ndarray,
    eval_y: np.ndarray,
    orientation: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    memory_scores = np.asarray(calib_scores, dtype=np.float64).reshape(-1).copy()
    memory_y = np.asarray(calib_y, dtype=np.int32).reshape(-1).copy()
    block_indices = _split_indices(int(eval_y.size), int(num_blocks))
    cumulative_tp = cumulative_fp = cumulative_tn = cumulative_fn = 0
    alpha_cal = float(alpha) * float(spec["alpha_cal_multiplier"])

    fixed_tau: Optional[float] = None
    if spec["refresh"] == "none":
        h0_for_tau = _tau_h0_scores(memory_scores, memory_y, spec)
        fixed_tau = select_tau(h0_for_tau, alpha_cal, orientation)

    for block_index, idx in enumerate(block_indices):
        block_scores = np.asarray(eval_scores, dtype=np.float64).reshape(-1)[idx]
        block_y = np.asarray(eval_y, dtype=np.int32).reshape(-1)[idx]

        h0_for_tau = _tau_h0_scores(memory_scores, memory_y, spec)
        invalid_reason = ""
        if h0_for_tau.size == 0:
            invalid_reason = "no H0 calibration examples available for threshold selection"
            tau = float("inf") if orientation == "higher" else float("-inf")
        elif fixed_tau is not None:
            tau = fixed_tau
        else:
            tau = select_tau(h0_for_tau, alpha_cal, orientation)

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
        block_drift = None if block_h0_p95 is None or calib_h0_p95 is None else block_h0_p95 - calib_h0_p95

        block_metrics = _metrics_from_counts(tp, fp, tn, fn, "block")
        cumulative_metrics = _metrics_from_counts(cumulative_tp, cumulative_fp, cumulative_tn, cumulative_fn, "cumulative")
        row: Dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "policy": spec["policy"],
            "alpha": float(alpha),
            "alpha_cal_multiplier": float(spec["alpha_cal_multiplier"]),
            "num_blocks": int(num_blocks),
            "rolling_K_H0": spec.get("rolling_K_H0", ""),
            "block_index": int(block_index),
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
            "invalid_reason": invalid_reason,
        }
        row.update(block_metrics)
        row.update(cumulative_metrics)
        row["safe_block"] = bool((_float(row.get("block_FPR")) is not None) and float(row["block_FPR"]) <= float(alpha))
        row["safe_cumulative"] = bool(
            (_float(row.get("cumulative_FPR")) is not None) and float(row["cumulative_FPR"]) <= float(alpha)
        )
        rows.append(row)

        if spec["refresh"] != "none":
            memory_scores = np.concatenate([memory_scores, block_scores])
            memory_y = np.concatenate([memory_y, block_y])

    return rows


def build_block_metadata(pair_dirs: Sequence[Path], num_blocks_values: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair_dir in pair_dirs:
        _train, _calib, eval_pairs, config = load_pairs_dir(pair_dir)
        dataset = _dataset_from_config(config)
        seed = _seed_from_config(config)
        for num_blocks in num_blocks_values:
            for block_index, idx in enumerate(_split_indices(int(eval_pairs.y.size), int(num_blocks))):
                y = eval_pairs.y[idx]
                h1, h0 = _counts(y)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "num_blocks": int(num_blocks),
                        "block_index": int(block_index),
                        "block_start_index": int(idx[0]) if idx.size else "",
                        "block_end_index": int(idx[-1]) if idx.size else "",
                        "block_n": int(idx.size),
                        "block_H0": h0,
                        "block_H1": h1,
                    }
                )
    return rows


def build_block_rows(
    pair_dirs: Sequence[Path],
    methods: Sequence[str],
    alpha: float,
    num_blocks_values: Sequence[int],
    rolling_values: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = _policy_specs(rolling_values)
    for pair_dir in pair_dirs:
        if not pair_dir.exists():
            raise FileNotFoundError(f"Missing required pair directory: {pair_dir}")
        for required in ["online_train_pairs.npz", "online_calib_pairs.npz", "online_eval_pairs.npz", "config.json"]:
            path = pair_dir / required
            if not path.exists():
                raise FileNotFoundError(f"Missing required threshold-refresh input: {path}")
        train, calib, eval_pairs, config = load_pairs_dir(pair_dir)
        dataset = _dataset_from_config(config)
        seed = _seed_from_config(config)
        for method in methods:
            print(f"Scoring {dataset} / {method}")
            calib_scores, eval_scores, orientation = _score_bundle(method, train, calib, eval_pairs, alpha=alpha, seed=seed)
            for num_blocks in num_blocks_values:
                for spec in specs:
                    rows.extend(
                        _simulate_policy(
                            dataset=dataset,
                            method=method,
                            alpha=alpha,
                            num_blocks=int(num_blocks),
                            spec=spec,
                            calib_scores=calib_scores,
                            calib_y=calib.y,
                            eval_scores=eval_scores,
                            eval_y=eval_pairs.y,
                            orientation=orientation,
                        )
                    )
    rows.sort(
        key=lambda r: (
            str(r.get("dataset")),
            str(r.get("method")),
            str(r.get("policy")),
            _float(r.get("alpha_cal_multiplier")) or 0.0,
            int(r.get("num_blocks") or 0),
            int(r.get("rolling_K_H0") or 0) if str(r.get("rolling_K_H0", "")).strip() else 0,
            int(r.get("block_index") or 0),
        )
    )
    return rows


def build_summary(rows: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted(
        {
            (
                r.get("dataset"),
                r.get("method"),
                r.get("policy"),
                r.get("alpha_cal_multiplier"),
                r.get("num_blocks"),
                r.get("rolling_K_H0"),
            )
            for r in rows
        }
    )
    for dataset, method, policy, mult, num_blocks, rolling_k in keys:
        group = [
            r
            for r in rows
            if r.get("dataset") == dataset
            and r.get("method") == method
            and r.get("policy") == policy
            and r.get("alpha_cal_multiplier") == mult
            and r.get("num_blocks") == num_blocks
            and r.get("rolling_K_H0") == rolling_k
        ]
        group.sort(key=lambda r: int(r.get("block_index") or 0))
        if not group:
            continue
        last = group[-1]
        block_fprs = np.asarray([_float(r.get("block_FPR")) for r in group if _float(r.get("block_FPR")) is not None], dtype=float)
        block_hits = np.asarray([_float(r.get("block_hit")) for r in group if _float(r.get("block_hit")) is not None], dtype=float)
        taus = np.asarray([_float(r.get("tau")) for r in group if _float(r.get("tau")) is not None and np.isfinite(float(r.get("tau")))], dtype=float)
        unsafe = [r for r in group if r.get("safe_block") is False]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "policy": policy,
                "alpha": float(alpha),
                "alpha_cal_multiplier": mult,
                "num_blocks": num_blocks,
                "rolling_K_H0": rolling_k,
                "final_cumulative_FPR": last.get("cumulative_FPR"),
                "final_cumulative_TPR": last.get("cumulative_TPR"),
                "final_cumulative_hit": last.get("cumulative_hit"),
                "final_cumulative_precision": last.get("cumulative_precision"),
                "final_cumulative_FP_over_n": last.get("cumulative_FP_over_n"),
                "mean_block_FPR": float(np.mean(block_fprs)) if block_fprs.size else None,
                "max_block_FPR": float(np.max(block_fprs)) if block_fprs.size else None,
                "p95_block_FPR": float(np.quantile(block_fprs, 0.95)) if block_fprs.size else None,
                "fraction_safe_blocks": float(np.mean(block_fprs <= float(alpha))) if block_fprs.size else None,
                "mean_block_hit": float(np.mean(block_hits)) if block_hits.size else None,
                "min_block_hit": float(np.min(block_hits)) if block_hits.size else None,
                "max_block_hit": float(np.max(block_hits)) if block_hits.size else None,
                "total_blocks": int(len(group)),
                "unsafe_blocks": int(len(unsafe)),
                "first_unsafe_block": int(unsafe[0].get("block_index")) if unsafe else "",
                "final_tau": last.get("tau"),
                "tau_min": float(np.min(taus)) if taus.size else None,
                "tau_max": float(np.max(taus)) if taus.size else None,
                "tau_drift": float(np.max(taus) - np.min(taus)) if taus.size else None,
                "final_safe": bool((_float(last.get("cumulative_FPR")) is not None) and float(last["cumulative_FPR"]) <= float(alpha)),
                "all_blocks_safe": bool(block_fprs.size > 0 and float(np.max(block_fprs)) <= float(alpha)),
            }
        )
    return out


def _best_summary_for_family(summary: Sequence[Dict[str, Any]], dataset: str, method: str, family: str) -> Optional[Dict[str, Any]]:
    candidates = [r for r in summary if r.get("dataset") == dataset and r.get("method") == method and r.get("policy") == family]
    if not candidates:
        return None
    safe = [r for r in candidates if r.get("final_safe") is True]
    pool = safe if safe else candidates
    return max(
        pool,
        key=lambda r: (
            1 if r.get("final_safe") is True else 0,
            _float(r.get("final_cumulative_hit")) or -1.0,
            -(_float(r.get("max_block_FPR")) or 1e9),
        ),
    )


def _read_risk_baselines(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    baselines: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return baselines
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset = str(row.get("dataset", ""))
            method = str(row.get("method", ""))
            if not dataset or not method:
                continue
            baselines[(dataset, method)] = {
                "baseline_policy": f"fixed_threshold_full_data_alpha_cal_x_{row.get('alpha_cal_multiplier', '')}",
                "baseline_final_FPR": _float(row.get("full_eval_FPR")),
                "baseline_hit": _float(row.get("full_eval_hit")),
                "baseline_safe": str(row.get("reference_is_safe", "")).lower() == "true",
            }
    return baselines


def build_fixed_vs_refresh(summary: Sequence[Dict[str, Any]], risk_ref_path: Path) -> List[Dict[str, Any]]:
    baselines = _read_risk_baselines(risk_ref_path)
    out: List[Dict[str, Any]] = []
    keys = sorted({(r.get("dataset"), r.get("method")) for r in summary})
    for dataset, method in keys:
        baseline = baselines.get((str(dataset), str(method)))
        if baseline is None:
            continue
        for family in ["cumulative_refresh", "rolling_refresh", "conservative_rolling_refresh"]:
            selected = _best_summary_for_family(summary, str(dataset), str(method), family)
            if selected is None:
                continue
            baseline_fpr = _float(baseline.get("baseline_final_FPR"))
            refresh_fpr = _float(selected.get("final_cumulative_FPR"))
            baseline_hit = _float(baseline.get("baseline_hit"))
            refresh_hit = _float(selected.get("final_cumulative_hit"))
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "baseline_policy": baseline.get("baseline_policy"),
                    "refresh_policy": _policy_label(selected),
                    "baseline_final_FPR": baseline_fpr,
                    "refresh_final_FPR": refresh_fpr,
                    "baseline_hit": baseline_hit,
                    "refresh_hit": refresh_hit,
                    "baseline_safe": baseline.get("baseline_safe"),
                    "refresh_safe": selected.get("final_safe"),
                    "delta_FPR": None if baseline_fpr is None or refresh_fpr is None else refresh_fpr - baseline_fpr,
                    "delta_hit": None if baseline_hit is None or refresh_hit is None else refresh_hit - baseline_hit,
                }
            )
    return out


def _policy_label(row: Dict[str, Any]) -> str:
    label = f"{row.get('policy')}@blocks={row.get('num_blocks')},alpha_cal_x={_fmt(row.get('alpha_cal_multiplier'))}"
    if str(row.get("rolling_K_H0", "")).strip():
        label += f",K_H0={row.get('rolling_K_H0')}"
    return label


def build_success(summary: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted({(r.get("dataset"), r.get("method")) for r in summary})
    for dataset, method in keys:
        one_shot = [
            r
            for r in summary
            if r.get("dataset") == dataset
            and r.get("method") == method
            and str(r.get("policy", "")).startswith("one_shot")
        ]
        safe_one_shot = [r for r in one_shot if r.get("final_safe") is True]
        reference_is_safe = bool(safe_one_shot)
        if safe_one_shot:
            ref = max(safe_one_shot, key=lambda r: _float(r.get("final_cumulative_hit")) or -1.0)
        elif one_shot:
            ref = max(one_shot, key=lambda r: _float(r.get("final_cumulative_hit")) or -1.0)
        else:
            continue
        reference_hit = _float(ref.get("final_cumulative_hit"))
        for row in [
            r
            for r in summary
            if r.get("dataset") == dataset and r.get("method") == method
        ]:
            hit = _float(row.get("final_cumulative_hit"))
            rel = hit / reference_hit if hit is not None and reference_hit and reference_hit > 0 else None
            final_fpr = _float(row.get("final_cumulative_FPR"))
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "policy": row.get("policy"),
                    "alpha_cal_multiplier": row.get("alpha_cal_multiplier"),
                    "num_blocks": row.get("num_blocks"),
                    "rolling_K_H0": row.get("rolling_K_H0"),
                    "final_cumulative_FPR": final_fpr,
                    "final_cumulative_hit": hit,
                    "reference_hit": reference_hit,
                    "reference_is_safe": reference_is_safe,
                    "relative_hit_vs_reference": rel,
                    "fraction_safe_blocks": row.get("fraction_safe_blocks"),
                    "max_block_FPR": row.get("max_block_FPR"),
                    "success_90": bool(final_fpr is not None and final_fpr <= float(alpha) and rel is not None and rel >= 0.90),
                    "success_95": bool(final_fpr is not None and final_fpr <= float(alpha) and rel is not None and rel >= 0.95),
                }
            )
    return out


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


def _top_rows(
    summary: Sequence[Dict[str, Any]],
    dataset: str,
    method: str,
    *,
    policies: Optional[Sequence[str]] = None,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    rows = [r for r in summary if r.get("dataset") == dataset and r.get("method") == method]
    if policies is not None:
        rows = [r for r in rows if r.get("policy") in set(policies)]
    rows.sort(
        key=lambda r: (
            0 if r.get("final_safe") is True else 1,
            -(_float(r.get("final_cumulative_hit")) or -1.0),
            _float(r.get("max_block_FPR")) or 1e9,
            str(r.get("policy")),
        )
    )
    return rows[:limit]


def _summary_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    body = []
    for r in rows:
        body.append(
            [
                _policy_label(r),
                _fmt(r.get("final_cumulative_FPR")),
                _fmt(r.get("final_cumulative_hit")),
                _fmt(r.get("fraction_safe_blocks")),
                _fmt(r.get("max_block_FPR")),
                _fmt(r.get("tau_drift")),
                str(r.get("final_safe")),
                str(r.get("all_blocks_safe")),
            ]
        )
    return _table(["Policy", "Final FPR", "Final hit", "Safe blocks", "Max block FPR", "Tau drift", "Final safe", "All blocks safe"], body)


def _success_table(rows: Sequence[Dict[str, Any]], dataset: str, method: str) -> List[str]:
    group = [r for r in rows if r.get("dataset") == dataset and r.get("method") == method and (r.get("success_90") or r.get("success_95"))]
    group.sort(
        key=lambda r: (
            0 if r.get("success_95") else 1,
            -(_float(r.get("relative_hit_vs_reference")) or -1),
            _float(r.get("final_cumulative_FPR")) or 1e9,
        )
    )
    body = [
        [
            _policy_label(r),
            _fmt(r.get("final_cumulative_FPR")),
            _fmt(r.get("final_cumulative_hit")),
            _fmt(r.get("relative_hit_vs_reference")),
            _fmt(r.get("fraction_safe_blocks")),
            _fmt(r.get("max_block_FPR")),
            str(r.get("success_90")),
            str(r.get("success_95")),
        ]
        for r in group[:10]
    ]
    return _table(["Policy", "Final FPR", "Hit", "Rel hit", "Safe blocks", "Max block FPR", "S90", "S95"], body)


def _strict_block_table(rows: Sequence[Dict[str, Any]], dataset: str, method: str) -> List[str]:
    group = [
        r
        for r in rows
        if r.get("dataset") == dataset
        and r.get("method") == method
        and r.get("all_blocks_safe") is True
    ]
    group.sort(key=lambda r: -(_float(r.get("final_cumulative_hit")) or -1.0))
    body = [
        [
            _policy_label(r),
            _fmt(r.get("final_cumulative_FPR")),
            _fmt(r.get("final_cumulative_hit")),
            _fmt(r.get("max_block_FPR")),
            _fmt(r.get("fraction_safe_blocks")),
        ]
        for r in group[:8]
    ]
    return _table(["Policy", "Final FPR", "Final hit", "Max block FPR", "Safe blocks"], body)


def _best_by_family(summary: Sequence[Dict[str, Any]], dataset: str, method: str) -> Dict[str, Optional[Dict[str, Any]]]:
    return {
        family: _best_summary_for_family(summary, dataset, method, family)
        for family in ["one_shot_naive", "one_shot_conservative", "cumulative_refresh", "rolling_refresh", "conservative_rolling_refresh"]
    }


def write_markdown(path: Path, summary: Sequence[Dict[str, Any]], success: Sequence[Dict[str, Any]], fixed: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lma_we = _best_by_family(summary, "SemCacheLMArena", "ours_weighted_ensemble")
    sq_we = _best_by_family(summary, "SemCacheSearchQueries", "ours_weighted_ensemble")

    lma_best_safe = [r for r in summary if r.get("dataset") == "SemCacheLMArena" and r.get("method") == "ours_weighted_ensemble" and r.get("final_safe") is True]
    sq_best_safe = [r for r in summary if r.get("dataset") == "SemCacheSearchQueries" and r.get("method") == "ours_weighted_ensemble" and r.get("final_safe") is True]
    lma_best_safe_row = max(lma_best_safe, key=lambda r: _float(r.get("final_cumulative_hit")) or -1.0) if lma_best_safe else None
    sq_best_safe_row = max(sq_best_safe, key=lambda r: _float(r.get("final_cumulative_hit")) or -1.0) if sq_best_safe else None
    lma_all_block_safe = [
        r
        for r in summary
        if r.get("dataset") == "SemCacheLMArena"
        and r.get("method") == "ours_weighted_ensemble"
        and r.get("all_blocks_safe") is True
    ]
    lma_best_all_block_safe = (
        max(lma_all_block_safe, key=lambda r: _float(r.get("final_cumulative_hit")) or -1.0)
        if lma_all_block_safe
        else None
    )

    lines: List[str] = [
        "# Freeze-Scorer, Refresh-Threshold",
        "",
        "## 1. Executive Summary",
        "",
        "This experiment trains each scorer once on `online_train`, freezes the scorer, and refreshes only the threshold `tau` using chronological calibration evidence. The deployment constraint is candidate-level `FPR = FP / (FP + TN) <= 0.05`.",
        "",
        "The result is single-seed evidence. It should not be described as guaranteed FPR control or robustness across seeds.",
        "",
    ]
    if lma_best_safe_row is None:
        lines.append("For SemCacheLMArena / WeightedEnsemble, no refresh policy in this run produced final cumulative FPR at or below 0.05.")
    else:
        lines.append(
            "For SemCacheLMArena / WeightedEnsemble, the best safe refresh row was "
            f"`{_policy_label(lma_best_safe_row)}` with final FPR `{_fmt(lma_best_safe_row.get('final_cumulative_FPR'))}` "
            f"and hit `{_fmt(lma_best_safe_row.get('final_cumulative_hit'))}`. This is cumulative-safe, not block-safe: "
            f"its max block FPR is `{_fmt(lma_best_safe_row.get('max_block_FPR'))}`."
        )
    if lma_best_all_block_safe is not None:
        lines.append(
            "Strict all-block safety is possible for SemCacheLMArena / WeightedEnsemble only at lower utility in this run: "
            f"`{_policy_label(lma_best_all_block_safe)}` has final FPR `{_fmt(lma_best_all_block_safe.get('final_cumulative_FPR'))}`, "
            f"max block FPR `{_fmt(lma_best_all_block_safe.get('max_block_FPR'))}`, and hit `{_fmt(lma_best_all_block_safe.get('final_cumulative_hit'))}`."
        )
    if sq_best_safe_row is None:
        lines.append("For SemCacheSearchQueries / WeightedEnsemble, no refresh policy in this run produced final cumulative FPR at or below 0.05.")
    else:
        lines.append(
            "For SemCacheSearchQueries / WeightedEnsemble, the best safe refresh row was "
            f"`{_policy_label(sq_best_safe_row)}` with final FPR `{_fmt(sq_best_safe_row.get('final_cumulative_FPR'))}` "
            f"and hit `{_fmt(sq_best_safe_row.get('final_cumulative_hit'))}`."
        )

    lines.extend(
        [
            "",
            "## 2. Motivation",
            "",
            "Fixed-threshold stopping failed under chronological drift because the scorer and threshold were frozen together. The audit showed the threshold selector and FPR denominator were correct, but the later H0 score distribution shifted upward. This experiment tests whether the scorer ranking can remain useful while only `tau` is refreshed.",
            "",
            "## 3. Protocol",
            "",
            "- `online_train` fits the scorer once.",
            "- `online_calib` initializes the threshold calibration memory.",
            "- `online_eval` is split chronologically into 5, 10, or 20 blocks.",
            "- For each block, `tau` is selected from current calibration-memory H0 scores only.",
            "- The block is evaluated before its labels are added to calibration memory.",
            "- Model weights are never updated after the initial training step.",
            "",
            "For higher-is-better scores, the threshold selector is the audited empirical rule: choose the first observed threshold whose H0 accept rate under `score >= tau` is at most `alpha_cal`.",
            "",
            "## 4. Policies Compared",
            "",
            "- `one_shot_naive`: calibrate once with `alpha_cal = alpha`; no refresh.",
            "- `one_shot_conservative`: calibrate once with `alpha_cal` in `{0.7 alpha, 0.5 alpha}`; no refresh.",
            "- `cumulative_refresh`: after each block, add all revealed block labels and recalibrate from all accumulated H0 scores.",
            "- `rolling_refresh`: recalibrate from the most recent K H0 scores, with K in `{1000, 2000, 5000, 10000}`.",
            "- `conservative_rolling_refresh`: rolling refresh with `alpha_cal` in `{0.7 alpha, 0.5 alpha}`.",
            "",
            "## 5. Main Results: SemCacheLMArena / WeightedEnsemble",
            "",
        ]
    )
    lines.extend(_summary_table(_top_rows(summary, "SemCacheLMArena", "ours_weighted_ensemble", limit=12)))
    lines.extend(
        [
            "",
            "Successful rows relative to the best one-shot reference:",
            "",
        ]
    )
    lines.extend(_success_table(success, "SemCacheLMArena", "ours_weighted_ensemble"))
    lines.extend(
        [
            "",
            "## 6. Main Results: SemCacheSearchQueries / WeightedEnsemble",
            "",
        ]
    )
    lines.extend(_summary_table(_top_rows(summary, "SemCacheSearchQueries", "ours_weighted_ensemble", limit=12)))
    lines.extend(["", "Successful rows relative to the best one-shot reference:", ""])
    lines.extend(_success_table(success, "SemCacheSearchQueries", "ours_weighted_ensemble"))
    lines.extend(
        [
            "",
            "## 7. Block-Level Safety",
            "",
            "Block-level safety is stricter than cumulative deployment safety. A policy can finish with cumulative FPR below alpha while still having individual unsafe blocks.",
            "",
            "The summary CSV reports `fraction_safe_blocks`, `max_block_FPR`, `final_safe`, and `all_blocks_safe` for every policy.",
            "",
            "Best strict all-block-safe rows for SemCacheLMArena / WeightedEnsemble:",
            "",
        ]
    )
    lines.extend(_strict_block_table(summary, "SemCacheLMArena", "ours_weighted_ensemble"))
    lines.extend(
        [
            "",
            "Best strict all-block-safe rows for SemCacheSearchQueries / WeightedEnsemble:",
            "",
        ]
    )
    lines.extend(_strict_block_table(summary, "SemCacheSearchQueries", "ours_weighted_ensemble"))
    lines.extend(
        [
            "",
            "## 8. Utility Tradeoff",
            "",
            "`threshold_refresh_success.csv` compares final hit rate against the best one-shot reference for each dataset/method. If no one-shot policy is safe, the reference is marked unsafe and success is measured relative to the best available one-shot hit.",
            "",
            "## 9. Comparison To Fixed-Threshold Stopping",
            "",
            "The fixed-threshold baseline rows come from the previous risk-aware stopping report, where the final evaluation was the final half of the old `online_eval` stream. The refresh rows here report cumulative metrics over the whole old `online_eval` stream split into blocks, so this table is useful as a directional reference but is not a perfectly matched stream comparison.",
            "",
        ]
    )
    fixed_rows = [
        [
            str(r.get("dataset")),
            str(r.get("method")),
            str(r.get("refresh_policy")),
            _fmt(r.get("baseline_final_FPR")),
            _fmt(r.get("refresh_final_FPR")),
            _fmt(r.get("baseline_hit")),
            _fmt(r.get("refresh_hit")),
            str(r.get("baseline_safe")),
            str(r.get("refresh_safe")),
            _fmt(r.get("delta_FPR")),
            _fmt(r.get("delta_hit")),
        ]
        for r in fixed
        if r.get("method") == "ours_weighted_ensemble"
    ]
    lines.extend(
        _table(
            ["Dataset", "Method", "Refresh policy", "Baseline FPR", "Refresh FPR", "Baseline hit", "Refresh hit", "Baseline safe", "Refresh safe", "Delta FPR", "Delta hit"],
            fixed_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 10. Failure Modes",
            "",
            "- Refresh can improve cumulative safety while leaving individual blocks unsafe.",
            "- Rolling windows can react to drift but may be noisy if the recent H0 window is small.",
            "- Conservative thresholds reduce hits and may still fail if the next block shifts further upward.",
            "- This run uses seed 42 only.",
            "",
            "## 11. Supported Claims",
            "",
            "- This experiment correctly separates scorer training from threshold refresh.",
            "- Thresholds are refreshed using past H0 evidence only; future block labels are not used for current-block threshold selection.",
            "- Any claim about improvement should be tied to the specific dataset, method, policy, block count, and rolling window reported in the tables.",
            "",
            "## 12. Unsupported Claims",
            "",
            "- Do not write that FPR control is guaranteed; block-level violations may remain.",
            "- Do not write that the method is robust until multiple seeds are run.",
            "- Do not claim vCache is beaten generally; this experiment does not evaluate vCache.",
            "",
            "## 13. Next Experiments",
            "",
            "- Rerun the threshold-refresh protocol for seeds 43-46.",
            "- Test shorter calibration refresh intervals or adaptive block sizes.",
            "- Add confidence-bound calibration on recent H0 windows.",
            "- Compare threshold refresh to retraining or local/context-conditioned thresholds.",
            "",
            "## Explicit Answers",
            "",
            "A. Does refreshing tau fix LMArena final FPR? "
            + (
                f"Yes for the checked full-stream cumulative criterion: `{_policy_label(lma_best_safe_row)}` reaches final FPR `{_fmt(lma_best_safe_row.get('final_cumulative_FPR'))}`. It does not solve strict block-level control because max block FPR is `{_fmt(lma_best_safe_row.get('max_block_FPR'))}`."
                if lma_best_safe_row
                else "No checked refresh policy reaches final cumulative FPR <= 0.05."
            ),
            "B. Is cumulative refresh enough, or is rolling refresh better? "
            + _family_answer(lma_we, "cumulative_refresh", "rolling_refresh")
            + " For LMArena, one-shot conservative calibration is also cumulative-safe on the full old eval stream, so refresh should not be described as strictly necessary for cumulative FPR in this run.",
            "C. Is conservative rolling refresh needed? "
            + _conservative_answer(lma_we)
            + " The highest-hit safe LMArena row is conservative rolling, but lower-hit non-conservative rolling is also cumulative-safe.",
            "D. Hit-rate loss relative to fixed threshold is reported in `fixed_vs_refresh_comparison.csv`; use the row matching the selected refresh policy.",
            "E. All-block safety is stricter than cumulative safety. See `all_blocks_safe` and `max_block_FPR`; do not infer block-level control from final cumulative FPR alone.",
            "F. SearchQueries remains safe only for the policies marked `final_safe`; the success table gives the corresponding utility relative to one-shot.",
            "G. The best rolling K is policy-specific; choose the safe row with the highest hit in the summary table rather than a single universal K.",
            "H. Tau drift is reported as `tau_drift` and plotted in `plot_tau_over_time.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _family_answer(families: Dict[str, Optional[Dict[str, Any]]], a: str, b: str) -> str:
    row_a = families.get(a)
    row_b = families.get(b)
    if row_a is None or row_b is None:
        return "The comparison is unavailable for one of the policy families."
    a_safe = row_a.get("final_safe") is True
    b_safe = row_b.get("final_safe") is True
    if a_safe and not b_safe:
        return f"`{a}` has a safe selected row while `{b}` does not."
    if b_safe and not a_safe:
        return f"`{b}` has a safe selected row while `{a}` does not."
    hit_a = _float(row_a.get("final_cumulative_hit")) or -1.0
    hit_b = _float(row_b.get("final_cumulative_hit")) or -1.0
    if abs(hit_a - hit_b) < 1e-12:
        return "The selected rows have the same final hit."
    return f"`{a}` has higher selected hit." if hit_a > hit_b else f"`{b}` has higher selected hit."


def _conservative_answer(families: Dict[str, Optional[Dict[str, Any]]]) -> str:
    conservative = families.get("conservative_rolling_refresh")
    rolling = families.get("rolling_refresh")
    if conservative is None:
        return "No conservative rolling row was available."
    if conservative.get("final_safe") is True and (rolling is None or rolling.get("final_safe") is not True):
        return "Yes for cumulative safety in this selected comparison: conservative rolling is safe while non-conservative rolling is not."
    if conservative.get("final_safe") is True:
        return "It is one safe option, but non-conservative rolling may also be safe in this run."
    return "No selected conservative rolling row is final-safe in this run."


def write_plot_csvs(out_dir: Path, block_rows: Sequence[Dict[str, Any]], summary_rows: Sequence[Dict[str, Any]]) -> List[Path]:
    paths: List[Path] = []
    common = ["dataset", "method", "policy", "alpha", "alpha_cal_multiplier", "num_blocks", "rolling_K_H0", "block_index"]

    path = out_dir / "plot_block_fpr.csv"
    rows = [{**{k: r.get(k) for k in common}, "block_FPR": r.get("block_FPR")} for r in block_rows]
    _write_csv(path, rows, [*common, "block_FPR"])
    paths.append(path)

    path = out_dir / "plot_cumulative_fpr.csv"
    rows = [{**{k: r.get(k) for k in common}, "cumulative_FPR": r.get("cumulative_FPR")} for r in block_rows]
    _write_csv(path, rows, [*common, "cumulative_FPR"])
    paths.append(path)

    path = out_dir / "plot_hit_over_time.csv"
    rows = []
    for r in block_rows:
        base = {k: r.get(k) for k in common}
        rows.append({**base, "metric": "block_hit", "hit": r.get("block_hit")})
        rows.append({**base, "metric": "cumulative_hit", "hit": r.get("cumulative_hit")})
    _write_csv(path, rows, [*common, "metric", "hit"])
    paths.append(path)

    path = out_dir / "plot_tau_over_time.csv"
    rows = [{**{k: r.get(k) for k in common}, "tau": r.get("tau")} for r in block_rows]
    _write_csv(path, rows, [*common, "tau"])
    paths.append(path)

    path = out_dir / "plot_h0_p95_drift.csv"
    rows = [{**{k: r.get(k) for k in common}, "block_minus_calib_H0_p95": r.get("block_minus_calib_H0_p95")} for r in block_rows]
    _write_csv(path, rows, [*common, "block_minus_calib_H0_p95"])
    paths.append(path)

    path = out_dir / "plot_policy_comparison.csv"
    fields = [
        "dataset",
        "method",
        "policy",
        "alpha",
        "alpha_cal_multiplier",
        "num_blocks",
        "rolling_K_H0",
        "metric",
        "value",
        "final_safe",
        "all_blocks_safe",
    ]
    rows = []
    for r in summary_rows:
        base = {
            "dataset": r.get("dataset"),
            "method": r.get("method"),
            "policy": r.get("policy"),
            "alpha": r.get("alpha"),
            "alpha_cal_multiplier": r.get("alpha_cal_multiplier"),
            "num_blocks": r.get("num_blocks"),
            "rolling_K_H0": r.get("rolling_K_H0"),
            "final_safe": r.get("final_safe"),
            "all_blocks_safe": r.get("all_blocks_safe"),
        }
        rows.append({**base, "metric": "final_cumulative_FPR", "value": r.get("final_cumulative_FPR")})
        rows.append({**base, "metric": "final_cumulative_hit", "value": r.get("final_cumulative_hit")})
    _write_csv(path, rows, fields)
    paths.append(path)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    pair_dirs = [Path(p) for p in args.pair_dirs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_metadata = build_block_metadata(pair_dirs, args.num_blocks)
    block_rows = build_block_rows(pair_dirs, args.methods, float(args.alpha), args.num_blocks, args.rolling_K_H0)
    summary_rows = build_summary(block_rows, float(args.alpha))
    fixed_rows = build_fixed_vs_refresh(summary_rows, Path("results/risk_aware_stopping/safe_full_data_reference.csv"))
    success_rows = build_success(summary_rows, float(args.alpha))

    outputs: List[Path] = []
    path = out_dir / "block_metadata.csv"
    _write_csv(path, block_metadata, BLOCK_METADATA_FIELDS)
    outputs.append(path)
    path = out_dir / "threshold_refresh_blocks.csv"
    _write_csv(path, block_rows, BLOCK_FIELDS)
    outputs.append(path)
    path = out_dir / "threshold_refresh_summary.csv"
    _write_csv(path, summary_rows, SUMMARY_FIELDS)
    outputs.append(path)
    path = out_dir / "fixed_vs_refresh_comparison.csv"
    _write_csv(path, fixed_rows, FIXED_VS_REFRESH_FIELDS)
    outputs.append(path)
    path = out_dir / "threshold_refresh_success.csv"
    _write_csv(path, success_rows, SUCCESS_FIELDS)
    outputs.append(path)
    path = out_dir / "threshold_refresh_report.md"
    write_markdown(path, summary_rows, success_rows, fixed_rows)
    outputs.append(path)
    outputs.extend(write_plot_csvs(out_dir, block_rows, summary_rows))

    print("Wrote threshold-refresh outputs:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
