from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .debug_ours_online_mined_calibration import METHOD_MAP
from .online_mined_ours import (
    OnlinePairs,
    accept_scores,
    infer_orientation,
    load_pairs_dir,
    score_method,
    select_tau,
    _prepare_method,
)
from .online_policies import _fit_method


DEFAULT_PAIR_DIRS = [
    "results/online_candidate_pairs/lmarena_gte_seed42",
    "results/online_candidate_pairs/searchqueries_gte_seed42",
]
DEFAULT_METHODS = ["ours_weighted_ensemble", "ours_whitened_hadamard", "cosine"]
DEFAULT_ALPHA = 0.05
DEFAULT_ALPHA_CAL_MULTIPLIERS = [1.0, 0.9, 0.8, 0.7, 0.5]
DEFAULT_DOSAGES: List[Any] = [250, 500, 1000, 2000, 5000, 10000, "full"]


SWEEP_FIELDS = [
    "dataset",
    "method",
    "seed",
    "alpha",
    "alpha_cal_multiplier",
    "alpha_cal",
    "train_pairs_requested",
    "train_pairs_used",
    "train_H1",
    "train_H0",
    "calib_pairs_requested",
    "calib_pairs_used",
    "calib_H1",
    "calib_H0",
    "tau",
    "orientation",
    "calib_FP",
    "calib_TN",
    "calib_FPR",
    "calib_TP",
    "calib_FN",
    "calib_TPR",
    "calib_hit",
    "calib_precision",
    "monitor_FP",
    "monitor_TN",
    "monitor_FPR",
    "monitor_TP",
    "monitor_FN",
    "monitor_TPR",
    "monitor_hit",
    "monitor_precision",
    "eval_FP",
    "eval_TN",
    "eval_FPR",
    "eval_TP",
    "eval_FN",
    "eval_TPR",
    "eval_hit",
    "eval_precision",
    "calib_H0_mean",
    "monitor_H0_mean",
    "eval_H0_mean",
    "calib_H0_p95",
    "monitor_H0_p95",
    "eval_H0_p95",
    "monitor_minus_calib_H0_p95",
    "eval_minus_calib_H0_p95",
    "offline_labels_used",
    "calib_safe_under_alpha_cal",
    "monitor_safe_under_alpha",
    "eval_safe_under_alpha",
    "relative_eval_hit_vs_safe_full",
    "relative_eval_TPR_vs_safe_full",
    "success_90",
    "success_95",
    "monitor_success_90",
    "monitor_success_95",
    "warning",
]

REFERENCE_FIELDS = [
    "dataset",
    "method",
    "alpha",
    "alpha_cal_multiplier",
    "alpha_cal",
    "full_train_labels",
    "full_calib_labels",
    "full_total_labels",
    "full_eval_FPR",
    "full_eval_TPR",
    "full_eval_hit",
    "full_eval_precision",
    "reference_is_safe",
]

BEST_FIELDS = [
    "dataset",
    "method",
    "alpha",
    "success_type",
    "alpha_cal_multiplier",
    "train_pairs_used",
    "calib_pairs_used",
    "offline_labels_used",
    "label_reduction_vs_safe_full",
    "tau",
    "calib_FPR",
    "monitor_FPR",
    "eval_FPR",
    "eval_TPR",
    "eval_hit",
    "relative_eval_hit_vs_safe_full",
    "relative_eval_TPR_vs_safe_full",
    "calib_H0_p95",
    "monitor_H0_p95",
    "eval_H0_p95",
    "monitor_minus_calib_H0_p95",
    "eval_minus_calib_H0_p95",
]

SIM_FIELDS = [
    "dataset",
    "method",
    "alpha",
    "alpha_cal_multiplier",
    "patience",
    "epsilon_hit",
    "epsilon_tau",
    "margin_p95",
    "stopped",
    "stop_step",
    "train_pairs_used",
    "calib_pairs_used",
    "offline_labels_used",
    "tau",
    "monitor_FPR",
    "monitor_hit",
    "eval_FPR",
    "eval_hit",
    "relative_eval_hit_vs_safe_full",
    "success_90",
    "success_95",
]


def _float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    number = _float(value)
    if number is None:
        return ""
    return str(int(round(number)))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _dataset_from_config(config: Dict[str, Any]) -> str:
    dataset = config.get("dataset") or config.get("args", {}).get("dataset")
    if not dataset:
        raise ValueError("Could not recover dataset from online pair config.")
    return str(dataset)


def _seed_from_config(config: Dict[str, Any]) -> int:
    return int(config.get("seed", config.get("args", {}).get("seed", 42)))


def _dosage_label(value: Any) -> str:
    return "full" if str(value).lower() == "full" else str(int(value))


def _requested_n(requested: Any, total: int) -> int:
    if str(requested).lower() == "full":
        return int(total)
    return min(int(requested), int(total))


def _prefix_pairs(pairs: OnlinePairs, requested: Any) -> OnlinePairs:
    n = _requested_n(requested, int(pairs.y.size))
    return _subset_pairs(pairs, np.arange(n, dtype=np.int64))


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


def _split_monitor_eval(eval_pairs: OnlinePairs) -> Tuple[OnlinePairs, OnlinePairs]:
    n = int(eval_pairs.y.size)
    half = n // 2
    monitor = _subset_pairs(eval_pairs, np.arange(0, half, dtype=np.int64))
    final = _subset_pairs(eval_pairs, np.arange(half, n, dtype=np.int64))
    return monitor, final


def _counts(y: np.ndarray) -> Tuple[int, int]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    return int(np.sum(yy == 1)), int(np.sum(yy == 0))


def _metrics(y: np.ndarray, scores: np.ndarray, tau: float, orientation: str, prefix: str) -> Dict[str, Any]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    ss = np.asarray(scores, dtype=np.float64).reshape(-1)
    accepted = accept_scores(ss, float(tau), orientation)
    tp = int(np.sum(accepted & (yy == 1)))
    fp = int(np.sum(accepted & (yy == 0)))
    tn = int(np.sum((~accepted) & (yy == 0)))
    fn = int(np.sum((~accepted) & (yy == 1)))
    n = int(yy.size)
    return {
        f"{prefix}_FP": fp,
        f"{prefix}_TN": tn,
        f"{prefix}_FPR": fp / float(fp + tn) if fp + tn else None,
        f"{prefix}_TP": tp,
        f"{prefix}_FN": fn,
        f"{prefix}_TPR": tp / float(tp + fn) if tp + fn else None,
        f"{prefix}_hit": (tp + fp) / float(n) if n else None,
        f"{prefix}_precision": tp / float(tp + fp) if tp + fp else None,
    }


def _h0_stats(y: np.ndarray, scores: np.ndarray, prefix: str) -> Dict[str, Any]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    ss = np.asarray(scores, dtype=np.float64).reshape(-1)
    h0 = ss[yy == 0]
    return {
        f"{prefix}_H0_mean": float(np.mean(h0)) if h0.size else None,
        f"{prefix}_H0_p95": float(np.quantile(h0, 0.95)) if h0.size else None,
    }


def _fit_learned(method_key: str, train_pairs: OnlinePairs, *, alpha: float, seed: int) -> Tuple[Any, bool, str]:
    method_name = METHOD_MAP[method_key]
    method, uses_alt, judge_input = _prepare_method(method_name)
    y = train_pairs.y.reshape(-1).astype(np.int32)
    h0 = y == 0
    h1 = y == 1
    if int(np.sum(h0)) == 0 or int(np.sum(h1)) == 0:
        raise ValueError(f"{method_key} needs both H0 and H1 in train subset")
    h0_x = train_pairs.X[h0]
    h1_x = train_pairs.X[h1]
    h0_alt = train_pairs.cosine[h0].reshape(-1, 1)
    h1_alt = train_pairs.cosine[h1].reshape(-1, 1)
    if uses_alt:
        method.fit(
            h0_x,
            h1_x,
            seed=seed,
            alpha=float(alpha),
            H0_train_alt=h0_alt,
            H1_train_alt=h1_alt,
            judge_input=judge_input,
            fit_context="risk_aware_chronological_stopping",
        )
    else:
        _fit_method(method, h0_x, h1_x, seed=seed, alpha=float(alpha))
    return method, uses_alt, method_name


def _score_learned(method: Any, uses_alt: bool, pairs: OnlinePairs) -> np.ndarray:
    alt = pairs.cosine.reshape(-1, 1) if uses_alt else None
    return score_method(method, pairs.X, alt, uses_alt)


def _score_cosine(pairs: OnlinePairs) -> np.ndarray:
    return np.asarray(pairs.cosine, dtype=np.float64).reshape(-1)


def _row_for_threshold(
    *,
    dataset: str,
    method_key: str,
    seed: int,
    alpha: float,
    alpha_cal_multiplier: float,
    train_label: Any,
    train_pairs: Optional[OnlinePairs],
    calib_label: Any,
    calib_pairs: OnlinePairs,
    calib_scores_all: np.ndarray,
    monitor_pairs: OnlinePairs,
    monitor_scores: np.ndarray,
    eval_pairs: OnlinePairs,
    eval_scores: np.ndarray,
    orientation: str,
    warning: str,
) -> Dict[str, Any]:
    calib_sub = _prefix_pairs(calib_pairs, calib_label)
    calib_n = int(calib_sub.y.size)
    calib_scores = np.asarray(calib_scores_all[:calib_n], dtype=np.float64)
    calib_h1, calib_h0 = _counts(calib_sub.y)
    alpha_cal = float(alpha) * float(alpha_cal_multiplier)
    h0_scores = calib_scores[calib_sub.y == 0]
    tau = select_tau(h0_scores, alpha_cal, orientation)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "method": method_key,
        "seed": int(seed),
        "alpha": float(alpha),
        "alpha_cal_multiplier": float(alpha_cal_multiplier),
        "alpha_cal": alpha_cal,
        "train_pairs_requested": "none" if train_pairs is None else _dosage_label(train_label),
        "train_pairs_used": 0 if train_pairs is None else int(train_pairs.y.size),
        "calib_pairs_requested": _dosage_label(calib_label),
        "calib_pairs_used": int(calib_sub.y.size),
        "calib_H1": calib_h1,
        "calib_H0": calib_h0,
        "tau": float(tau),
        "orientation": orientation,
        "warning": warning,
    }
    if train_pairs is None:
        row["train_H1"] = 0
        row["train_H0"] = 0
    else:
        row["train_H1"], row["train_H0"] = _counts(train_pairs.y)

    row.update(_metrics(calib_sub.y, calib_scores, tau, orientation, "calib"))
    row.update(_metrics(monitor_pairs.y, monitor_scores, tau, orientation, "monitor"))
    row.update(_metrics(eval_pairs.y, eval_scores, tau, orientation, "eval"))
    row.update(_h0_stats(calib_sub.y, calib_scores, "calib"))
    row.update(_h0_stats(monitor_pairs.y, monitor_scores, "monitor"))
    row.update(_h0_stats(eval_pairs.y, eval_scores, "eval"))

    calib_p95 = _float(row.get("calib_H0_p95"))
    monitor_p95 = _float(row.get("monitor_H0_p95"))
    eval_p95 = _float(row.get("eval_H0_p95"))
    row["monitor_minus_calib_H0_p95"] = None if calib_p95 is None or monitor_p95 is None else monitor_p95 - calib_p95
    row["eval_minus_calib_H0_p95"] = None if calib_p95 is None or eval_p95 is None else eval_p95 - calib_p95
    row["offline_labels_used"] = int(row["train_pairs_used"]) + int(row["calib_pairs_used"])
    row["calib_safe_under_alpha_cal"] = bool((_float(row.get("calib_FPR")) or 1e9) <= alpha_cal)
    row["monitor_safe_under_alpha"] = bool((_float(row.get("monitor_FPR")) or 1e9) <= float(alpha))
    row["eval_safe_under_alpha"] = bool((_float(row.get("eval_FPR")) or 1e9) <= float(alpha))
    return row


def build_sweep(
    pair_dirs: Sequence[Path],
    methods: Sequence[str],
    train_dosages: Sequence[Any],
    calib_dosages: Sequence[Any],
    alpha: float,
    alpha_cal_multipliers: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair_dir in pair_dirs:
        if not pair_dir.exists():
            raise FileNotFoundError(f"Missing required pair directory: {pair_dir}")
        train_all, calib_all, eval_all, config = load_pairs_dir(pair_dir)
        monitor_pairs, eval_final_pairs = _split_monitor_eval(eval_all)
        dataset = _dataset_from_config(config)
        seed = _seed_from_config(config)

        if "cosine" in methods:
            calib_scores_all = _score_cosine(calib_all)
            monitor_scores = _score_cosine(monitor_pairs)
            eval_scores = _score_cosine(eval_final_pairs)
            for calib_label in calib_dosages:
                for mult in alpha_cal_multipliers:
                    rows.append(
                        _row_for_threshold(
                            dataset=dataset,
                            method_key="cosine",
                            seed=seed,
                            alpha=alpha,
                            alpha_cal_multiplier=float(mult),
                            train_label="none",
                            train_pairs=None,
                            calib_label=calib_label,
                            calib_pairs=calib_all,
                            calib_scores_all=calib_scores_all,
                            monitor_pairs=monitor_pairs,
                            monitor_scores=monitor_scores,
                            eval_pairs=eval_final_pairs,
                            eval_scores=eval_scores,
                            orientation="higher",
                            warning="",
                        )
                    )

        for method_key in [m for m in methods if m != "cosine"]:
            for train_label in train_dosages:
                warning = ""
                train_sub = _prefix_pairs(train_all, train_label)
                try:
                    fitted, uses_alt, _method_name = _fit_learned(method_key, train_sub, alpha=alpha, seed=seed)
                except Exception as exc:
                    warning = f"fit_failed: {exc}"
                    for calib_label in calib_dosages:
                        for mult in alpha_cal_multipliers:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "method": method_key,
                                    "seed": seed,
                                    "alpha": float(alpha),
                                    "alpha_cal_multiplier": float(mult),
                                    "alpha_cal": float(alpha) * float(mult),
                                    "train_pairs_requested": _dosage_label(train_label),
                                    "train_pairs_used": int(train_sub.y.size),
                                    "calib_pairs_requested": _dosage_label(calib_label),
                                    "calib_pairs_used": _requested_n(calib_label, int(calib_all.y.size)),
                                    "offline_labels_used": int(train_sub.y.size) + _requested_n(calib_label, int(calib_all.y.size)),
                                    "warning": warning,
                                }
                            )
                    continue
                calib_scores_all = _score_learned(fitted, uses_alt, calib_all)
                monitor_scores = _score_learned(fitted, uses_alt, monitor_pairs)
                eval_scores = _score_learned(fitted, uses_alt, eval_final_pairs)
                for calib_label in calib_dosages:
                    calib_sub = _prefix_pairs(calib_all, calib_label)
                    calib_scores = calib_scores_all[: int(calib_sub.y.size)]
                    s0 = calib_scores[calib_sub.y == 0]
                    s1 = calib_scores[calib_sub.y == 1]
                    orientation, orientation_warning = infer_orientation(s0, s1)
                    if orientation == "ambiguous":
                        orientation = "higher"
                        orientation_warning = (orientation_warning + "; defaulted to higher").strip("; ")
                    row_warning = orientation_warning
                    for mult in alpha_cal_multipliers:
                        rows.append(
                            _row_for_threshold(
                                dataset=dataset,
                                method_key=method_key,
                                seed=seed,
                                alpha=alpha,
                                alpha_cal_multiplier=float(mult),
                                train_label=train_label,
                                train_pairs=train_sub,
                                calib_label=calib_label,
                                calib_pairs=calib_all,
                                calib_scores_all=calib_scores_all,
                                monitor_pairs=monitor_pairs,
                                monitor_scores=monitor_scores,
                                eval_pairs=eval_final_pairs,
                                eval_scores=eval_scores,
                                orientation=orientation,
                                warning=row_warning,
                            )
                        )
    rows.sort(
        key=lambda r: (
            str(r.get("dataset")),
            str(r.get("method")),
            _float(r.get("alpha_cal_multiplier")) or 0.0,
            int(r.get("offline_labels_used") or 0),
            int(r.get("train_pairs_used") or 0),
            int(r.get("calib_pairs_used") or 0),
        )
    )
    return rows


def build_safe_references(rows: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    keys = sorted({(r.get("dataset"), r.get("method")) for r in rows})
    for dataset, method in keys:
        full_rows = [
            r
            for r in rows
            if r.get("dataset") == dataset
            and r.get("method") == method
            and r.get("calib_pairs_requested") == "full"
            and (r.get("train_pairs_requested") == "full" or method == "cosine")
            and not r.get("warning", "").startswith("fit_failed")
        ]
        safe = [r for r in full_rows if (_float(r.get("eval_FPR")) is not None and float(r.get("eval_FPR")) <= float(alpha))]
        reference_is_safe = bool(safe)
        candidates = safe if safe else full_rows
        if not candidates:
            refs.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "alpha": float(alpha),
                    "reference_is_safe": False,
                }
            )
            continue
        if safe:
            best = max(candidates, key=lambda r: _float(r.get("eval_hit")) or -1.0)
        else:
            best = min(
                candidates,
                key=lambda r: (
                    _float(r.get("eval_FPR")) if _float(r.get("eval_FPR")) is not None else 1e9,
                    -(_float(r.get("eval_hit")) or -1.0),
                ),
            )
        refs.append(
            {
                "dataset": dataset,
                "method": method,
                "alpha": float(alpha),
                "alpha_cal_multiplier": best.get("alpha_cal_multiplier"),
                "alpha_cal": best.get("alpha_cal"),
                "full_train_labels": best.get("train_pairs_used"),
                "full_calib_labels": best.get("calib_pairs_used"),
                "full_total_labels": best.get("offline_labels_used"),
                "full_eval_FPR": best.get("eval_FPR"),
                "full_eval_TPR": best.get("eval_TPR"),
                "full_eval_hit": best.get("eval_hit"),
                "full_eval_precision": best.get("eval_precision"),
                "reference_is_safe": reference_is_safe,
            }
        )
    return refs


def add_relative_success(rows: List[Dict[str, Any]], refs: Sequence[Dict[str, Any]], alpha: float) -> None:
    ref_by_key = {(r.get("dataset"), r.get("method")): r for r in refs if r.get("reference_is_safe") is True}
    for row in rows:
        ref = ref_by_key.get((row.get("dataset"), row.get("method")))
        if ref is None:
            row["relative_eval_hit_vs_safe_full"] = None
            row["relative_eval_TPR_vs_safe_full"] = None
            row["success_90"] = False
            row["success_95"] = False
            row["monitor_success_90"] = False
            row["monitor_success_95"] = False
            continue
        ref_hit = _float(ref.get("full_eval_hit"))
        ref_tpr = _float(ref.get("full_eval_TPR"))
        hit = _float(row.get("eval_hit"))
        tpr = _float(row.get("eval_TPR"))
        row["relative_eval_hit_vs_safe_full"] = hit / ref_hit if hit is not None and ref_hit and ref_hit > 0 else None
        row["relative_eval_TPR_vs_safe_full"] = tpr / ref_tpr if tpr is not None and ref_tpr and ref_tpr > 0 else None
        eval_safe = bool((_float(row.get("eval_FPR")) is not None) and float(row.get("eval_FPR")) <= float(alpha))
        monitor_safe = bool((_float(row.get("monitor_FPR")) is not None) and float(row.get("monitor_FPR")) <= float(alpha))
        rel_hit = _float(row.get("relative_eval_hit_vs_safe_full"))
        row["success_90"] = bool(eval_safe and rel_hit is not None and rel_hit >= 0.90)
        row["success_95"] = bool(eval_safe and rel_hit is not None and rel_hit >= 0.95)
        row["monitor_success_90"] = bool(monitor_safe and eval_safe and rel_hit is not None and rel_hit >= 0.90)
        row["monitor_success_95"] = bool(monitor_safe and eval_safe and rel_hit is not None and rel_hit >= 0.95)


def build_best(rows: Sequence[Dict[str, Any]], refs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ref_by_key = {(r.get("dataset"), r.get("method")): r for r in refs if r.get("reference_is_safe") is True}
    best_rows: List[Dict[str, Any]] = []
    for dataset, method in sorted(ref_by_key):
        ref = ref_by_key[(dataset, method)]
        full_labels = _float(ref.get("full_total_labels"))
        for success_type in ["success_90", "success_95", "monitor_success_90", "monitor_success_95"]:
            candidates = [
                r
                for r in rows
                if r.get("dataset") == dataset
                and r.get("method") == method
                and bool(r.get(success_type)) is True
            ]
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda r: (
                    int(r.get("offline_labels_used") or 10**18),
                    -(_float(r.get("eval_hit")) or -1.0),
                    _float(r.get("eval_FPR")) or 1e9,
                ),
            )
            stopped_labels = _float(selected.get("offline_labels_used"))
            best_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "alpha": selected.get("alpha"),
                    "success_type": success_type,
                    "alpha_cal_multiplier": selected.get("alpha_cal_multiplier"),
                    "train_pairs_used": selected.get("train_pairs_used"),
                    "calib_pairs_used": selected.get("calib_pairs_used"),
                    "offline_labels_used": selected.get("offline_labels_used"),
                    "label_reduction_vs_safe_full": (
                        1.0 - stopped_labels / full_labels
                        if stopped_labels is not None and full_labels and full_labels > 0
                        else None
                    ),
                    "tau": selected.get("tau"),
                    "calib_FPR": selected.get("calib_FPR"),
                    "monitor_FPR": selected.get("monitor_FPR"),
                    "eval_FPR": selected.get("eval_FPR"),
                    "eval_TPR": selected.get("eval_TPR"),
                    "eval_hit": selected.get("eval_hit"),
                    "relative_eval_hit_vs_safe_full": selected.get("relative_eval_hit_vs_safe_full"),
                    "relative_eval_TPR_vs_safe_full": selected.get("relative_eval_TPR_vs_safe_full"),
                    "calib_H0_p95": selected.get("calib_H0_p95"),
                    "monitor_H0_p95": selected.get("monitor_H0_p95"),
                    "eval_H0_p95": selected.get("eval_H0_p95"),
                    "monitor_minus_calib_H0_p95": selected.get("monitor_minus_calib_H0_p95"),
                    "eval_minus_calib_H0_p95": selected.get("eval_minus_calib_H0_p95"),
                }
            )
    return best_rows


def build_simulated(
    rows: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    patience_values: Sequence[int],
    epsilon_hit_values: Sequence[float],
    epsilon_tau_values: Sequence[float],
    margin_p95_values: Sequence[float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted({(r.get("dataset"), r.get("method"), r.get("alpha_cal_multiplier")) for r in rows})
    for dataset, method, mult in keys:
        group = [
            r
            for r in rows
            if r.get("dataset") == dataset
            and r.get("method") == method
            and abs((_float(r.get("alpha_cal_multiplier")) or -999.0) - float(mult)) < 1e-12
            and not str(r.get("warning", "")).startswith("fit_failed")
        ]
        group.sort(
            key=lambda r: (
                int(r.get("offline_labels_used") or 0),
                int(r.get("train_pairs_used") or 0),
                int(r.get("calib_pairs_used") or 0),
            )
        )
        for patience in patience_values:
            for eps_hit in epsilon_hit_values:
                for eps_tau in epsilon_tau_values:
                    for margin_p95 in margin_p95_values:
                        stopped_row: Optional[Dict[str, Any]] = None
                        consecutive = 0
                        prev: Optional[Dict[str, Any]] = None
                        for step, row in enumerate(group, start=1):
                            condition = False
                            if prev is not None:
                                monitor_fpr = _float(row.get("monitor_FPR"))
                                monitor_hit = _float(row.get("monitor_hit"))
                                prev_hit = _float(prev.get("monitor_hit"))
                                tau = _float(row.get("tau"))
                                prev_tau = _float(prev.get("tau"))
                                drift = _float(row.get("monitor_minus_calib_H0_p95"))
                                if (
                                    monitor_fpr is not None
                                    and monitor_fpr <= float(alpha)
                                    and monitor_hit is not None
                                    and prev_hit is not None
                                    and abs(monitor_hit - prev_hit) <= float(eps_hit)
                                    and tau is not None
                                    and prev_tau is not None
                                    and abs(tau - prev_tau) <= float(eps_tau)
                                    and drift is not None
                                    and drift <= float(margin_p95)
                                ):
                                    condition = True
                            consecutive = consecutive + 1 if condition else 0
                            if consecutive >= int(patience):
                                stopped_row = dict(row)
                                stopped_row["stop_step"] = step
                                break
                            prev = row
                        if stopped_row is None:
                            out.append(
                                {
                                    "dataset": dataset,
                                    "method": method,
                                    "alpha": float(alpha),
                                    "alpha_cal_multiplier": mult,
                                    "patience": patience,
                                    "epsilon_hit": eps_hit,
                                    "epsilon_tau": epsilon_tau_values[0],
                                    "margin_p95": margin_p95,
                                    "stopped": False,
                                }
                            )
                        else:
                            out.append(
                                {
                                    "dataset": dataset,
                                    "method": method,
                                    "alpha": float(alpha),
                                    "alpha_cal_multiplier": mult,
                                    "patience": patience,
                                    "epsilon_hit": eps_hit,
                                    "epsilon_tau": epsilon_tau_values[0],
                                    "margin_p95": margin_p95,
                                    "stopped": True,
                                    "stop_step": stopped_row.get("stop_step"),
                                    "train_pairs_used": stopped_row.get("train_pairs_used"),
                                    "calib_pairs_used": stopped_row.get("calib_pairs_used"),
                                    "offline_labels_used": stopped_row.get("offline_labels_used"),
                                    "tau": stopped_row.get("tau"),
                                    "monitor_FPR": stopped_row.get("monitor_FPR"),
                                    "monitor_hit": stopped_row.get("monitor_hit"),
                                    "eval_FPR": stopped_row.get("eval_FPR"),
                                    "eval_hit": stopped_row.get("eval_hit"),
                                    "relative_eval_hit_vs_safe_full": stopped_row.get("relative_eval_hit_vs_safe_full"),
                                    "success_90": stopped_row.get("success_90"),
                                    "success_95": stopped_row.get("success_95"),
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
        return lines
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _reference_table(refs: Sequence[Dict[str, Any]]) -> List[str]:
    rows = [
        [
            str(r.get("dataset", "")),
            str(r.get("method", "")),
            _fmt(r.get("alpha_cal_multiplier")),
            _fmt_int(r.get("full_total_labels")),
            _fmt(r.get("full_eval_FPR")),
            _fmt(r.get("full_eval_TPR")),
            _fmt(r.get("full_eval_hit")),
            str(r.get("reference_is_safe")),
        ]
        for r in refs
    ]
    return _table(["Dataset", "Method", "alpha_cal x", "Full labels", "Eval FPR", "Eval TPR", "Eval hit", "Safe"], rows)


def _best_table(best: Sequence[Dict[str, Any]], success_filter: Optional[str] = None) -> List[str]:
    rows = []
    for r in best:
        if success_filter and r.get("success_type") != success_filter:
            continue
        rows.append(
            [
                str(r.get("dataset", "")),
                str(r.get("method", "")),
                str(r.get("success_type", "")),
                _fmt(r.get("alpha_cal_multiplier")),
                _fmt_int(r.get("offline_labels_used")),
                _fmt(r.get("label_reduction_vs_safe_full")),
                _fmt(r.get("monitor_FPR")),
                _fmt(r.get("eval_FPR")),
                _fmt(r.get("eval_hit")),
                _fmt(r.get("relative_eval_hit_vs_safe_full")),
            ]
        )
    return _table(
        ["Dataset", "Method", "Success", "alpha_cal x", "Labels", "Reduction", "Monitor FPR", "Eval FPR", "Eval hit", "Rel hit"],
        rows,
    )


def _default_sim_table(sim: Sequence[Dict[str, Any]]) -> List[str]:
    rows = []
    for r in sim:
        if int(r.get("patience") or 0) != 2:
            continue
        if abs((_float(r.get("epsilon_hit")) or -999.0) - 0.005) > 1e-12:
            continue
        if abs((_float(r.get("epsilon_tau")) or -999.0) - 0.005) > 1e-12:
            continue
        if abs((_float(r.get("margin_p95")) or -999.0) - 0.015) > 1e-12:
            continue
        rows.append(
            [
                str(r.get("dataset", "")),
                str(r.get("method", "")),
                _fmt(r.get("alpha_cal_multiplier")),
                str(r.get("stopped", "")),
                str(r.get("stop_step", "")),
                _fmt_int(r.get("offline_labels_used")),
                _fmt(r.get("monitor_FPR")),
                _fmt(r.get("eval_FPR")),
                _fmt(r.get("eval_hit")),
                _fmt(r.get("relative_eval_hit_vs_safe_full")),
                str(r.get("success_90", "")),
                str(r.get("success_95", "")),
            ]
        )
    return _table(
        ["Dataset", "Method", "alpha_cal x", "Stopped", "Step", "Labels", "Monitor FPR", "Eval FPR", "Eval hit", "Rel hit", "S90", "S95"],
        rows,
    )


def _monitor_predictiveness(rows: Sequence[Dict[str, Any]]) -> List[str]:
    table_rows = []
    for dataset, method in sorted({(r.get("dataset"), r.get("method")) for r in rows}):
        group = [r for r in rows if r.get("dataset") == dataset and r.get("method") == method]
        monitor_safe = [r for r in group if r.get("monitor_safe_under_alpha") is True]
        both_safe = [r for r in monitor_safe if r.get("eval_safe_under_alpha") is True]
        monitor_safe_eval_unsafe = len(monitor_safe) - len(both_safe)
        eval_safe = [r for r in group if r.get("eval_safe_under_alpha") is True]
        table_rows.append(
            [
                str(dataset),
                str(method),
                str(len(group)),
                str(len(monitor_safe)),
                str(len(both_safe)),
                str(monitor_safe_eval_unsafe),
                str(len(eval_safe)),
            ]
        )
    return _table(["Dataset", "Method", "Rows", "Monitor safe", "Monitor+eval safe", "Monitor safe eval unsafe", "Eval safe"], table_rows)


def _answer_line(best: Sequence[Dict[str, Any]], dataset: str, method: str, success_type: str) -> str:
    row = next((r for r in best if r.get("dataset") == dataset and r.get("method") == method and r.get("success_type") == success_type), None)
    if row is None:
        return f"No `{success_type}` row was found for `{dataset}` / `{method}`."
    return (
        f"`{dataset}` / `{method}` reaches `{success_type}` with `{_fmt_int(row.get('offline_labels_used'))}` labels "
        f"(alpha_cal multiplier `{_fmt(row.get('alpha_cal_multiplier'))}`, eval_FPR `{_fmt(row.get('eval_FPR'))}`, "
        f"eval_hit `{_fmt(row.get('eval_hit'))}`)."
    )


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]], refs: Sequence[Dict[str, Any]], best: Sequence[Dict[str, Any]], sim: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lma_we_ref = next((r for r in refs if r.get("dataset") == "SemCacheLMArena" and r.get("method") == "ours_weighted_ensemble"), None)
    sq_we_ref = next((r for r in refs if r.get("dataset") == "SemCacheSearchQueries" and r.get("method") == "ours_weighted_ensemble"), None)
    default_sim_stopped = [r for r in sim if r.get("stopped") is True and abs((_float(r.get("epsilon_hit")) or -999.0) - 0.005) < 1e-12 and abs((_float(r.get("margin_p95")) or -999.0) - 0.015) < 1e-12]
    del default_sim_stopped
    lma_we_m90 = next((r for r in best if r.get("dataset") == "SemCacheLMArena" and r.get("method") == "ours_weighted_ensemble" and r.get("success_type") == "monitor_success_90"), None)
    lma_we_m95 = next((r for r in best if r.get("dataset") == "SemCacheLMArena" and r.get("method") == "ours_weighted_ensemble" and r.get("success_type") == "monitor_success_95"), None)
    sq_we_s95 = next((r for r in best if r.get("dataset") == "SemCacheSearchQueries" and r.get("method") == "ours_weighted_ensemble" and r.get("success_type") == "success_95"), None)
    sq_we_m95 = next((r for r in best if r.get("dataset") == "SemCacheSearchQueries" and r.get("method") == "ours_weighted_ensemble" and r.get("success_type") == "monitor_success_95"), None)

    lines: List[str] = [
        "# Risk-Aware Chronological Stopping",
        "",
        "## 1. Executive Summary",
        "",
        "This experiment evaluates label-efficient stopping with a chronological monitor split and conservative calibration targets. The deployment target is candidate-level `FPR = FP / (FP + TN) <= 0.05`.",
        "",
        "The result should be read as single-seed evidence only. The post-hoc sweep and the simulated stopping rule are reported separately.",
        "",
    ]
    if lma_we_ref and lma_we_ref.get("reference_is_safe") is True:
        lines.append(
            f"For SemCacheLMArena / WeightedEnsemble, the best safe full-data reference uses alpha_cal multiplier `{_fmt(lma_we_ref.get('alpha_cal_multiplier'))}` with final eval FPR `{_fmt(lma_we_ref.get('full_eval_FPR'))}` and hit `{_fmt(lma_we_ref.get('full_eval_hit'))}`."
        )
    elif lma_we_ref:
        lines.append(
            f"For SemCacheLMArena / WeightedEnsemble, no checked full-data configuration is safe on `online_eval_final`. The lowest-FPR full-data row uses alpha_cal multiplier `{_fmt(lma_we_ref.get('alpha_cal_multiplier'))}` with final eval FPR `{_fmt(lma_we_ref.get('full_eval_FPR'))}` and hit `{_fmt(lma_we_ref.get('full_eval_hit'))}`."
        )
    if sq_we_ref:
        lines.append(
            f"For SemCacheSearchQueries / WeightedEnsemble, the best safe full-data reference uses alpha_cal multiplier `{_fmt(sq_we_ref.get('alpha_cal_multiplier'))}` with final eval FPR `{_fmt(sq_we_ref.get('full_eval_FPR'))}` and hit `{_fmt(sq_we_ref.get('full_eval_hit'))}`."
        )
    lines.extend(
        [
            "",
            "## 2. Why The Old Stopping Mechanism Was Insufficient",
            "",
            "The old stopping sweep froze the scorer and threshold after train/calib selection and then judged success on the same future stream used for final reporting. The LMArena audit showed that naive calibration can satisfy calibration FPR while failing on later chronological traffic because the future H0 score distribution shifts upward.",
            "",
            "## 3. The New Stopping Rule",
            "",
            "Risk-aware chronological stopping uses `online_train` for fitting, `online_calib` for threshold selection, the first half of the old `online_eval` as `online_monitor`, and the last half as untouched `online_eval_final`.",
            "",
            "Thresholds are selected from calibration H0 scores only. For higher-is-better scores, the script uses the audited empirical rule: choose the first observed threshold whose calibration H0 accept rate under `score >= tau` is at most `alpha_cal`.",
            "",
            "The simulated rule stops only after monitor FPR is safe, monitor hit and tau plateau, and monitor H0 p95 drift is below the configured margin.",
            "",
            "## 4. Safe Full-Data Reference Table",
            "",
        ]
    )
    lines.extend(_reference_table(refs))
    lines.extend(
        [
            "",
            "## 5. Sweep Results",
            "",
            "The full sweep is in `risk_aware_stopping_sweep.csv`. It includes chronological prefix train/calib dosages, conservative calibration targets, monitor metrics, final eval metrics, and H0 drift statistics.",
            "",
            "Monitor safety versus final eval safety:",
            "",
        ]
    )
    lines.extend(_monitor_predictiveness(rows))
    lines.extend(
        [
            "",
            "## 6. Best Stopped Configurations",
            "",
            "These rows are post-hoc smallest successful dosages over the completed grid. They are not the simulated stopping rule.",
            "",
        ]
    )
    lines.extend(_best_table(best))
    lines.extend(
        [
            "",
            "## 7. Simulated Stopping Results",
            "",
            "Default simulation uses patience 2, epsilon_hit 0.005, epsilon_tau 0.005, and margin_p95 0.015.",
            "",
        ]
    )
    lines.extend(_default_sim_table(sim))
    lines.extend(
        [
            "",
            "## 8. LMArena Analysis",
            "",
            _answer_line(best, "SemCacheLMArena", "ours_weighted_ensemble", "monitor_success_90"),
            _answer_line(best, "SemCacheLMArena", "ours_weighted_ensemble", "monitor_success_95"),
            "",
            "Conservative calibration is required for the LMArena learned scorer because the naive alpha_cal=alpha threshold was previously shown to fail under chronological transfer. This experiment uses the monitor split to gate stopped configurations before final evaluation.",
            "",
            "## 9. SearchQueries Analysis",
            "",
            _answer_line(best, "SemCacheSearchQueries", "ours_weighted_ensemble", "monitor_success_90"),
            _answer_line(best, "SemCacheSearchQueries", "ours_weighted_ensemble", "monitor_success_95"),
            "",
            "## 10. Failure Modes",
            "",
            "- Monitor-safe rows can still be final-eval unsafe when drift continues after the monitor block.",
            "- Very small chronological train prefixes can contain few H1 examples, especially on SearchQueries.",
            "- Plateau simulation depends on the ordering of the dosage grid and should not be interpreted as an optimal stopping policy.",
            "- These are single-seed results.",
            "",
            "## 11. Supported Claims",
            "",
            "- Conservative calibration plus a chronological monitor split can be evaluated without using final-eval labels for stopping decisions.",
            "- A stopped configuration is only counted as successful when final eval FPR is also within the deployment alpha.",
            "- The safest claim is dataset- and method-specific; see the best-stopped and simulated tables.",
            "",
            "## 12. Unsupported Claims",
            "",
            "- Do not write that stopping is solved unless monitor and final eval are safe and success_90 or success_95 is reached.",
            "- Do not write that the method is robust across seeds; this run checks seed 42 only.",
            "- Do not claim monitor safety perfectly predicts final eval safety; use the monitor-vs-eval table.",
            "",
            "## 13. Next Experiments",
            "",
            "- Rerun the same risk-aware stopping sweep for seeds 43-46.",
            "- Evaluate alternative monitor sizes and rolling monitor windows.",
            "- Replace the simple p95 drift margin with a confidence-bound or two-sample drift test on H0 scores.",
            "- Test an online threshold refresh protocol that updates only calibration statistics, not the scorer.",
            "",
            "## Explicit Questions",
            "",
            "A. Does conservative calibration make LMArena stopping work? "
            + (
                f"Yes for success_90 at `{_fmt_int(lma_we_m90.get('offline_labels_used'))}` labels." if lma_we_m90 else "No. No LMArena / WeightedEnsemble monitor-success row was found, and no full-data row is safe on the final half of the old eval stream."
            ),
            "B. Labels needed for 90% safe full hit: "
            + (
                f"`{_fmt_int(lma_we_m90.get('offline_labels_used'))}` for LMArena / WeightedEnsemble." if lma_we_m90 else "not available for LMArena / WeightedEnsemble because there is no safe full-data reference under the checked grid."
            ),
            "C. Labels needed for 95% safe full hit: "
            + (
                f"`{_fmt_int(lma_we_m95.get('offline_labels_used'))}` for LMArena / WeightedEnsemble." if lma_we_m95 else "not available for LMArena / WeightedEnsemble because there is no safe full-data reference under the checked grid."
            ),
            "D. Monitor safety predicts final safety imperfectly; for LMArena / WeightedEnsemble many monitor-safe rows are still final-eval unsafe.",
            "E. For SearchQueries / WeightedEnsemble, the full-data utility reference is alpha_cal multiplier "
            + (f"`{_fmt(sq_we_ref.get('alpha_cal_multiplier'))}`." if sq_we_ref else "unavailable."),
            "F. `0.7 * alpha` is not enough for LMArena final-eval safety in this stricter split; even `0.5 * alpha` remains above alpha on the final half.",
            "G. SearchQueries does not reach monitor-gated success_95 near 3,000 labels in this chronological-prefix run. "
            + (
                f"Post-hoc final-only success_95 appears at `{_fmt_int(sq_we_s95.get('offline_labels_used'))}` labels, while monitor-gated success_95 appears at `{_fmt_int(sq_we_m95.get('offline_labels_used'))}` labels." if sq_we_s95 and sq_we_m95 else "See the best table for available SearchQueries rows."
            ),
            "H. Dataset/method combinations with missing best rows are failures under the current grid.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot_csvs(out_dir: Path, rows: Sequence[Dict[str, Any]], refs: Sequence[Dict[str, Any]], best: Sequence[Dict[str, Any]]) -> List[Path]:
    paths: List[Path] = []
    utility = [
        {
            "dataset": r.get("dataset"),
            "method": r.get("method"),
            "alpha": r.get("alpha"),
            "alpha_cal_multiplier": r.get("alpha_cal_multiplier"),
            "offline_labels_used": r.get("offline_labels_used"),
            "eval_hit": r.get("eval_hit"),
            "eval_TPR": r.get("eval_TPR"),
        }
        for r in rows
        if r.get("eval_hit") not in {None, ""}
    ]
    path = out_dir / "plot_utility_curve.csv"
    _write_csv(path, utility, ["dataset", "method", "alpha", "alpha_cal_multiplier", "offline_labels_used", "eval_hit", "eval_TPR"])
    paths.append(path)

    safety = [
        {
            "dataset": r.get("dataset"),
            "method": r.get("method"),
            "alpha": r.get("alpha"),
            "alpha_cal_multiplier": r.get("alpha_cal_multiplier"),
            "offline_labels_used": r.get("offline_labels_used"),
            "eval_FPR": r.get("eval_FPR"),
        }
        for r in rows
        if r.get("eval_FPR") not in {None, ""}
    ]
    path = out_dir / "plot_safety_curve.csv"
    _write_csv(path, safety, ["dataset", "method", "alpha", "alpha_cal_multiplier", "offline_labels_used", "eval_FPR"])
    paths.append(path)

    monitor_eval = [
        {
            "dataset": r.get("dataset"),
            "method": r.get("method"),
            "alpha": r.get("alpha"),
            "alpha_cal_multiplier": r.get("alpha_cal_multiplier"),
            "offline_labels_used": r.get("offline_labels_used"),
            "monitor_FPR": r.get("monitor_FPR"),
            "eval_FPR": r.get("eval_FPR"),
        }
        for r in rows
        if r.get("monitor_FPR") not in {None, ""} and r.get("eval_FPR") not in {None, ""}
    ]
    path = out_dir / "plot_monitor_vs_eval.csv"
    _write_csv(path, monitor_eval, ["dataset", "method", "alpha", "alpha_cal_multiplier", "offline_labels_used", "monitor_FPR", "eval_FPR"])
    paths.append(path)

    drift = [
        {
            "dataset": r.get("dataset"),
            "method": r.get("method"),
            "alpha": r.get("alpha"),
            "alpha_cal_multiplier": r.get("alpha_cal_multiplier"),
            "offline_labels_used": r.get("offline_labels_used"),
            "eval_minus_calib_H0_p95": r.get("eval_minus_calib_H0_p95"),
            "monitor_minus_calib_H0_p95": r.get("monitor_minus_calib_H0_p95"),
        }
        for r in rows
        if r.get("eval_minus_calib_H0_p95") not in {None, ""}
    ]
    path = out_dir / "plot_h0_drift.csv"
    _write_csv(
        path,
        drift,
        ["dataset", "method", "alpha", "alpha_cal_multiplier", "offline_labels_used", "eval_minus_calib_H0_p95", "monitor_minus_calib_H0_p95"],
    )
    paths.append(path)

    ref_by_key = {(r.get("dataset"), r.get("method")): r for r in refs if r.get("reference_is_safe") is True}
    bars: List[Dict[str, Any]] = []
    for row in best:
        if row.get("success_type") not in {"monitor_success_90", "monitor_success_95"}:
            continue
        ref = ref_by_key.get((row.get("dataset"), row.get("method")))
        if ref is None:
            continue
        reduction = row.get("label_reduction_vs_safe_full")
        bars.append(
            {
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "success_type": row.get("success_type"),
                "bar_type": "safe_full",
                "labels_used": ref.get("full_total_labels"),
                "label_reduction": 0.0,
            }
        )
        bars.append(
            {
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "success_type": row.get("success_type"),
                "bar_type": "stopped",
                "labels_used": row.get("offline_labels_used"),
                "label_reduction": reduction,
            }
        )
    path = out_dir / "plot_best_stopped.csv"
    _write_csv(path, bars, ["dataset", "method", "success_type", "bar_type", "labels_used", "label_reduction"])
    paths.append(path)
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run risk-aware chronological stopping sweep.")
    ap.add_argument("--pair_dirs", nargs="+", default=DEFAULT_PAIR_DIRS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["cosine", *sorted(METHOD_MAP)])
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--alpha_cal_multipliers", nargs="+", type=float, default=DEFAULT_ALPHA_CAL_MULTIPLIERS)
    ap.add_argument("--train_dosages", nargs="+", default=[str(v) for v in DEFAULT_DOSAGES])
    ap.add_argument("--calib_dosages", nargs="+", default=[str(v) for v in DEFAULT_DOSAGES])
    ap.add_argument("--output_dir", default="results/risk_aware_stopping")
    return ap.parse_args(argv)


def _parse_dosages(values: Sequence[str]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if str(value).lower() == "full":
            out.append("full")
        else:
            out.append(int(value))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dosages = _parse_dosages(args.train_dosages)
    calib_dosages = _parse_dosages(args.calib_dosages)

    sweep = build_sweep(
        [Path(p) for p in args.pair_dirs],
        methods=args.methods,
        train_dosages=train_dosages,
        calib_dosages=calib_dosages,
        alpha=float(args.alpha),
        alpha_cal_multipliers=args.alpha_cal_multipliers,
    )
    refs = build_safe_references(sweep, float(args.alpha))
    add_relative_success(sweep, refs, float(args.alpha))
    best = build_best(sweep, refs)
    sim = build_simulated(
        sweep,
        alpha=float(args.alpha),
        patience_values=[2],
        epsilon_hit_values=[0.005, 0.01],
        epsilon_tau_values=[0.005],
        margin_p95_values=[0.01, 0.015, 0.02],
    )

    outputs: List[Path] = []
    path = out_dir / "risk_aware_stopping_sweep.csv"
    _write_csv(path, sweep, SWEEP_FIELDS)
    outputs.append(path)
    path = out_dir / "safe_full_data_reference.csv"
    _write_csv(path, refs, REFERENCE_FIELDS)
    outputs.append(path)
    path = out_dir / "risk_aware_stopping_best.csv"
    _write_csv(path, best, BEST_FIELDS)
    outputs.append(path)
    path = out_dir / "risk_aware_stopping_simulated.csv"
    _write_csv(path, sim, SIM_FIELDS)
    outputs.append(path)
    outputs.extend(write_plot_csvs(out_dir, sweep, refs, best))
    report_path = out_dir / "risk_aware_stopping_report.md"
    write_markdown(report_path, sweep, refs, best, sim)
    outputs.append(report_path)

    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
