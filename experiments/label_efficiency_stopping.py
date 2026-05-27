from __future__ import annotations

import argparse
import copy
import csv
import json
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
DEFAULT_ALPHAS = [0.05]
DEFAULT_DOSAGES: List[Any] = [250, 500, 1000, 2000, 5000, 10000, "full"]

FIELDS = [
    "dataset",
    "method",
    "seed",
    "alpha",
    "train_pairs_requested",
    "train_pairs_used",
    "train_H1",
    "train_H0",
    "calib_pairs_requested",
    "calib_pairs_used",
    "calib_H1",
    "calib_H0",
    "threshold_or_param",
    "score_orientation",
    "calib_FPR",
    "calib_TPR",
    "calib_hit_rate",
    "eval_FPR",
    "eval_TPR",
    "eval_hit_rate",
    "eval_precision",
    "eval_FP_over_n",
    "TP",
    "FP",
    "TN",
    "FN",
    "n",
    "offline_labels_used",
    "eval_constraint_satisfied",
    "full_eval_hit_rate",
    "full_eval_TPR",
    "relative_hit_vs_full",
    "relative_TPR_vs_full",
    "success_90",
    "success_95",
    "warning",
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


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _dataset_from_config(config: Dict[str, Any]) -> str:
    dataset = config.get("dataset") or config.get("args", {}).get("dataset")
    if not dataset:
        raise ValueError("Could not recover dataset from pairs config.")
    return str(dataset)


def _seed_from_config(config: Dict[str, Any]) -> int:
    return int(config.get("args", {}).get("seed", 42))


def _requested_to_n(requested: Any, total: int) -> int:
    if str(requested).lower() == "full":
        return int(total)
    return min(int(requested), int(total))


def _sample_stratified(y: np.ndarray, requested: Any, *, seed: int) -> np.ndarray:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    n = _requested_to_n(requested, int(yy.size))
    if n >= yy.size:
        return np.arange(yy.size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    h0 = np.where(yy == 0)[0]
    h1 = np.where(yy == 1)[0]
    if h0.size == 0 or h1.size == 0:
        idx = np.arange(yy.size, dtype=np.int64)
        rng.shuffle(idx)
        return np.sort(idx[:n])
    h1_target = int(round(n * (h1.size / yy.size)))
    h1_target = max(1, min(int(h1.size), h1_target))
    h0_target = n - h1_target
    if h0_target <= 0:
        h0_target = 1
        h1_target = n - 1
    if h0_target > h0.size:
        h0_target = int(h0.size)
        h1_target = min(int(h1.size), n - h0_target)
    if h1_target > h1.size:
        h1_target = int(h1.size)
        h0_target = min(int(h0.size), n - h1_target)
    chosen = np.concatenate(
        [
            rng.choice(h0, size=h0_target, replace=False),
            rng.choice(h1, size=h1_target, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return np.sort(chosen.astype(np.int64))


def _counts(y: np.ndarray, idx: np.ndarray) -> Tuple[int, int]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    sub = yy[idx]
    h1 = int(np.sum(sub == 1))
    h0 = int(np.sum(sub == 0))
    return h1, h0


def _pair_eval_metrics(y: np.ndarray, accepted: np.ndarray) -> Dict[str, Any]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    aa = np.asarray(accepted).reshape(-1).astype(bool)
    tp = int(np.sum(aa & (yy == 1)))
    fp = int(np.sum(aa & (yy == 0)))
    tn = int(np.sum((~aa) & (yy == 0)))
    fn = int(np.sum((~aa) & (yy == 1)))
    n = int(yy.size)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "n": n,
        "precision": None if tp + fp == 0 else tp / float(tp + fp),
        "TPR": None if tp + fn == 0 else tp / float(tp + fn),
        "FPR": None if fp + tn == 0 else fp / float(fp + tn),
        "hit_rate": None if n == 0 else (tp + fp) / float(n),
        "FP_over_n": None if n == 0 else fp / float(n),
    }


def _calib_metrics(y: np.ndarray, accepted: np.ndarray) -> Dict[str, Any]:
    m = _pair_eval_metrics(y, accepted)
    return {
        "calib_FPR": m["FPR"],
        "calib_TPR": m["TPR"],
        "calib_hit_rate": m["hit_rate"],
        "calib_FP": m["FP"],
    }


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


def _fit_learned(method_key: str, train_pairs: OnlinePairs, *, alpha: float, seed: int) -> Tuple[Any, bool, str, str]:
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
            fit_context="label_efficiency_stopping",
        )
    else:
        _fit_method(method, h0_x, h1_x, seed=seed, alpha=float(alpha))
    return method, uses_alt, method_name, ""


def _scores_for_method(method: Any, uses_alt: bool, pairs: OnlinePairs) -> np.ndarray:
    alt = pairs.cosine.reshape(-1, 1) if uses_alt else None
    return score_method(method, pairs.X, alt, uses_alt)


def _score_cosine(pairs: OnlinePairs) -> np.ndarray:
    return np.asarray(pairs.cosine, dtype=np.float64).reshape(-1)


def _select_and_eval(
    *,
    scores_calib: np.ndarray,
    scores_eval: np.ndarray,
    calib_pairs: OnlinePairs,
    eval_pairs: OnlinePairs,
    calib_idx: np.ndarray,
    alpha: float,
    orientation: str,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    calib_sub_scores = np.asarray(scores_calib[calib_idx], dtype=np.float64)
    calib_sub_y = calib_pairs.y[calib_idx]
    h0_scores = calib_sub_scores[calib_sub_y == 0]
    tau = select_tau(h0_scores, float(alpha), orientation)
    calib_accept = accept_scores(calib_sub_scores, tau, orientation)
    eval_accept = accept_scores(scores_eval, tau, orientation)
    return float(tau), _calib_metrics(calib_sub_y, calib_accept), _pair_eval_metrics(eval_pairs.y, eval_accept)


def _dosage_label(value: Any) -> str:
    return "full" if str(value).lower() == "full" else str(int(value))


def build_rows(
    *,
    pair_dirs: Sequence[Path],
    methods: Sequence[str],
    alphas: Sequence[float],
    train_dosages: Sequence[Any],
    calib_dosages: Sequence[Any],
    seed_offset: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pairs_dir in pair_dirs:
        if not pairs_dir.exists():
            raise FileNotFoundError(f"Missing required pairs directory: {pairs_dir}")
        train_full, calib_full, eval_pairs, config = load_pairs_dir(pairs_dir)
        dataset = _dataset_from_config(config)
        seed = _seed_from_config(config)

        train_indices = {
            _dosage_label(d): _sample_stratified(train_full.y, d, seed=seed + seed_offset + int(i) * 17)
            for i, d in enumerate(train_dosages)
        }
        calib_indices = {
            _dosage_label(d): _sample_stratified(calib_full.y, d, seed=seed + seed_offset + int(i) * 31 + 1000)
            for i, d in enumerate(calib_dosages)
        }

        dataset_rows: List[Dict[str, Any]] = []
        for alpha in alphas:
            # Cosine reference: no training labels, threshold selected on calibration subset.
            if "cosine" in methods:
                scores_calib = _score_cosine(calib_full)
                scores_eval = _score_cosine(eval_pairs)
                for calib_req in calib_dosages:
                    calib_label = _dosage_label(calib_req)
                    calib_idx = calib_indices[calib_label]
                    calib_h1, calib_h0 = _counts(calib_full.y, calib_idx)
                    tau, calib_m, eval_m = _select_and_eval(
                        scores_calib=scores_calib,
                        scores_eval=scores_eval,
                        calib_pairs=calib_full,
                        eval_pairs=eval_pairs,
                        calib_idx=calib_idx,
                        alpha=float(alpha),
                        orientation="higher",
                    )
                    dataset_rows.append(
                        {
                            "dataset": dataset,
                            "method": "cosine",
                            "seed": seed,
                            "alpha": float(alpha),
                            "train_pairs_requested": "none",
                            "train_pairs_used": 0,
                            "train_H1": 0,
                            "train_H0": 0,
                            "calib_pairs_requested": calib_label,
                            "calib_pairs_used": int(calib_idx.size),
                            "calib_H1": calib_h1,
                            "calib_H0": calib_h0,
                            "threshold_or_param": tau,
                            "score_orientation": "higher",
                            **calib_m,
                            "eval_FPR": eval_m["FPR"],
                            "eval_TPR": eval_m["TPR"],
                            "eval_hit_rate": eval_m["hit_rate"],
                            "eval_precision": eval_m["precision"],
                            "eval_FP_over_n": eval_m["FP_over_n"],
                            "TP": eval_m["TP"],
                            "FP": eval_m["FP"],
                            "TN": eval_m["TN"],
                            "FN": eval_m["FN"],
                            "n": eval_m["n"],
                            "offline_labels_used": int(calib_idx.size),
                            "eval_constraint_satisfied": bool((eval_m["FPR"] is not None) and (eval_m["FPR"] <= float(alpha))),
                            "warning": "",
                        }
                    )

            for method_key in [m for m in methods if m != "cosine"]:
                for train_req in train_dosages:
                    train_label = _dosage_label(train_req)
                    train_idx = train_indices[train_label]
                    train_pairs = _subset_pairs(train_full, train_idx)
                    train_h1, train_h0 = _counts(train_full.y, train_idx)
                    try:
                        model, uses_alt, _, warning = _fit_learned(method_key, train_pairs, alpha=float(alpha), seed=seed)
                        scores_calib = _scores_for_method(model, uses_alt, calib_full)
                        scores_eval = _scores_for_method(model, uses_alt, eval_pairs)
                        s0 = scores_calib[calib_full.y == 0]
                        s1 = scores_calib[calib_full.y == 1]
                        orientation, orientation_warning = infer_orientation(s0, s1)
                        if orientation == "ambiguous":
                            orientation = "higher"
                            warning = (warning + "; " + orientation_warning).strip("; ")
                    except Exception as exc:
                        for calib_req in calib_dosages:
                            calib_label = _dosage_label(calib_req)
                            calib_idx = calib_indices[calib_label]
                            calib_h1, calib_h0 = _counts(calib_full.y, calib_idx)
                            dataset_rows.append(
                                {
                                    "dataset": dataset,
                                    "method": method_key,
                                    "seed": seed,
                                    "alpha": float(alpha),
                                    "train_pairs_requested": train_label,
                                    "train_pairs_used": int(train_idx.size),
                                    "train_H1": train_h1,
                                    "train_H0": train_h0,
                                    "calib_pairs_requested": calib_label,
                                    "calib_pairs_used": int(calib_idx.size),
                                    "calib_H1": calib_h1,
                                    "calib_H0": calib_h0,
                                    "offline_labels_used": int(train_idx.size + calib_idx.size),
                                    "eval_constraint_satisfied": False,
                                    "success_90": False,
                                    "success_95": False,
                                    "warning": f"fit_or_score_failed: {exc}",
                                }
                            )
                        continue

                    for calib_req in calib_dosages:
                        calib_label = _dosage_label(calib_req)
                        calib_idx = calib_indices[calib_label]
                        calib_h1, calib_h0 = _counts(calib_full.y, calib_idx)
                        tau, calib_m, eval_m = _select_and_eval(
                            scores_calib=scores_calib,
                            scores_eval=scores_eval,
                            calib_pairs=calib_full,
                            eval_pairs=eval_pairs,
                            calib_idx=calib_idx,
                            alpha=float(alpha),
                            orientation=orientation,
                        )
                        dataset_rows.append(
                            {
                                "dataset": dataset,
                                "method": method_key,
                                "seed": seed,
                                "alpha": float(alpha),
                                "train_pairs_requested": train_label,
                                "train_pairs_used": int(train_idx.size),
                                "train_H1": train_h1,
                                "train_H0": train_h0,
                                "calib_pairs_requested": calib_label,
                                "calib_pairs_used": int(calib_idx.size),
                                "calib_H1": calib_h1,
                                "calib_H0": calib_h0,
                                "threshold_or_param": tau,
                                "score_orientation": orientation,
                                **calib_m,
                                "eval_FPR": eval_m["FPR"],
                                "eval_TPR": eval_m["TPR"],
                                "eval_hit_rate": eval_m["hit_rate"],
                                "eval_precision": eval_m["precision"],
                                "eval_FP_over_n": eval_m["FP_over_n"],
                                "TP": eval_m["TP"],
                                "FP": eval_m["FP"],
                                "TN": eval_m["TN"],
                                "FN": eval_m["FN"],
                                "n": eval_m["n"],
                                "offline_labels_used": int(train_idx.size + calib_idx.size),
                                "eval_constraint_satisfied": bool((eval_m["FPR"] is not None) and (eval_m["FPR"] <= float(alpha))),
                                "warning": warning,
                            }
                        )

        # Relative performance uses full-label WeightedEnsemble on the same dataset/alpha.
        for alpha in alphas:
            full_ref = next(
                (
                    r
                    for r in dataset_rows
                    if r.get("method") == "ours_weighted_ensemble"
                    and _dosage_label(r.get("train_pairs_requested")) == "full"
                    and _dosage_label(r.get("calib_pairs_requested")) == "full"
                    and abs(float(r.get("alpha")) - float(alpha)) < 1e-12
                    and not r.get("warning")
                ),
                None,
            )
            if full_ref is None:
                continue
            full_hit = _float(full_ref.get("eval_hit_rate"))
            full_tpr = _float(full_ref.get("eval_TPR"))
            for row in dataset_rows:
                if abs((_float(row.get("alpha")) or -999.0) - float(alpha)) > 1e-12:
                    continue
                hit = _float(row.get("eval_hit_rate"))
                tpr = _float(row.get("eval_TPR"))
                rel_hit = (hit / full_hit) if hit is not None and full_hit and full_hit > 0 else None
                rel_tpr = (tpr / full_tpr) if tpr is not None and full_tpr and full_tpr > 0 else None
                row["full_eval_hit_rate"] = full_hit
                row["full_eval_TPR"] = full_tpr
                row["relative_hit_vs_full"] = rel_hit
                row["relative_TPR_vs_full"] = rel_tpr
                safe = bool(row.get("eval_constraint_satisfied"))
                row["success_90"] = bool(safe and rel_hit is not None and rel_hit >= 0.90)
                row["success_95"] = bool(safe and rel_hit is not None and rel_hit >= 0.95)
        rows.extend(dataset_rows)
    rows.sort(
        key=lambda r: (
            str(r.get("dataset")),
            str(r.get("method")),
            _float(r.get("alpha")) or 0.0,
            str(r.get("train_pairs_requested")),
            str(r.get("calib_pairs_requested")),
        )
    )
    return rows


def _full_reference_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for row in rows:
        method = row.get("method")
        if method == "cosine":
            if row.get("calib_pairs_requested") != "full":
                continue
        else:
            if row.get("train_pairs_requested") != "full" or row.get("calib_pairs_requested") != "full":
                continue
        refs.append(row)
    return refs


def _best_stopped_rows(rows: Sequence[Dict[str, Any]], criterion: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted({(r.get("dataset"), r.get("method"), r.get("alpha")) for r in rows if r.get(method_key := "method") != "cosine"})
    for dataset, method, alpha in keys:
        candidates = [
            r
            for r in rows
            if r.get("dataset") == dataset
            and r.get("method") == method
            and abs((_float(r.get("alpha")) or -999.0) - float(alpha)) < 1e-12
            and bool(r.get(criterion)) is True
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda r: (
                int(r.get("offline_labels_used") or 10**18),
                -(_float(r.get("eval_hit_rate")) or -1.0),
            ),
        )
        full = next(
            (
                r
                for r in rows
                if r.get("dataset") == dataset
                and r.get("method") == method
                and abs((_float(r.get("alpha")) or -999.0) - float(alpha)) < 1e-12
                and r.get("train_pairs_requested") == "full"
                and r.get("calib_pairs_requested") == "full"
            ),
            None,
        )
        if full:
            best = dict(best)
            best["criterion"] = criterion
            best["full_labels"] = full.get("offline_labels_used")
            best["full_FPR"] = full.get("eval_FPR")
            best["full_hit"] = full.get("eval_hit_rate")
            out.append(best)
    return out


def _table(rows: Sequence[Dict[str, Any]], fields: Sequence[Tuple[str, str]], *, limit: Optional[int] = None) -> List[str]:
    numeric = {
        "alpha",
        "calib_FPR",
        "calib_TPR",
        "calib_hit_rate",
        "eval_FPR",
        "eval_TPR",
        "eval_hit_rate",
        "eval_precision",
        "eval_FP_over_n",
        "relative_hit_vs_full",
        "relative_TPR_vs_full",
        "full_eval_hit_rate",
        "full_eval_TPR",
        "label_reduction",
    }
    take = list(rows[:limit] if limit is not None else rows)
    lines = [
        "| " + " | ".join(title for title, _ in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in take:
        vals = []
        for _, key in fields:
            vals.append(_fmt(row.get(key)) if key in numeric else str(row.get(key, "")))
        lines.append("| " + " | ".join(vals) + " |")
    if not take:
        lines.append("| " + " | ".join("" for _ in fields) + " |")
    return lines


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_refs = _full_reference_rows(rows)
    stopped: List[Dict[str, Any]] = []
    for criterion in ["success_90", "success_95"]:
        for row in _best_stopped_rows(rows, criterion):
            full_labels = _float(row.get("full_labels"))
            stopped_labels = _float(row.get("offline_labels_used"))
            row["label_reduction"] = (
                1.0 - stopped_labels / full_labels
                if full_labels is not None and stopped_labels is not None and full_labels > 0
                else None
            )
            stopped.append(row)

    lines: List[str] = [
        "# Label-Efficiency / Stopping Sweep",
        "",
        "Goal: recover most of the full-label performance with fewer online-mined labels while using the candidate-level constraint `FPR = FP / (FP + TN) <= alpha`.",
        "",
        "Thresholds are selected only on the requested calibration subset. The fixed `online_eval` candidate-pair split is used only for final measurement.",
        "",
        "The stopped rows below are post-hoc label-efficiency summaries over the completed sweep. They show the smallest measured dosage satisfying the stated eval-side success criterion; they are not an online stopping rule by themselves.",
        "",
        "## Full-Data Reference",
        "",
    ]
    lines.extend(
        _table(
            full_refs,
            [
                ("Dataset", "dataset"),
                ("Method", "method"),
                ("alpha", "alpha"),
                ("Full train labels", "train_pairs_used"),
                ("Full calib labels", "calib_pairs_used"),
                ("Full total labels", "offline_labels_used"),
                ("Full eval FPR", "eval_FPR"),
                ("Full eval TPR", "eval_TPR"),
                ("Full hit", "eval_hit_rate"),
            ],
        )
    )
    lines.extend(["", "## Best Stopped Configurations", ""])
    lines.extend(
        _table(
            stopped,
            [
                ("Dataset", "dataset"),
                ("Method", "method"),
                ("alpha", "alpha"),
                ("Full labels", "full_labels"),
                ("Stopped labels", "offline_labels_used"),
                ("Label reduction", "label_reduction"),
                ("Full hit", "full_hit"),
                ("Stopped hit", "eval_hit_rate"),
                ("Full FPR", "full_FPR"),
                ("Stopped FPR", "eval_FPR"),
                ("Success criterion", "criterion"),
            ],
        )
    )
    lines.extend(["", "## Label-Efficiency Curve", ""])
    curve_rows = [r for r in rows if r.get("method") != "cosine"]
    lines.extend(
        _table(
            curve_rows,
            [
                ("Dataset", "dataset"),
                ("Method", "method"),
                ("Train", "train_pairs_requested"),
                ("Calib", "calib_pairs_requested"),
                ("Labels", "offline_labels_used"),
                ("Eval FPR", "eval_FPR"),
                ("Eval TPR", "eval_TPR"),
                ("Hit", "eval_hit_rate"),
                ("Rel hit", "relative_hit_vs_full"),
                ("Safe", "eval_constraint_satisfied"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A row is considered safe only when `eval_FPR <= alpha` on the fixed eval split.",
            "- `success_90` and `success_95` additionally require at least 90% or 95% of the full-label WeightedEnsemble hit rate.",
            "- If a small configuration has strong hit rate but violates eval FPR, it is not counted as a successful stopped configuration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_dosages(values: Sequence[str]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if value.lower() == "full":
            out.append("full")
        else:
            out.append(int(value))
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run label-efficiency sweeps on online-mined candidate pairs.")
    ap.add_argument("--pair_dirs", nargs="+", default=DEFAULT_PAIR_DIRS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--train_pairs", nargs="+", default=[str(v) for v in DEFAULT_DOSAGES])
    ap.add_argument("--calib_pairs", nargs="+", default=[str(v) for v in DEFAULT_DOSAGES])
    ap.add_argument("--seed_offset", type=int, default=9100)
    ap.add_argument("--output_dir", default="results/label_efficiency_stopping")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_rows(
        pair_dirs=[Path(p) for p in args.pair_dirs],
        methods=args.methods,
        alphas=args.alphas,
        train_dosages=_parse_dosages(args.train_pairs),
        calib_dosages=_parse_dosages(args.calib_pairs),
        seed_offset=int(args.seed_offset),
    )
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "label_efficiency_stopping.csv"
    md_path = out_dir / "label_efficiency_stopping.md"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
