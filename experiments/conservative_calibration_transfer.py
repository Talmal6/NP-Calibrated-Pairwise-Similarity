from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .debug_ours_online_mined_calibration import METHOD_MAP
from .label_efficiency_stopping import (
    _dataset_from_config,
    _fit_learned,
    _pair_eval_metrics,
    _scores_for_method,
    _seed_from_config,
)
from .online_mined_ours import accept_scores, infer_orientation, load_pairs_dir, select_tau


DEFAULT_PAIR_DIRS = [
    "results/online_candidate_pairs/lmarena_gte_seed42",
    "results/online_candidate_pairs/searchqueries_gte_seed42",
]
DEFAULT_METHODS = ["ours_weighted_ensemble", "ours_whitened_hadamard"]
DEFAULT_ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.08]
DEFAULT_TARGET_FACTORS = [1.0, 0.9, 0.8, 0.7, 0.5]

FIELDS = [
    "dataset",
    "method",
    "seed",
    "alpha",
    "calibration_rule",
    "calib_target",
    "threshold_or_param",
    "score_orientation",
    "calib_H0",
    "calib_FP",
    "calib_FPR",
    "calib_FPR_upper_bound",
    "eval_H0",
    "eval_FP",
    "eval_FPR",
    "eval_TPR",
    "eval_hit_rate",
    "eval_precision",
    "eval_constraint_satisfied",
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


def _wilson_upper(k: int, n: int, *, confidence: float = 0.95) -> float:
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import norm

        z = float(norm.ppf(confidence))
    except Exception:
        z = 1.6448536269514722
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return float(min(1.0, max(0.0, center + radius)))


def _accept_count_for_tau(scores_h0: np.ndarray, tau: float, orientation: str) -> int:
    return int(np.sum(accept_scores(scores_h0, tau, orientation)))


def _select_tau_ucb(scores_h0: np.ndarray, alpha: float, orientation: str) -> Tuple[float, float, int]:
    s = np.asarray(scores_h0, dtype=np.float64).reshape(-1)
    n = int(s.size)
    if n == 0:
        return (float("inf") if orientation == "higher" else float("-inf")), 1.0, 0
    if orientation == "higher":
        candidates = np.unique(s)
    elif orientation == "lower":
        candidates = np.unique(s)[::-1]
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    fallback_tau = float("inf") if orientation == "higher" else float("-inf")
    fallback_k = 0
    for tau in candidates:
        k = _accept_count_for_tau(s, float(tau), orientation)
        upper = _wilson_upper(k, n, confidence=0.95)
        if upper <= float(alpha):
            return float(tau), upper, k
    return fallback_tau, _wilson_upper(fallback_k, n, confidence=0.95), fallback_k


def _calib_counts(y: np.ndarray, accepted: np.ndarray) -> Tuple[int, int, float]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    aa = np.asarray(accepted).reshape(-1).astype(bool)
    h0 = yy == 0
    n_h0 = int(np.sum(h0))
    fp = int(np.sum(aa & h0))
    return n_h0, fp, (fp / n_h0 if n_h0 else 0.0)


def build_rows(
    *,
    pair_dirs: Sequence[Path],
    methods: Sequence[str],
    alphas: Sequence[float],
    target_factors: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pairs_dir in pair_dirs:
        if not pairs_dir.exists():
            raise FileNotFoundError(f"Missing required pairs directory: {pairs_dir}")
        train_pairs, calib_pairs, eval_pairs, config = load_pairs_dir(pairs_dir)
        dataset = _dataset_from_config(config)
        seed = _seed_from_config(config)
        for method_key in methods:
            for alpha in alphas:
                model, uses_alt, _, warning = _fit_learned(method_key, train_pairs, alpha=float(alpha), seed=seed)
                scores_calib = _scores_for_method(model, uses_alt, calib_pairs)
                scores_eval = _scores_for_method(model, uses_alt, eval_pairs)
                s0 = scores_calib[calib_pairs.y == 0]
                s1 = scores_calib[calib_pairs.y == 1]
                orientation, orientation_warning = infer_orientation(s0, s1)
                if orientation == "ambiguous":
                    orientation = "higher"

                for factor in target_factors:
                    calib_target = float(alpha) * float(factor)
                    tau = select_tau(s0, calib_target, orientation)
                    accepted_calib = accept_scores(scores_calib, tau, orientation)
                    accepted_eval = accept_scores(scores_eval, tau, orientation)
                    calib_h0, calib_fp, calib_fpr = _calib_counts(calib_pairs.y, accepted_calib)
                    eval_m = _pair_eval_metrics(eval_pairs.y, accepted_eval)
                    eval_h0 = int(eval_m["FP"] + eval_m["TN"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method_key,
                            "seed": seed,
                            "alpha": float(alpha),
                            "calibration_rule": f"empirical_target_{factor:g}x_alpha",
                            "calib_target": calib_target,
                            "threshold_or_param": float(tau),
                            "score_orientation": orientation,
                            "calib_H0": calib_h0,
                            "calib_FP": calib_fp,
                            "calib_FPR": calib_fpr,
                            "calib_FPR_upper_bound": _wilson_upper(calib_fp, calib_h0),
                            "eval_H0": eval_h0,
                            "eval_FP": eval_m["FP"],
                            "eval_FPR": eval_m["FPR"],
                            "eval_TPR": eval_m["TPR"],
                            "eval_hit_rate": eval_m["hit_rate"],
                            "eval_precision": eval_m["precision"],
                            "eval_constraint_satisfied": bool(eval_m["FPR"] is not None and eval_m["FPR"] <= float(alpha)),
                        }
                    )

                tau, upper, _ = _select_tau_ucb(s0, float(alpha), orientation)
                accepted_calib = accept_scores(scores_calib, tau, orientation)
                accepted_eval = accept_scores(scores_eval, tau, orientation)
                calib_h0, calib_fp, calib_fpr = _calib_counts(calib_pairs.y, accepted_calib)
                eval_m = _pair_eval_metrics(eval_pairs.y, accepted_eval)
                eval_h0 = int(eval_m["FP"] + eval_m["TN"])
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method_key,
                        "seed": seed,
                        "alpha": float(alpha),
                        "calibration_rule": "ucb_wilson_95",
                        "calib_target": float(alpha),
                        "threshold_or_param": float(tau),
                        "score_orientation": orientation,
                        "calib_H0": calib_h0,
                        "calib_FP": calib_fp,
                        "calib_FPR": calib_fpr,
                        "calib_FPR_upper_bound": upper,
                        "eval_H0": eval_h0,
                        "eval_FP": eval_m["FP"],
                        "eval_FPR": eval_m["FPR"],
                        "eval_TPR": eval_m["TPR"],
                        "eval_hit_rate": eval_m["hit_rate"],
                        "eval_precision": eval_m["precision"],
                        "eval_constraint_satisfied": bool(eval_m["FPR"] is not None and eval_m["FPR"] <= float(alpha)),
                    }
                )
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["method"]), float(r["alpha"]), str(r["calibration_rule"])))
    return rows


def _best_safe(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted({(r["dataset"], r["method"], r["alpha"]) for r in rows})
    for dataset, method, alpha in keys:
        safe = [
            r
            for r in rows
            if r["dataset"] == dataset
            and r["method"] == method
            and abs(float(r["alpha"]) - float(alpha)) < 1e-12
            and r.get("eval_constraint_satisfied") is True
        ]
        if safe:
            out.append(max(safe, key=lambda r: _float(r.get("eval_hit_rate")) or -1.0))
    return out


def _table(rows: Sequence[Dict[str, Any]], fields: Sequence[Tuple[str, str]], *, limit: Optional[int] = None) -> List[str]:
    numeric = {
        "alpha",
        "calib_target",
        "calib_FPR",
        "calib_FPR_upper_bound",
        "eval_FPR",
        "eval_TPR",
        "eval_hit_rate",
        "eval_precision",
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
    baseline = [r for r in rows if r.get("calibration_rule") == "empirical_target_1x_alpha"]
    safe_count = sum(1 for r in rows if r.get("eval_constraint_satisfied") is True)
    baseline_safe = sum(1 for r in baseline if r.get("eval_constraint_satisfied") is True)
    best_safe = _best_safe(rows)
    lines: List[str] = [
        "# Conservative Calibration Transfer",
        "",
        "This report tests whether more conservative calibration rules improve eval-side candidate-level FPR satisfaction.",
        "",
        "Candidate-level FPR is `FP / (FP + TN)`. Thresholds are selected on `online_calib`; `online_eval` is used only for final measurement.",
        "",
        f"Baseline strict satisfaction: {baseline_safe}/{len(baseline)} rows.",
        f"All conservative-rule strict satisfaction: {safe_count}/{len(rows)} rows.",
        "",
        "## Baseline Calibration Transfer",
        "",
    ]
    fields = [
        ("Dataset", "dataset"),
        ("Method", "method"),
        ("alpha", "alpha"),
        ("Rule", "calibration_rule"),
        ("Calib target", "calib_target"),
        ("Calib FPR", "calib_FPR"),
        ("Calib UCB", "calib_FPR_upper_bound"),
        ("Eval FPR", "eval_FPR"),
        ("Eval TPR", "eval_TPR"),
        ("Hit", "eval_hit_rate"),
        ("Safe", "eval_constraint_satisfied"),
    ]
    lines.extend(_table(baseline, fields))
    lines.extend(["", "## Conservative Target Comparison", ""])
    lines.extend(_table([r for r in rows if str(r.get("calibration_rule", "")).startswith("empirical_target_")], fields))
    lines.extend(["", "## Confidence-Bound Calibration Comparison", ""])
    lines.extend(_table([r for r in rows if r.get("calibration_rule") == "ucb_wilson_95"], fields))
    lines.extend(["", "## Best Safe Rule Per Dataset/Method", ""])
    lines.extend(_table(best_safe, fields))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Conservative calibration improves safety only when it reduces eval FPR below the target without eliminating too much hit rate.",
            "- A rule is not considered safe unless `eval_FPR <= alpha` on the fixed eval split.",
            "- These are still single-seed calibration-transfer results; they do not prove robust calibration across seeds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate conservative online-mined calibration rules.")
    ap.add_argument("--pair_dirs", nargs="+", default=DEFAULT_PAIR_DIRS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--target_factors", nargs="+", type=float, default=DEFAULT_TARGET_FACTORS)
    ap.add_argument("--output_dir", default="results/conservative_calibration_transfer")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_rows(
        pair_dirs=[Path(p) for p in args.pair_dirs],
        methods=args.methods,
        alphas=args.alphas,
        target_factors=args.target_factors,
    )
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "conservative_calibration_transfer.csv"
    md_path = out_dir / "conservative_calibration_transfer.md"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
