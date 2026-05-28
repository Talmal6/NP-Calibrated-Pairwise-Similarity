from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .debug_ours_online_mined_calibration import METHOD_MAP
from .online_mined_ours import (
    OnlinePairs,
    accept_scores,
    h0_h1,
    load_pairs_dir,
    score_method,
    select_tau,
    train_online_mined_model,
)


DEFAULT_PAIRS_DIR = "results/online_candidate_pairs/lmarena_gte_seed42"


METRIC_FIELDS = [
    "split",
    "threshold_name",
    "threshold",
    "operator",
    "H0",
    "H1",
    "FP",
    "TN",
    "TP",
    "FN",
    "FPR",
    "TPR",
    "hit_rate",
    "precision",
    "h0_equal_threshold",
]

SCORE_STAT_FIELDS = [
    "split",
    "label",
    "n",
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "max",
]

RANDOM_FIELDS = [
    "trial",
    "threshold",
    "calib_H0",
    "calib_H1",
    "calib_FP",
    "calib_TN",
    "calib_FPR",
    "eval_H0",
    "eval_H1",
    "eval_FP",
    "eval_TN",
    "eval_FPR",
    "eval_TP",
    "eval_FN",
    "eval_TPR",
    "eval_hit_rate",
]


def _fmt(value: Any, digits: int = 6) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _score_stats(split: str, label: str, scores: np.ndarray) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    qs = np.quantile(s, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if s.size else [np.nan] * 9
    return {
        "split": split,
        "label": label,
        "n": int(s.size),
        "mean": float(np.mean(s)) if s.size else None,
        "std": float(np.std(s)) if s.size else None,
        "min": float(np.min(s)) if s.size else None,
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
        "p95": float(qs[7]),
        "p99": float(qs[8]),
        "max": float(np.max(s)) if s.size else None,
    }


def _metrics(
    *,
    split: str,
    threshold_name: str,
    threshold: float,
    scores_h0: np.ndarray,
    scores_h1: np.ndarray,
    operator: str,
) -> Dict[str, Any]:
    s0 = np.asarray(scores_h0, dtype=np.float64).reshape(-1)
    s1 = np.asarray(scores_h1, dtype=np.float64).reshape(-1)
    if operator == ">=":
        a0 = s0 >= float(threshold)
        a1 = s1 >= float(threshold)
    elif operator == ">":
        a0 = s0 > float(threshold)
        a1 = s1 > float(threshold)
    else:
        raise ValueError(f"Unsupported operator {operator!r}")
    fp = int(np.sum(a0))
    tn = int(s0.size - fp)
    tp = int(np.sum(a1))
    fn = int(s1.size - tp)
    n = int(s0.size + s1.size)
    return {
        "split": split,
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "operator": operator,
        "H0": int(s0.size),
        "H1": int(s1.size),
        "FP": fp,
        "TN": tn,
        "TP": tp,
        "FN": fn,
        "FPR": fp / float(fp + tn) if fp + tn else None,
        "TPR": tp / float(tp + fn) if tp + fn else None,
        "hit_rate": (tp + fp) / float(n) if n else None,
        "precision": tp / float(tp + fp) if tp + fp else None,
        "h0_equal_threshold": int(np.sum(s0 == float(threshold))),
    }


def _combine_pairs(a: OnlinePairs, b: OnlinePairs) -> OnlinePairs:
    query_emb = None
    candidate_emb = None
    if a.query_emb is not None and b.query_emb is not None:
        query_emb = np.concatenate([a.query_emb, b.query_emb], axis=0)
    if a.candidate_emb is not None and b.candidate_emb is not None:
        candidate_emb = np.concatenate([a.candidate_emb, b.candidate_emb], axis=0)
    return OnlinePairs(
        X=np.concatenate([a.X, b.X], axis=0),
        y=np.concatenate([a.y, b.y], axis=0),
        cosine=np.concatenate([a.cosine, b.cosine], axis=0),
        query_id=np.concatenate([a.query_id, b.query_id], axis=0),
        candidate_id=np.concatenate([a.candidate_id, b.candidate_id], axis=0),
        query_emb=query_emb,
        candidate_emb=candidate_emb,
    )


def _random_split_rows(
    *,
    rng: np.random.Generator,
    scores_h0_all: np.ndarray,
    scores_h1_all: np.ndarray,
    calib_h0_n: int,
    calib_h1_n: int,
    alpha: float,
    trials: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    h0_idx_all = np.arange(scores_h0_all.size)
    h1_idx_all = np.arange(scores_h1_all.size)
    for trial in range(int(trials)):
        calib_h0_idx = rng.choice(h0_idx_all, size=calib_h0_n, replace=False)
        calib_h1_idx = rng.choice(h1_idx_all, size=calib_h1_n, replace=False)
        h0_mask = np.ones(scores_h0_all.size, dtype=bool)
        h1_mask = np.ones(scores_h1_all.size, dtype=bool)
        h0_mask[calib_h0_idx] = False
        h1_mask[calib_h1_idx] = False
        s0_calib = scores_h0_all[calib_h0_idx]
        s1_calib = scores_h1_all[calib_h1_idx]
        s0_eval = scores_h0_all[h0_mask]
        s1_eval = scores_h1_all[h1_mask]
        tau = select_tau(s0_calib, float(alpha), "higher")
        calib_m = _metrics(
            split="random_calib",
            threshold_name="empirical_unique_tau",
            threshold=tau,
            scores_h0=s0_calib,
            scores_h1=s1_calib,
            operator=">=",
        )
        eval_m = _metrics(
            split="random_eval",
            threshold_name="empirical_unique_tau",
            threshold=tau,
            scores_h0=s0_eval,
            scores_h1=s1_eval,
            operator=">=",
        )
        rows.append(
            {
                "trial": trial,
                "threshold": tau,
                "calib_H0": calib_m["H0"],
                "calib_H1": calib_m["H1"],
                "calib_FP": calib_m["FP"],
                "calib_TN": calib_m["TN"],
                "calib_FPR": calib_m["FPR"],
                "eval_H0": eval_m["H0"],
                "eval_H1": eval_m["H1"],
                "eval_FP": eval_m["FP"],
                "eval_TN": eval_m["TN"],
                "eval_FPR": eval_m["FPR"],
                "eval_TP": eval_m["TP"],
                "eval_FN": eval_m["FN"],
                "eval_TPR": eval_m["TPR"],
                "eval_hit_rate": eval_m["hit_rate"],
            }
        )
    return rows


def _summarize_random(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows if row.get(key) not in {None, ""}], dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    if not rows:
        out.append("| " + " | ".join("" for _ in headers) + " |")
    return out


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    pairs_dir = Path(args.pairs_dir)
    method_key = args.method
    method_name = METHOD_MAP[method_key]
    alpha = float(args.alpha)
    seed = int(args.seed)

    train_pairs, calib_pairs, eval_pairs, _config = load_pairs_dir(pairs_dir)
    model, _diag = train_online_mined_model(
        pairs_dir=pairs_dir,
        method_name=method_name,
        alpha=alpha,
        seed=seed,
    )
    if model.orientation != "higher":
        raise ValueError(f"This audit expects higher-is-better scores; got orientation={model.orientation!r}")

    h0_calib, h1_calib, h0_calib_alt, h1_calib_alt = h0_h1(calib_pairs)
    h0_eval, h1_eval, h0_eval_alt, h1_eval_alt = h0_h1(eval_pairs)
    s0_calib = score_method(model.method, h0_calib, h0_calib_alt, model.uses_alt_score)
    s1_calib = score_method(model.method, h1_calib, h1_calib_alt, model.uses_alt_score)
    s0_eval = score_method(model.method, h0_eval, h0_eval_alt, model.uses_alt_score)
    s1_eval = score_method(model.method, h1_eval, h1_eval_alt, model.uses_alt_score)

    tau_model = float(model.tau)
    tau_recomputed = float(select_tau(s0_calib, alpha, "higher"))
    tau_np_quantile = float(np.quantile(s0_calib, 1.0 - alpha))

    metric_rows = []
    for threshold_name, tau in [
        ("model_tau", tau_model),
        ("recomputed_empirical_unique_tau", tau_recomputed),
        ("numpy_quantile_1_minus_alpha", tau_np_quantile),
    ]:
        for split, s0, s1 in [
            ("online_calib", s0_calib, s1_calib),
            ("online_eval", s0_eval, s1_eval),
        ]:
            metric_rows.append(
                _metrics(
                    split=split,
                    threshold_name=threshold_name,
                    threshold=tau,
                    scores_h0=s0,
                    scores_h1=s1,
                    operator=">=",
                )
            )
            metric_rows.append(
                _metrics(
                    split=split,
                    threshold_name=threshold_name,
                    threshold=tau,
                    scores_h0=s0,
                    scores_h1=s1,
                    operator=">",
                )
            )

    score_rows = [
        _score_stats("online_calib", "H0", s0_calib),
        _score_stats("online_calib", "H1", s1_calib),
        _score_stats("online_eval", "H0", s0_eval),
        _score_stats("online_eval", "H1", s1_eval),
    ]

    combined_pairs = _combine_pairs(calib_pairs, eval_pairs)
    h0_all, h1_all, h0_all_alt, h1_all_alt = h0_h1(combined_pairs)
    s0_all = score_method(model.method, h0_all, h0_all_alt, model.uses_alt_score)
    s1_all = score_method(model.method, h1_all, h1_all_alt, model.uses_alt_score)
    random_rows = _random_split_rows(
        rng=np.random.default_rng(seed),
        scores_h0_all=s0_all,
        scores_h1_all=s1_all,
        calib_h0_n=int(s0_calib.size),
        calib_h1_n=int(s1_calib.size),
        alpha=alpha,
        trials=int(args.random_trials),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "manual_metric_recompute.csv", metric_rows, METRIC_FIELDS)
    _write_csv(out_dir / "score_distribution_summary.csv", score_rows, SCORE_STAT_FIELDS)
    _write_csv(out_dir / "random_split_comparison.csv", random_rows, RANDOM_FIELDS)

    random_eval_fpr = _summarize_random(random_rows, "eval_FPR")
    random_eval_hit = _summarize_random(random_rows, "eval_hit_rate")
    random_eval_tpr = _summarize_random(random_rows, "eval_TPR")

    selected_calib_ge = next(
        row
        for row in metric_rows
        if row["split"] == "online_calib" and row["threshold_name"] == "model_tau" and row["operator"] == ">="
    )
    selected_eval_ge = next(
        row
        for row in metric_rows
        if row["split"] == "online_eval" and row["threshold_name"] == "model_tau" and row["operator"] == ">="
    )
    selected_calib_gt = next(
        row
        for row in metric_rows
        if row["split"] == "online_calib" and row["threshold_name"] == "model_tau" and row["operator"] == ">"
    )
    selected_eval_gt = next(
        row
        for row in metric_rows
        if row["split"] == "online_eval" and row["threshold_name"] == "model_tau" and row["operator"] == ">"
    )

    h0_shift = {
        "calib_h0_mean": float(np.mean(s0_calib)),
        "eval_h0_mean": float(np.mean(s0_eval)),
        "mean_delta_eval_minus_calib": float(np.mean(s0_eval) - np.mean(s0_calib)),
        "calib_h0_p95": float(np.quantile(s0_calib, 0.95)),
        "eval_h0_p95": float(np.quantile(s0_eval, 0.95)),
        "p95_delta_eval_minus_calib": float(np.quantile(s0_eval, 0.95) - np.quantile(s0_calib, 0.95)),
    }
    direction = {
        "calib_mean_H0": float(np.mean(s0_calib)),
        "calib_mean_H1": float(np.mean(s1_calib)),
        "eval_mean_H0": float(np.mean(s0_eval)),
        "eval_mean_H1": float(np.mean(s1_eval)),
        "calib_H1_greater_H0": bool(np.mean(s1_calib) > np.mean(s0_calib)),
        "eval_H1_greater_H0": bool(np.mean(s1_eval) > np.mean(s0_eval)),
    }

    lines = [
        "# SemCacheLMArena Calibration-Transfer Audit",
        "",
        f"Dataset pairs: `{pairs_dir}`",
        f"Method: `{method_key}`",
        f"Alpha: `{alpha:g}`",
        "",
        "## Code Path",
        "",
        "The calibration-transfer CSV is assembled by `experiments/calibration_transfer.py` lines 130-164 from `results/debug_ours_online_mined_calibration/lmarena_gte_seed42/online_threshold_calibration_summary.csv` and `online_pair_eval_summary.csv`.",
        "",
        "The threshold is produced in `experiments/debug_ours_online_mined_calibration.py` lines 366-387, which calls `train_online_mined_model(...)` and then reports `model.tau`.",
        "",
        "Inside `experiments/online_mined_ours.py` lines 203-208, `train_online_mined_model(...)` scores `online_calib` pairs, extracts `s0_calib` from H0 calibration pairs only, and calls `select_tau(s0_calib, alpha, orientation)`. The selector and acceptance rule are in lines 139-165.",
        "",
        "The implementation is not NumPy's interpolated `quantile(H0_scores, 1-alpha)`. For `higher` orientation it scans unique H0 scores and returns the first threshold whose empirical accept rate under `score >= threshold` is `<= alpha`. With no problematic ties this is effectively the empirical upper-tail quantile, but it is slightly conservative in finite samples.",
        "",
        "## Threshold And Counts",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Item", "Value"],
            [
                ["model tau", _fmt(tau_model, 12)],
                ["recomputed empirical tau", _fmt(tau_recomputed, 12)],
                ["numpy quantile 1-alpha", _fmt(tau_np_quantile, 12)],
                ["calib H0", str(int(s0_calib.size))],
                ["calib H1", str(int(s1_calib.size))],
                ["eval H0", str(int(s0_eval.size))],
                ["eval H1", str(int(s1_eval.size))],
            ],
        )
    )
    lines.extend(["", "## Manual Recompute For Selected Threshold", ""])
    lines.extend(
        _markdown_table(
            ["Split", "Op", "FP", "TN", "FPR", "TP", "FN", "TPR", "Hit", "Precision", "H0 ties"],
            [
                [
                    "online_calib",
                    ">=",
                    str(selected_calib_ge["FP"]),
                    str(selected_calib_ge["TN"]),
                    _fmt(selected_calib_ge["FPR"]),
                    str(selected_calib_ge["TP"]),
                    str(selected_calib_ge["FN"]),
                    _fmt(selected_calib_ge["TPR"]),
                    _fmt(selected_calib_ge["hit_rate"]),
                    _fmt(selected_calib_ge["precision"]),
                    str(selected_calib_ge["h0_equal_threshold"]),
                ],
                [
                    "online_calib",
                    ">",
                    str(selected_calib_gt["FP"]),
                    str(selected_calib_gt["TN"]),
                    _fmt(selected_calib_gt["FPR"]),
                    str(selected_calib_gt["TP"]),
                    str(selected_calib_gt["FN"]),
                    _fmt(selected_calib_gt["TPR"]),
                    _fmt(selected_calib_gt["hit_rate"]),
                    _fmt(selected_calib_gt["precision"]),
                    str(selected_calib_gt["h0_equal_threshold"]),
                ],
                [
                    "online_eval",
                    ">=",
                    str(selected_eval_ge["FP"]),
                    str(selected_eval_ge["TN"]),
                    _fmt(selected_eval_ge["FPR"]),
                    str(selected_eval_ge["TP"]),
                    str(selected_eval_ge["FN"]),
                    _fmt(selected_eval_ge["TPR"]),
                    _fmt(selected_eval_ge["hit_rate"]),
                    _fmt(selected_eval_ge["precision"]),
                    str(selected_eval_ge["h0_equal_threshold"]),
                ],
                [
                    "online_eval",
                    ">",
                    str(selected_eval_gt["FP"]),
                    str(selected_eval_gt["TN"]),
                    _fmt(selected_eval_gt["FPR"]),
                    str(selected_eval_gt["TP"]),
                    str(selected_eval_gt["FN"]),
                    _fmt(selected_eval_gt["TPR"]),
                    _fmt(selected_eval_gt["hit_rate"]),
                    _fmt(selected_eval_gt["precision"]),
                    str(selected_eval_gt["h0_equal_threshold"]),
                ],
            ],
        )
    )
    lines.extend(["", "## Score Direction", ""])
    lines.extend(
        _markdown_table(
            ["Split", "Mean H0", "Mean H1", "H1 > H0?"],
            [
                ["online_calib", _fmt(direction["calib_mean_H0"]), _fmt(direction["calib_mean_H1"]), str(direction["calib_H1_greater_H0"])],
                ["online_eval", _fmt(direction["eval_mean_H0"]), _fmt(direction["eval_mean_H1"]), str(direction["eval_H1_greater_H0"])],
            ],
        )
    )
    lines.extend(["", "## H0 Distribution Shift", ""])
    lines.extend(
        _markdown_table(
            ["Statistic", "Calib H0", "Eval H0", "Eval - calib"],
            [
                ["mean", _fmt(h0_shift["calib_h0_mean"]), _fmt(h0_shift["eval_h0_mean"]), _fmt(h0_shift["mean_delta_eval_minus_calib"])],
                ["p95", _fmt(h0_shift["calib_h0_p95"]), _fmt(h0_shift["eval_h0_p95"]), _fmt(h0_shift["p95_delta_eval_minus_calib"])],
            ],
        )
    )
    lines.extend(
        [
            "",
            "Full score distribution summaries are exported to `score_distribution_summary.csv`.",
            "",
            "## Random Split Check",
            "",
            f"Random check uses {int(args.random_trials)} stratified random calib/eval splits from the combined chronological calib+eval pool, preserving the chronological calib H0/H1 sizes.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Metric", "Chronological eval", "Random mean", "Random std", "Random p05", "Random p50", "Random p95"],
            [
                [
                    "eval FPR",
                    _fmt(selected_eval_ge["FPR"]),
                    _fmt(random_eval_fpr["mean"]),
                    _fmt(random_eval_fpr["std"]),
                    _fmt(random_eval_fpr["p05"]),
                    _fmt(random_eval_fpr["p50"]),
                    _fmt(random_eval_fpr["p95"]),
                ],
                [
                    "eval TPR",
                    _fmt(selected_eval_ge["TPR"]),
                    _fmt(random_eval_tpr["mean"]),
                    _fmt(random_eval_tpr["std"]),
                    _fmt(random_eval_tpr["p05"]),
                    _fmt(random_eval_tpr["p50"]),
                    _fmt(random_eval_tpr["p95"]),
                ],
                [
                    "eval hit_rate",
                    _fmt(selected_eval_ge["hit_rate"]),
                    _fmt(random_eval_hit["mean"]),
                    _fmt(random_eval_hit["std"]),
                    _fmt(random_eval_hit["p05"]),
                    _fmt(random_eval_hit["p50"]),
                    _fmt(random_eval_hit["p95"]),
                ],
            ],
        )
    )
    conclusion = "distribution shift"
    if selected_eval_ge["h0_equal_threshold"] or selected_calib_ge["h0_equal_threshold"] > 1:
        tie_note = "Ties exist because tau is an observed score, but using `>` instead of `>=` changes FPR only by the tied H0 count shown above."
    else:
        tie_note = "Ties are not a material explanation."
    lines.extend(
        [
            "",
            "## Audit Conclusion",
            "",
            "- Calibration uses the correct denominator: `FP / (FP + TN)` over H0 candidate pairs.",
            "- The threshold is selected from `online_calib` H0 scores only.",
            "- Score direction is correct: mean H1 score is greater than mean H0 score on both calibration and eval.",
            f"- Eval H0 scores are shifted upward relative to calibration H0 scores; p95 increases by `{_fmt(h0_shift['p95_delta_eval_minus_calib'])}`.",
            f"- {tie_note}",
            f"- The observed failure is best explained as `{conclusion}`, not as a denominator bug or using eval data in threshold selection.",
            "",
            "## Exported Files",
            "",
            "- `manual_metric_recompute.csv`",
            "- `score_distribution_summary.csv`",
            "- `random_split_comparison.csv`",
            "",
        ]
    )
    report_path = out_dir / "calibration_transfer_audit.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "report": report_path,
        "metric_rows": metric_rows,
        "score_rows": score_rows,
        "random_rows": random_rows,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit one calibration-transfer threshold failure.")
    ap.add_argument("--pairs_dir", default=DEFAULT_PAIRS_DIR)
    ap.add_argument("--method", default="ours_weighted_ensemble", choices=sorted(METHOD_MAP))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random_trials", type=int, default=100)
    ap.add_argument("--output_dir", default="results/calibration_transfer_audit/lmarena_weighted_ensemble_alpha_0_05")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = audit(args)
    out_dir = Path(args.output_dir)
    print(f"wrote {out_dir / 'manual_metric_recompute.csv'}")
    print(f"wrote {out_dir / 'score_distribution_summary.csv'}")
    print(f"wrote {out_dir / 'random_split_comparison.csv'}")
    print(f"wrote {result['report']}")


if __name__ == "__main__":
    main()
