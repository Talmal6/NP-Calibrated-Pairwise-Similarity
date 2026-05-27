from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vcache_metrics import read_csv_dicts, write_csv


FIELDS = [
    "dataset",
    "subset_name",
    "budget",
    "method",
    "selected_delta_or_threshold",
    "n",
    "TP",
    "FP",
    "TN",
    "FN",
    "error_rate_stream",
    "hit_rate",
    "precision",
    "false_positive_rate",
    "true_positive_rate",
    "average_latency",
    "p50_latency",
    "p95_latency",
    "llm_calls",
    "judge_calls",
    "threshold_source",
    "score_orientation",
    "calibration_source",
    "calibration_train_pairs",
    "calibration_calib_pairs",
    "calibration_eval_pairs",
    "calibration_labels_used",
    "calibration_split_size",
    "calibration_equivalence_mode",
    "source_dir",
]


AGG_FIELDS = [
    "dataset",
    "subset_name",
    "budget",
    "method",
    "n_seeds",
    "mean_hit_rate",
    "std_hit_rate",
    "mean_stream_error",
    "std_stream_error",
    "mean_precision",
    "std_precision",
    "mean_fpr",
    "std_fpr",
    "mean_tpr",
    "std_tpr",
    "mean_average_latency",
    "std_average_latency",
    "mean_llm_calls",
    "std_llm_calls",
    "mean_judge_calls",
    "std_judge_calls",
]


def _to_float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _mean_std(values: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
    nums = [_to_float(v) for v in values]
    arr = np.asarray([x for x in nums if x is not None], dtype=float)
    if arr.size == 0:
        return None, None
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def select_matched_budget(rows: Sequence[Dict[str, Any]], budgets: Sequence[float]) -> List[Dict[str, Any]]:
    keys = sorted({(r.get("dataset", ""), r.get("subset_name", "")) for r in rows})
    methods = sorted({r.get("method", "") for r in rows})
    out: List[Dict[str, Any]] = []
    for dataset, subset_name in keys:
        for budget in budgets:
            for method in methods:
                candidates = []
                for row in rows:
                    if row.get("dataset") != dataset or row.get("subset_name") != subset_name or row.get("method") != method:
                        continue
                    err = _to_float(row.get("error_rate_stream"))
                    hit = _to_float(row.get("hit_rate"))
                    if err is None or hit is None:
                        continue
                    if err <= float(budget):
                        candidates.append((hit, -err, row))
                if not candidates:
                    continue
                _, _, selected = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)[0]
                out.append(
                    {
                        "dataset": dataset,
                        "subset_name": subset_name,
                        "budget": float(budget),
                        "method": method,
                        "selected_delta_or_threshold": selected.get("delta_or_threshold"),
                        "n": selected.get("n"),
                        "TP": selected.get("TP"),
                        "FP": selected.get("FP"),
                        "TN": selected.get("TN"),
                        "FN": selected.get("FN"),
                        "error_rate_stream": selected.get("error_rate_stream"),
                        "hit_rate": selected.get("hit_rate"),
                        "precision": selected.get("precision"),
                        "false_positive_rate": selected.get("false_positive_rate"),
                        "true_positive_rate": selected.get("true_positive_rate"),
                        "average_latency": selected.get("average_latency"),
                        "p50_latency": selected.get("p50_latency"),
                        "p95_latency": selected.get("p95_latency"),
                        "llm_calls": selected.get("llm_calls"),
                        "judge_calls": selected.get("judge_calls"),
                        "threshold_source": selected.get("threshold_source"),
                        "score_orientation": selected.get("score_orientation"),
                        "calibration_source": selected.get("calibration_source"),
                        "calibration_train_pairs": selected.get("calibration_train_pairs"),
                        "calibration_calib_pairs": selected.get("calibration_calib_pairs"),
                        "calibration_eval_pairs": selected.get("calibration_eval_pairs"),
                        "calibration_labels_used": selected.get("calibration_labels_used"),
                        "calibration_split_size": selected.get("calibration_split_size"),
                        "calibration_equivalence_mode": selected.get("calibration_equivalence_mode"),
                        "source_dir": selected.get("source_dir", ""),
                    }
                )
    return out


def aggregate_matched_budget(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset"), row.get("subset_name"), row.get("budget"), row.get("method"))].append(row)

    out: List[Dict[str, Any]] = []
    for (dataset, subset_name, budget, method), group in sorted(grouped.items()):
        hit_m, hit_s = _mean_std([r.get("hit_rate") for r in group])
        err_m, err_s = _mean_std([r.get("error_rate_stream") for r in group])
        prec_m, prec_s = _mean_std([r.get("precision") for r in group])
        fpr_m, fpr_s = _mean_std([r.get("false_positive_rate") for r in group])
        tpr_m, tpr_s = _mean_std([r.get("true_positive_rate") for r in group])
        lat_m, lat_s = _mean_std([r.get("average_latency") for r in group])
        llm_m, llm_s = _mean_std([r.get("llm_calls") for r in group])
        judge_m, judge_s = _mean_std([r.get("judge_calls") for r in group])
        out.append(
            {
                "dataset": dataset,
                "subset_name": subset_name,
                "budget": budget,
                "method": method,
                "n_seeds": len(group),
                "mean_hit_rate": hit_m,
                "std_hit_rate": hit_s,
                "mean_stream_error": err_m,
                "std_stream_error": err_s,
                "mean_precision": prec_m,
                "std_precision": prec_s,
                "mean_fpr": fpr_m,
                "std_fpr": fpr_s,
                "mean_tpr": tpr_m,
                "std_tpr": tpr_s,
                "mean_average_latency": lat_m,
                "std_average_latency": lat_s,
                "mean_llm_calls": llm_m,
                "std_llm_calls": llm_s,
                "mean_judge_calls": judge_m,
                "std_judge_calls": judge_s,
            }
        )
    return out


def _safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unnamed"


def _plot_matched_budget_bars(rows: Sequence[Dict[str, Any]], output_dir: Path, *, aggregated: bool) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    groups: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("dataset"), row.get("subset_name"), row.get("budget"))].append(row)

    for (dataset, subset_name, budget), group in groups.items():
        group = sorted(group, key=lambda r: str(r.get("method")))
        methods = [str(r.get("method")) for r in group]
        if aggregated:
            values = [_to_float(r.get("mean_hit_rate")) or 0.0 for r in group]
            errors = [_to_float(r.get("std_hit_rate")) or 0.0 for r in group]
        else:
            values = [_to_float(r.get("hit_rate")) or 0.0 for r in group]
            errors = None
        plot_dir = output_dir / "plots" / _safe_name(subset_name)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8.0, 4.8))
        x = np.arange(len(methods))
        plt.bar(x, values, yerr=errors, capsize=3 if errors is not None else 0, color="#4c78a8")
        plt.xticks(x, methods, rotation=30, ha="right")
        plt.ylabel("hit_rate")
        plt.xlabel("method")
        plt.title(f"{subset_name} matched budget {budget}")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"matched_budget_hit_rate_budget_{budget}.png", dpi=180)
        plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select and aggregate best real-dataset runs at matched stream-error budgets.")
    ap.add_argument("--input_dir", default=None, help="Single comparison result directory.")
    ap.add_argument("--input_dirs", nargs="+", default=None, help="Multiple comparison result directories to aggregate.")
    ap.add_argument("--output_dir", default=None, help="Output directory for multi-input aggregation.")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05])
    return ap.parse_args(argv)


def _load_summary(input_dir: Path) -> List[Dict[str, Any]]:
    summary_path = input_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary metrics: {summary_path}")
    rows = read_csv_dicts(summary_path)
    for row in rows:
        row["source_dir"] = str(input_dir)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.input_dirs:
        input_dirs = [Path(p) for p in args.input_dirs]
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/vcache_real_datasets_comparison_aggregated")
        output_dir.mkdir(parents=True, exist_ok=True)
        all_matched: List[Dict[str, Any]] = []
        for input_dir in input_dirs:
            matched = select_matched_budget(_load_summary(input_dir), args.budgets)
            for row in matched:
                row["source_dir"] = str(input_dir)
            all_matched.extend(matched)
        aggregated = aggregate_matched_budget(all_matched)
        write_csv(output_dir / "matched_budget_summary.csv", all_matched, FIELDS)
        write_csv(output_dir / "matched_budget_aggregated_summary.csv", aggregated, AGG_FIELDS)
        _plot_matched_budget_bars(aggregated, output_dir, aggregated=True)
        return

    input_dir = Path(args.input_dir or "results/vcache_real_datasets_comparison")
    rows = _load_summary(input_dir)
    matched = select_matched_budget(rows, args.budgets)
    write_csv(input_dir / "matched_budget_summary.csv", matched, FIELDS)
    _plot_matched_budget_bars(matched, input_dir, aggregated=False)


if __name__ == "__main__":
    main(sys.argv[1:])
