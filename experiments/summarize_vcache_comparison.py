from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .vcache_metrics import read_csv_dicts, write_csv


FIELDS = [
    "dataset",
    "budget",
    "method",
    "selected_delta_or_threshold",
    "error_rate_stream",
    "hit_rate",
    "precision",
    "false_positive_rate",
    "true_positive_rate",
    "average_latency",
    "llm_calls",
    "judge_calls",
]


def _to_float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def select_matched_budget(rows: Sequence[Dict[str, Any]], budgets: Sequence[float]) -> List[Dict[str, Any]]:
    datasets = sorted({r.get("dataset", "") for r in rows})
    methods = sorted({r.get("method", "") for r in rows})
    out: List[Dict[str, Any]] = []
    for dataset in datasets:
        for budget in budgets:
            for method in methods:
                candidates = []
                for row in rows:
                    if row.get("dataset") != dataset or row.get("method") != method:
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
                        "budget": float(budget),
                        "method": method,
                        "selected_delta_or_threshold": selected.get("delta_or_threshold"),
                        "error_rate_stream": selected.get("error_rate_stream"),
                        "hit_rate": selected.get("hit_rate"),
                        "precision": selected.get("precision"),
                        "false_positive_rate": selected.get("false_positive_rate"),
                        "true_positive_rate": selected.get("true_positive_rate"),
                        "average_latency": selected.get("average_latency"),
                        "llm_calls": selected.get("llm_calls"),
                        "judge_calls": selected.get("judge_calls"),
                    }
                )
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select best runs at matched empirical stream-error budgets.")
    ap.add_argument("--input_dir", default="results/vcache_comparison")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05])
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary metrics: {summary_path}")
    rows = read_csv_dicts(summary_path)
    matched = select_matched_budget(rows, args.budgets)
    write_csv(input_dir / "matched_budget_summary.csv", matched, FIELDS)


if __name__ == "__main__":
    main(sys.argv[1:])

