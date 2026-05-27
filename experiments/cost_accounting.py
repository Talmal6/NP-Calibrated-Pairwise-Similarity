from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SELECTION = "results/candidate_fpr_matched_summary/candidate_fpr_matched_summary.csv"
DEFAULT_RAW_FILES = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/raw_decisions_minimal.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/raw_decisions_minimal.csv",
]
DEFAULT_SUMMARY_FILES = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/summary_metrics.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/summary_metrics.csv",
]

FIELDS = [
    "dataset",
    "subset_name",
    "alpha",
    "method",
    "selected_delta_or_threshold",
    "n",
    "hit_rate",
    "online_llm_calls_cache_misses",
    "online_judge_calls",
    "calibration_judge_calls",
    "evaluation_judge_calls",
    "total_offline_labeling_calls",
    "raw_judge_calls_sum",
    "summary_judge_calls",
    "calibration_source",
    "calibration_train_pairs",
    "calibration_calib_pairs",
    "calibration_eval_pairs",
    "calibration_judge_calls_status",
    "offline_labeling_status",
    "judge_calls_note",
    "source_file",
]


def _float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _sum_field(target: Dict[str, float], name: str, value: Any) -> None:
    number = _float(value)
    if number is not None:
        target[name] += number


def load_selected(selection_path: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_csv(selection_path):
        if row.get("selected_delta_or_threshold") in {None, ""}:
            continue
        rows.append(row)
    return rows


def load_summary_metadata(paths: Iterable[Path]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for path in paths:
        for row in _read_csv(path):
            key = (
                str(row.get("dataset")),
                str(row.get("subset_name")),
                str(row.get("method")),
                str(row.get("delta_or_threshold")),
            )
            out[key] = row
    return out


def collect_raw_costs(raw_files: Sequence[Path], selected: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    wanted = {
        (
            str(row.get("dataset")),
            str(row.get("subset_name")),
            str(row.get("method")),
            str(row.get("selected_delta_or_threshold")),
        )
        for row in selected
    }
    costs: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "online_llm_calls_cache_misses": 0.0,
            "online_judge_calls": 0.0,
            "evaluation_judge_calls": 0.0,
            "raw_judge_calls_sum": 0.0,
        }
    )
    for path in raw_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw decisions file: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (
                    str(row.get("dataset")),
                    str(row.get("subset_name")),
                    str(row.get("method")),
                    str(row.get("delta_or_threshold")),
                )
                if key not in wanted:
                    continue
                target = costs[key]
                _sum_field(target, "online_llm_calls_cache_misses", row.get("llm_calls"))
                _sum_field(target, "online_judge_calls", row.get("online_judge_calls"))
                _sum_field(target, "evaluation_judge_calls", row.get("evaluation_judge_calls"))
                _sum_field(target, "raw_judge_calls_sum", row.get("judge_calls"))
    return costs


def build_rows(selection_path: Path, raw_files: Sequence[Path], summary_files: Sequence[Path]) -> List[Dict[str, Any]]:
    selected = load_selected(selection_path)
    summary_meta = load_summary_metadata(summary_files)
    raw_costs = collect_raw_costs(raw_files, selected)
    rows: List[Dict[str, Any]] = []
    for row in selected:
        key = (
            str(row.get("dataset")),
            str(row.get("subset_name")),
            str(row.get("method")),
            str(row.get("selected_delta_or_threshold")),
        )
        meta = summary_meta.get(key, {})
        costs = raw_costs.get(key, {})
        calibration_labels = meta.get("calibration_labels_used")
        method = row.get("method")
        calibration_source = meta.get("calibration_source")
        calibration_status = "not_logged"
        offline_status = "not_applicable"
        if calibration_labels not in {None, ""}:
            offline_status = "reported_as_calibration_labels_used_not_runtime_judge_calls"
        if method in {"ours_whitened_hadamard", "ours_weighted_ensemble"} and not calibration_labels:
            offline_status = "missing_calibration_label_count"
        rows.append(
            {
                "dataset": row.get("dataset"),
                "subset_name": row.get("subset_name"),
                "alpha": row.get("alpha"),
                "method": method,
                "selected_delta_or_threshold": row.get("selected_delta_or_threshold"),
                "n": row.get("n"),
                "hit_rate": row.get("hit_rate"),
                "online_llm_calls_cache_misses": costs.get("online_llm_calls_cache_misses"),
                "online_judge_calls": costs.get("online_judge_calls"),
                "calibration_judge_calls": "",
                "evaluation_judge_calls": costs.get("evaluation_judge_calls"),
                "total_offline_labeling_calls": calibration_labels,
                "raw_judge_calls_sum": costs.get("raw_judge_calls_sum"),
                "summary_judge_calls": row.get("judge_calls"),
                "calibration_source": calibration_source,
                "calibration_train_pairs": meta.get("calibration_train_pairs"),
                "calibration_calib_pairs": meta.get("calibration_calib_pairs"),
                "calibration_eval_pairs": meta.get("calibration_eval_pairs"),
                "calibration_judge_calls_status": calibration_status,
                "offline_labeling_status": offline_status,
                "judge_calls_note": "raw judge_calls mixes online_judge_calls and evaluation_judge_calls; do not use it as deployment cost",
                "source_file": row.get("source_file"),
            }
        )
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["subset_name"]), _float(r["alpha"]) or 0.0, str(r["method"])))
    return rows


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fields = [
        ("Dataset", "dataset"),
        ("Subset", "subset_name"),
        ("Alpha", "alpha"),
        ("Method", "method"),
        ("Param", "selected_delta_or_threshold"),
        ("LLM/cache misses", "online_llm_calls_cache_misses"),
        ("Online judge", "online_judge_calls"),
        ("Eval judge", "evaluation_judge_calls"),
        ("Offline labels", "total_offline_labeling_calls"),
        ("Calib judge status", "calibration_judge_calls_status"),
    ]
    numeric = {"alpha", "online_llm_calls_cache_misses", "online_judge_calls", "evaluation_judge_calls", "total_offline_labeling_calls"}
    lines = [
        "| " + " | ".join(title for title, _ in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for _, key in fields:
            values.append(_fmt(row.get(key), 0) if key in numeric and key != "alpha" else (_fmt(row.get(key)) if key == "alpha" else str(row.get(key, ""))))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_rows = [
        row
        for row in rows
        if str(row.get("subset_name", "")).endswith("_full")
        and abs((_float(row.get("alpha")) or -1.0) - 0.05) < 1e-12
        and row.get("method") in {"cosine", "ours_weighted_ensemble", "ours_whitened_hadamard"}
    ]
    lines = [
        "# Cost Accounting",
        "",
        "This report separates deployment-like online cache behavior from evaluation and calibration accounting.",
        "",
        "- `online_llm_calls_cache_misses` is the number of online misses that would call the backing LLM in the cache simulation.",
        "- `online_judge_calls` is logged separately when the online policy itself calls a judge.",
        "- `evaluation_judge_calls` are measurement calls used to label whether a candidate hit would have been correct; these are not deployment savings.",
        "- `calibration_judge_calls` are not separately logged in the current raw files.",
        "- `total_offline_labeling_calls` uses `calibration_labels_used` when available, but that field is not a runtime judge-call log.",
        "",
        "## Full-Stream Alpha 0.05 Snapshot",
        "",
    ]
    lines.extend(_markdown_table(table_rows))
    lines.extend(
        [
            "",
            "## TODO for Next Rerun",
            "",
            "- Log `online_llm_calls` or `cache_miss_llm_calls` separately from any judge accounting.",
            "- Log `online_judge_calls` only for calls made by the deployed online policy.",
            "- Log `calibration_judge_calls` separately from train/calibration/eval pair counts.",
            "- Log `evaluation_judge_calls` only for offline measurement labels.",
            "- Log `offline_label_source` so cluster/equivalence labels are not conflated with LLM judge calls.",
            "",
            "The existing `judge_calls` field should not be used directly for deployment-cost claims because it mixes online and evaluation concepts in the raw decision logs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Separate online, evaluation, and calibration cost accounting.")
    ap.add_argument("--selection_csv", default=DEFAULT_SELECTION)
    ap.add_argument("--raw_files", nargs="+", default=DEFAULT_RAW_FILES)
    ap.add_argument("--summary_files", nargs="+", default=DEFAULT_SUMMARY_FILES)
    ap.add_argument("--output_dir", default="results/cost_accounting")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_rows(
        Path(args.selection_csv),
        [Path(p) for p in args.raw_files],
        [Path(p) for p in args.summary_files],
    )
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "cost_accounting.csv"
    md_path = out_dir / "cost_accounting.md"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
