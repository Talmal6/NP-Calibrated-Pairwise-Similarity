from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_SELECTION = "results/candidate_fpr_matched_summary/candidate_fpr_matched_summary.csv"
DEFAULT_SUMMARY_FILES = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/summary_metrics.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/summary_metrics.csv",
]
DEFAULT_RAW_FILES = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/raw_decisions_minimal.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/raw_decisions_minimal.csv",
]

FIELDS = [
    "dataset",
    "method",
    "seed",
    "alpha",
    "train_labels_used",
    "calibration_labels_used",
    "total_offline_labels",
    "online_llm_calls",
    "online_cache_hits",
    "online_judge_calls",
    "calibration_judge_calls",
    "evaluation_judge_calls",
    "total_judge_calls",
    "deployment_cost_excludes_eval_judge_calls",
    "notes",
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


def _fmt(value: Any, digits: int = 0) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _summary_meta(paths: Sequence[Path]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
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


def _collect_raw(raw_files: Sequence[Path], wanted: set[Tuple[str, str, str, str]]) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "online_llm_calls": 0.0,
            "online_cache_hits": 0.0,
            "online_judge_calls": 0.0,
            "evaluation_judge_calls": 0.0,
            "total_judge_calls": 0.0,
            "seed": 0.0,
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
                target = out[key]
                for field in ["llm_calls", "online_judge_calls", "evaluation_judge_calls", "judge_calls"]:
                    val = _float(row.get(field))
                    if val is not None:
                        if field == "llm_calls":
                            target["online_llm_calls"] += val
                        elif field == "judge_calls":
                            target["total_judge_calls"] += val
                        else:
                            target[field] += val
                if row.get("decision") in {"hit", "exploit"}:
                    target["online_cache_hits"] += 1.0
                seed = _float(row.get("seed"))
                if seed is not None:
                    target["seed"] = seed
    return out


def build_rows(selection_csv: Path, summary_files: Sequence[Path], raw_files: Sequence[Path]) -> List[Dict[str, Any]]:
    selected = [
        row
        for row in _read_csv(selection_csv)
        if row.get("metric_reliability") == "reliable"
        and str(row.get("subset_name", "")).endswith("_full")
        and row.get("selected_delta_or_threshold") not in {None, ""}
    ]
    meta = _summary_meta(summary_files)
    wanted = {
        (
            str(row.get("dataset")),
            str(row.get("subset_name")),
            str(row.get("method")),
            str(row.get("selected_delta_or_threshold")),
        )
        for row in selected
    }
    raw = _collect_raw(raw_files, wanted)
    rows: List[Dict[str, Any]] = []
    for row in selected:
        key = (
            str(row.get("dataset")),
            str(row.get("subset_name")),
            str(row.get("method")),
            str(row.get("selected_delta_or_threshold")),
        )
        m = meta.get(key, {})
        r = raw.get(key, {})
        train_labels = _float(m.get("calibration_train_pairs")) or 0.0
        calib_labels = _float(m.get("calibration_calib_pairs")) or 0.0
        total_offline = train_labels + calib_labels if train_labels or calib_labels else ""
        note = "calibration_judge_calls are not logged separately; offline labels are pair labels, not necessarily runtime judge calls"
        rows.append(
            {
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "seed": int(r.get("seed", 42)),
                "alpha": row.get("alpha"),
                "train_labels_used": int(train_labels) if train_labels else "",
                "calibration_labels_used": int(calib_labels) if calib_labels else "",
                "total_offline_labels": int(total_offline) if total_offline != "" else "",
                "online_llm_calls": int(r.get("online_llm_calls", 0.0)),
                "online_cache_hits": int(r.get("online_cache_hits", 0.0)),
                "online_judge_calls": int(r.get("online_judge_calls", 0.0)),
                "calibration_judge_calls": "",
                "evaluation_judge_calls": int(r.get("evaluation_judge_calls", 0.0)),
                "total_judge_calls": int(r.get("total_judge_calls", 0.0)),
                "deployment_cost_excludes_eval_judge_calls": "true",
                "notes": note,
            }
        )
    rows.sort(key=lambda r: (str(r["dataset"]), float(r["alpha"]), str(r["method"])))
    return rows


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha05 = [r for r in rows if abs((_float(r.get("alpha")) or -1.0) - 0.05) < 1e-12]
    lines = [
        "# Clean Cost Accounting",
        "",
        "This report separates offline setup labels, online runtime calls, and evaluation-only measurement calls.",
        "",
        "- Evaluation judge calls are measurement overhead, not deployment runtime cost.",
        "- Offline train/calibration labels are real setup cost.",
        "- `calibration_judge_calls` are not separately logged in the current raw files.",
        "- Deployment-cost comparisons must exclude evaluation judge calls.",
        "",
        "## Full-Stream Alpha 0.05",
        "",
        "| Dataset | Method | Train labels | Calib labels | Offline labels | Online LLM | Cache hits | Online judge | Eval judge | Excludes eval? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in alpha05:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("method", "")),
                    _fmt(row.get("train_labels_used")),
                    _fmt(row.get("calibration_labels_used")),
                    _fmt(row.get("total_offline_labels")),
                    _fmt(row.get("online_llm_calls")),
                    _fmt(row.get("online_cache_hits")),
                    _fmt(row.get("online_judge_calls")),
                    _fmt(row.get("evaluation_judge_calls")),
                    str(row.get("deployment_cost_excludes_eval_judge_calls", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Missing Fields for Next Rerun",
            "",
            "- `calibration_judge_calls` as a separate setup-cost field.",
            "- `offline_label_source` to distinguish cluster labels, static labels, human labels, and LLM judge labels.",
            "- Per-row boolean flags for `online_judge_called` and `eval_judge_called`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write clean cost accounting from selected full-stream rows.")
    ap.add_argument("--selection_csv", default=DEFAULT_SELECTION)
    ap.add_argument("--summary_files", nargs="+", default=DEFAULT_SUMMARY_FILES)
    ap.add_argument("--raw_files", nargs="+", default=DEFAULT_RAW_FILES)
    ap.add_argument("--output_dir", default="results/cost_accounting_clean")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_rows(Path(args.selection_csv), [Path(p) for p in args.summary_files], [Path(p) for p in args.raw_files])
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "cost_accounting_clean.csv"
    md_path = out_dir / "cost_accounting_clean.md"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
