from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RAW_FILES = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/raw_decisions_minimal.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/raw_decisions_minimal.csv",
]

FIELDS = [
    "dataset",
    "subset_name",
    "seed",
    "delta_or_policy_param",
    "total_rows",
    "rows_missing_nearest_candidate_id",
    "rows_missing_nearest_candidate_label",
    "rows_missing_nearest_candidate_cosine",
    "rows_usable_for_candidate_fpr",
    "fraction_usable_for_candidate_fpr_metrics",
    "candidate_metrics_reliability",
    "source_file",
]


def _present(value: Any) -> bool:
    return value not in {None, "", "None", "nan"}


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if _present(value):
            return value
    return ""


def _read_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required raw decisions file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def build_rows(raw_files: Sequence[Path]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str, str, str, str], Dict[str, int]] = defaultdict(
        lambda: {
            "total_rows": 0,
            "missing_id": 0,
            "missing_label": 0,
            "missing_cosine": 0,
            "usable": 0,
        }
    )
    for path in raw_files:
        for row in _read_csv_rows(path):
            if row.get("method") != "vcache":
                continue
            key = (
                str(row.get("dataset")),
                str(row.get("subset_name")),
                str(row.get("seed")),
                str(row.get("delta_or_threshold")),
                str(path),
            )
            c = counts[key]
            c["total_rows"] += 1
            has_id = _present(_first_present(row, ["nearest_candidate_id", "nearest_prompt_id"]))
            has_label = _present(_first_present(row, ["nearest_candidate_label_H1_or_H0", "nearest_would_be_correct"]))
            has_cosine = _present(_first_present(row, ["nearest_candidate_cosine", "similarity_score"]))
            if not has_id:
                c["missing_id"] += 1
            if not has_label:
                c["missing_label"] += 1
            if not has_cosine:
                c["missing_cosine"] += 1
            if has_id and has_label and has_cosine:
                c["usable"] += 1

    out: List[Dict[str, Any]] = []
    for (dataset, subset, seed, param, source), c in sorted(counts.items()):
        total = c["total_rows"]
        fraction = c["usable"] / total if total else 0.0
        out.append(
            {
                "dataset": dataset,
                "subset_name": subset,
                "seed": seed,
                "delta_or_policy_param": param,
                "total_rows": total,
                "rows_missing_nearest_candidate_id": c["missing_id"],
                "rows_missing_nearest_candidate_label": c["missing_label"],
                "rows_missing_nearest_candidate_cosine": c["missing_cosine"],
                "rows_usable_for_candidate_fpr": c["usable"],
                "fraction_usable_for_candidate_fpr_metrics": fraction,
                "candidate_metrics_reliability": "reliable" if fraction >= 0.99 else "unreliable",
                "source_file": source,
            }
        )
    return out


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = sum(int(r.get("total_rows") or 0) for r in rows)
    usable = sum(int(r.get("rows_usable_for_candidate_fpr") or 0) for r in rows)
    fraction = usable / total_rows if total_rows else 0.0
    status = "reliable" if fraction >= 0.99 else "unreliable"
    lines = [
        "# vCache Metadata Audit",
        "",
        "This audit checks whether current vCache `explore`/`exploit` decision rows contain enough nearest-candidate metadata to compute candidate-level TP/FP/TN/FN reliably.",
        "",
        f"Overall usable fraction for candidate-FPR metrics: `{_fmt(fraction)}`.",
        f"Overall status: `{status}`.",
        "",
        "vCache must not be used for main candidate-level FPR claims unless this usable fraction is close to 100%.",
        "",
        "This audit reads existing raw outputs. The vCache adapter now emits explicit candidate metadata fields for future reruns; rerun vCache before using these metrics for claims.",
        "",
        "| Dataset | Subset | Param | Rows | Missing ID | Missing label | Missing cosine | Usable fraction | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("subset_name", "")),
                    str(row.get("delta_or_policy_param", "")),
                    str(row.get("total_rows", "")),
                    str(row.get("rows_missing_nearest_candidate_id", "")),
                    str(row.get("rows_missing_nearest_candidate_label", "")),
                    str(row.get("rows_missing_nearest_candidate_cosine", "")),
                    _fmt(row.get("fraction_usable_for_candidate_fpr_metrics")),
                    str(row.get("candidate_metrics_reliability", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Required Fixed Logging Fields",
            "",
            "- `decision_type = explore/exploit` and an explicit `cache_hit_or_miss` field.",
            "- `nearest_candidate_id`, `nearest_candidate_cosine`, and `nearest_candidate_label_H1_or_H0` for every row where a candidate exists.",
            "- Per-row TP/FP/TN/FN indicators derived from the nearest-candidate label and decision.",
            "- Separate `online_judge_called` and `eval_judge_called` flags.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit vCache raw decision metadata for candidate-FPR usability.")
    ap.add_argument("--raw_files", nargs="+", default=DEFAULT_RAW_FILES)
    ap.add_argument("--output_dir", default="results/vcache_metadata_audit")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_rows([Path(p) for p in args.raw_files])
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "vcache_metadata_audit.csv"
    md_path = out_dir / "vcache_metadata_audit.md"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
