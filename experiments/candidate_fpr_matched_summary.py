from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUTS = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/summary_metrics.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/summary_metrics.csv",
]
DEFAULT_RAW_FILES = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/raw_decisions_minimal.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/raw_decisions_minimal.csv",
]
DEFAULT_ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.10]
MAIN_METHODS = ["cosine", "ours_whitened_hadamard", "ours_weighted_ensemble"]

SUMMARY_FIELDS = [
    "dataset",
    "subset_name",
    "alpha",
    "method",
    "selected_delta_or_threshold",
    "n",
    "TP",
    "FP",
    "TN",
    "FN",
    "false_positive_rate",
    "true_positive_rate",
    "hit_rate",
    "precision",
    "error_rate_stream",
    "llm_calls",
    "judge_calls",
    "metric_reliability",
    "reliability_note",
    "source_file",
]

DELTA_FIELDS = [
    "dataset",
    "subset_name",
    "alpha",
    "ours_method",
    "cosine_selected_delta_or_threshold",
    "ours_selected_delta_or_threshold",
    "cosine_false_positive_rate",
    "ours_false_positive_rate",
    "cosine_true_positive_rate",
    "ours_true_positive_rate",
    "delta_tpr",
    "cosine_hit_rate",
    "ours_hit_rate",
    "delta_hit_rate",
    "relative_hit_gain",
    "cosine_precision",
    "ours_precision",
    "cosine_error_rate_stream",
    "ours_error_rate_stream",
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


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def load_summary(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for row in _read_csv(path):
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def inspect_vcache_reliability(raw_files: Iterable[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in raw_files:
        if not path.exists():
            continue
        counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"missing": 0, "total": 0})
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("method") != "vcache":
                    continue
                key = (str(row.get("dataset")), str(row.get("subset_name")))
                has_similarity = row.get("similarity_score") not in {None, ""}
                has_nearest = row.get("nearest_prompt_id") not in {None, ""}
                if has_similarity:
                    counts[key]["total"] += 1
                    if not has_nearest:
                        counts[key]["missing"] += 1
        for key, count in counts.items():
            missing = count["missing"]
            total = count["total"]
            out[key] = {
                "missing_nearest_with_similarity": missing,
                "rows_with_similarity": total,
                "reliable": missing == 0,
            }
    return out


def select_fpr_matched(
    rows: Sequence[Dict[str, Any]],
    *,
    alphas: Sequence[float],
    methods: Sequence[str],
    vcache_status: Dict[Tuple[str, str], Dict[str, Any]],
    include_vcache_reference: bool,
) -> List[Dict[str, Any]]:
    wanted = set(methods)
    if include_vcache_reference:
        wanted.add("vcache")
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    dataset_subsets: set[Tuple[str, str]] = set()
    for row in rows:
        dataset_subset = (str(row.get("dataset")), str(row.get("subset_name")))
        dataset_subsets.add(dataset_subset)
        method = str(row.get("method"))
        if method not in wanted:
            continue
        grouped[(dataset_subset[0], dataset_subset[1], method)].append(row)

    out: List[Dict[str, Any]] = []
    for dataset, subset in sorted(dataset_subsets):
        for method in sorted(wanted):
            group = grouped.get((dataset, subset, method), [])
            if not group:
                for alpha in alphas:
                    out.append(
                        {
                            "dataset": dataset,
                            "subset_name": subset,
                            "alpha": float(alpha),
                            "method": method,
                            "metric_reliability": "no_candidate_rows",
                            "reliability_note": "No rows for this method in the source summary.",
                        }
                    )
                continue

            for alpha in alphas:
                valid: List[Dict[str, Any]] = []
                for row in group:
                    fpr = _float(row.get("false_positive_rate"))
                    tpr = _float(row.get("true_positive_rate"))
                    if fpr is None or tpr is None:
                        continue
                    if fpr <= float(alpha):
                        valid.append(row)
                if not valid:
                    out.append(
                        {
                            "dataset": dataset,
                            "subset_name": subset,
                            "alpha": float(alpha),
                            "method": method,
                            "metric_reliability": "no_valid_config",
                            "reliability_note": "No row had false_positive_rate <= alpha.",
                            "source_file": group[0].get("source_file") if group else "",
                        }
                    )
                    continue
                selected = sorted(
                    valid,
                    key=lambda r: (
                        _float(r.get("true_positive_rate")) or -1.0,
                        _float(r.get("hit_rate")) or -1.0,
                        -(_float(r.get("false_positive_rate")) or 1e9),
                    ),
                    reverse=True,
                )[0]
                reliability = "reliable"
                note = ""
                if method == "vcache":
                    status = vcache_status.get((dataset, subset), {})
                    if not status.get("reliable", False):
                        reliability = "unreliable_candidate_metrics"
                        note = (
                            "vCache raw logs have explore rows with similarity_score but missing nearest_prompt_id; "
                            f"missing={status.get('missing_nearest_with_similarity', 'unknown')}"
                        )
                out.append(
                    {
                        "dataset": dataset,
                        "subset_name": subset,
                        "alpha": float(alpha),
                        "method": method,
                        "selected_delta_or_threshold": selected.get("delta_or_threshold"),
                        "n": selected.get("n"),
                        "TP": selected.get("TP"),
                        "FP": selected.get("FP"),
                        "TN": selected.get("TN"),
                        "FN": selected.get("FN"),
                        "false_positive_rate": selected.get("false_positive_rate"),
                        "true_positive_rate": selected.get("true_positive_rate"),
                        "hit_rate": selected.get("hit_rate"),
                        "precision": selected.get("precision"),
                        "error_rate_stream": selected.get("error_rate_stream"),
                        "llm_calls": selected.get("llm_calls"),
                        "judge_calls": selected.get("judge_calls"),
                        "metric_reliability": reliability,
                        "reliability_note": note,
                        "source_file": selected.get("source_file"),
                    }
                )
    return out


def build_delta_rows(selected: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {
        (r["dataset"], r["subset_name"], str(r["alpha"]), r["method"]): r
        for r in selected
        if r.get("metric_reliability") == "reliable"
    }
    out: List[Dict[str, Any]] = []
    for dataset, subset, alpha, method in sorted(by_key):
        if method not in {"ours_whitened_hadamard", "ours_weighted_ensemble"}:
            continue
        ours = by_key[(dataset, subset, alpha, method)]
        cosine = by_key.get((dataset, subset, alpha, "cosine"))
        if cosine is None:
            continue
        ours_tpr = _float(ours.get("true_positive_rate"))
        cosine_tpr = _float(cosine.get("true_positive_rate"))
        ours_hit = _float(ours.get("hit_rate"))
        cosine_hit = _float(cosine.get("hit_rate"))
        if None in {ours_tpr, cosine_tpr, ours_hit, cosine_hit}:
            continue
        out.append(
            {
                "dataset": dataset,
                "subset_name": subset,
                "alpha": _float(alpha),
                "ours_method": method,
                "cosine_selected_delta_or_threshold": cosine.get("selected_delta_or_threshold"),
                "ours_selected_delta_or_threshold": ours.get("selected_delta_or_threshold"),
                "cosine_false_positive_rate": cosine.get("false_positive_rate"),
                "ours_false_positive_rate": ours.get("false_positive_rate"),
                "cosine_true_positive_rate": cosine_tpr,
                "ours_true_positive_rate": ours_tpr,
                "delta_tpr": ours_tpr - cosine_tpr,  # type: ignore[operator]
                "cosine_hit_rate": cosine_hit,
                "ours_hit_rate": ours_hit,
                "delta_hit_rate": ours_hit - cosine_hit,  # type: ignore[operator]
                "relative_hit_gain": (ours_hit / cosine_hit) if cosine_hit and cosine_hit > 0 else None,  # type: ignore[operator]
                "cosine_precision": cosine.get("precision"),
                "ours_precision": ours.get("precision"),
                "cosine_error_rate_stream": cosine.get("error_rate_stream"),
                "ours_error_rate_stream": ours.get("error_rate_stream"),
                "source_file": ours.get("source_file"),
            }
        )
    out.sort(
        key=lambda r: (
            _float(r.get("delta_tpr")) or -1e9,
            _float(r.get("delta_hit_rate")) or -1e9,
        ),
        reverse=True,
    )
    return out


def _table(rows: Sequence[Dict[str, Any]], fields: Sequence[Tuple[str, str]], limit: Optional[int] = None) -> List[str]:
    numeric_keys = {
        "alpha",
        "false_positive_rate",
        "true_positive_rate",
        "hit_rate",
        "precision",
        "error_rate_stream",
        "delta_tpr",
        "delta_hit_rate",
        "relative_hit_gain",
        "ours_false_positive_rate",
        "cosine_false_positive_rate",
        "ours_true_positive_rate",
        "cosine_true_positive_rate",
    }
    take = list(rows[:limit] if limit is not None else rows)
    lines = [
        "| " + " | ".join(title for title, _ in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in take:
        vals = []
        for _, key in fields:
            val = row.get(key)
            if key in numeric_keys:
                vals.append(_fmt(val))
            elif key == "metric_reliability" and val == "reliable":
                vals.append("")
            else:
                vals.append(str(val if val is not None else ""))
        lines.append("| " + " | ".join(vals) + " |")
    if not take:
        lines.append("| " + " | ".join("" for _ in fields) + " |")
    return lines


def write_markdown(path: Path, selected: Sequence[Dict[str, Any]], deltas: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Candidate-Level FPR-Matched Summary",
        "",
        "This report selects configurations using candidate-level false-positive rate:",
        "",
        "```text",
        "FPR = FP / (FP + TN)",
        "```",
        "",
        "This is different from stream-level error:",
        "",
        "```text",
        "FP / n",
        "```",
        "",
        "For each dataset/subset/method/alpha, rows with `false_positive_rate <= alpha` are eligible, and the selected row is the eligible row with the highest `true_positive_rate`.",
        "",
        "vCache is shown only as a secondary reference. Current vCache candidate-level metrics are marked unreliable when raw logs have missing nearest-candidate metadata for explore decisions.",
        "",
    ]
    best = next((r for r in deltas if r.get("ours_method") == "ours_weighted_ensemble"), deltas[0] if deltas else None)
    if best:
        lines.extend(
            [
                "## Strongest Valid Ours-vs-Cosine Result",
                "",
                (
                    f"`{best['ours_method']}` on `{best['dataset']}` / `{best['subset_name']}` at alpha `{_fmt(best['alpha'], 2)}` "
                    f"has TPR `{_fmt(best['ours_true_positive_rate'])}` versus cosine `{_fmt(best['cosine_true_positive_rate'])}`, "
                    f"delta TPR `{_fmt(best['delta_tpr'])}`."
                ),
                "",
            ]
        )

    fields = [
        ("Dataset", "dataset"),
        ("Subset", "subset_name"),
        ("Alpha", "alpha"),
        ("Method", "method"),
        ("Param", "selected_delta_or_threshold"),
        ("FPR", "false_positive_rate"),
        ("TPR", "true_positive_rate"),
        ("Hit", "hit_rate"),
        ("Precision", "precision"),
        ("FP/n", "error_rate_stream"),
        ("Status", "metric_reliability"),
    ]
    for title, predicate in [
        ("SemCacheLMArena Full Stream", lambda r: r["dataset"] == "SemCacheLMArena" and r["subset_name"].endswith("_full") and r["method"] in MAIN_METHODS),
        ("SemCacheSearchQueries Full Stream", lambda r: r["dataset"] == "SemCacheSearchQueries" and r["subset_name"].endswith("_full") and r["method"] in MAIN_METHODS),
        ("SemCacheLMArena Hard Subsets", lambda r: r["dataset"] == "SemCacheLMArena" and "hard_cos" in r["subset_name"] and r["method"] in MAIN_METHODS),
        ("SemCacheSearchQueries Hard Subsets", lambda r: r["dataset"] == "SemCacheSearchQueries" and "hard_cos" in r["subset_name"] and r["method"] in MAIN_METHODS),
    ]:
        section_rows = sorted(
            [r for r in selected if predicate(r)],
            key=lambda r: (str(r["subset_name"]), float(r["alpha"]), str(r["method"])),
        )
        lines.extend(["## " + title, ""])
        lines.extend(_table(section_rows, fields))
        lines.append("")

    delta_fields = [
        ("Dataset", "dataset"),
        ("Subset", "subset_name"),
        ("Alpha", "alpha"),
        ("Ours", "ours_method"),
        ("Delta TPR", "delta_tpr"),
        ("Delta Hit", "delta_hit_rate"),
        ("Rel Hit", "relative_hit_gain"),
        ("Ours FPR", "ours_false_positive_rate"),
        ("Cos FPR", "cosine_false_positive_rate"),
        ("Ours TPR", "ours_true_positive_rate"),
        ("Cos TPR", "cosine_true_positive_rate"),
    ]
    lines.extend(["## Ours vs Cosine", ""])
    lines.extend(_table(deltas, delta_fields, limit=40))
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This report is candidate-FPR matched. It should not be mixed with the earlier stream-level `FP/n` matched report.",
            "- vCache candidate-FPR rows are not used for main claims until vCache is rerun with complete nearest-candidate metadata.",
            "- The current checked result files are single-seed unless the multi-seed status report says otherwise.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select rows by candidate-level FPR and maximize TPR.")
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--raw_files", nargs="*", default=DEFAULT_RAW_FILES)
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--methods", nargs="+", default=MAIN_METHODS)
    ap.add_argument("--include_vcache_reference", action="store_true", default=True)
    ap.add_argument("--output_dir", default="results/candidate_fpr_matched_summary")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    rows = load_summary(Path(p) for p in args.inputs)
    vcache_status = inspect_vcache_reliability(Path(p) for p in args.raw_files)
    selected = select_fpr_matched(
        rows,
        alphas=args.alphas,
        methods=args.methods,
        vcache_status=vcache_status,
        include_vcache_reference=bool(args.include_vcache_reference),
    )
    deltas = build_delta_rows(selected)
    summary_path = out_dir / "candidate_fpr_matched_summary.csv"
    delta_path = out_dir / "candidate_fpr_matched_ours_vs_cosine.csv"
    md_path = out_dir / "candidate_fpr_matched_summary.md"
    _write_csv(summary_path, selected, SUMMARY_FIELDS)
    _write_csv(delta_path, deltas, DELTA_FIELDS)
    write_markdown(md_path, selected, deltas)
    print(f"wrote {summary_path}")
    print(f"wrote {delta_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
