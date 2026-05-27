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
DEFAULT_SELECTION = "results/candidate_fpr_matched_summary/candidate_fpr_matched_summary.csv"
METHODS = ["cosine", "ours_weighted_ensemble"]
BUCKETS = [
    ("[0.70,0.80)", 0.70, 0.80, False),
    ("[0.80,0.85)", 0.80, 0.85, False),
    ("[0.85,0.90)", 0.85, 0.90, False),
    ("[0.90,0.95)", 0.90, 0.95, False),
    ("[0.95,1.00]", 0.95, 1.000001, True),
]

FIELDS = [
    "dataset",
    "bucket",
    "constraint_used",
    "cosine_selected_param",
    "ours_weighted_ensemble_selected_param",
    "number_of_examples",
    "ours_number_of_examples",
    "H1_ratio",
    "H0_ratio",
    "cosine_selected_TPR",
    "ours_weighted_ensemble_selected_TPR",
    "cosine_selected_FPR",
    "ours_weighted_ensemble_selected_FPR",
    "cosine_hit_rate",
    "ours_weighted_ensemble_hit_rate",
    "cosine_TP",
    "cosine_FP",
    "cosine_TN",
    "cosine_FN",
    "ours_weighted_ensemble_TP",
    "ours_weighted_ensemble_FP",
    "ours_weighted_ensemble_TN",
    "ours_weighted_ensemble_FN",
]


def _float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _truth(value: Any) -> Optional[bool]:
    if value in {True, "True", "true", "1", 1}:
        return True
    if value in {False, "False", "false", "0", 0}:
        return False
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


def _bucket_for(similarity: float) -> Optional[str]:
    for name, lo, hi, inclusive_hi in BUCKETS:
        if inclusive_hi:
            if lo <= similarity <= hi:
                return name
        elif lo <= similarity < hi:
            return name
    return None


def load_selected_params(selection_path: Path, alpha: float) -> Dict[Tuple[str, str], str]:
    rows = _read_csv(selection_path)
    selected: Dict[Tuple[str, str], str] = {}
    for row in rows:
        if row.get("method") not in METHODS:
            continue
        if row.get("metric_reliability") != "reliable":
            continue
        if not str(row.get("subset_name", "")).endswith("_full"):
            continue
        row_alpha = _float(row.get("alpha"))
        if row_alpha is None or abs(row_alpha - alpha) > 1e-12:
            continue
        param = row.get("selected_delta_or_threshold")
        if param in {None, ""}:
            continue
        selected[(str(row.get("dataset")), str(row.get("method")))] = str(param)
    missing = [
        f"{dataset}/{method}"
        for dataset in ["SemCacheLMArena", "SemCacheSearchQueries"]
        for method in METHODS
        if (dataset, method) not in selected
    ]
    if missing:
        raise ValueError(
            "Candidate-FPR selections are missing for alpha "
            f"{alpha}: {', '.join(missing)}. Run python -m experiments.candidate_fpr_matched_summary first."
        )
    return selected


def _empty_counts() -> Dict[str, int]:
    return {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "H1": 0, "H0": 0}


def _add_counts(counts: Dict[str, int], decision: str, nearest_correct: bool, correctness: Optional[bool]) -> None:
    if nearest_correct:
        counts["H1"] += 1
    else:
        counts["H0"] += 1

    is_hit = decision in {"hit", "exploit"}
    if is_hit:
        if correctness is True:
            counts["TP"] += 1
        else:
            counts["FP"] += 1
    else:
        if nearest_correct:
            counts["FN"] += 1
        else:
            counts["TN"] += 1


def collect_counts(raw_files: Sequence[Path], selected: Dict[Tuple[str, str], str]) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    counts: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(_empty_counts)
    for path in raw_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw decisions file: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                dataset = str(row.get("dataset"))
                method = str(row.get("method"))
                if method not in METHODS:
                    continue
                if not str(row.get("subset_name", "")).endswith("_full"):
                    continue
                selected_param = selected.get((dataset, method))
                if selected_param is None or str(row.get("delta_or_threshold")) != selected_param:
                    continue
                similarity = _float(row.get("similarity_score"))
                if similarity is None:
                    continue
                bucket = _bucket_for(similarity)
                if bucket is None:
                    continue
                nearest_correct = _truth(row.get("nearest_would_be_correct"))
                if nearest_correct is None:
                    continue
                correctness = _truth(row.get("correctness"))
                _add_counts(counts[(dataset, method, bucket)], str(row.get("decision")), nearest_correct, correctness)
    return counts


def _rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _metrics(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp = counts["TP"]
    fp = counts["FP"]
    tn = counts["TN"]
    fn = counts["FN"]
    n = tp + fp + tn + fn
    return {
        "n": float(n),
        "H1_ratio": _rate(counts["H1"], counts["H1"] + counts["H0"]),
        "H0_ratio": _rate(counts["H0"], counts["H1"] + counts["H0"]),
        "TPR": _rate(tp, tp + fn),
        "FPR": _rate(fp, fp + tn),
        "hit_rate": _rate(tp + fp, n),
    }


def build_rows(raw_files: Sequence[Path], selection_path: Path, alpha: float) -> List[Dict[str, Any]]:
    selected = load_selected_params(selection_path, alpha)
    counts = collect_counts(raw_files, selected)
    rows: List[Dict[str, Any]] = []
    for dataset in ["SemCacheLMArena", "SemCacheSearchQueries"]:
        for bucket, *_ in BUCKETS:
            cosine_counts = counts.get((dataset, "cosine", bucket), _empty_counts())
            ours_counts = counts.get((dataset, "ours_weighted_ensemble", bucket), _empty_counts())
            cosine_metrics = _metrics(cosine_counts)
            ours_metrics = _metrics(ours_counts)
            rows.append(
                {
                    "dataset": dataset,
                    "bucket": bucket,
                    "constraint_used": f"candidate_fpr_alpha_{alpha:g}",
                    "cosine_selected_param": selected.get((dataset, "cosine")),
                    "ours_weighted_ensemble_selected_param": selected.get((dataset, "ours_weighted_ensemble")),
                    "number_of_examples": int(cosine_metrics["n"] or 0),
                    "ours_number_of_examples": int(ours_metrics["n"] or 0),
                    "H1_ratio": cosine_metrics["H1_ratio"],
                    "H0_ratio": cosine_metrics["H0_ratio"],
                    "cosine_selected_TPR": cosine_metrics["TPR"],
                    "ours_weighted_ensemble_selected_TPR": ours_metrics["TPR"],
                    "cosine_selected_FPR": cosine_metrics["FPR"],
                    "ours_weighted_ensemble_selected_FPR": ours_metrics["FPR"],
                    "cosine_hit_rate": cosine_metrics["hit_rate"],
                    "ours_weighted_ensemble_hit_rate": ours_metrics["hit_rate"],
                    "cosine_TP": cosine_counts["TP"],
                    "cosine_FP": cosine_counts["FP"],
                    "cosine_TN": cosine_counts["TN"],
                    "cosine_FN": cosine_counts["FN"],
                    "ours_weighted_ensemble_TP": ours_counts["TP"],
                    "ours_weighted_ensemble_FP": ours_counts["FP"],
                    "ours_weighted_ensemble_TN": ours_counts["TN"],
                    "ours_weighted_ensemble_FN": ours_counts["FN"],
                }
            )
    return rows


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]], alpha: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("Dataset", "dataset"),
        ("Bucket", "bucket"),
        ("n", "number_of_examples"),
        ("H1 ratio", "H1_ratio"),
        ("H0 ratio", "H0_ratio"),
        ("Cos TPR", "cosine_selected_TPR"),
        ("Ours TPR", "ours_weighted_ensemble_selected_TPR"),
        ("Cos FPR", "cosine_selected_FPR"),
        ("Ours FPR", "ours_weighted_ensemble_selected_FPR"),
        ("Cos hit", "cosine_hit_rate"),
        ("Ours hit", "ours_weighted_ensemble_hit_rate"),
    ]
    numeric = {
        "H1_ratio",
        "H0_ratio",
        "cosine_selected_TPR",
        "ours_weighted_ensemble_selected_TPR",
        "cosine_selected_FPR",
        "ours_weighted_ensemble_selected_FPR",
        "cosine_hit_rate",
        "ours_weighted_ensemble_hit_rate",
    }
    lines = [
        "# Hard-Negative Bucket Analysis",
        "",
        f"Rows use the full-stream configurations selected under candidate-level FPR target alpha `{alpha:g}`.",
        "",
        "Buckets are based on the nearest-neighbor cosine score recorded in each method's online cache-policy raw decisions. Because cache states can diverge after different hit/miss decisions, `n` and H1/H0 ratios are taken from the cosine-selected rows; the CSV also includes `ours_number_of_examples`.",
        "",
        "| " + " | ".join(title for title, _ in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for _, key in fields:
            values.append(_fmt(row.get(key)) if key in numeric else str(row.get(key, "")))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a bucketed decomposition of the selected streaming policies, not a fresh threshold selection inside each bucket.",
            "- The bucket table should be read as evidence about where the selected learned scorer gains or loses relative to raw cosine, not as a proof that all cosine ranges improve.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bucket selected policies by nearest-neighbor cosine range.")
    ap.add_argument("--raw_files", nargs="+", default=DEFAULT_RAW_FILES)
    ap.add_argument("--selection_csv", default=DEFAULT_SELECTION)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output_dir", default="results/hard_negative_bucket_analysis")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_rows([Path(p) for p in args.raw_files], Path(args.selection_csv), args.alpha)
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "hard_negative_bucket_analysis.csv"
    md_path = out_dir / "hard_negative_bucket_analysis.md"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows, args.alpha)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
