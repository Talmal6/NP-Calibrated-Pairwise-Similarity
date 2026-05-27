from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_STATIC_SUMMARY = "results/debug_ours_calibration/online_smoke_test_summary.csv"
DEFAULT_STATIC_CONFIG = "results/debug_ours_calibration/config.json"
DEFAULT_ONLINE_DIRS = [
    "results/debug_ours_online_mined_calibration/lmarena_gte_seed42",
    "results/debug_ours_online_mined_calibration/searchqueries_gte_seed42",
]

FIELDS = [
    "dataset",
    "method",
    "calibration_source",
    "eval_distribution",
    "target",
    "threshold_or_param",
    "eval_fpr",
    "eval_fp_over_n",
    "eval_tpr",
    "hit_rate",
    "precision",
    "TP",
    "FP",
    "TN",
    "FN",
    "n",
    "comparison_status",
    "notes",
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
        raise FileNotFoundError(
            f"Missing required input: {path}. "
            "If static/random calibration was not run, rerun it on the same online eval stream and pass --static_summary."
        )
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


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_from_online_config(config: Dict[str, Any]) -> str:
    dataset = config.get("pairs_config", {}).get("args", {}).get("dataset")
    if not dataset:
        raise ValueError("Could not recover dataset from online calibration config.")
    return str(dataset)


def _target_from_row(row: Dict[str, Any]) -> Any:
    return row.get("target_fpr") or row.get("target_alpha") or row.get("target")


def _fp_over_n(row: Dict[str, Any]) -> Optional[float]:
    fp = _float(row.get("FP"))
    n = _float(row.get("n"))
    if fp is not None and n:
        return fp / n
    return _float(row.get("error_rate_stream"))


def build_static_rows(static_summary: Path, static_config: Path) -> List[Dict[str, Any]]:
    config = _load_json(static_config)
    dataset = str(config.get("stream_dataset") or config.get("args", {}).get("stream_dataset") or "")
    if not dataset:
        raise ValueError("Could not recover stream_dataset from static calibration config.")
    max_examples = config.get("args", {}).get("max_examples")
    source = str(config.get("args", {}).get("pairwise_data", "static/random pair file"))
    threshold_path = static_summary.parent / "threshold_calibration_summary.csv"
    threshold_rows = _read_csv(threshold_path) if threshold_path.exists() else []
    thresholds = {
        (row.get("method"), row.get("target_fpr")): row.get("model_threshold")
        for row in threshold_rows
    }
    rows: List[Dict[str, Any]] = []
    for row in _read_csv(static_summary):
        method = row.get("method")
        if method not in {"ours_whitened_hadamard", "ours_weighted_ensemble"}:
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "calibration_source": "static_random_pair_calibration",
                "eval_distribution": f"online_smoke_stream_first_{max_examples}",
                "target": _target_from_row(row),
                "threshold_or_param": thresholds.get((method, _target_from_row(row)), ""),
                "eval_fpr": row.get("false_positive_rate"),
                "eval_fp_over_n": _fp_over_n(row),
                "eval_tpr": row.get("true_positive_rate"),
                "hit_rate": row.get("hit_rate"),
                "precision": row.get("precision"),
                "TP": row.get("TP"),
                "FP": row.get("FP"),
                "TN": row.get("TN"),
                "FN": row.get("FN"),
                "n": row.get("n"),
                "comparison_status": "not_same_full_eval_stream",
                "notes": f"Static calibration came from {source}; checked output is a smoke stream, not the full online eval stream.",
                "source_file": str(static_summary),
            }
        )
    return rows


def build_online_rows(online_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for online_dir in online_dirs:
        if not online_dir.exists():
            raise FileNotFoundError(f"Missing required online calibration directory: {online_dir}")
        config = _load_json(online_dir / "config.json")
        dataset = _dataset_from_online_config(config)
        eval_rows = _read_csv(online_dir / "online_pair_eval_summary.csv")
        thresholds = {
            (row.get("method"), row.get("target_fpr")): row.get("tau")
            for row in _read_csv(online_dir / "online_threshold_calibration_summary.csv")
        }
        for row in eval_rows:
            method = row.get("method")
            if method not in {"ours_whitened_hadamard", "ours_weighted_ensemble"}:
                continue
            target = row.get("target_fpr")
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "calibration_source": "online_mined_nearest_neighbor_calibration",
                    "eval_distribution": "online_mined_eval_pairs",
                    "target": target,
                    "threshold_or_param": thresholds.get((method, target), ""),
                    "eval_fpr": row.get("false_positive_rate"),
                    "eval_fp_over_n": _fp_over_n(row),
                    "eval_tpr": row.get("true_positive_rate"),
                    "hit_rate": row.get("hit_rate"),
                    "precision": row.get("precision"),
                    "TP": row.get("TP"),
                    "FP": row.get("FP"),
                    "TN": row.get("TN"),
                    "FN": row.get("FN"),
                    "n": row.get("n"),
                    "comparison_status": "same_online_pair_eval_distribution",
                    "notes": "Threshold selected on online_calib and evaluated on online_eval candidate pairs.",
                    "source_file": str(online_dir / "online_pair_eval_summary.csv"),
                }
            )
    return rows


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fields = [
        ("Calibration source", "calibration_source"),
        ("Eval distribution", "eval_distribution"),
        ("Dataset", "dataset"),
        ("Method", "method"),
        ("Target", "target"),
        ("Eval FPR", "eval_fpr"),
        ("Eval FP/n", "eval_fp_over_n"),
        ("Eval TPR", "eval_tpr"),
        ("Hit rate", "hit_rate"),
        ("Precision", "precision"),
        ("Status", "comparison_status"),
    ]
    numeric = {"target", "eval_fpr", "eval_fp_over_n", "eval_tpr", "hit_rate", "precision"}
    lines = [
        "| " + " | ".join(title for title, _ in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for _, key in fields:
            values.append(_fmt(row.get(key)) if key in numeric else str(row.get(key, "")))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_comparable_static = any(row.get("comparison_status") == "same_online_pair_eval_distribution" and row.get("calibration_source", "").startswith("static") for row in rows)
    lines = [
        "# Static vs Online-Mined Calibration",
        "",
        "This report compares available static/random-pair calibration outputs with online-mined nearest-neighbor calibration outputs.",
        "",
        "The current checked static/random output is not a full apples-to-apples comparison: it is a smoke-stream evaluation on `SemCacheLMArena` rather than the same full online eval-pair distribution used by the online-mined calibration files.",
        "",
        "Therefore this artifact does not force the conclusion that online-mined calibration is necessary. It records the current evidence and the missing rerun.",
        "",
        f"Comparable static full-online evaluation present: {'yes' if has_comparable_static else 'no'}.",
        "",
        "## Available Rows",
        "",
    ]
    lines.extend(_markdown_table(rows))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Online-mined calibration transfer is available for both datasets and learned methods.",
            "- Static/random calibration currently shows very high eval FPR on the LMArena smoke stream, but because the eval stream is not the same full online eval-pair distribution, this is diagnostic rather than a final controlled comparison.",
            "",
            "## TODO for a Controlled Test",
            "",
            "- Rerun static/random-pair calibration using the same train/calib/eval split boundaries and the same online eval nearest-neighbor candidate stream.",
            "- Log the selected static threshold, calibration FPR/TPR, and full online eval FPR/FP/n/TPR/hit_rate/precision for each dataset, method, and target.",
            "- Pass the resulting full static summary with `--static_summary` and `--static_config`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_command_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Template for the missing controlled static/random calibration comparison.",
        "# The current checked static output is not evaluated on the same full online_eval",
        "# candidate-pair distribution, so it is not comparable.",
        "",
        "# Expected future command shape:",
        "# python -m experiments.static_random_online_eval \\",
        "#   --pairwise_data results/vcache_real_datasets_comparison/pairwise_cache/SemCacheLMArena_GteLargeENv1_5_hadamard_s42_k8000.npz \\",
        "#   --online_pairs_dir results/online_candidate_pairs/lmarena_gte_seed42 \\",
        "#   --methods ours_whitened_hadamard ours_weighted_ensemble \\",
        "#   --target_fprs 0.01 0.02 0.03 0.05 0.08 \\",
        "#   --output_dir results/static_vs_online_calibration/lmarena_static_full_online",
        "",
        "# python -m experiments.static_random_online_eval \\",
        "#   --pairwise_data <searchqueries_static_random_pairwise_file.npz> \\",
        "#   --online_pairs_dir results/online_candidate_pairs/searchqueries_gte_seed42 \\",
        "#   --methods ours_whitened_hadamard ours_weighted_ensemble \\",
        "#   --target_fprs 0.01 0.02 0.03 0.05 0.08 \\",
        "#   --output_dir results/static_vs_online_calibration/searchqueries_static_full_online",
        "",
        "# The future output must include: calibration_source, eval_stream, train_pairs_used,",
        "# calib_pairs_used, calib_H0, calib_H1, threshold_or_param, eval_FPR, eval_TPR,",
        "# eval_hit_rate, eval_precision, and eval_FP_over_n.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare static/random and online-mined calibration outputs.")
    ap.add_argument("--static_summary", default=DEFAULT_STATIC_SUMMARY)
    ap.add_argument("--static_config", default=DEFAULT_STATIC_CONFIG)
    ap.add_argument("--online_dirs", nargs="+", default=DEFAULT_ONLINE_DIRS)
    ap.add_argument("--output_dir", default="results/static_vs_online_calibration")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_static_rows(Path(args.static_summary), Path(args.static_config))
    rows.extend(build_online_rows([Path(p) for p in args.online_dirs]))
    rows.sort(
        key=lambda r: (
            str(r.get("dataset")),
            str(r.get("method")),
            str(r.get("calibration_source")),
            _float(r.get("target")) or 0.0,
        )
    )
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "static_vs_online_calibration.csv"
    md_path = out_dir / "static_vs_online_calibration.md"
    template_path = out_dir / "rerun_static_online_commands.sh"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    write_command_template(template_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {template_path}")


if __name__ == "__main__":
    main()
