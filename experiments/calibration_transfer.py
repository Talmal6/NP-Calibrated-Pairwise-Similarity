from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CALIB_DIRS = [
    "results/debug_ours_online_mined_calibration/lmarena_gte_seed42",
    "results/debug_ours_online_mined_calibration/searchqueries_gte_seed42",
]

FIELDS = [
    "dataset",
    "method",
    "target_alpha",
    "threshold_param",
    "orientation",
    "calib_fpr",
    "eval_fpr",
    "calib_tpr",
    "eval_tpr",
    "calib_hit_rate",
    "eval_hit_rate",
    "eval_fp_over_n",
    "eval_precision",
    "TP",
    "FP",
    "TN",
    "FN",
    "n",
    "constraint_satisfied_on_eval",
    "tolerance",
    "source_dir",
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


def _load_config(path: Path) -> Dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing required input: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _dataset_from_config(config: Dict[str, Any]) -> str:
    dataset = (
        config.get("pairs_config", {})
        .get("args", {})
        .get("dataset")
    )
    if not dataset:
        raise ValueError("Could not recover dataset from calibration config.")
    return str(dataset)


def _pair_stats_path(config: Dict[str, Any]) -> Path:
    pairs_dir = config.get("args", {}).get("pairs_dir")
    if not pairs_dir:
        pairs_dir = (
            config.get("pairs_config", {})
            .get("args", {})
            .get("output_dir")
        )
    if not pairs_dir:
        raise ValueError("Could not recover pairs_dir from calibration config.")
    return Path(pairs_dir) / "online_candidate_pair_stats.csv"


def _split_counts(stats_rows: Iterable[Dict[str, Any]], split_name: str) -> Tuple[float, float]:
    for row in stats_rows:
        if row.get("split_name") == split_name:
            n_h1 = _float(row.get("n_H1"))
            n_h0 = _float(row.get("n_H0"))
            if n_h1 is None or n_h0 is None:
                raise ValueError(f"Missing H1/H0 counts for split {split_name}.")
            return n_h1, n_h0
    raise ValueError(f"Missing split {split_name} in online candidate pair stats.")


def _hit_rate_from_accept_rates(h1_rate: Any, h0_rate: Any, n_h1: float, n_h0: float) -> Optional[float]:
    h1 = _float(h1_rate)
    h0 = _float(h0_rate)
    denom = n_h1 + n_h0
    if h1 is None or h0 is None or denom <= 0:
        return None
    return (h1 * n_h1 + h0 * n_h0) / denom


def build_rows(calib_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for calib_dir in calib_dirs:
        if not calib_dir.exists():
            raise FileNotFoundError(f"Missing required calibration directory: {calib_dir}")
        config = _load_config(calib_dir)
        dataset = _dataset_from_config(config)
        threshold_rows = _read_csv(calib_dir / "online_threshold_calibration_summary.csv")
        eval_rows = _read_csv(calib_dir / "online_pair_eval_summary.csv")
        eval_by_key = {
            (row.get("method"), row.get("target_fpr")): row
            for row in eval_rows
        }
        stats_rows = _read_csv(_pair_stats_path(config))
        calib_h1, calib_h0 = _split_counts(stats_rows, "online_calib")

        for row in threshold_rows:
            method = row.get("method")
            target = row.get("target_fpr")
            eval_row = eval_by_key.get((method, target), {})
            target_float = _float(target)
            eval_fpr = _float(row.get("eval_FPR"))
            eval_fp = _float(eval_row.get("FP"))
            eval_n = _float(eval_row.get("n"))
            calib_hit_rate = _hit_rate_from_accept_rates(
                row.get("calib_H1_accept_rate"),
                row.get("calib_H0_accept_rate"),
                calib_h1,
                calib_h0,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "target_alpha": target,
                    "threshold_param": row.get("tau"),
                    "orientation": row.get("orientation"),
                    "calib_fpr": row.get("calib_H0_accept_rate"),
                    "eval_fpr": row.get("eval_FPR"),
                    "calib_tpr": row.get("calib_H1_accept_rate"),
                    "eval_tpr": row.get("eval_TPR"),
                    "calib_hit_rate": calib_hit_rate,
                    "eval_hit_rate": eval_row.get("hit_rate"),
                    "eval_fp_over_n": (eval_fp / eval_n) if eval_fp is not None and eval_n else None,
                    "eval_precision": eval_row.get("precision") or row.get("eval_precision"),
                    "TP": eval_row.get("TP"),
                    "FP": eval_row.get("FP"),
                    "TN": eval_row.get("TN"),
                    "FN": eval_row.get("FN"),
                    "n": eval_row.get("n"),
                    "constraint_satisfied_on_eval": (
                        "yes"
                        if eval_fpr is not None and target_float is not None and eval_fpr <= target_float
                        else "no"
                    ),
                    "tolerance": row.get("tolerance"),
                    "source_dir": str(calib_dir),
                }
            )
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["method"]), _float(r["target_alpha"]) or 0.0))
    return rows


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fields = [
        ("Dataset", "dataset"),
        ("Method", "method"),
        ("Target alpha", "target_alpha"),
        ("Threshold/param", "threshold_param"),
        ("Calib FPR", "calib_fpr"),
        ("Eval FPR", "eval_fpr"),
        ("Calib TPR", "calib_tpr"),
        ("Eval TPR", "eval_tpr"),
        ("Calib hit", "calib_hit_rate"),
        ("Eval hit", "eval_hit_rate"),
        ("Constraint satisfied on eval?", "constraint_satisfied_on_eval"),
    ]
    numeric = {
        "target_alpha",
        "threshold_param",
        "calib_fpr",
        "eval_fpr",
        "calib_tpr",
        "eval_tpr",
        "calib_hit_rate",
        "eval_hit_rate",
    }
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
    n_rows = len(rows)
    n_satisfied = sum(1 for row in rows if row.get("constraint_satisfied_on_eval") == "yes")
    lines = [
        "# Calibration Transfer",
        "",
        "This report checks whether thresholds selected on `online_calib` satisfy the same candidate-level FPR target on `online_eval`.",
        "",
        "Candidate-level FPR is `FP / (FP + TN)`. The eval constraint is counted as satisfied only when `eval_FPR <= target_alpha`; the calibration tolerance column is retained in the CSV but is not used for this strict yes/no check.",
        "",
        f"Strict eval satisfaction: {n_satisfied}/{n_rows} rows.",
        "",
        "The current threshold calibration files cover the learned methods. Cosine threshold transfer is not included because the checked calibration files do not contain a cosine calibration-threshold table.",
        "",
        "## Transfer Table",
        "",
    ]
    lines.extend(_markdown_table(rows))
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Some learned thresholds satisfy the target on calibration but exceed it on the eval stream, especially on SemCacheLMArena.",
            "- This table evaluates transfer of thresholds from online-mined calibration pairs to online-mined eval pairs; it is separate from the full streaming cache-policy summary.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Report online calibration-to-eval FPR transfer.")
    ap.add_argument("--calib_dirs", nargs="+", default=DEFAULT_CALIB_DIRS)
    ap.add_argument("--output_dir", default="results/calibration_transfer")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = build_rows([Path(p) for p in args.calib_dirs])
    out_dir = Path(args.output_dir)
    csv_path = out_dir / "calibration_transfer.csv"
    md_path = out_dir / "calibration_transfer.md"
    _write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
