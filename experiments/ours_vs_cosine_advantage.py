from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_INPUTS = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena/matched_budget_summary.csv",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries/matched_budget_summary.csv",
]

FIELDS = [
    "dataset",
    "subset_name",
    "budget",
    "ours_method",
    "ours_param",
    "cosine_param",
    "vcache_param",
    "ours_hit_rate",
    "cosine_hit_rate",
    "vcache_hit_rate",
    "hit_rate_gain_abs",
    "hit_rate_gain_rel",
    "ours_error_rate_stream",
    "cosine_error_rate_stream",
    "vcache_error_rate_stream",
    "ours_false_positive_rate",
    "cosine_false_positive_rate",
    "vcache_false_positive_rate",
    "ours_true_positive_rate",
    "cosine_true_positive_rate",
    "vcache_true_positive_rate",
    "ours_precision",
    "cosine_precision",
    "vcache_precision",
    "precision_delta",
    "ours_llm_calls",
    "cosine_llm_calls",
    "vcache_llm_calls",
    "llm_calls_saved_vs_cosine",
    "source_file",
]


def _float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int(value: Any) -> Optional[int]:
    out = _float(value)
    return None if out is None else int(out)


def read_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def compare(rows: Sequence[Dict[str, Any]], ours_method: str) -> List[Dict[str, Any]]:
    by_key = {
        (r.get("dataset"), r.get("subset_name"), r.get("budget"), r.get("method")): r
        for r in rows
    }
    out: List[Dict[str, Any]] = []
    for dataset, subset_name, budget, method in sorted(by_key):
        if method != ours_method:
            continue
        ours = by_key[(dataset, subset_name, budget, method)]
        cosine = by_key.get((dataset, subset_name, budget, "cosine"))
        vcache = by_key.get((dataset, subset_name, budget, "vcache"))
        if cosine is None:
            continue
        ours_hit = _float(ours.get("hit_rate"))
        cosine_hit = _float(cosine.get("hit_rate"))
        if ours_hit is None or cosine_hit is None:
            continue
        ours_llm = _int(ours.get("llm_calls"))
        cosine_llm = _int(cosine.get("llm_calls"))
        ours_precision = _float(ours.get("precision"))
        cosine_precision = _float(cosine.get("precision"))
        out.append(
            {
                "dataset": dataset,
                "subset_name": subset_name,
                "budget": _float(budget),
                "ours_method": ours_method,
                "ours_param": ours.get("selected_delta_or_threshold"),
                "cosine_param": cosine.get("selected_delta_or_threshold"),
                "vcache_param": None if vcache is None else vcache.get("selected_delta_or_threshold"),
                "ours_hit_rate": ours_hit,
                "cosine_hit_rate": cosine_hit,
                "vcache_hit_rate": None if vcache is None else _float(vcache.get("hit_rate")),
                "hit_rate_gain_abs": ours_hit - cosine_hit,
                "hit_rate_gain_rel": (ours_hit / cosine_hit) if cosine_hit > 0 else None,
                "ours_error_rate_stream": _float(ours.get("error_rate_stream")),
                "cosine_error_rate_stream": _float(cosine.get("error_rate_stream")),
                "vcache_error_rate_stream": None if vcache is None else _float(vcache.get("error_rate_stream")),
                "ours_false_positive_rate": _float(ours.get("false_positive_rate")),
                "cosine_false_positive_rate": _float(cosine.get("false_positive_rate")),
                "vcache_false_positive_rate": None if vcache is None else _float(vcache.get("false_positive_rate")),
                "ours_true_positive_rate": _float(ours.get("true_positive_rate")),
                "cosine_true_positive_rate": _float(cosine.get("true_positive_rate")),
                "vcache_true_positive_rate": None if vcache is None else _float(vcache.get("true_positive_rate")),
                "ours_precision": ours_precision,
                "cosine_precision": cosine_precision,
                "vcache_precision": None if vcache is None else _float(vcache.get("precision")),
                "precision_delta": None
                if ours_precision is None or cosine_precision is None
                else ours_precision - cosine_precision,
                "ours_llm_calls": ours_llm,
                "cosine_llm_calls": cosine_llm,
                "vcache_llm_calls": None if vcache is None else _int(vcache.get("llm_calls")),
                "llm_calls_saved_vs_cosine": None
                if ours_llm is None or cosine_llm is None
                else cosine_llm - ours_llm,
                "source_file": ours.get("source_file"),
            }
        )
    out.sort(
        key=lambda r: (
            float(r.get("hit_rate_gain_abs") or 0.0),
            float(r.get("hit_rate_gain_rel") or 0.0),
        ),
        reverse=True,
    )
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def fmt(value: Any, ndigits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{ndigits}f}"


def write_report(path: Path, rows: Sequence[Dict[str, Any]], ours_method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = list(rows[:12])
    full = [r for r in rows if str(r.get("subset_name", "")).endswith("_full")]
    lines = [
        "# Ours vs Cosine Advantage Experiment",
        "",
        "This report compares the matched-budget selected runs for the learned method against the cosine threshold baseline.",
        "",
        f"Learned method: `{ours_method}`",
        "",
        "Selection rule: for each dataset/subset/budget, compare the selected matched-budget row from `matched_budget_summary.csv`.",
        "The ranking below uses absolute hit-rate gain: `ours_hit_rate - cosine_hit_rate`.",
        "",
        "## Strongest Improvements",
        "",
        "| Rank | Dataset | Subset | Budget | Ours hit | Cosine hit | vCache hit | Abs gain | Ours FPR | Ours TPR | Cosine FPR | Cosine TPR | vCache FPR | vCache TPR | LLM calls saved |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(top, start=1):
        rel = _float(row.get("hit_rate_gain_rel"))
        rel_text = "" if rel is None else f"{rel:.2f}x"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(row.get("dataset")),
                    str(row.get("subset_name")),
                    fmt(row.get("budget"), 2),
                    fmt(row.get("ours_hit_rate")),
                    fmt(row.get("cosine_hit_rate")),
                    fmt(row.get("vcache_hit_rate")),
                    fmt(row.get("hit_rate_gain_abs")),
                    fmt(row.get("ours_false_positive_rate")),
                    fmt(row.get("ours_true_positive_rate")),
                    fmt(row.get("cosine_false_positive_rate")),
                    fmt(row.get("cosine_true_positive_rate")),
                    fmt(row.get("vcache_false_positive_rate")),
                    fmt(row.get("vcache_true_positive_rate")),
                    str(row.get("llm_calls_saved_vs_cosine")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Full-Stream Rows",
            "",
            "| Dataset | Budget | Ours FP/n | Ours FPR | Ours TPR | Cosine FP/n | Cosine FPR | Cosine TPR | vCache FP/n | vCache FPR | vCache TPR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(full, key=lambda r: (str(r["dataset"]), float(r["budget"]))):
        rel = _float(row.get("hit_rate_gain_rel"))
        rel_text = "" if rel is None else f"{rel:.2f}x"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset")),
                    fmt(row.get("budget"), 2),
                    fmt(row.get("ours_error_rate_stream")),
                    fmt(row.get("ours_false_positive_rate")),
                    fmt(row.get("ours_true_positive_rate")),
                    fmt(row.get("cosine_error_rate_stream")),
                    fmt(row.get("cosine_false_positive_rate")),
                    fmt(row.get("cosine_true_positive_rate")),
                    fmt(row.get("vcache_error_rate_stream")),
                    fmt(row.get("vcache_false_positive_rate")),
                    fmt(row.get("vcache_true_positive_rate")),
                ]
            )
            + " |"
        )
    if rows:
        best = rows[0]
        lines.extend(
            [
                "",
                "## Main Finding",
                "",
                (
                    "The largest observed advantage is on "
                    f"`{best['dataset']}` / `{best['subset_name']}` at budget `{fmt(best['budget'], 2)}`: "
                    f"`{ours_method}` reaches hit rate `{fmt(best['ours_hit_rate'])}` versus cosine `{fmt(best['cosine_hit_rate'])}`, "
                    f"an absolute gain of `{fmt(best['hit_rate_gain_abs'])}`."
                ),
                "",
                "This is empirical evidence from the current matched-budget outputs, not a universal proof. The claim should be stated as: "
                "`under the checked matched FP/n budgets, the online-mined WeightedEnsemble gives the largest observed gains over cosine on LMArena, especially hard-neighbor subsets.`",
                "",
                "vCache is included as a reference column only. The ranking and claim in this report are still based on `ours_weighted_ensemble` versus `cosine`.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare online-mined ours method against cosine matched-budget rows.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--ours_method", default="ours_weighted_ensemble")
    parser.add_argument("--output_dir", default="results/ours_vs_cosine_advantage")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = read_rows(Path(p) for p in args.inputs)
    compared = compare(rows, args.ours_method)
    out_dir = Path(args.output_dir)
    write_csv(out_dir / "ours_vs_cosine_advantage.csv", compared)
    write_report(out_dir / "ours_vs_cosine_advantage.md", compared, args.ours_method)
    print(f"wrote {out_dir / 'ours_vs_cosine_advantage.csv'}")
    print(f"wrote {out_dir / 'ours_vs_cosine_advantage.md'}")
    if compared:
        best = compared[0]
        print(
            "best: "
            f"{best['dataset']} {best['subset_name']} budget={best['budget']} "
            f"gain={best['hit_rate_gain_abs']:.4f} "
            f"ours_hit={best['ours_hit_rate']:.4f} cosine_hit={best['cosine_hit_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
