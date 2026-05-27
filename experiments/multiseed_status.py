from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RESULT_DIRS = [
    "results/vcache_real_datasets_comparison_online_mined_lmarena",
    "results/vcache_real_datasets_comparison_online_mined_searchqueries",
    "results/debug_ours_online_mined_calibration/lmarena_gte_seed42",
    "results/debug_ours_online_mined_calibration/searchqueries_gte_seed42",
    "results/online_candidate_pairs/lmarena_gte_seed42",
    "results/online_candidate_pairs/searchqueries_gte_seed42",
]
SEEDS = [42, 43, 44, 45, 46]

SUMMARY_FIELDS = [
    "dataset",
    "subset_name",
    "method",
    "budget_or_alpha",
    "n_seeds",
    "mean_hit_rate",
    "std_hit_rate",
    "mean_FP_over_n",
    "std_FP_over_n",
    "mean_FPR",
    "std_FPR",
    "mean_TPR",
    "std_TPR",
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
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_seeds(paths: Iterable[Path]) -> List[int]:
    seeds: set[int] = set()
    for path in paths:
        text = str(path)
        for match in re.finditer(r"seed[_-]?(\d+)|_s(\d+)|_seed(\d+)", text):
            for group in match.groups():
                if group:
                    seeds.add(int(group))
        config = _load_json(path / "config.json") if path.is_dir() else None
        if config:
            for candidate in [
                config.get("args", {}).get("seed"),
                config.get("pairs_config", {}).get("args", {}).get("seed"),
            ]:
                if candidate is not None:
                    seeds.add(int(candidate))
        for csv_name in ["matched_budget_aggregated_summary.csv", "multiseed_summary.csv"]:
            for row in _read_csv(path / csv_name if path.is_dir() else path):
                n_seeds = _float(row.get("n_seeds"))
                if n_seeds and n_seeds > 1:
                    seeds.update(range(1, int(n_seeds) + 1))
    return sorted(seeds)


def _candidate_summary_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for path in paths:
        if path.is_dir():
            candidate = path / "matched_budget_summary.csv"
            if candidate.exists():
                out.append(candidate)
        elif path.name == "matched_budget_summary.csv":
            out.append(path)
    return out


def aggregate_if_multiseed(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    raw_rows: List[Dict[str, Any]] = []
    for path in _candidate_summary_paths(paths):
        for row in _read_csv(path):
            config = _load_json(path.parent / "config.json") or {}
            seed = config.get("args", {}).get("seed")
            if seed is None:
                seed = 42
            row["_seed"] = str(seed)
            raw_rows.append(row)
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in raw_rows:
        key = (
            str(row.get("dataset")),
            str(row.get("subset_name")),
            str(row.get("method")),
            str(row.get("budget") or row.get("alpha")),
        )
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        seeds = {row.get("_seed") for row in group}
        if len(seeds) <= 1:
            continue
        hit = [_float(row.get("hit_rate")) for row in group]
        fp_n = [_float(row.get("error_rate_stream")) for row in group]
        fpr = [_float(row.get("false_positive_rate")) for row in group]
        tpr = [_float(row.get("true_positive_rate")) for row in group]

        def stats(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
            clean = [v for v in values if v is not None]
            if not clean:
                return None, None
            return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0

        mean_hit, std_hit = stats(hit)
        mean_fp_n, std_fp_n = stats(fp_n)
        mean_fpr, std_fpr = stats(fpr)
        mean_tpr, std_tpr = stats(tpr)
        out.append(
            {
                "dataset": key[0],
                "subset_name": key[1],
                "method": key[2],
                "budget_or_alpha": key[3],
                "n_seeds": len(seeds),
                "mean_hit_rate": mean_hit,
                "std_hit_rate": std_hit,
                "mean_FP_over_n": mean_fp_n,
                "std_FP_over_n": std_fp_n,
                "mean_FPR": mean_fpr,
                "std_FPR": std_fpr,
                "mean_TPR": mean_tpr,
                "std_TPR": std_tpr,
            }
        )
    return out


def write_rerun_commands(path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Multi-seed rerun plan for the main methods plus the metadata-fixed vCache reference.",
        "# vCache remains secondary until the rerun audit confirms near-complete nearest-candidate metadata.",
        "",
        "for seed in 42 43 44 45 46; do",
        "  python -m experiments.mine_online_candidate_pairs --dataset SemCacheLMArena --embedding_model GTE --output_dir results/online_candidate_pairs/lmarena_gte_seed${seed} --seed ${seed} --cache_size 4096 --eviction_policy mru --split_ratios 0.4 0.2 0.4 --equivalence_mode cluster",
        "  python -m experiments.debug_ours_online_mined_calibration --pairs_dir results/online_candidate_pairs/lmarena_gte_seed${seed} --methods ours_whitened_hadamard ours_weighted_ensemble --target_fprs 0.01 0.02 0.03 0.05 0.08 --seed ${seed} --output_dir results/debug_ours_online_mined_calibration/lmarena_gte_seed${seed}",
        "  python -m experiments.label_efficiency_stopping --pair_dirs results/online_candidate_pairs/lmarena_gte_seed${seed} --alphas 0.05 --output_dir results/label_efficiency_stopping/lmarena_seed${seed}",
        "  python -m experiments.conservative_calibration_transfer --pair_dirs results/online_candidate_pairs/lmarena_gte_seed${seed} --output_dir results/conservative_calibration_transfer/lmarena_seed${seed}",
        "  python -m experiments.compare_vcache_real_datasets --datasets SemCacheLMArena --methods vcache cosine ours_whitened_hadamard ours_weighted_ensemble --thresholds 0.8 0.85 0.9 0.93 0.95 0.97 0.98 0.99 0.995 0.999 --target_budgets 0.01 0.02 0.03 0.05 --hard_neighbor_thresholds 0.80 0.85 0.90 --embedding_model GTE --seed ${seed} --cache_size 4096 --eviction_policy mru --ours_calibration_source online_mined --online_pairs_dir results/online_candidate_pairs/lmarena_gte_seed${seed} --raw_log_mode minimal --output_dir results/vcache_real_datasets_comparison_online_mined_lmarena_seed${seed}",
        "",
        "  python -m experiments.mine_online_candidate_pairs --dataset SemCacheSearchQueries --embedding_model GTE --output_dir results/online_candidate_pairs/searchqueries_gte_seed${seed} --seed ${seed} --cache_size 4096 --eviction_policy mru --split_ratios 0.4 0.2 0.4 --equivalence_mode cluster",
        "  python -m experiments.debug_ours_online_mined_calibration --pairs_dir results/online_candidate_pairs/searchqueries_gte_seed${seed} --methods ours_whitened_hadamard ours_weighted_ensemble --target_fprs 0.01 0.02 0.03 0.05 0.08 --seed ${seed} --output_dir results/debug_ours_online_mined_calibration/searchqueries_gte_seed${seed}",
        "  python -m experiments.label_efficiency_stopping --pair_dirs results/online_candidate_pairs/searchqueries_gte_seed${seed} --alphas 0.05 --output_dir results/label_efficiency_stopping/searchqueries_seed${seed}",
        "  python -m experiments.conservative_calibration_transfer --pair_dirs results/online_candidate_pairs/searchqueries_gte_seed${seed} --output_dir results/conservative_calibration_transfer/searchqueries_seed${seed}",
        "  python -m experiments.compare_vcache_real_datasets --datasets SemCacheSearchQueries --methods vcache cosine ours_whitened_hadamard ours_weighted_ensemble --thresholds 0.8 0.85 0.9 0.93 0.95 0.97 0.98 0.99 0.995 0.999 --target_budgets 0.01 0.02 0.03 0.05 --hard_neighbor_thresholds 0.80 0.85 0.90 --embedding_model GTE --seed ${seed} --cache_size 4096 --eviction_policy mru --ours_calibration_source online_mined --online_pairs_dir results/online_candidate_pairs/searchqueries_gte_seed${seed} --raw_log_mode minimal --output_dir results/vcache_real_datasets_comparison_online_mined_searchqueries_seed${seed}",
        "done",
        "",
        "# After all seeds finish, regenerate:",
        "# python -m experiments.vcache_metadata_audit --raw_files results/vcache_real_datasets_comparison_online_mined_lmarena_seed*/raw_decisions_minimal.csv results/vcache_real_datasets_comparison_online_mined_searchqueries_seed*/raw_decisions_minimal.csv",
        "# python -m experiments.candidate_fpr_matched_summary --inputs results/vcache_real_datasets_comparison_online_mined_lmarena_seed*/summary_metrics.csv results/vcache_real_datasets_comparison_online_mined_searchqueries_seed*/summary_metrics.csv",
        "# python -m experiments.multiseed_status --result_dirs results/vcache_real_datasets_comparison_online_mined_lmarena_seed* results/vcache_real_datasets_comparison_online_mined_searchqueries_seed*",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def write_status(path: Path, seeds: Sequence[int], commands_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multi-Seed Status",
        "",
        f"Detected seeds: {', '.join(str(s) for s in seeds) if seeds else 'none'}",
        "",
        "Current checked results are single-seed only.",
        "",
        "The available aggregate file also reports `n_seeds=1`, so no statistical seed variance can be claimed from the checked outputs.",
        "",
        f"Rerun commands for seeds 42, 43, 44, 45, and 46 are written to `{commands_path}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Multi-Seed Summary",
        "",
        "Multiple seeds were detected and aggregated by dataset, subset, method, and budget/alpha.",
        "",
        "| Dataset | Subset | Method | Budget/alpha | n seeds | Mean hit | Std hit | Mean FP/n | Std FP/n | Mean FPR | Std FPR | Mean TPR | Std TPR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("subset_name", "")),
                    str(row.get("method", "")),
                    str(row.get("budget_or_alpha", "")),
                    str(row.get("n_seeds", "")),
                    _fmt(row.get("mean_hit_rate")),
                    _fmt(row.get("std_hit_rate")),
                    _fmt(row.get("mean_FP_over_n")),
                    _fmt(row.get("std_FP_over_n")),
                    _fmt(row.get("mean_FPR")),
                    _fmt(row.get("std_FPR")),
                    _fmt(row.get("mean_TPR")),
                    _fmt(row.get("std_TPR")),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check and aggregate multi-seed result status.")
    ap.add_argument("--result_dirs", nargs="+", default=DEFAULT_RESULT_DIRS)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = [Path(p) for p in args.result_dirs]
    seeds = discover_seeds(paths)
    aggregate_rows = aggregate_if_multiseed(paths)
    if aggregate_rows:
        out_dir = Path("results/multiseed_summary")
        csv_path = out_dir / "multiseed_summary.csv"
        md_path = out_dir / "multiseed_summary.md"
        _write_csv(csv_path, aggregate_rows)
        write_summary_markdown(md_path, aggregate_rows)
        print(f"wrote {csv_path}")
        print(f"wrote {md_path}")
    else:
        out_dir = Path("results/multiseed_status")
        md_path = out_dir / "multiseed_status.md"
        commands_path = out_dir / "rerun_commands.sh"
        write_rerun_commands(commands_path)
        write_status(md_path, seeds, commands_path)
        print(f"wrote {md_path}")
        print(f"wrote {commands_path}")


if __name__ == "__main__":
    main()
