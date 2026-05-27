from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CANDIDATE_DELTAS = "results/candidate_fpr_matched_summary/candidate_fpr_matched_ours_vs_cosine.csv"
DEFAULT_LABEL_EFFICIENCY = "results/label_efficiency_stopping/label_efficiency_stopping.csv"
DEFAULT_CONSERVATIVE = "results/conservative_calibration_transfer/conservative_calibration_transfer.csv"
DEFAULT_STATIC_ONLINE = "results/static_vs_online_calibration/static_vs_online_calibration.csv"
DEFAULT_MULTISEED_STATUS = "results/multiseed_status/multiseed_status.md"
DEFAULT_VCACHE_AUDIT = "results/vcache_metadata_audit/vcache_metadata_audit.csv"
DEFAULT_COST = "results/cost_accounting_clean/cost_accounting_clean.csv"
DEFAULT_OUTPUT = "results/research_missing_data_report.md"


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path.read_text(encoding="utf-8")


def _float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int(value: Any) -> Optional[int]:
    number = _float(value)
    if number is None:
        return None
    return int(round(number))


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    number = _int(value)
    return "" if number is None else str(number)


def _find(rows: Sequence[Dict[str, Any]], **filters: Any) -> Optional[Dict[str, Any]]:
    for row in rows:
        ok = True
        for key, value in filters.items():
            row_value = row.get(key)
            if key in {"alpha", "target_alpha"}:
                rv = _float(row_value)
                ok = rv is not None and abs(rv - float(value)) < 1e-12
            else:
                ok = row_value == value
            if not ok:
                break
        if ok:
            return row
    return None


def _rows(rows: Sequence[Dict[str, Any]], **filters: Any) -> List[Dict[str, Any]]:
    return [row for row in rows if _find([row], **filters) is not None]


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows:
        out.append("| " + " | ".join("" for _ in headers) + " |")
        return out
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def _candidate_snapshot(candidate_rows: Sequence[Dict[str, Any]]) -> List[str]:
    picks = [
        ("SemCacheLMArena", "SemCacheLMArena_full", 0.05, "ours_weighted_ensemble"),
        ("SemCacheLMArena", "SemCacheLMArena_hard_cos_ge_0.90", 0.05, "ours_weighted_ensemble"),
        ("SemCacheSearchQueries", "SemCacheSearchQueries_full", 0.05, "ours_weighted_ensemble"),
        ("SemCacheSearchQueries", "SemCacheSearchQueries_hard_cos_ge_0.80", 0.05, "ours_weighted_ensemble"),
    ]
    rows: List[List[str]] = []
    for dataset, subset, alpha, method in picks:
        row = _find(candidate_rows, dataset=dataset, subset_name=subset, alpha=alpha, ours_method=method)
        if not row:
            continue
        rows.append(
            [
                dataset,
                subset,
                _fmt(alpha),
                _fmt(row.get("cosine_false_positive_rate")),
                _fmt(row.get("ours_false_positive_rate")),
                _fmt(row.get("cosine_true_positive_rate")),
                _fmt(row.get("ours_true_positive_rate")),
                _fmt(row.get("cosine_hit_rate")),
                _fmt(row.get("ours_hit_rate")),
                _fmt(row.get("relative_hit_gain")),
            ]
        )
    return _table(
        [
            "Dataset",
            "Subset",
            "Alpha",
            "Cos FPR",
            "WE FPR",
            "Cos TPR",
            "WE TPR",
            "Cos hit",
            "WE hit",
            "Rel hit",
        ],
        rows,
    )


def _full_label_rows(label_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in label_rows:
        method = row.get("method")
        if method == "cosine":
            if row.get("calib_pairs_requested") == "full":
                out.append(row)
            continue
        if row.get("train_pairs_requested") == "full" and row.get("calib_pairs_requested") == "full":
            out.append(row)
    return out


def _label_full_snapshot(label_rows: Sequence[Dict[str, Any]]) -> List[str]:
    rows: List[List[str]] = []
    for dataset in ["SemCacheLMArena", "SemCacheSearchQueries"]:
        for method in ["ours_weighted_ensemble", "ours_whitened_hadamard", "cosine"]:
            row = next(
                (
                    r
                    for r in _full_label_rows(label_rows)
                    if r.get("dataset") == dataset
                    and r.get("method") == method
                    and abs((_float(r.get("alpha")) or -999.0) - 0.05) < 1e-12
                ),
                None,
            )
            if not row:
                continue
            rows.append(
                [
                    dataset,
                    method,
                    _fmt(row.get("alpha")),
                    _fmt_int(row.get("offline_labels_used")),
                    _fmt(row.get("eval_FPR")),
                    _fmt(row.get("eval_TPR")),
                    _fmt(row.get("eval_hit_rate")),
                    str(_truth(row.get("eval_constraint_satisfied"))),
                ]
            )
    return _table(
        ["Dataset", "Method", "Alpha", "Labels", "Eval FPR", "Eval TPR", "Hit", "Safe"],
        rows,
    )


def _best_stopped(label_rows: Sequence[Dict[str, Any]], criterion: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted(
        {
            (row.get("dataset"), row.get("method"), row.get("alpha"))
            for row in label_rows
            if row.get("method") not in {"cosine", ""}
        }
    )
    for dataset, method, alpha in keys:
        candidates = [
            row
            for row in label_rows
            if row.get("dataset") == dataset
            and row.get("method") == method
            and abs((_float(row.get("alpha")) or -999.0) - float(alpha)) < 1e-12
            and _truth(row.get(criterion))
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda row: (
                _int(row.get("offline_labels_used")) or 10**18,
                -(_float(row.get("eval_hit_rate")) or -1.0),
            ),
        )
        full = next(
            (
                row
                for row in label_rows
                if row.get("dataset") == dataset
                and row.get("method") == method
                and abs((_float(row.get("alpha")) or -999.0) - float(alpha)) < 1e-12
                and row.get("train_pairs_requested") == "full"
                and row.get("calib_pairs_requested") == "full"
            ),
            None,
        )
        merged = dict(best)
        if full:
            merged["full_labels"] = full.get("offline_labels_used")
            merged["full_hit"] = full.get("eval_hit_rate")
            merged["full_FPR"] = full.get("eval_FPR")
        merged["criterion"] = criterion
        out.append(merged)
    return out


def _label_efficiency_snapshot(label_rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    stopped = _best_stopped(label_rows, "success_90") + _best_stopped(label_rows, "success_95")
    rows: List[List[str]] = []
    for row in stopped:
        full_labels = _float(row.get("full_labels"))
        stopped_labels = _float(row.get("offline_labels_used"))
        reduction = None
        if full_labels and stopped_labels is not None:
            reduction = 1.0 - stopped_labels / full_labels
        rows.append(
            [
                str(row.get("dataset")),
                str(row.get("method")),
                _fmt(row.get("alpha")),
                str(row.get("criterion")),
                _fmt_int(row.get("full_labels")),
                _fmt_int(row.get("offline_labels_used")),
                _fmt(reduction),
                _fmt(row.get("full_hit")),
                _fmt(row.get("eval_hit_rate")),
                _fmt(row.get("full_FPR")),
                _fmt(row.get("eval_FPR")),
            ]
        )

    notes = []
    for dataset in ["SemCacheLMArena", "SemCacheSearchQueries"]:
        for method in ["ours_weighted_ensemble", "ours_whitened_hadamard"]:
            if not any(r.get("dataset") == dataset and r.get("method") == method for r in stopped):
                notes.append(f"No success_90 or success_95 stopped row was found for `{dataset}` / `{method}` at alpha 0.05.")
    return (
        _table(
            [
                "Dataset",
                "Method",
                "Alpha",
                "Criterion",
                "Full labels",
                "Stopped labels",
                "Label reduction",
                "Full hit",
                "Stopped hit",
                "Full FPR",
                "Stopped FPR",
            ],
            rows,
        ),
        notes,
    )


def _safe_conservative_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = sorted(
        {
            (row.get("dataset"), row.get("method"), row.get("alpha"))
            for row in rows
            if abs((_float(row.get("alpha")) or -999.0) - 0.05) < 1e-12
        }
    )
    for dataset, method, alpha in keys:
        candidates = [
            row
            for row in rows
            if row.get("dataset") == dataset
            and row.get("method") == method
            and abs((_float(row.get("alpha")) or -999.0) - float(alpha)) < 1e-12
            and _truth(row.get("eval_constraint_satisfied"))
        ]
        if not candidates:
            continue
        out.append(max(candidates, key=lambda row: _float(row.get("eval_hit_rate")) or -1.0))
    return out


def _conservative_snapshot(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], str]:
    baseline_rows = [row for row in rows if row.get("calibration_rule") == "empirical_target_1x_alpha"]
    baseline_safe = sum(1 for row in baseline_rows if _truth(row.get("eval_constraint_satisfied")))
    total_safe = sum(1 for row in rows if _truth(row.get("eval_constraint_satisfied")))
    best_safe = _safe_conservative_rows(rows)
    table_rows = [
        [
            str(row.get("dataset")),
            str(row.get("method")),
            _fmt(row.get("alpha")),
            str(row.get("calibration_rule")),
            _fmt(row.get("calib_target")),
            _fmt(row.get("eval_FPR")),
            _fmt(row.get("eval_TPR")),
            _fmt(row.get("eval_hit_rate")),
        ]
        for row in best_safe
    ]
    summary = (
        f"Baseline 1x calibration satisfied eval FPR in {baseline_safe}/{len(baseline_rows)} rows; "
        f"all conservative rules together satisfied it in {total_safe}/{len(rows)} rows."
    )
    return (
        _table(
            ["Dataset", "Method", "Alpha", "Rule", "Calib target", "Eval FPR", "Eval TPR", "Hit"],
            table_rows,
        ),
        summary,
    )


def _static_status(rows: Sequence[Dict[str, Any]]) -> str:
    comparable = any(
        row.get("calibration_source") == "static_random_pair_calibration"
        and row.get("comparison_status") == "same_online_pair_eval_distribution"
        for row in rows
    )
    if comparable:
        return "A comparable static/random calibration result exists in the checked CSV."
    return (
        "Current static/random calibration output is not comparable because it is not evaluated on "
        "the same full online eval distribution."
    )


def _vcache_summary(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], str]:
    total_rows = sum(_int(row.get("total_rows")) or 0 for row in rows)
    usable_rows = sum(_int(row.get("rows_usable_for_candidate_fpr")) or 0 for row in rows)
    usable_fraction = usable_rows / total_rows if total_rows > 0 else 0.0
    full_rows = [
        row
        for row in rows
        if row.get("subset_name", "").endswith("_full")
        and abs((_float(row.get("delta_or_policy_param")) or -999.0) - 0.05) < 1e-12
    ]
    table_rows = [
        [
            str(row.get("dataset")),
            str(row.get("subset_name")),
            _fmt(row.get("delta_or_policy_param")),
            _fmt_int(row.get("total_rows")),
            _fmt_int(row.get("rows_missing_nearest_candidate_id")),
            _fmt_int(row.get("rows_missing_nearest_candidate_label")),
            _fmt(row.get("fraction_usable_for_candidate_fpr_metrics")),
            str(row.get("candidate_metrics_reliability")),
        ]
        for row in full_rows
    ]
    summary = f"Overall usable fraction for candidate-FPR metrics is {usable_fraction:.4f}."
    return (
        _table(
            [
                "Dataset",
                "Subset",
                "Param",
                "Rows",
                "Missing ID",
                "Missing label",
                "Usable fraction",
                "Status",
            ],
            table_rows,
        ),
        summary,
    )


def _cost_snapshot(rows: Sequence[Dict[str, Any]]) -> List[str]:
    table_rows: List[List[str]] = []
    for dataset in ["SemCacheLMArena", "SemCacheSearchQueries"]:
        for method in ["cosine", "ours_weighted_ensemble", "ours_whitened_hadamard"]:
            row = _find(rows, dataset=dataset, method=method, alpha=0.05)
            if not row:
                continue
            table_rows.append(
                [
                    dataset,
                    method,
                    _fmt_int(row.get("total_offline_labels")),
                    _fmt_int(row.get("online_llm_calls")),
                    _fmt_int(row.get("online_cache_hits")),
                    _fmt_int(row.get("online_judge_calls")),
                    _fmt_int(row.get("evaluation_judge_calls")),
                    str(row.get("deployment_cost_excludes_eval_judge_calls")),
                ]
            )
    return _table(
        [
            "Dataset",
            "Method",
            "Offline labels",
            "Online LLM",
            "Cache hits",
            "Online judge",
            "Eval judge",
            "Excludes eval?",
        ],
        table_rows,
    )


def build_report(args: argparse.Namespace) -> str:
    candidate_rows = _read_csv(Path(args.candidate_deltas))
    label_rows = _read_csv(Path(args.label_efficiency))
    conservative_rows = _read_csv(Path(args.conservative_calibration))
    static_rows = _read_csv(Path(args.static_vs_online))
    multiseed_status = _read_text(Path(args.multiseed_status))
    vcache_rows = _read_csv(Path(args.vcache_audit))
    cost_rows = _read_csv(Path(args.cost_accounting))

    stopped_table, stopped_notes = _label_efficiency_snapshot(label_rows)
    conservative_table, conservative_summary = _conservative_snapshot(conservative_rows)
    vcache_table, vcache_summary = _vcache_summary(vcache_rows)
    single_seed = "single-seed only" in multiseed_status

    lines: List[str] = [
        "# Research Missing Data Report",
        "",
        "## 1. Executive summary",
        "",
        "This report consolidates the new missing-data experiments for safe semantic-cache reuse under candidate-level false-positive control.",
        "",
        "The main new label-efficiency result is mixed. On SemCacheSearchQueries, WeightedEnsemble reaches the success_95 criterion at alpha 0.05 with 3,000 online-mined labels, using 3.33% of the full train+calib label budget. On SemCacheLMArena, the full-label calibration-selected WeightedEnsemble row itself exceeds eval FPR 0.05, so the stopped configurations are not safe under the strict eval-side constraint.",
        "",
        "Conservative calibration improves eval-side FPR satisfaction, especially on SemCacheLMArena, but it lowers hit rate relative to the unconservative 1x calibration.",
        "",
        "The checked outputs are still seed 42 only. Existing vCache candidate-FPR metrics remain unreliable; the adapter has been patched to log explicit nearest-candidate metadata, but vCache must be rerun before those metrics can support claims.",
        "",
        "## 2. What was already known from candidate-FPR matched results",
        "",
        "Candidate-level FPR is `FP / (FP + TN)`. This differs from stream-level error `FP / n`.",
        "",
        "The existing candidate-FPR matched report selects only rows satisfying `false_positive_rate <= alpha`, then picks the maximum TPR row for each method.",
        "",
    ]
    lines.extend(_candidate_snapshot(candidate_rows))
    lines.extend(
        [
            "",
            "Interpretation: Under matched candidate-level FPR constraints, online-mined WeightedEnsemble improves TPR and hit rate over cosine on SemCacheLMArena full and hard subsets. SemCacheSearchQueries full stream improves modestly at alpha 0.05, while hard subsets are mixed.",
            "",
            "## 3. Label-efficiency / stopping results",
            "",
            "Goal: recover most of the full-label performance with fewer online-mined labels.",
            "",
            "Thresholds are selected on the sampled calibration split only. The fixed `online_eval` split is used for final measurement.",
            "",
            "Full-data calibration-selected reference rows at alpha 0.05:",
            "",
        ]
    )
    lines.extend(_label_full_snapshot(label_rows))
    lines.extend(
        [
            "",
            "Smallest measured stopped configurations satisfying the eval-side success criterion:",
            "",
        ]
    )
    lines.extend(stopped_table)
    if stopped_notes:
        lines.extend(["", "Missing stopped successes:"])
        lines.extend([f"- {note}" for note in stopped_notes])
    lines.extend(
        [
            "",
            "Conservative interpretation: the stopped configuration recovers at least 95% of the full-label WeightedEnsemble hit rate on SemCacheSearchQueries while using only 3.33% of the online-mined labels. The same conclusion is not supported on SemCacheLMArena because the strict eval-FPR constraint is not met by the full-label calibration-selected row.",
            "",
            "The stopped table is a label-efficiency sweep result over measured dosages. It estimates the minimum checked label budget; it is not by itself a deployable online stopping rule.",
            "",
            "Detailed artifact: `results/label_efficiency_stopping/label_efficiency_stopping.md`.",
            "",
            "## 4. Conservative calibration transfer results",
            "",
            conservative_summary,
            "",
        ]
    )
    lines.extend(conservative_table)
    lines.extend(
        [
            "",
            "At alpha 0.05, SemCacheLMArena WeightedEnsemble becomes eval-safe with the empirical 0.7x alpha calibration target: eval FPR 0.0484, TPR 0.7223, hit rate 0.5502. The unconservative full-label row had hit rate 0.6086 but eval FPR 0.0677.",
            "",
            "Conservative calibration improves eval-side FPR satisfaction, at the cost of lower hit rate. These are still single-seed results.",
            "",
            "Detailed artifact: `results/conservative_calibration_transfer/conservative_calibration_transfer.md`.",
            "",
            "## 5. Multi-seed status or results",
            "",
            "Current checked results are single-seed only." if single_seed else "Multiple seeds were detected in the checked outputs.",
            "",
            "Rerun commands for seeds 42, 43, 44, 45, and 46 are in `results/multiseed_status/rerun_commands.sh`.",
            "",
            "## 6. Static-vs-online calibration comparison",
            "",
            _static_status(static_rows),
            "",
            "The current static/random rows are diagnostic only and should not be used to claim that online-mined calibration is definitively better. A controlled rerun must evaluate both calibration sources on the same full online eval nearest-neighbor distribution.",
            "",
            "Command templates are in `results/static_vs_online_calibration/rerun_static_online_commands.sh`.",
            "",
            "## 7. vCache metadata audit",
            "",
            vcache_summary,
            "",
        ]
    )
    lines.extend(vcache_table)
    lines.extend(
        [
            "",
            "This audit uses the existing vCache explore/exploit decision logs. The main learned-method experiments do not rely on vCache for claims. vCache should remain a secondary reference until it is rerun with the patched logging and every explore/exploit row preserves candidate ID, candidate label, and candidate cosine when a nearest candidate exists.",
            "",
            "Detailed artifact: `results/vcache_metadata_audit/vcache_metadata_audit.md`.",
            "",
            "## 8. Clean cost accounting",
            "",
            "Evaluation judge calls are measurement overhead, not deployment runtime cost. Offline train/calibration labels are setup cost. Calibration judge calls are not separately logged in the current raw files.",
            "",
        ]
    )
    lines.extend(_cost_snapshot(cost_rows))
    lines.extend(
        [
            "",
            "Detailed artifact: `results/cost_accounting_clean/cost_accounting_clean.md`.",
            "",
            "## 9. Supported claims",
            "",
            "- Under matched candidate-level FPR constraints, online-mined WeightedEnsemble improves TPR and hit rate over cosine on SemCacheLMArena.",
            "- On SemCacheSearchQueries at alpha 0.05, the stopped WeightedEnsemble configuration recovers at least 95% of the full-label hit rate while using 3,000 of 89,998 online-mined train+calib labels.",
            "- Conservative calibration improves eval-side FPR satisfaction, at the cost of lower hit rate.",
            "- The current vCache explore/exploit logs are insufficient for reliable candidate-level FPR comparison.",
            "",
            "## 10. Unsupported claims",
            "",
            "- Do not write that WeightedEnsemble always beats cosine.",
            "- Do not write that calibration is solved; these results are single-seed and not uniformly safe without conservative targets.",
            "- Do not write that label-efficient stopping is proven on SemCacheLMArena at alpha 0.05.",
            "- Do not write that static/random calibration has been fairly beaten until both calibration sources are evaluated on the same full online eval stream.",
            "- Do not write that we beat vCache generally until vCache is rerun with metadata-fixed logs.",
            "- Do not write that deployment cost is reduced using evaluation-only judge calls.",
            "",
            "## 11. Next required reruns",
            "",
            "- Run seeds 43, 44, 45, and 46 using `results/multiseed_status/rerun_commands.sh`.",
            "- Rerun conservative calibration and label-efficiency summaries for every seed after mining the seed-specific online pairs.",
            "- Generate a controlled static/random calibration file evaluated on the same full online eval stream.",
            "- Rerun vCache with the patched explore/exploit logging, then rerun the vCache metadata audit and candidate-FPR comparison.",
            "- Add separate raw fields for calibration judge calls, evaluation judge calls, and online runtime judge calls.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate the paper-ready missing-data research report.")
    ap.add_argument("--candidate_deltas", default=DEFAULT_CANDIDATE_DELTAS)
    ap.add_argument("--label_efficiency", default=DEFAULT_LABEL_EFFICIENCY)
    ap.add_argument("--conservative_calibration", default=DEFAULT_CONSERVATIVE)
    ap.add_argument("--static_vs_online", default=DEFAULT_STATIC_ONLINE)
    ap.add_argument("--multiseed_status", default=DEFAULT_MULTISEED_STATUS)
    ap.add_argument("--vcache_audit", default=DEFAULT_VCACHE_AUDIT)
    ap.add_argument("--cost_accounting", default=DEFAULT_COST)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
