from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_STREAM_ADVANTAGE = "results/ours_vs_cosine_advantage/ours_vs_cosine_advantage.csv"
DEFAULT_CANDIDATE_DELTAS = "results/candidate_fpr_matched_summary/candidate_fpr_matched_ours_vs_cosine.csv"
DEFAULT_CALIBRATION = "results/calibration_transfer/calibration_transfer.csv"
DEFAULT_STATIC_ONLINE = "results/static_vs_online_calibration/static_vs_online_calibration.csv"
DEFAULT_BUCKETS = "results/hard_negative_bucket_analysis/hard_negative_bucket_analysis.csv"
DEFAULT_COST = "results/cost_accounting/cost_accounting.csv"
DEFAULT_MULTISEED_STATUS = "results/multiseed_status/multiseed_status.md"


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


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _find(rows: Sequence[Dict[str, Any]], **filters: Any) -> Optional[Dict[str, Any]]:
    for row in rows:
        ok = True
        for key, value in filters.items():
            if key in {"alpha", "budget"}:
                if abs((_float(row.get(key)) or -999.0) - float(value)) > 1e-12:
                    ok = False
                    break
            elif row.get(key) != value:
                ok = False
                break
        if ok:
            return row
    return None


def _best_by(rows: Sequence[Dict[str, Any]], key: str, method: Optional[str] = None) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if method is None or row.get("ours_method") == method]
    if not candidates:
        return None
    return max(candidates, key=lambda r: _float(r.get(key)) or -1e9)


def _candidate_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    picks = [
        ("SemCacheLMArena", "SemCacheLMArena_full", 0.05, "ours_weighted_ensemble"),
        ("SemCacheSearchQueries", "SemCacheSearchQueries_full", 0.05, "ours_weighted_ensemble"),
        ("SemCacheLMArena", "SemCacheLMArena_hard_cos_ge_0.90", 0.05, "ours_weighted_ensemble"),
        ("SemCacheSearchQueries", "SemCacheSearchQueries_hard_cos_ge_0.80", 0.05, "ours_weighted_ensemble"),
        ("SemCacheSearchQueries", "SemCacheSearchQueries_hard_cos_ge_0.90", 0.05, "ours_weighted_ensemble"),
    ]
    out = [
        "| Dataset | Subset | Alpha | Method | Delta TPR | Delta hit | Rel hit | Ours FPR | Cos FPR | Ours TPR | Cos TPR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset, subset, alpha, method in picks:
        row = _find(rows, dataset=dataset, subset_name=subset, alpha=alpha, ours_method=method)
        if not row:
            continue
        out.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    subset,
                    _fmt(alpha),
                    method,
                    _fmt(row.get("delta_tpr")),
                    _fmt(row.get("delta_hit_rate")),
                    _fmt(row.get("relative_hit_gain")),
                    _fmt(row.get("ours_false_positive_rate")),
                    _fmt(row.get("cosine_false_positive_rate")),
                    _fmt(row.get("ours_true_positive_rate")),
                    _fmt(row.get("cosine_true_positive_rate")),
                ]
            )
            + " |"
        )
    return out


def _cost_snapshot(rows: Sequence[Dict[str, Any]]) -> List[str]:
    out = [
        "| Dataset | Method | Alpha | LLM/cache misses | Online judge | Eval judge | Offline labels |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset in ["SemCacheLMArena", "SemCacheSearchQueries"]:
        for method in ["cosine", "ours_weighted_ensemble", "ours_whitened_hadamard"]:
            subset = f"{dataset}_full"
            row = _find(rows, dataset=dataset, subset_name=subset, alpha=0.05, method=method)
            if not row:
                continue
            out.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        method,
                        _fmt(row.get("alpha")),
                        _fmt(row.get("online_llm_calls_cache_misses"), 0),
                        _fmt(row.get("online_judge_calls"), 0),
                        _fmt(row.get("evaluation_judge_calls"), 0),
                        _fmt(row.get("total_offline_labeling_calls"), 0),
                    ]
                )
                + " |"
            )
    return out


def _bucket_snapshot(rows: Sequence[Dict[str, Any]]) -> List[str]:
    picks = [
        ("SemCacheLMArena", "[0.90,0.95)"),
        ("SemCacheLMArena", "[0.95,1.00]"),
        ("SemCacheSearchQueries", "[0.85,0.90)"),
        ("SemCacheSearchQueries", "[0.90,0.95)"),
    ]
    out = [
        "| Dataset | Bucket | Cos TPR | Ours TPR | Cos FPR | Ours FPR | Cos hit | Ours hit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset, bucket in picks:
        row = _find(rows, dataset=dataset, bucket=bucket)
        if not row:
            continue
        out.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    bucket,
                    _fmt(row.get("cosine_selected_TPR")),
                    _fmt(row.get("ours_weighted_ensemble_selected_TPR")),
                    _fmt(row.get("cosine_selected_FPR")),
                    _fmt(row.get("ours_weighted_ensemble_selected_FPR")),
                    _fmt(row.get("cosine_hit_rate")),
                    _fmt(row.get("ours_weighted_ensemble_hit_rate")),
                ]
            )
            + " |"
        )
    return out


def build_report(args: argparse.Namespace) -> str:
    stream_rows = _read_csv(Path(args.stream_advantage))
    candidate_rows = _read_csv(Path(args.candidate_deltas))
    calibration_rows = _read_csv(Path(args.calibration_transfer))
    static_rows = _read_csv(Path(args.static_vs_online))
    bucket_rows = _read_csv(Path(args.hard_buckets))
    cost_rows = _read_csv(Path(args.cost_accounting))
    multiseed_status = Path(args.multiseed_status).read_text(encoding="utf-8") if Path(args.multiseed_status).exists() else ""

    best_stream = _best_by(stream_rows, "hit_rate_gain_abs")
    best_candidate = _best_by(candidate_rows, "delta_tpr", method="ours_weighted_ensemble")
    calib_satisfied = sum(1 for row in calibration_rows if row.get("constraint_satisfied_on_eval") == "yes")
    comparable_static = any(
        row.get("calibration_source", "").startswith("static")
        and row.get("comparison_status") == "same_online_pair_eval_distribution"
        for row in static_rows
    )
    single_seed = "single-seed only" in multiseed_status

    lines: List[str] = [
        "# Research Next Results Report",
        "",
        "## 1. Executive summary",
        "",
        "The previous strongest result was a stream-level comparison under `FP / n <= B`. The new candidate-level report now selects configurations using `FPR = FP / (FP + TN) <= alpha`, which is the constraint needed for the formal safe-reuse claim.",
        "",
        "The checked candidate-FPR results support a stronger but still bounded claim: online-mined WeightedEnsemble beats cosine on SemCacheLMArena full and hard subsets under matched candidate-level FPR constraints. On SemCacheSearchQueries the full-stream gain is positive but smaller, while hard subsets are mixed.",
        "",
        f"Calibration transfer remains a weakness: only {calib_satisfied}/{len(calibration_rows)} learned-threshold rows strictly satisfy `eval_FPR <= target_alpha` after being selected on `online_calib`.",
        "",
        "vCache remains secondary only. Current vCache candidate-level metrics are marked unreliable because raw logs do not consistently preserve nearest-candidate metadata for explore decisions.",
        "",
        "## 2. Current confirmed result: stream-level FP/n matched comparison",
        "",
        "Definition used by the earlier report:",
        "",
        "```text",
        "error_rate_stream = FP / n <= B",
        "```",
        "",
        "This is not the same as candidate-level FPR. It measures bad cache reuses per stream item, not false accepts among H0 candidate pairs.",
        "",
    ]
    if best_stream:
        lines.extend(
            [
                (
                    f"Strongest stream-level row: `{best_stream.get('ours_method')}` on `{best_stream.get('dataset')}` / `{best_stream.get('subset_name')}` "
                    f"at B `{_fmt(best_stream.get('budget'))}` had hit rate `{_fmt(best_stream.get('ours_hit_rate'))}` vs cosine `{_fmt(best_stream.get('cosine_hit_rate'))}` "
                    f"with FP/n `{_fmt(best_stream.get('ours_error_rate_stream'))}` vs cosine `{_fmt(best_stream.get('cosine_error_rate_stream'))}`."
                ),
                "",
                "Supported stream-level wording: Under matched stream-level error budgets, online-mined WeightedEnsemble recovers more cache hits than cosine on the checked datasets, with the strongest gains on SemCacheLMArena.",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. New result: candidate-level FPR matched comparison",
            "",
            "Definition used here:",
            "",
            "```text",
            "FPR = FP / (FP + TN)",
            "```",
            "",
            "For each dataset/subset/method/alpha, the summarizer keeps rows with `false_positive_rate <= alpha` and selects the eligible row with maximum `true_positive_rate`.",
            "",
        ]
    )
    if best_candidate:
        lines.extend(
            [
                (
                    f"Strongest valid candidate-FPR row: `ours_weighted_ensemble` on `{best_candidate.get('dataset')}` / `{best_candidate.get('subset_name')}` "
                    f"at alpha `{_fmt(best_candidate.get('alpha'))}` had TPR `{_fmt(best_candidate.get('ours_true_positive_rate'))}` vs cosine `{_fmt(best_candidate.get('cosine_true_positive_rate'))}` "
                    f"and hit rate `{_fmt(best_candidate.get('ours_hit_rate'))}` vs cosine `{_fmt(best_candidate.get('cosine_hit_rate'))}`."
                ),
                "",
            ]
        )
    lines.extend(_candidate_table(candidate_rows))
    lines.extend(
        [
            "",
            "Result interpretation: Under matched candidate-level FPR constraints, online-mined WeightedEnsemble achieves higher TPR/hit rate than cosine on SemCacheLMArena full and hard subsets, and on SemCacheSearchQueries full stream at alpha 0.05. SemCacheSearchQueries hard subsets are mixed, so the result should not be stated as universal.",
            "",
            "Detailed artifact: `results/candidate_fpr_matched_summary/candidate_fpr_matched_summary.md`.",
            "",
            "## 4. Calibration transfer analysis",
            "",
            f"Strict eval satisfaction is {calib_satisfied}/{len(calibration_rows)} rows. On SemCacheLMArena, the learned thresholds selected on `online_calib` exceed the target on `online_eval` for all checked learned-method targets. On SemCacheSearchQueries, WeightedEnsemble satisfies the strict eval constraint for target alpha 0.02, 0.05, and 0.08, but not for every target.",
            "",
            "This means the post-hoc candidate-FPR matched table supports method comparison under a common constraint, but the current calibration procedure does not yet prove reliable target transfer on every dataset.",
            "",
            "Detailed artifact: `results/calibration_transfer/calibration_transfer.md`.",
            "",
            "## 5. Static vs online-mined calibration analysis",
            "",
            f"Comparable static full-online evaluation present: {'yes' if comparable_static else 'no'}.",
            "",
            "The current static/random calibration output is a SemCacheLMArena smoke-stream diagnostic, not the same full online eval-pair distribution. It shows very high FPR on that smoke stream, but because the eval distribution differs, it cannot by itself prove the need for online-mined calibration.",
            "",
            "Detailed artifact: `results/static_vs_online_calibration/static_vs_online_calibration.md`.",
            "",
            "## 6. Hard-negative bucket analysis",
            "",
            "The bucket report uses the alpha 0.05 candidate-FPR-selected full-stream configurations and buckets examples by nearest-neighbor cosine.",
            "",
        ]
    )
    lines.extend(_bucket_snapshot(bucket_rows))
    lines.extend(
        [
            "",
            "Interpretation: On SemCacheLMArena, WeightedEnsemble recovers many hits in the 0.85 to 0.95 cosine ranges where the selected cosine threshold makes no hits. In the highest bucket it also improves TPR while reducing bucket FPR. On SemCacheSearchQueries, WeightedEnsemble is more selective than cosine in the >=0.90 buckets, but the full-stream gain is modest and bucket behavior is mixed.",
            "",
            "Detailed artifact: `results/hard_negative_bucket_analysis/hard_negative_bucket_analysis.md`.",
            "",
            "## 7. Cost accounting",
            "",
            "The cost report separates cache misses, online judge calls, evaluation judge calls, and offline labels. Evaluation judge calls are measurement cost and must not be counted as deployment savings.",
            "",
        ]
    )
    lines.extend(_cost_snapshot(cost_rows))
    lines.extend(
        [
            "",
            "Calibration judge calls are not separately logged. The available `calibration_labels_used` field is reported as offline label accounting, not runtime judge usage.",
            "",
            "Detailed artifact: `results/cost_accounting/cost_accounting.md`.",
            "",
            "## 8. Multi-seed status",
            "",
            "Current checked results are single-seed only." if single_seed else "Multiple seeds were detected; see the multi-seed summary artifact.",
            "",
            "Rerun commands for seeds 42, 43, 44, 45, and 46 are in `results/multiseed_status/rerun_commands.sh`.",
            "",
            "## 9. What claims are supported",
            "",
            "- Under matched stream-level error budgets, online-mined WeightedEnsemble recovers more cache hits than cosine on the checked datasets, with the strongest gains on SemCacheLMArena.",
            "- Under matched candidate-level FPR constraints, online-mined WeightedEnsemble achieves higher TPR/hit rate than cosine on SemCacheLMArena full and hard subsets, and on SemCacheSearchQueries full stream at the checked alpha 0.05 row.",
            "- The tested vCache reference configuration is not a main-claim baseline for candidate-level FPR because its current nearest-candidate metadata is incomplete.",
            "",
            "## 10. What claims are not yet supported",
            "",
            "- Do not claim WeightedEnsemble always beats cosine. SemCacheSearchQueries hard subsets contain mixed rows.",
            "- Do not claim calibration transfer is solved. Most learned thresholds do not strictly satisfy the target on eval after being selected on calibration.",
            "- Do not claim static/random calibration has been fairly beaten on a controlled full online eval stream. That rerun is still missing.",
            "- Do not claim multi-seed robustness. The checked outputs are seed 42 only.",
            "- Do not claim deployment-cost savings using evaluation judge calls. Those calls are offline measurement overhead.",
            "- Do not claim that we beat vCache generally. At most, use vCache as an audited secondary reference after rerunning it with reliable nearest-candidate metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate the combined research follow-up report.")
    ap.add_argument("--stream_advantage", default=DEFAULT_STREAM_ADVANTAGE)
    ap.add_argument("--candidate_deltas", default=DEFAULT_CANDIDATE_DELTAS)
    ap.add_argument("--calibration_transfer", default=DEFAULT_CALIBRATION)
    ap.add_argument("--static_vs_online", default=DEFAULT_STATIC_ONLINE)
    ap.add_argument("--hard_buckets", default=DEFAULT_BUCKETS)
    ap.add_argument("--cost_accounting", default=DEFAULT_COST)
    ap.add_argument("--multiseed_status", default=DEFAULT_MULTISEED_STATUS)
    ap.add_argument("--output", default="results/research_next_results_report.md")
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
