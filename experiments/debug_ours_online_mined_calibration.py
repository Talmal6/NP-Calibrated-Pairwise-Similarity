from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .equivalence import EquivalenceJudge
from .online_mined_ours import (
    OnlinePairs,
    accept_scores,
    h0_h1,
    infer_orientation,
    load_eval_stream_from_pairs_dir,
    load_pairs_dir,
    score_method,
    select_tau,
    train_online_mined_model,
)
from .online_policies import ExactVectorCache, run_cosine_policy, run_ours_policy
from .vcache_metrics import summarize_run, write_csv, write_json


METHOD_MAP = {
    "ours_whitened_hadamard": "WhitenedCosine",
    "ours_weighted_ensemble": "WeightedEnsemble",
    "ours_xgboost": "XGBoost",
    "ours_lda": "LDA",
    "ours_tiny_mlp": "Tiny MLP",
}

THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]

SCORE_FIELDS = [
    "method",
    "target_fpr",
    "split",
    "label",
    "n",
    "mean",
    "median",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
    "orientation",
    "warning",
]

THRESHOLD_FIELDS = [
    "method",
    "target_fpr",
    "tau",
    "orientation",
    "calib_H0_accept_rate",
    "calib_H1_accept_rate",
    "eval_H0_accept_rate",
    "eval_H1_accept_rate",
    "eval_precision",
    "eval_TPR",
    "eval_FPR",
    "tolerance",
]

PAIR_EVAL_FIELDS = [
    "method",
    "target_fpr",
    "n",
    "TP",
    "FP",
    "TN",
    "FN",
    "precision",
    "true_positive_rate",
    "false_positive_rate",
    "hit_rate",
    "warning",
]

SMOKE_FIELDS = [
    "method",
    "target_fpr",
    "n",
    "TP",
    "FP",
    "TN",
    "FN",
    "stream_error",
    "FPR",
    "TPR",
    "precision",
    "hit_rate",
    "miss_rate",
    "warning",
]

DEBUG_FIELDS = [
    "method",
    "target_fpr",
    "prompt_id",
    "candidate_id",
    "correctness",
    "score",
    "threshold",
    "orientation",
    "decision",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Debug ours calibration on online-mined nearest-neighbor candidate pairs.")
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True, choices=sorted(METHOD_MAP))
    ap.add_argument("--target_fprs", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.08])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_smoke_examples", type=int, default=1000)
    ap.add_argument("--cache_size", type=int, default=None)
    ap.add_argument("--eviction_policy", default=None, choices=["mru", "lru", "fifo"])
    return ap.parse_args(argv)


def _quantiles(scores: np.ndarray) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    qs = np.quantile(s, [0.01, 0.05, 0.50, 0.95, 0.99]) if s.size else [np.nan] * 5
    return {
        "n": int(s.size),
        "mean": float(np.mean(s)) if s.size else None,
        "median": float(np.median(s)) if s.size else None,
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p50": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
    }


def _score_rows(method: str, alpha: float, split: str, label: str, scores: np.ndarray, orientation: str, warning: str) -> Dict[str, Any]:
    return {
        "method": method,
        "target_fpr": float(alpha),
        "split": split,
        "label": label,
        **_quantiles(scores),
        "orientation": orientation,
        "warning": warning,
    }


def pair_eval_metrics(y: np.ndarray, accepted: np.ndarray) -> Dict[str, Any]:
    yy = np.asarray(y).reshape(-1).astype(np.int32)
    aa = np.asarray(accepted).reshape(-1).astype(bool)
    tp = int(np.sum(aa & (yy == 1)))
    fp = int(np.sum(aa & (yy == 0)))
    tn = int(np.sum((~aa) & (yy == 0)))
    fn = int(np.sum((~aa) & (yy == 1)))
    n = int(yy.size)
    return {
        "n": n,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": None if tp + fp == 0 else tp / float(tp + fp),
        "true_positive_rate": None if tp + fn == 0 else tp / float(tp + fn),
        "false_positive_rate": None if fp + tn == 0 else fp / float(fp + tn),
        "hit_rate": None if n == 0 else (tp + fp) / float(n),
    }


def smoke_row_from_summary(method: str, alpha: Any, summary: Dict[str, Any], warning: str = "") -> Dict[str, Any]:
    return {
        "method": method,
        "target_fpr": alpha,
        "n": summary.get("n"),
        "TP": summary.get("TP"),
        "FP": summary.get("FP"),
        "TN": summary.get("TN"),
        "FN": summary.get("FN"),
        "stream_error": summary.get("error_rate_stream"),
        "FPR": summary.get("false_positive_rate"),
        "TPR": summary.get("true_positive_rate"),
        "precision": summary.get("precision"),
        "hit_rate": summary.get("hit_rate"),
        "miss_rate": summary.get("miss_rate"),
        "warning": warning,
    }


def run_always_policy(stream: Sequence[Any], *, always_hit: bool, cache_size: int, eviction_policy: str) -> Dict[str, Any]:
    from .online_policies import _base_record, _evaluate_decision
    from .vcache_metrics import add_cumulative_fields

    cache = ExactVectorCache(capacity=cache_size, eviction_policy=eviction_policy)
    judge = EquivalenceJudge(mode="cluster")
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(stream):
        nearest, sim = cache.nearest(ex.embedding)
        is_hit = bool(always_hit and nearest is not None)
        if is_hit:
            returned = nearest.example.gold_response
            llm_calls = 0
        else:
            returned = ex.gold_response
            llm_calls = 1
            cache.add(ex)
        correctness, would_correct, eval_calls = _evaluate_decision(judge, ex, nearest, is_hit=is_hit, returned_response=returned)
        rows.append(
            _base_record(
                dataset="online_mined_debug",
                method="always_hit" if always_hit else "always_miss",
                seed=0,
                param="baseline",
                request_index=i,
                example=ex,
                nearest=nearest,
                decision="hit" if is_hit else "miss",
                returned_response=returned,
                correctness=correctness,
                would_correct=would_correct,
                similarity_score=sim,
                method_score=sim,
                latency=0.0,
                llm_calls=llm_calls,
                online_judge_calls=0,
                evaluation_judge_calls=eval_calls,
                cache_size=cache_size,
                embedding_model="debug",
                judge_name=judge.name,
                stream_hash="debug",
            )
        )
    add_cumulative_fields(rows)
    return summarize_run(rows)


def _plot_hist(scores_h0: np.ndarray, scores_h1: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.hist(scores_h0, bins=60, alpha=0.65, label="H0", density=True)
    plt.hist(scores_h1, bins=60, alpha=0.65, label="H1", density=True)
    plt.xlabel("score")
    plt.ylabel("density")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_accept_rates(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        xs = [float(r["target_fpr"]) for r in rr]
        h0 = [float(r["calib_H0_accept_rate"]) for r in rr]
        h1 = [float(r["calib_H1_accept_rate"]) for r in rr]
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(xs, h0, marker="o", label="H0")
        plt.plot(xs, h1, marker="o", label="H1")
        plt.plot(xs, xs, linestyle="--", color="black", linewidth=1, label="target")
        plt.xlabel("target_fpr")
        plt.ylabel("accept_rate")
        plt.title(f"{method} online-mined accept rates")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{method}_accept_rates.png", dpi=160)
        plt.close()


def _plot_pair_tradeoffs(pair_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 4.5))
    for method in sorted({r["method"] for r in pair_rows}):
        rr = [r for r in pair_rows if r["method"] == method]
        xs = [float(r["false_positive_rate"]) for r in rr if r.get("false_positive_rate") not in {None, ""}]
        ys = [float(r["true_positive_rate"]) for r in rr if r.get("true_positive_rate") not in {None, ""}]
        if xs and len(xs) == len(ys):
            plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Online-mined pair ROC-like tradeoff")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "tpr_vs_fpr_online_pairs.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    for method in sorted({r["method"] for r in pair_rows}):
        rr = [r for r in pair_rows if r["method"] == method]
        xs = [float(r["hit_rate"]) for r in rr if r.get("hit_rate") not in {None, ""}]
        ys = [float(r["precision"]) for r in rr if r.get("precision") not in {None, ""}]
        if xs and len(xs) == len(ys):
            plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("hit_rate")
    plt.ylabel("precision")
    plt.title("Precision vs hit rate")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_vs_hit_rate.png", dpi=160)
    plt.close()


def _plot_cosine_distribution(eval_pairs: OnlinePairs, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    h0 = eval_pairs.cosine[eval_pairs.y == 0]
    h1 = eval_pairs.cosine[eval_pairs.y == 1]
    plt.figure(figsize=(7, 4.5))
    if h0.size:
        plt.hist(h0, bins=60, alpha=0.65, label="H0", density=True)
    if h1.size:
        plt.hist(h1, bins=60, alpha=0.65, label="H1", density=True)
    plt.xlabel("nearest_neighbor_cosine")
    plt.ylabel("density")
    plt.title("Online candidate cosine distribution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_distribution_h1_vs_h0.png", dpi=160)
    plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, calib_pairs, eval_pairs, pairs_config = load_pairs_dir(pairs_dir)
    cache_size = int(args.cache_size or pairs_config.get("cache_size", pairs_config.get("args", {}).get("cache_size", 4096)))
    eviction_policy = str(args.eviction_policy or pairs_config.get("eviction_policy", pairs_config.get("args", {}).get("eviction_policy", "mru")))
    eval_stream, _info, _cfg, embedding_model_name = load_eval_stream_from_pairs_dir(
        pairs_dir,
        limit=int(args.max_smoke_examples),
    )

    warnings: List[str] = []
    failures: List[str] = []
    score_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    h0_calib, h1_calib, h0_calib_alt, h1_calib_alt = h0_h1(calib_pairs)
    h0_eval, h1_eval, h0_eval_alt, h1_eval_alt = h0_h1(eval_pairs)

    previous_by_method: Dict[str, np.ndarray] = {}
    for method_label in args.methods:
        method_name = METHOD_MAP[method_label]
        for alpha in args.target_fprs:
            model, diag = train_online_mined_model(
                pairs_dir=pairs_dir,
                method_name=method_name,
                alpha=float(alpha),
                seed=args.seed,
            )
            method = model.method
            s0_calib = score_method(method, h0_calib, h0_calib_alt, model.uses_alt_score)
            s1_calib = score_method(method, h1_calib, h1_calib_alt, model.uses_alt_score)
            s0_eval = score_method(method, h0_eval, h0_eval_alt, model.uses_alt_score)
            s1_eval = score_method(method, h1_eval, h1_eval_alt, model.uses_alt_score)
            orientation, orientation_warning = infer_orientation(s0_calib, s1_calib)
            if orientation == "ambiguous":
                failures.append(f"{method_label} alpha={alpha}: score orientation ambiguous: {orientation_warning}")
                orientation = model.orientation
            if orientation_warning:
                warnings.append(f"{method_label} alpha={alpha}: {orientation_warning}")
            tau = model.tau
            calib_h0_acc = float(np.mean(accept_scores(s0_calib, tau, orientation)))
            calib_h1_acc = float(np.mean(accept_scores(s1_calib, tau, orientation)))
            eval_h0_acc = float(np.mean(accept_scores(s0_eval, tau, orientation)))
            eval_h1_acc = float(np.mean(accept_scores(s1_eval, tau, orientation)))
            tolerance = max(0.005, 0.25 * float(alpha))
            if calib_h0_acc > float(alpha) + tolerance:
                failures.append(
                    f"{method_label} alpha={alpha}: calib_H0_accept_rate={calib_h0_acc:.6f} "
                    f"> alpha+tolerance={float(alpha) + tolerance:.6f}"
                )

            for split_name, label, scores in [
                ("online_calib", "H0", s0_calib),
                ("online_calib", "H1", s1_calib),
                ("online_eval", "H0", s0_eval),
                ("online_eval", "H1", s1_eval),
            ]:
                score_rows.append(_score_rows(method_label, float(alpha), split_name, label, scores, orientation, orientation_warning))

            eval_scores = np.empty(eval_pairs.y.shape[0], dtype=np.float64)
            eval_scores[eval_pairs.y == 0] = s0_eval
            eval_scores[eval_pairs.y == 1] = s1_eval
            accepted_eval = accept_scores(eval_scores, tau, orientation)
            pair_metrics = pair_eval_metrics(eval_pairs.y, accepted_eval)
            eval_fpr = pair_metrics.get("false_positive_rate")
            eval_tpr = pair_metrics.get("true_positive_rate")
            eval_precision = pair_metrics.get("precision")

            threshold_rows.append(
                {
                    "method": method_label,
                    "target_fpr": float(alpha),
                    "tau": float(tau),
                    "orientation": orientation,
                    "calib_H0_accept_rate": calib_h0_acc,
                    "calib_H1_accept_rate": calib_h1_acc,
                    "eval_H0_accept_rate": eval_h0_acc,
                    "eval_H1_accept_rate": eval_h1_acc,
                    "eval_precision": eval_precision,
                    "eval_TPR": eval_tpr,
                    "eval_FPR": eval_fpr,
                    "tolerance": tolerance,
                }
            )

            pair_warning = ""
            if eval_fpr is not None and float(alpha) <= 0.05 and float(eval_fpr) > 0.20:
                pair_warning = f"eval_FPR {float(eval_fpr):.6f} > 0.20 for alpha={alpha}"
                failures.append(f"{method_label} alpha={alpha}: {pair_warning}")
            pair_rows.append({"method": method_label, "target_fpr": float(alpha), **pair_metrics, "warning": pair_warning})

            _plot_hist(s0_calib, s1_calib, plots_dir / f"{method_label}_alpha_{alpha:g}_calib_score_hist.png", f"{method_label} calib alpha={alpha}")
            _plot_hist(s0_eval, s1_eval, plots_dir / f"{method_label}_alpha_{alpha:g}_eval_score_hist.png", f"{method_label} eval alpha={alpha}")

            judge = EquivalenceJudge(mode="cluster")
            records = run_ours_policy(
                eval_stream,
                model=model,
                method_label=method_label,
                dataset=str(pairs_config["dataset"]),
                seed=args.seed,
                cache_size=cache_size,
                eviction_policy=eviction_policy,
                judge=judge,
                embedding_model=embedding_model_name,
                stream_hash="online_mined_eval_smoke",
            )
            summary = summarize_run(records)
            smoke_warning_bits: List[str] = []
            smoke_failure_bits: List[str] = []
            smoke_fpr = summary.get("false_positive_rate")
            stream_error = summary.get("error_rate_stream")
            hit_rate = summary.get("hit_rate")
            if smoke_fpr is not None and float(alpha) <= 0.05 and float(smoke_fpr) > 0.20:
                smoke_failure_bits.append(f"online_smoke_FPR {float(smoke_fpr):.6f} > 0.20")
            if stream_error is not None and float(alpha) <= 0.05 and float(stream_error) > 0.20:
                smoke_failure_bits.append(f"online_smoke_stream_error {float(stream_error):.6f} > 0.20")
            current_decisions = np.asarray([1 if r.get("decision") == "hit" else 0 for r in records], dtype=np.int8)
            prev = previous_by_method.get(method_label)
            if prev is not None and prev.shape == current_decisions.shape:
                changed = float(np.mean(prev != current_decisions))
                if changed <= 0.005:
                    smoke_warning_bits.append(f"decisions nearly identical to previous alpha: changed_fraction={changed:.6f}")
            previous_by_method[method_label] = current_decisions

            always_hit = run_always_policy(eval_stream, always_hit=True, cache_size=cache_size, eviction_policy=eviction_policy)
            always_hit_rate = float(always_hit.get("hit_rate") or 0.0)
            if hit_rate is not None and abs(float(hit_rate) - always_hit_rate) <= 0.05:
                smoke_failure_bits.append("ERROR: learned method behaves like always_hit even after online-mined calibration.")
            smoke_note = "; ".join(smoke_warning_bits + smoke_failure_bits)
            if smoke_failure_bits:
                failures.append(f"{method_label} alpha={alpha} online smoke: " + "; ".join(smoke_failure_bits))
            elif smoke_warning_bits:
                warnings.append(f"{method_label} alpha={alpha} online smoke: " + "; ".join(smoke_warning_bits))
            smoke_rows.append(smoke_row_from_summary(method_label, float(alpha), summary, smoke_note))
            for row in [r for r in records if r.get("method_score") not in {None, ""}][:20]:
                debug_rows.append(
                    {
                        "method": method_label,
                        "target_fpr": float(alpha),
                        "prompt_id": row.get("prompt_id"),
                        "candidate_id": row.get("nearest_prompt_id"),
                        "correctness": row.get("correctness") if row.get("decision") == "hit" else row.get("nearest_would_be_correct"),
                        "score": row.get("method_score"),
                        "threshold": float(tau),
                        "orientation": orientation,
                        "decision": row.get("decision"),
                    }
                )

    always_hit = run_always_policy(eval_stream, always_hit=True, cache_size=cache_size, eviction_policy=eviction_policy)
    always_miss = run_always_policy(eval_stream, always_hit=False, cache_size=cache_size, eviction_policy=eviction_policy)
    smoke_rows.append(smoke_row_from_summary("always_hit", "baseline", always_hit))
    smoke_rows.append(smoke_row_from_summary("always_miss", "baseline", always_miss))
    for threshold in THRESHOLDS:
        rows = run_cosine_policy(
            eval_stream,
            threshold=float(threshold),
            dataset=str(pairs_config["dataset"]),
            seed=args.seed,
            cache_size=cache_size,
            eviction_policy=eviction_policy,
            judge=EquivalenceJudge(mode="cluster"),
            embedding_model=embedding_model_name,
            stream_hash="online_mined_eval_smoke",
        )
        smoke_rows.append(smoke_row_from_summary("cosine", float(threshold), summarize_run(rows)))

        pair_accepted = eval_pairs.cosine >= float(threshold)
        pair_rows.append({"method": "cosine", "target_fpr": float(threshold), **pair_eval_metrics(eval_pairs.y, pair_accepted), "warning": ""})

    _plot_accept_rates(threshold_rows, plots_dir)
    _plot_pair_tradeoffs(pair_rows, plots_dir)
    _plot_cosine_distribution(eval_pairs, plots_dir)

    write_json(out_dir / "config.json", {"args": vars(args), "pairs_config": pairs_config})
    write_csv(out_dir / "online_pair_score_distribution_summary.csv", score_rows, SCORE_FIELDS)
    write_csv(out_dir / "online_threshold_calibration_summary.csv", threshold_rows, THRESHOLD_FIELDS)
    write_csv(out_dir / "online_pair_eval_summary.csv", pair_rows, PAIR_EVAL_FIELDS)
    write_csv(out_dir / "online_smoke_test_summary.csv", smoke_rows, SMOKE_FIELDS)
    write_csv(out_dir / "online_smoke_debug_examples.csv", debug_rows, DEBUG_FIELDS)
    with (out_dir / "warnings.txt").open("w", encoding="utf-8") as f:
        for msg in warnings + failures:
            f.write(str(msg) + "\n")

    if failures:
        print("ONLINE-MINED CALIBRATION FAILED", file=sys.stderr)
        for msg in failures:
            print(f"- {msg}", file=sys.stderr)
        raise SystemExit(1)
    print("ONLINE-MINED CALIBRATION PASSED", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
