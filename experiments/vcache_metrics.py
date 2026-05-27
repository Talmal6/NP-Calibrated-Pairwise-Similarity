from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RAW_DECISION_FIELDS = [
    "dataset",
    "subset_name",
    "method",
    "seed",
    "delta_or_threshold",
    "delta_or_policy_param",
    "request_index",
    "stream_index",
    "prompt_id",
    "prompt",
    "query_id",
    "query_text",
    "nearest_prompt_id",
    "nearest_prompt",
    "nearest_candidate_id",
    "nearest_candidate_text_id",
    "nearest_candidate_response_id",
    "nearest_candidate_cosine",
    "nearest_candidate_label_H1_or_H0",
    "decision",
    "decision_type",
    "cache_hit_or_miss",
    "returned_response",
    "gold_response",
    "correctness",
    "nearest_would_be_correct",
    "TP",
    "FP",
    "TN",
    "FN",
    "similarity_score",
    "method_score",
    "latency",
    "llm_calls",
    "online_judge_calls",
    "evaluation_judge_calls",
    "online_judge_called",
    "eval_judge_called",
    "judge_calls",
    "total_estimated_cost",
    "cache_size",
    "embedding_model",
    "judge",
    "stream_hash",
    "cumulative_stream_error",
    "cumulative_hit_rate",
]

RAW_DECISION_MINIMAL_FIELDS = [
    "dataset",
    "subset_name",
    "method",
    "seed",
    "delta_or_threshold",
    "delta_or_policy_param",
    "request_index",
    "stream_index",
    "prompt_id",
    "query_id",
    "nearest_prompt_id",
    "nearest_candidate_id",
    "nearest_candidate_text_id",
    "nearest_candidate_response_id",
    "nearest_candidate_cosine",
    "nearest_candidate_label_H1_or_H0",
    "decision",
    "decision_type",
    "cache_hit_or_miss",
    "correctness",
    "nearest_would_be_correct",
    "TP",
    "FP",
    "TN",
    "FN",
    "similarity_score",
    "method_score",
    "latency",
    "llm_calls",
    "online_judge_calls",
    "evaluation_judge_calls",
    "online_judge_called",
    "eval_judge_called",
    "judge_calls",
    "total_estimated_cost",
    "cache_size",
    "embedding_model",
    "judge",
    "stream_hash",
    "threshold_source",
    "score_orientation",
    "calibration_source",
    "cumulative_stream_error",
    "cumulative_hit_rate",
]

SUMMARY_FIELDS = [
    "dataset",
    "subset_name",
    "method",
    "seed",
    "delta_or_threshold",
    "n",
    "TP",
    "FP",
    "TN",
    "FN",
    "error_rate_stream",
    "hit_rate",
    "precision",
    "false_positive_rate",
    "true_positive_rate",
    "miss_rate",
    "average_latency",
    "p50_latency",
    "p95_latency",
    "llm_calls",
    "judge_calls",
    "total_estimated_cost",
    "threshold_source",
    "score_orientation",
    "calibration_source",
    "calibration_train_pairs",
    "calibration_calib_pairs",
    "calibration_eval_pairs",
    "calibration_labels_used",
    "calibration_split_size",
    "calibration_equivalence_mode",
]


def safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)


def as_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def add_cumulative_fields(records: List[Dict[str, Any]]) -> None:
    fp = 0
    hits = 0
    for i, row in enumerate(records, start=1):
        is_hit = row.get("decision") in {"hit", "exploit"}
        correct = as_bool(row.get("correctness"))
        if is_hit:
            hits += 1
            if correct is False:
                fp += 1
        row["cumulative_stream_error"] = safe_div(fp, i)
        row["cumulative_hit_rate"] = safe_div(hits, i)


def summarize_run(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize empty run")
    n = len(records)
    tp = fp = tn = fn = 0
    for row in records:
        is_hit = row.get("decision") in {"hit", "exploit"}
        correct = as_bool(row.get("correctness"))
        would_correct = as_bool(row.get("nearest_would_be_correct"))
        if is_hit:
            if correct is True:
                tp += 1
            elif correct is False:
                fp += 1
        else:
            if would_correct is True:
                fn += 1
            elif would_correct is False:
                tn += 1

    latencies = np.asarray([float(r.get("latency") or 0.0) for r in records], dtype=float)
    llm_calls = int(sum(int(r.get("llm_calls") or 0) for r in records))
    judge_calls = int(sum(int(r.get("judge_calls") or 0) for r in records))
    cost_values = [r.get("total_estimated_cost") for r in records]
    cost = None
    if any(v not in {None, ""} for v in cost_values):
        cost = float(sum(float(v or 0.0) for v in cost_values))

    first = records[0]
    hit_rate = safe_div(tp + fp, n)
    return {
        "dataset": first.get("dataset"),
        "subset_name": first.get("subset_name"),
        "method": first.get("method"),
        "seed": first.get("seed"),
        "delta_or_threshold": first.get("delta_or_threshold"),
        "n": n,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "error_rate_stream": safe_div(fp, n),
        "hit_rate": hit_rate,
        "precision": safe_div(tp, tp + fp),
        "false_positive_rate": safe_div(fp, fp + tn),
        "true_positive_rate": safe_div(tp, tp + fn),
        "miss_rate": None if hit_rate is None else 1.0 - hit_rate,
        "average_latency": float(np.mean(latencies)) if latencies.size else None,
        "p50_latency": float(np.quantile(latencies, 0.50)) if latencies.size else None,
        "p95_latency": float(np.quantile(latencies, 0.95)) if latencies.size else None,
        "llm_calls": llm_calls,
        "judge_calls": judge_calls,
        "total_estimated_cost": cost,
        "threshold_source": first.get("threshold_source"),
        "score_orientation": first.get("score_orientation"),
        "calibration_source": first.get("calibration_source"),
        "calibration_train_pairs": first.get("calibration_train_pairs"),
        "calibration_calib_pairs": first.get("calibration_calib_pairs"),
        "calibration_eval_pairs": first.get("calibration_eval_pairs"),
        "calibration_labels_used": first.get("calibration_labels_used"),
        "calibration_split_size": first.get("calibration_split_size"),
        "calibration_equivalence_mode": first.get("calibration_equivalence_mode"),
    }


def summarize_all(raw_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_records:
        key = (
            row.get("dataset"),
            row.get("subset_name"),
            row.get("method"),
            row.get("seed"),
            row.get("delta_or_threshold"),
        )
        grouped[key].append(row)
    return [summarize_run(rows) for rows in grouped.values()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_none(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _plot_scatter(summary: Sequence[Dict[str, Any]], x: str, y: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.0, 5.0))
    for row in summary:
        xv = _float_or_none(row.get(x))
        yv = _float_or_none(row.get(y))
        if xv is None or yv is None:
            continue
        label = f"{row.get('method')} {row.get('delta_or_threshold')}"
        plt.scatter([xv], [yv], s=38)
        plt.annotate(label, (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_over_time(raw_records: Sequence[Dict[str, Any]], y: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_records:
        grouped[(row.get("method"), row.get("delta_or_threshold"))].append(row)

    plt.figure(figsize=(8.0, 5.0))
    for (method, param), rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r.get("request_index") or 0))
        xs = [int(r.get("request_index") or 0) + 1 for r in rows]
        ys = [_float_or_none(r.get(y)) for r in rows]
        if not any(v is not None for v in ys):
            continue
        plt.plot(xs, [np.nan if v is None else v for v in ys], linewidth=1.4, label=f"{method} {param}")
    plt.xlabel("processed_requests")
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    if len(grouped) <= 18:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_plots(summary: Sequence[Dict[str, Any]], raw_records: Sequence[Dict[str, Any]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".matplotlib"))

    _plot_scatter(
        summary,
        x="error_rate_stream",
        y="hit_rate",
        path=plots_dir / "hit_rate_vs_stream_error_rate.png",
        title="Hit Rate vs Stream Error Rate",
    )
    _plot_scatter(
        summary,
        x="hit_rate",
        y="precision",
        path=plots_dir / "precision_vs_hit_rate.png",
        title="Precision vs Hit Rate",
    )
    if any(_float_or_none(r.get("false_positive_rate")) is not None for r in summary) and any(
        _float_or_none(r.get("true_positive_rate")) is not None for r in summary
    ):
        _plot_scatter(
            summary,
            x="false_positive_rate",
            y="true_positive_rate",
            path=plots_dir / "tpr_vs_fpr.png",
            title="TPR vs FPR",
        )
    _plot_scatter(
        summary,
        x="hit_rate",
        y="average_latency",
        path=plots_dir / "latency_vs_hit_rate.png",
        title="Latency vs Hit Rate",
    )
    _plot_over_time(
        raw_records,
        y="cumulative_stream_error",
        path=plots_dir / "stream_error_rate_over_time.png",
        title="Stream Error Rate Over Time",
    )
    _plot_over_time(
        raw_records,
        y="cumulative_hit_rate",
        path=plots_dir / "hit_rate_over_time.png",
        title="Hit Rate Over Time",
    )
