from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compare_vcache_real_datasets import SPECS, _embedding_choice, load_real_stream
from .equivalence import EquivalenceJudge
from .online_policies import (
    ExactVectorCache,
    OursModel,
    _base_record,
    _evaluate_decision,
    run_ours_policy,
    train_ours_model,
)
from .vcache_metrics import add_cumulative_fields, summarize_run, write_csv, write_json


METHOD_MAP = {
    "ours_whitened_hadamard": "WhitenedCosine",
    "ours_weighted_ensemble": "WeightedEnsemble",
    "ours_xgboost": "XGBoost",
    "ours_lda": "LDA",
    "ours_tiny_mlp": "Tiny MLP",
}


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
    "separation_warning",
]


THRESHOLD_FIELDS = [
    "method",
    "target_fpr",
    "orientation",
    "model_threshold",
    "reference_threshold_full_calib",
    "threshold_source",
    "calib_accept_rate_H0",
    "calib_accept_rate_H1",
    "eval_accept_rate_H0",
    "eval_accept_rate_H1",
    "tolerance",
]


SMOKE_FIELDS = [
    "method",
    "target_fpr",
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
    "llm_calls",
    "judge_calls",
    "warning",
]


DEBUG_EXAMPLE_FIELDS = [
    "method",
    "target_fpr",
    "prompt_id",
    "candidate_id",
    "label_or_correctness",
    "score",
    "threshold",
    "orientation",
    "decision",
    "expected_decision_from_sanity_logic",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Debug learned-pairwise calibration before online cache evaluation.")
    ap.add_argument("--pairwise_data", required=True)
    ap.add_argument("--method", nargs="+", required=True, choices=sorted(METHOD_MAP))
    ap.add_argument("--pair_feature", default="hadamard", choices=["hadamard", "absdiff", "concat", "cosine"])
    ap.add_argument("--target_fprs", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.08])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/debug_ours_calibration")
    ap.add_argument("--feature_key", default="emb")
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--alt_feature_key", default="cosine_to_anchor")
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_calib", type=int, default=1200)
    ap.add_argument("--n_eval", type=int, default=1200)
    ap.add_argument("--stream_dataset", default="auto", choices=["auto", "none", *sorted(SPECS)])
    ap.add_argument("--embedding_model", default="GTE")
    ap.add_argument("--max_examples", type=int, default=1000)
    ap.add_argument("--cache_size", type=int, default=4096)
    ap.add_argument("--eviction_policy", default="mru", choices=["mru", "lru", "fifo"])
    ap.add_argument("--allow_missing_pairwise_model", action="store_true")
    return ap.parse_args(argv)


def _shape(value: np.ndarray) -> str:
    return "x".join(str(x) for x in value.shape)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _read_npz_metadata(path: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    with np.load(path, allow_pickle=True) as ds:
        arrays = {key: ds[key] for key in ds.files}
    rows = []
    for key, value in arrays.items():
        rows.append(
            {
                "key": key,
                "shape": _shape(np.asarray(value)),
                "dtype": str(np.asarray(value).dtype),
            }
        )
    return arrays, rows


def _infer_dataset_from_path(path: Path) -> str:
    name = path.name.lower()
    parent = str(path.parent).lower()
    text = f"{parent}/{name}"
    if "searchqueries" in text or "search_queries" in text:
        return "SemCacheSearchQueries"
    if "lmarena" in text or "lmarena" in text or "arena" in text:
        return "SemCacheLMArena"
    return "none"


def _as_feature_matrix(value: np.ndarray, key: str) -> np.ndarray:
    X = np.asarray(value, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Feature key {key!r} must be 2D, got shape={X.shape}")
    return X


def _as_labels(value: np.ndarray, key: str) -> np.ndarray:
    y = np.asarray(value).reshape(-1)
    unique = set(np.unique(y).tolist())
    if unique - {0, 1}:
        raise SystemExit(f"Label mapping ambiguous: {key!r} contains labels {sorted(unique)}, expected only 0/1")
    return y.astype(np.int32, copy=False)


@dataclass
class SplitData:
    h0_train: np.ndarray
    h1_train: np.ndarray
    h0_calib: np.ndarray
    h1_calib: np.ndarray
    h0_eval: np.ndarray
    h1_eval: np.ndarray
    h0_train_alt: Optional[np.ndarray]
    h1_train_alt: Optional[np.ndarray]
    h0_calib_alt: Optional[np.ndarray]
    h1_calib_alt: Optional[np.ndarray]
    h0_eval_alt: Optional[np.ndarray]
    h1_eval_alt: Optional[np.ndarray]


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_alt: Optional[np.ndarray],
    n_train: int,
    n_calib: int,
    n_eval: int,
    seed: int,
) -> SplitData:
    rng = np.random.default_rng(int(seed))
    h0_idx = np.flatnonzero(y == 0)
    h1_idx = np.flatnonzero(y == 1)
    rng.shuffle(h0_idx)
    rng.shuffle(h1_idx)
    need = int(n_train) + int(n_calib) + int(n_eval)
    if h0_idx.size < need or h1_idx.size < need:
        raise SystemExit(
            "Not enough pairwise rows for debug splits: "
            f"H0={h0_idx.size}, H1={h1_idx.size}, need_each={need}"
        )

    h0_train_idx = h0_idx[:n_train]
    h0_calib_idx = h0_idx[n_train : n_train + n_calib]
    h0_eval_idx = h0_idx[n_train + n_calib : need]
    h1_train_idx = h1_idx[:n_train]
    h1_calib_idx = h1_idx[n_train : n_train + n_calib]
    h1_eval_idx = h1_idx[n_train + n_calib : need]

    def take_alt(idx: np.ndarray) -> Optional[np.ndarray]:
        if X_alt is None:
            return None
        return X_alt[idx]

    return SplitData(
        h0_train=X[h0_train_idx],
        h1_train=X[h1_train_idx],
        h0_calib=X[h0_calib_idx],
        h1_calib=X[h1_calib_idx],
        h0_eval=X[h0_eval_idx],
        h1_eval=X[h1_eval_idx],
        h0_train_alt=take_alt(h0_train_idx),
        h1_train_alt=take_alt(h1_train_idx),
        h0_calib_alt=take_alt(h0_calib_idx),
        h1_calib_alt=take_alt(h1_calib_idx),
        h0_eval_alt=take_alt(h0_eval_idx),
        h1_eval_alt=take_alt(h1_eval_idx),
    )


def score_model(model: OursModel, X: np.ndarray, X_alt: Optional[np.ndarray]) -> np.ndarray:
    if model.uses_alt_score:
        if X_alt is None:
            raise ValueError("Model requires alt scores but no alt feature matrix is available")
        return np.asarray(model.method.score(X, X_alt=X_alt), dtype=np.float64).reshape(-1)
    return np.asarray(model.method.score(X), dtype=np.float64).reshape(-1)


def select_threshold(scores_h0: np.ndarray, alpha: float, orientation: str) -> float:
    s = np.asarray(scores_h0, dtype=np.float64).reshape(-1)
    if orientation == "higher":
        return float(np.quantile(s, 1.0 - float(alpha)))
    if orientation == "lower":
        return float(np.quantile(s, float(alpha)))
    raise ValueError(f"Unknown orientation {orientation!r}")


def accept(scores: np.ndarray, threshold: float, orientation: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if orientation == "higher":
        return s >= float(threshold)
    if orientation == "lower":
        return s <= float(threshold)
    raise ValueError(f"Unknown orientation {orientation!r}")


def quantile_row(method: str, alpha: float, split: str, label: str, scores: np.ndarray, orientation: str) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    qs = np.quantile(s, [0.01, 0.05, 0.50, 0.95, 0.99]) if s.size else [math.nan] * 5
    return {
        "method": method,
        "target_fpr": float(alpha),
        "split": split,
        "label": label,
        "n": int(s.size),
        "mean": float(np.mean(s)) if s.size else None,
        "median": float(np.median(s)) if s.size else None,
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p50": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
        "orientation": orientation,
        "separation_warning": "",
    }


def infer_orientation(scores_h0: np.ndarray, scores_h1: np.ndarray) -> Tuple[str, str]:
    mean0 = float(np.mean(scores_h0))
    mean1 = float(np.mean(scores_h1))
    pooled = np.asarray(np.concatenate([scores_h0, scores_h1]), dtype=np.float64)
    spread = float(np.std(pooled))
    diff = mean1 - mean0
    if abs(diff) <= max(1e-8, 0.02 * spread):
        return "ambiguous", f"score distributions have little separation: mean_H1={mean1}, mean_H0={mean0}, std={spread}"
    if diff > 0:
        return "higher", ""
    return "lower", ""


def _plot_score_hist(
    out_path: Path,
    *,
    method: str,
    alpha: float,
    scores_h0: np.ndarray,
    scores_h1: np.ndarray,
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.5))
    plt.hist(scores_h0, bins=60, alpha=0.65, label="H0", density=True)
    plt.hist(scores_h1, bins=60, alpha=0.65, label="H1", density=True)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label="threshold")
    plt.title(f"{method} score distribution alpha={alpha}")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_accept_rates(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    by_method = sorted({r["method"] for r in rows})
    for method in by_method:
        rr = [r for r in rows if r["method"] == method]
        xs = [float(r["target_fpr"]) for r in rr]
        h0 = [float(r["calib_accept_rate_H0"]) for r in rr]
        h1 = [float(r["calib_accept_rate_H1"]) for r in rr]
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(xs, h0, marker="o", label="calib H0 accept")
        plt.plot(xs, h1, marker="o", label="calib H1 accept")
        plt.plot(xs, xs, linestyle="--", color="black", linewidth=1.0, label="target")
        plt.xlabel("target_fpr")
        plt.ylabel("accept_rate")
        plt.title(f"{method} accept rates")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{method}_accept_rates.png", dpi=160)
        plt.close()


def _plot_roc_like(method: str, scores_h0: np.ndarray, scores_h1: np.ndarray, orientation: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    all_scores = np.unique(np.concatenate([scores_h0, scores_h1]))
    if all_scores.size > 800:
        all_scores = np.quantile(all_scores, np.linspace(0, 1, 800))
    fprs = []
    tprs = []
    for tau in all_scores:
        fprs.append(float(np.mean(accept(scores_h0, tau, orientation))))
        tprs.append(float(np.mean(accept(scores_h1, tau, orientation))))
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(fprs, tprs)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{method} ROC-like curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _detect_raw_threshold_bug(model: OursModel, alpha: float, failures: List[str], warnings: List[str]) -> str:
    threshold = float(model.tau)
    if abs(threshold - float(alpha)) <= 1e-12:
        failures.append(
            f"{getattr(model.method, 'name', type(model.method).__name__)} alpha={alpha}: "
            "model threshold equals target_fpr exactly; possible target_fpr-as-raw-threshold bug"
        )
        return "direct_alpha_suspected"
    source_train = inspect.getsource(train_ours_model)
    source_online = inspect.getsource(run_ours_policy)
    if "tau=alpha" in source_train.replace(" ", "") or "threshold=alpha" in source_train.replace(" ", ""):
        failures.append("Current train_ours_model source appears to assign alpha directly as threshold")
    if "model.tau" not in source_online:
        failures.append("Current run_ours_policy source does not use model.tau for decisions")
    if "_select_np_tau" not in source_train and ".fit(" not in source_train:
        warnings.append("Could not verify calibrated-threshold path statically from train_ours_model source")
    return "calibrated_not_direct"


def run_always_policy(stream: Sequence[Any], *, always_hit: bool, cache_size: int, eviction_policy: str) -> Dict[str, Any]:
    cache = ExactVectorCache(capacity=cache_size, eviction_policy=eviction_policy)
    judge = EquivalenceJudge(mode="cluster")
    records: List[Dict[str, Any]] = []
    for i, example in enumerate(stream):
        nearest, sim = cache.nearest(example.embedding)
        is_hit = bool(always_hit and nearest is not None)
        if is_hit:
            returned = nearest.example.gold_response
            llm_calls = 0
        else:
            returned = example.gold_response
            llm_calls = 1
            cache.add(example)
        correctness, would_correct, eval_calls = _evaluate_decision(
            judge, example, nearest, is_hit=is_hit, returned_response=returned
        )
        records.append(
            _base_record(
                dataset="debug_online_smoke",
                method="always_hit" if always_hit else "always_miss",
                seed=0,
                param="baseline",
                request_index=i,
                example=example,
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
    add_cumulative_fields(records)
    return summarize_run(records)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    failures: List[str] = []
    score_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    pairwise_path = Path(args.pairwise_data)
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Pairwise data not found: {pairwise_path}")

    arrays, npz_rows = _read_npz_metadata(pairwise_path)
    write_csv(output_dir / "npz_keys.csv", npz_rows, ["key", "shape", "dtype"])

    if args.feature_key not in arrays:
        raise SystemExit(f"Missing feature key {args.feature_key!r}. Available keys={sorted(arrays)}")
    if args.label_key not in arrays:
        raise SystemExit(f"Missing label key {args.label_key!r}. Available keys={sorted(arrays)}")

    X = _as_feature_matrix(arrays[args.feature_key], args.feature_key)
    y = _as_labels(arrays[args.label_key], args.label_key)
    if X.shape[0] != y.shape[0]:
        raise SystemExit(f"Feature/label row mismatch: {X.shape[0]} vs {y.shape[0]}")

    X_alt = None
    if args.alt_feature_key and args.alt_feature_key in arrays:
        X_alt = np.asarray(arrays[args.alt_feature_key], dtype=np.float32).reshape(-1, 1)
        if X_alt.shape[0] != X.shape[0]:
            raise SystemExit(f"Alt feature/label row mismatch: {X_alt.shape[0]} vs {y.shape[0]}")

    n_h1 = int(np.sum(y == 1))
    n_h0 = int(np.sum(y == 0))
    ratio_h1 = n_h1 / max(1, int(y.size))
    if ratio_h1 <= 0.05 or ratio_h1 >= 0.95:
        warnings.append(f"label distribution is extremely imbalanced: H1 ratio={ratio_h1:.6f}")

    embedding_key, stream_embedding_model = _embedding_choice(args.embedding_model)
    pairwise_model = None
    if "model" in arrays:
        raw_model = np.asarray(arrays["model"]).reshape(-1)
        pairwise_model = str(raw_model[0]) if raw_model.size else None
    if pairwise_model is None:
        msg = "Pairwise embedding model metadata is missing; pass --allow_missing_pairwise_model only for explicit debugging"
        if args.allow_missing_pairwise_model:
            warnings.append(msg)
        else:
            failures.append(msg)
    elif pairwise_model != stream_embedding_model:
        failures.append(
            f"embedding mismatch detected: stream embeddings={stream_embedding_model}, pairwise data={pairwise_model}"
        )

    raw_query_keys = [k for k in arrays if k in {"query_emb", "q_emb", "emb_q", "query_embedding"}]
    raw_anchor_keys = [k for k in arrays if k in {"anchor_emb", "c_emb", "emb_c", "candidate_embedding"}]
    feature_kind = "unknown"
    if raw_query_keys and raw_anchor_keys:
        q = np.asarray(arrays[raw_query_keys[0]], dtype=np.float32)
        c = np.asarray(arrays[raw_anchor_keys[0]], dtype=np.float32)
        if q.shape == c.shape and q.shape == X.shape:
            diff = float(np.max(np.abs((q[: min(10, len(q))] * c[: min(10, len(c))]) - X[: min(10, len(X))])))
            feature_kind = "hadamard_pair" if diff <= 1e-5 else "not_hadamard_pair"
            if diff > 1e-5:
                warnings.append(f"stored feature key {args.feature_key!r} does not match manual Hadamard on sampled raw pairs")
    else:
        if "hadamard" in pairwise_path.name.lower():
            feature_kind = "precomputed_hadamard_claimed_by_filename"
        warnings.append(
            f"Pairwise file has no raw query/anchor embedding keys; cannot manually prove {args.feature_key!r} is Hadamard"
        )

    split = make_splits(
        X,
        y,
        X_alt=X_alt,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_eval=args.n_eval,
        seed=args.seed,
    )

    dataset_name = args.stream_dataset
    if dataset_name == "auto":
        dataset_name = _infer_dataset_from_path(pairwise_path)
    stream = []
    if dataset_name != "none":
        stream, info = load_real_stream(
            SPECS[dataset_name],
            embedding_key=embedding_key,
            response_key=SPECS[dataset_name].default_response_key,
            seed=args.seed,
            limit=min(int(args.max_examples), 1000),
        )
        if pairwise_model is not None and info.embedding_key != embedding_key:
            failures.append(f"stream embedding key mismatch: loaded {info.embedding_key}, expected {embedding_key}")
    else:
        warnings.append("No stream dataset could be inferred; online smoke test skipped")

    config = {
        "args": vars(args),
        "pairwise_data": str(pairwise_path),
        "label_semantics": {
            "1": "H1 / reusable / correct cache reuse",
            "0": "H0 / non-reusable / incorrect cache reuse",
            "n_H1": n_h1,
            "n_H0": n_h0,
            "H1_ratio": ratio_h1,
            "H0_ratio": n_h0 / max(1, int(y.size)),
        },
        "npz_keys": npz_rows,
        "feature_kind": feature_kind,
        "feature_dim": int(X.shape[1]),
        "stream_embedding_model": stream_embedding_model,
        "pairwise_embedding_model": pairwise_model,
        "stream_dataset": dataset_name,
    }

    print(json.dumps(config["label_semantics"], indent=2, sort_keys=True), flush=True)
    print(f"feature_kind={feature_kind} feature_dim={X.shape[1]}", flush=True)
    print(f"stream_embedding_model={stream_embedding_model} pairwise_embedding_model={pairwise_model}", flush=True)

    for method_label in args.method:
        method_name = METHOD_MAP[method_label]
        last_eval_scores: Optional[Tuple[np.ndarray, np.ndarray, str]] = None
        previous_decisions: Optional[np.ndarray] = None
        for alpha in args.target_fprs:
            model = train_ours_model(
                pairwise_data=str(pairwise_path),
                method_name=method_name,
                feature_key=args.feature_key,
                label_key=args.label_key,
                alt_feature_key=args.alt_feature_key,
                n_train=args.n_train,
                n_calib=args.n_calib,
                seed=args.seed,
                alpha=float(alpha),
                pair_feature=args.pair_feature,
            )

            s0_cal = score_model(model, split.h0_calib, split.h0_calib_alt)
            s1_cal = score_model(model, split.h1_calib, split.h1_calib_alt)
            s0_eval = score_model(model, split.h0_eval, split.h0_eval_alt)
            s1_eval = score_model(model, split.h1_eval, split.h1_eval_alt)
            orientation, sep_warning = infer_orientation(s0_eval, s1_eval)
            if orientation == "ambiguous":
                failures.append(f"{method_label} alpha={alpha}: score orientation ambiguous: {sep_warning}")
                orientation = "higher"
            if sep_warning:
                warnings.append(f"{method_label} alpha={alpha}: {sep_warning}")

            model_threshold = float(model.tau)
            reference_threshold = select_threshold(s0_cal, float(alpha), orientation)
            threshold_source = _detect_raw_threshold_bug(model, float(alpha), failures, warnings)

            calib_h0_accept = float(np.mean(accept(s0_cal, model_threshold, orientation)))
            calib_h1_accept = float(np.mean(accept(s1_cal, model_threshold, orientation)))
            eval_h0_accept = float(np.mean(accept(s0_eval, model_threshold, orientation)))
            eval_h1_accept = float(np.mean(accept(s1_eval, model_threshold, orientation)))
            tolerance = max(0.005, 0.25 * float(alpha))
            if calib_h0_accept > float(alpha) + tolerance:
                failures.append(
                    f"{method_label} alpha={alpha}: calibration H0 accept rate {calib_h0_accept:.6f} "
                    f"exceeds alpha+tolerance {float(alpha) + tolerance:.6f}"
                )

            row_warning = sep_warning
            for split_name, label, scores in [
                ("calib", "H0", s0_cal),
                ("calib", "H1", s1_cal),
                ("eval", "H0", s0_eval),
                ("eval", "H1", s1_eval),
            ]:
                row = quantile_row(method_label, float(alpha), split_name, label, scores, orientation)
                row["separation_warning"] = row_warning
                score_rows.append(row)

            threshold_rows.append(
                {
                    "method": method_label,
                    "target_fpr": float(alpha),
                    "orientation": orientation,
                    "model_threshold": model_threshold,
                    "reference_threshold_full_calib": reference_threshold,
                    "threshold_source": threshold_source,
                    "calib_accept_rate_H0": calib_h0_accept,
                    "calib_accept_rate_H1": calib_h1_accept,
                    "eval_accept_rate_H0": eval_h0_accept,
                    "eval_accept_rate_H1": eval_h1_accept,
                    "tolerance": tolerance,
                }
            )

            _plot_score_hist(
                plots_dir / f"{method_label}_alpha_{alpha:g}_score_hist.png",
                method=method_label,
                alpha=float(alpha),
                scores_h0=s0_eval,
                scores_h1=s1_eval,
                threshold=model_threshold,
            )
            last_eval_scores = (s0_eval, s1_eval, orientation)

            if stream:
                judge = EquivalenceJudge(mode="cluster")
                records = run_ours_policy(
                    stream,
                    model=model,
                    method_label=method_label,
                    dataset=dataset_name,
                    seed=args.seed,
                    cache_size=args.cache_size,
                    eviction_policy=args.eviction_policy,
                    judge=judge,
                    embedding_model=stream_embedding_model,
                    stream_hash="debug",
                )
                summary = summarize_run(records)
                current_decisions = np.asarray([1 if r.get("decision") == "hit" else 0 for r in records], dtype=np.int8)
                warning_bits = []
                fpr = summary.get("false_positive_rate")
                hit_rate = summary.get("hit_rate")
                precision = summary.get("precision")
                fn = int(summary.get("FN") or 0)
                if fpr is not None and float(alpha) <= 0.05 and float(fpr) > 0.2:
                    warning_bits.append(f"FPR {float(fpr):.6f} > 0.2 for target_fpr {alpha}")
                if hit_rate is not None and precision is not None and float(hit_rate) > 0.95 and float(precision) < 0.5:
                    warning_bits.append("hit_rate > 0.95 and precision < 0.5")
                if fn == 0 and fpr is not None and float(fpr) > 0.5:
                    warning_bits.append("FN == 0 and FPR > 0.5")
                if previous_decisions is not None and previous_decisions.shape == current_decisions.shape:
                    diff = float(np.mean(previous_decisions != current_decisions))
                    if diff <= 0.005:
                        warning_bits.append(f"decisions nearly identical to previous target_fpr: changed_fraction={diff:.6f}")
                previous_decisions = current_decisions

                always_hit_summary = run_always_policy(
                    stream,
                    always_hit=True,
                    cache_size=args.cache_size,
                    eviction_policy=args.eviction_policy,
                )
                always_miss_summary = run_always_policy(
                    stream,
                    always_hit=False,
                    cache_size=args.cache_size,
                    eviction_policy=args.eviction_policy,
                )
                if hit_rate is not None:
                    ah_hit = float(always_hit_summary.get("hit_rate") or 0.0)
                    if abs(float(hit_rate) - ah_hit) <= 0.05:
                        warning_bits.append(
                            "ERROR: learned method behaves like always_hit; calibration or decision orientation is likely broken."
                        )

                if warning_bits:
                    failures.append(f"{method_label} alpha={alpha} online smoke: " + "; ".join(warning_bits))

                smoke_rows.append(
                    {
                        "method": method_label,
                        "target_fpr": float(alpha),
                        "n": summary.get("n"),
                        "TP": summary.get("TP"),
                        "FP": summary.get("FP"),
                        "TN": summary.get("TN"),
                        "FN": summary.get("FN"),
                        "error_rate_stream": summary.get("error_rate_stream"),
                        "hit_rate": summary.get("hit_rate"),
                        "precision": summary.get("precision"),
                        "false_positive_rate": summary.get("false_positive_rate"),
                        "true_positive_rate": summary.get("true_positive_rate"),
                        "llm_calls": summary.get("llm_calls"),
                        "judge_calls": summary.get("judge_calls"),
                        "warning": "; ".join(warning_bits),
                    }
                )
                smoke_rows.append(
                    {
                        "method": "always_hit",
                        "target_fpr": float(alpha),
                        "n": always_hit_summary.get("n"),
                        "TP": always_hit_summary.get("TP"),
                        "FP": always_hit_summary.get("FP"),
                        "TN": always_hit_summary.get("TN"),
                        "FN": always_hit_summary.get("FN"),
                        "error_rate_stream": always_hit_summary.get("error_rate_stream"),
                        "hit_rate": always_hit_summary.get("hit_rate"),
                        "precision": always_hit_summary.get("precision"),
                        "false_positive_rate": always_hit_summary.get("false_positive_rate"),
                        "true_positive_rate": always_hit_summary.get("true_positive_rate"),
                        "llm_calls": always_hit_summary.get("llm_calls"),
                        "judge_calls": always_hit_summary.get("judge_calls"),
                        "warning": "",
                    }
                )
                smoke_rows.append(
                    {
                        "method": "always_miss",
                        "target_fpr": float(alpha),
                        "n": always_miss_summary.get("n"),
                        "TP": always_miss_summary.get("TP"),
                        "FP": always_miss_summary.get("FP"),
                        "TN": always_miss_summary.get("TN"),
                        "FN": always_miss_summary.get("FN"),
                        "error_rate_stream": always_miss_summary.get("error_rate_stream"),
                        "hit_rate": always_miss_summary.get("hit_rate"),
                        "precision": always_miss_summary.get("precision"),
                        "false_positive_rate": always_miss_summary.get("false_positive_rate"),
                        "true_positive_rate": always_miss_summary.get("true_positive_rate"),
                        "llm_calls": always_miss_summary.get("llm_calls"),
                        "judge_calls": always_miss_summary.get("judge_calls"),
                        "warning": "",
                    }
                )

                for r in [x for x in records if x.get("method_score") not in {None, ""}][:20]:
                    score = float(r["method_score"])
                    expected_hit = bool(accept(np.asarray([score]), model_threshold, orientation)[0])
                    actual_decision = "hit" if r.get("decision") == "hit" else "miss"
                    expected_decision = "hit" if expected_hit else "miss"
                    if actual_decision != expected_decision:
                        failures.append(
                            f"{method_label} alpha={alpha}: online adapter decision disagreement for prompt_id={r.get('prompt_id')}"
                        )
                    debug_rows.append(
                        {
                            "method": method_label,
                            "target_fpr": float(alpha),
                            "prompt_id": r.get("prompt_id"),
                            "candidate_id": r.get("nearest_prompt_id"),
                            "label_or_correctness": r.get("correctness") if r.get("decision") == "hit" else r.get("nearest_would_be_correct"),
                            "score": score,
                            "threshold": model_threshold,
                            "orientation": orientation,
                            "decision": actual_decision,
                            "expected_decision_from_sanity_logic": expected_decision,
                        }
                    )

        if last_eval_scores is not None:
            s0_eval, s1_eval, orientation = last_eval_scores
            _plot_roc_like(method_label, s0_eval, s1_eval, orientation, plots_dir / f"{method_label}_roc_like.png")

    _plot_accept_rates(threshold_rows, plots_dir)

    write_csv(output_dir / "score_distribution_summary.csv", score_rows, SCORE_FIELDS)
    write_csv(output_dir / "threshold_calibration_summary.csv", threshold_rows, THRESHOLD_FIELDS)
    write_csv(output_dir / "online_smoke_test_summary.csv", smoke_rows, SMOKE_FIELDS)
    write_csv(output_dir / "debug_examples.csv", debug_rows, DEBUG_EXAMPLE_FIELDS)
    write_json(output_dir / "config.json", config)

    all_warnings = warnings + failures
    with (output_dir / "warnings.txt").open("w", encoding="utf-8") as f:
        for msg in all_warnings:
            f.write(str(msg) + "\n")

    if failures:
        print("DEBUG CALIBRATION FAILED", file=sys.stderr)
        for msg in failures:
            print(f"- {msg}", file=sys.stderr)
        raise SystemExit(1)

    print("DEBUG CALIBRATION PASSED", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
