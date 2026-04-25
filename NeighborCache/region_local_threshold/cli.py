"""CLI entry point: argument parsing and main experiment orchestration."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from np_bench.utils import make_run_dir, save_csv_rows, save_json
from np_bench.methods.base import OnlineBaseMethod

from .io_helpers import resolve_npz_path, load_npz, resolve_features
from .io_helpers import resolve_text_pairs
from .methods import build_methods, needs_weights
from .splits import (
    GlobalSplit,
    filter_global_split_by_score_range,
    filter_region_splits_by_score_range,
    filter_region_splits_by_score_range_detailed,
    split_indices_per_region,
    split_indices_per_region_detailed,
    split_global,
)
from .evaluation import fit_all_methods, evaluate_methods, evaluate_methods_global, aggregate_ranking, _select_tau, apply_threshold
from .display import print_trial_table, print_ranking
from .stopping_mechanism import OnlineStopper, StopConfig
from .ocats_baselines import run_ocats_baselines_for_split

NC_ROOT = ROOT / "NeighborCache"
OUT_BASE = NC_ROOT / "outputs" / "region_local_threshold"

REGION_KEY_ALIASES = {
}


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(denom, eps)).astype(np.float32, copy=False)


def _select_region_anchor_vectors(
    X_main: np.ndarray,
    region_id: np.ndarray,
    *,
    strategy: str,
    seed: int,
) -> Dict[int, np.ndarray]:
    """Select one L2-normalized anchor vector per region."""
    Xn = _l2_normalize_rows(X_main)
    region_id = np.asarray(region_id, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(seed)

    anchors_by_rid: Dict[int, np.ndarray] = {}
    for rid in np.unique(region_id):
        idx = np.flatnonzero(region_id == rid)
        if idx.size == 0:
            continue
        Xr = Xn[idx]

        if strategy == "random":
            anchor_local = int(rng.integers(0, idx.size))
        else:
            centroid = np.mean(Xr, axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            sims = Xr @ centroid
            anchor_local = int(np.argmax(sims))

        anchors_by_rid[int(rid)] = np.asarray(Xr[anchor_local], dtype=np.float32).copy()

    return anchors_by_rid


def _build_hadamard_features(
    X_main: np.ndarray,
    region_id: np.ndarray,
    *,
    strategy: str,
    seed: int,
    use_delta_vec: bool = False,
    use_abs_diff: bool = False,
    abs_diff_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build region-anchor features and cosine-to-anchor from feature vectors.

    Returns:
      features: (N, D | 2D) float32 depending on enabled parts
      cosine_to_anchor: (N, 1) float32
    """
    Xn = _l2_normalize_rows(X_main)
    region_id = np.asarray(region_id, dtype=np.int64).reshape(-1)
    N, D = Xn.shape

    anchors_by_rid = _select_region_anchor_vectors(
        X_main,
        region_id,
        strategy=strategy,
        seed=seed,
    )
    anchors = np.zeros((N, D), dtype=np.float32)
    for rid in np.unique(region_id):
        idx = np.flatnonzero(region_id == rid)
        if idx.size == 0:
            continue
        anchor = anchors_by_rid.get(int(rid))
        if anchor is None:
            continue
        anchors[idx] = anchor

    had = (Xn * anchors).astype(np.float32, copy=False)
    parts: List[np.ndarray] = []
    if not abs_diff_only:
        parts.append(had)
    if use_delta_vec:
        delta = (Xn - anchors).astype(np.float32, copy=False)
        parts.append(delta)
    elif use_abs_diff:
        abs_diff = np.abs(Xn - anchors).astype(np.float32, copy=False)
        parts.append(abs_diff)

    if not parts:
        raise ValueError(
            "No anchor-feature components selected. "
            "Enable --hadamard_preprocess and/or --use_abs_diff/--use_delta_vec."
        )
    X_pair = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    # Keep cosine-to-anchor defined from the Hadamard term only.
    cos = np.sum(had, axis=1, keepdims=True).astype(np.float32, copy=False)
    return X_pair, cos


def _kmeans_numpy(
    X: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int = 60,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = int(max(1, min(n_clusters, n)))
    rng = np.random.default_rng(seed)

    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if np.any(m):
                centers[j] = X[m].mean(axis=0)
            else:
                centers[j] = X[rng.integers(0, n)]
    return labels


def _build_tau_cluster_map(
    *,
    splits: List[Any],
    X_anchor_source: np.ndarray,
    region_id: np.ndarray,
    anchor_strategy: str,
    n_clusters: int,
    seed: int,
) -> Dict[int, int]:
    if not splits:
        return {}

    anchors_by_rid = _select_region_anchor_vectors(
        X_anchor_source,
        region_id,
        strategy=anchor_strategy,
        seed=seed,
    )

    rids = sorted({int(s.rid) for s in splits})
    if not rids:
        return {}

    anchor_rows: List[np.ndarray] = []
    used_rids: List[int] = []
    for rid in rids:
        a = anchors_by_rid.get(rid)
        if a is None:
            continue
        anchor_rows.append(np.asarray(a, dtype=np.float32))
        used_rids.append(int(rid))

    if not used_rids:
        return {}

    A = np.asarray(anchor_rows, dtype=np.float32)
    labels = _kmeans_numpy(A, n_clusters=n_clusters, seed=seed)
    return {rid: int(labels[i]) for i, rid in enumerate(used_rids)}


def _build_semantic_buckets_from_regions(
    X_main: np.ndarray,
    source_region_id: np.ndarray,
    *,
    n_buckets: int,
    seed: int,
    anchor_strategy: str,
) -> tuple[np.ndarray, Dict[int, int]]:
    """Build coarse semantic buckets by clustering per-region anchor vectors.

    Returns per-sample bucket ids and the source-region to bucket mapping.
    """
    X = np.asarray(X_main, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] <= 1:
        raise ValueError(
            "On-the-fly sem_bucket generation requires embedding-like features "
            f"with shape (N,D), D>1; got {X.shape}"
        )

    src_rid = np.asarray(source_region_id, dtype=np.int64).reshape(-1)
    if src_rid.shape[0] != X.shape[0]:
        raise ValueError(
            f"sem_bucket source-region length mismatch: {src_rid.shape[0]} vs {X.shape[0]}"
        )

    anchors_by_rid = _select_region_anchor_vectors(
        X,
        src_rid,
        strategy=anchor_strategy,
        seed=seed,
    )
    src_regions = sorted(anchors_by_rid.keys())
    if not src_regions:
        return np.zeros(X.shape[0], dtype=np.int64), {}

    A = np.asarray([anchors_by_rid[r] for r in src_regions], dtype=np.float32)
    labels = _kmeans_numpy(A, n_clusters=max(1, int(n_buckets)), seed=seed)
    rid_to_bucket = {int(r): int(labels[i]) for i, r in enumerate(src_regions)}

    bucket_id = np.fromiter(
        (rid_to_bucket[int(r)] for r in src_rid.tolist()),
        dtype=np.int64,
        count=src_rid.shape[0],
    )
    return bucket_id, rid_to_bucket


def _concat_indices(parts: List[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(p, dtype=np.int64) for p in parts if p is not None and p.size > 0]
    if not valid:
        return np.array([], dtype=np.int64)
    return np.concatenate(valid).astype(np.int64, copy=False)


def _build_global_split_from_local_regions(
    splits: List[Any],
    tested_region_ids: List[int],
) -> GlobalSplit:
    tested = set(int(r) for r in tested_region_ids)
    chosen = [s for s in splits if int(s.rid) in tested]
    return GlobalSplit(
        H0_train=_concat_indices([s.H0_train for s in chosen]),
        H1_train=_concat_indices([s.H1_train for s in chosen]),
        H0_calib=_concat_indices([s.H0_calib for s in chosen]),
        H1_calib=_concat_indices([s.H1_calib for s in chosen]),
        H0_eval=_concat_indices([s.H0_eval for s in chosen]),
        H1_eval=_concat_indices([s.H1_eval for s in chosen]),
    )


def _print_matched_comparison(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    print("\n=== Matched Comparison: Local vs Global on Local-Tested Regions ===")
    print("Method                   | L-TPR  | L-FPR  | G-TPR  | G-FPR  | L-MacroTPR | G-MacroTPR | Regions")
    print("-" * 104)
    for r in rows:
        print(
            f"{str(r['method']):<24} | "
            f"{float(r['local_micro_tpr']):.4f} | {float(r['local_micro_fpr']):.4f} | "
            f"{float(r['global_micro_tpr']):.4f} | {float(r['global_micro_fpr']):.4f} | "
            f"{float(r['local_macro_tpr']):.4f}     | {float(r['global_macro_tpr']):.4f}     | "
            f"{int(r['tested_regions'])}"
        )


def _sample_monitor_from_global_eval(
    gs: GlobalSplit,
    *,
    n_monitor_h0: int,
    n_monitor_h1: int,
    seed: int,
) -> tuple[GlobalSplit, np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    h0_eval = np.asarray(gs.H0_eval, dtype=np.int64).copy()
    h1_eval = np.asarray(gs.H1_eval, dtype=np.int64).copy()
    rng.shuffle(h0_eval)
    rng.shuffle(h1_eval)

    n0 = int(min(max(n_monitor_h0, 0), h0_eval.size))
    n1 = int(min(max(n_monitor_h1, 0), h1_eval.size))

    h0_monitor = h0_eval[:n0]
    h1_monitor = h1_eval[:n1]
    h0_eval_rem = h0_eval[n0:]
    h1_eval_rem = h1_eval[n1:]

    gs_out = GlobalSplit(
        H0_train=gs.H0_train,
        H1_train=gs.H1_train,
        H0_calib=gs.H0_calib,
        H1_calib=gs.H1_calib,
        H0_eval=h0_eval_rem,
        H1_eval=h1_eval_rem,
    )
    stats = {
        "monitor_h0": int(h0_monitor.size),
        "monitor_h1": int(h1_monitor.size),
        "eval_h0_remaining": int(h0_eval_rem.size),
        "eval_h1_remaining": int(h1_eval_rem.size),
    }
    return gs_out, h0_monitor, h1_monitor, stats


def _sample_monitor_from_region_splits(
    splits: List[Any],
    *,
    n_monitor_h0: int,
    n_monitor_h1: int,
    min_h0_eval: int,
    min_h1_eval: int,
    seed: int,
) -> tuple[List[Any], np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)

    h0_candidates: List[int] = []
    h1_candidates: List[int] = []
    for s in splits:
        h0_ev = np.asarray(s.H0_eval, dtype=np.int64)
        h1_ev = np.asarray(s.H1_eval, dtype=np.int64)
        h0_excess = max(0, int(h0_ev.size) - int(min_h0_eval))
        h1_excess = max(0, int(h1_ev.size) - int(min_h1_eval))
        if h0_excess > 0:
            idx0 = h0_ev.copy()
            rng.shuffle(idx0)
            h0_candidates.extend(idx0[:h0_excess].tolist())
        if h1_excess > 0:
            idx1 = h1_ev.copy()
            rng.shuffle(idx1)
            h1_candidates.extend(idx1[:h1_excess].tolist())

    rng.shuffle(h0_candidates)
    rng.shuffle(h1_candidates)

    n0 = int(min(max(n_monitor_h0, 0), len(h0_candidates)))
    n1 = int(min(max(n_monitor_h1, 0), len(h1_candidates)))

    h0_monitor = np.asarray(h0_candidates[:n0], dtype=np.int64)
    h1_monitor = np.asarray(h1_candidates[:n1], dtype=np.int64)

    h0_monitor_set = set(int(i) for i in h0_monitor.tolist())
    h1_monitor_set = set(int(i) for i in h1_monitor.tolist())

    splits_out: List[Any] = []
    for s in splits:
        h0_ev = np.asarray(s.H0_eval, dtype=np.int64)
        h1_ev = np.asarray(s.H1_eval, dtype=np.int64)
        h0_keep = h0_ev[~np.isin(h0_ev, list(h0_monitor_set))]
        h1_keep = h1_ev[~np.isin(h1_ev, list(h1_monitor_set))]
        splits_out.append(
            type(s)(
                rid=int(s.rid),
                H0_train=s.H0_train,
                H1_train=s.H1_train,
                H0_calib=s.H0_calib,
                H1_calib=s.H1_calib,
                H0_eval=h0_keep,
                H1_eval=h1_keep,
            )
        )

    stats = {
        "monitor_h0": int(h0_monitor.size),
        "monitor_h1": int(h1_monitor.size),
        "eval_h0_remaining": int(sum(int(s.H0_eval.size) for s in splits_out)),
        "eval_h1_remaining": int(sum(int(s.H1_eval.size) for s in splits_out)),
    }
    return splits_out, h0_monitor, h1_monitor, stats


def _run_online_stopping(
    X_main: np.ndarray,
    y: np.ndarray,
    *,
    h0_train_idx: np.ndarray,
    h1_train_idx: np.ndarray,
    h0_calib_idx: np.ndarray,
    h0_monitor_idx: np.ndarray,
    h1_monitor_idx: np.ndarray,
    alpha: float,
    tie_mode: str,
    tau_guardrail: str,
    tau_guardrail_delta: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Optional[OnlineBaseMethod], List[Dict[str, Any]], Dict[str, Any]]:
    if h0_train_idx.size == 0 or h1_train_idx.size == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_train",
        }
    if h0_calib_idx.size == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_calib_h0",
        }
    if h0_monitor_idx.size == 0 or h1_monitor_idx.size == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_monitor",
        }

    rng = np.random.default_rng(seed)
    h0_idx = np.asarray(h0_train_idx, dtype=np.int64).copy()
    h1_idx = np.asarray(h1_train_idx, dtype=np.int64).copy()
    rng.shuffle(h0_idx)
    rng.shuffle(h1_idx)

    n_init_h0 = int(min(max(args.online_init_h0, 0), h0_idx.size))
    n_init_h1 = int(min(max(args.online_init_h1, 0), h1_idx.size))
    if n_init_h0 == 0 or n_init_h1 == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_init",
        }

    init_h0_idx = h0_idx[:n_init_h0]
    init_h1_idx = h1_idx[:n_init_h1]
    rem_h0_idx = h0_idx[n_init_h0:]
    rem_h1_idx = h1_idx[n_init_h1:]

    online = OnlineBaseMethod(
        mem_cap_H0=int(args.online_mem_cap),
        mem_cap_H1=int(args.online_mem_cap),
        update_mode=str(args.online_update_mode),
        update_every=1,
        hill_lr=float(args.online_hill_lr),
        seed=seed,
    )
    online.initialize(X_main[init_h0_idx], X_main[init_h1_idx])

    check_every = max(1, int(args.stop_check_every))
    stopper = OnlineStopper(
        StopConfig(
            stop_check_every=check_every,
            stop_window=max(1, int(args.stop_window)),
            stop_patience=max(1, int(args.stop_patience)),
            stop_eps_tpr=float(args.stop_eps_tpr),
            stop_eps_fpr=float(args.stop_eps_fpr),
            stop_eps_tau=float(args.stop_eps_tau),
            stop_fpr_margin=float(args.stop_fpr_margin),
            alpha=float(alpha),
        )
    )

    rem_idx = np.concatenate([rem_h0_idx, rem_h1_idx]) if rem_h0_idx.size + rem_h1_idx.size > 0 else np.array([], dtype=np.int64)
    rng.shuffle(rem_idx)

    history_rows: List[Dict[str, Any]] = []
    total_updates = 0
    stopped = False
    stop_reason = "stream_exhausted"
    last_tau = float("inf")
    samples_streamed = 0
    samples_init = int(init_h0_idx.size + init_h1_idx.size)
    total_stream_available = int(rem_idx.size)

    def _checkpoint(checkpoint_idx: int) -> bool:
        nonlocal last_tau
        sc0_cal = np.asarray(online.score(X_main[h0_calib_idx]), dtype=np.float32).reshape(-1)
        tau = _select_tau(
            sc0_cal,
            alpha=float(alpha),
            tie_mode=tie_mode,
            guardrail=tau_guardrail,
            guardrail_delta=tau_guardrail_delta,
        )
        if not np.isfinite(tau):
            tau = float(np.quantile(sc0_cal, 1.0 - float(alpha)))
        last_tau = float(tau)

        sc0_mon = np.asarray(online.score(X_main[h0_monitor_idx]), dtype=np.float32).reshape(-1)
        sc1_mon = np.asarray(online.score(X_main[h1_monitor_idx]), dtype=np.float32).reshape(-1)
        p0 = apply_threshold(sc0_mon, tau, tie_mode)
        p1 = apply_threshold(sc1_mon, tau, tie_mode)
        fpr_monitor = float(np.mean(p0 == 1))
        tpr_monitor = float(np.mean(p1 == 1))

        entry = stopper.update(checkpoint_idx, tpr_monitor, fpr_monitor, float(tau))
        history_rows.append(
            {
                "checkpoint": int(entry.checkpoint),
                "tpr_monitor": float(entry.tpr_monitor),
                "fpr_monitor": float(entry.fpr_monitor),
                "tau": float(entry.tau),
                "slope_tpr": float(entry.slope_tpr),
                "slope_fpr": float(entry.slope_fpr),
                "slope_tau": float(entry.slope_tau),
                "condition_passed": bool(entry.condition_passed),
                "stop_streak": int(entry.stop_streak),
                "should_stop": bool(entry.should_stop),
            }
        )
        return bool(entry.should_stop)

    if rem_idx.size == 0:
        stopped = _checkpoint(0)
        stop_reason = "no_stream_data"
    else:
        batch_size = int(max(1, args.online_batch_size))
        for start in range(0, rem_idx.size, batch_size):
            end = min(start + batch_size, rem_idx.size)
            idx = rem_idx[start:end]
            online.update(X_main[idx], y[idx])
            total_updates += 1
            samples_streamed += int(idx.size)
            if total_updates % check_every == 0 or end == rem_idx.size:
                if _checkpoint(total_updates):
                    stopped = True
                    stop_reason = "stability_reached"
                    break

    summary = {
        "stopped": bool(stopped),
        "reason": str(stop_reason),
        "updates": int(total_updates),
        "final_tau": float(last_tau),
        "history_len": int(len(history_rows)),
        "samples_init": int(samples_init),
        "samples_streamed": int(samples_streamed),
        "samples_total_used": int(samples_init + samples_streamed),
        "samples_stream_available": int(total_stream_available),
    }
    return online, history_rows, summary


def _parse_float_csv(raw: Optional[str]) -> List[float]:
    if raw is None:
        return []
    txt = str(raw).strip()
    if not txt:
        return []

    out: List[float] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _resolve_ocats_methods(method_flag: str, ocats_methods_raw: str) -> List[str]:
    if method_flag in {"ocats_knn", "ocats_mlp"}:
        return [method_flag]

    parsed = [m.strip() for m in str(ocats_methods_raw).split(",") if m.strip()]
    allowed = {"ocats_knn", "ocats_mlp"}
    out = [m for m in parsed if m in allowed]
    if not out:
        out = ["ocats_knn", "ocats_mlp"]
    return out


def _run_ocats_for_split(
    *,
    X_main: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    trial: int,
    seed: int,
    region_key: str,
    tau_mode: str,
    comparison_scope: str,
    args: argparse.Namespace,
    selected_methods: List[str],
    lambda_values: List[float],
) -> Dict[str, List[Dict[str, Any]]]:
    train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    calib_idx = np.asarray(calib_idx, dtype=np.int64).reshape(-1)
    eval_idx = np.asarray(eval_idx, dtype=np.int64).reshape(-1)

    if train_idx.size == 0 or eval_idx.size == 0:
        return {
            "trial_rows": [],
            "tuning_rows": [],
            "curve_rows": [],
        }

    e_grid = _parse_float_csv(args.e_thresh_grid)
    d_grid = _parse_float_csv(args.d_thresh_grid)

    out = run_ocats_baselines_for_split(
        X=X_main,
        y=y,
        train_idx=train_idx,
        calib_idx=calib_idx,
        eval_idx=eval_idx,
        seed=seed,
        methods=selected_methods,
        lambdas=lambda_values,
        tune_ocats=bool(args.tune_ocats),
        cache_k=int(args.cache_k),
        e_thresh=float(args.e_thresh),
        d_thresh=float(args.d_thresh),
        e_thresh_grid=e_grid,
        d_thresh_grid=d_grid,
        knn_weight_power=float(args.knn_weight_power),
        mlp_hidden_dim=int(args.mlp_hidden_dim),
        mlp_dropout=float(args.mlp_dropout),
        mlp_lr=float(args.mlp_lr),
        mlp_epochs=int(args.mlp_epochs),
        mlp_batch_size=int(args.mlp_batch_size),
        mlp_weight_decay=float(args.mlp_weight_decay),
        online_retrain_interval=int(args.online_retrain_interval),
        online_retrain_last_p=int(args.online_retrain_last_p),
        alpha=float(args.alpha),
        record_curve=bool(args.ocats_record_curve),
        progress_every=int(args.ocats_progress_every),
    )

    trial_rows: List[Dict[str, Any]] = []
    for r in out.get("trial_rows", []):
        row = dict(r)
        row.update(
            {
                "trial": int(trial),
                "seed": int(seed),
                "region_key": str(region_key),
                "tau_mode": str(tau_mode),
                "comparison_scope": str(comparison_scope),
            }
        )
        trial_rows.append(row)

    tuning_rows: List[Dict[str, Any]] = []
    for r in out.get("tuning_rows", []):
        row = dict(r)
        row.update(
            {
                "trial": int(trial),
                "seed": int(seed),
                "region_key": str(region_key),
                "tau_mode": str(tau_mode),
                "comparison_scope": str(comparison_scope),
            }
        )
        tuning_rows.append(row)

    curve_rows: List[Dict[str, Any]] = []
    for r in out.get("curve_rows", []):
        row = dict(r)
        row.update(
            {
                "trial": int(trial),
                "seed": int(seed),
                "region_key": str(region_key),
                "tau_mode": str(tau_mode),
                "comparison_scope": str(comparison_scope),
            }
        )
        curve_rows.append(row)

    return {
        "trial_rows": trial_rows,
        "tuning_rows": tuning_rows,
        "curve_rows": curve_rows,
    }


def _build_ocats_comparison_rows(
    ocats_rows: List[Dict[str, Any]],
    *,
    shared_regions: int,
    dropped_regions_for_comparability: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    """Map OCATS trial rows into the main trial-summary schema.

    For each (trial, seed, method, region_key, tau_mode, comparison_scope), keep
    the alpha-feasible row (fpr <= alpha) with the best discounted score.
    If none is feasible, fall back to the lowest-FPR row.
    """
    grouped_by_key: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in ocats_rows:
        key = (
            int(r.get("trial", -1)),
            int(r.get("seed", -1)),
            str(r.get("method", "ocats")),
            str(r.get("region_key", "unknown")),
            str(r.get("tau_mode", "unknown")),
            str(r.get("comparison_scope", "unknown")),
        )
        grouped_by_key.setdefault(key, []).append(r)

    best_by_key: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    alpha_eps = float(alpha) + 1e-12
    for key, rows in grouped_by_key.items():
        feasible = [
            r for r in rows
            if np.isfinite(float(r.get("fpr", float("inf"))))
            and float(r.get("fpr", float("inf"))) <= alpha_eps
        ]
        if feasible:
            best = max(
                feasible,
                key=lambda r: (
                    float(r.get("discounted_score", float("-inf"))),
                    -int(r.get("calls", 0)),
                ),
            )
        else:
            best = max(
                rows,
                key=lambda r: (
                    -float(r.get("fpr", float("inf"))),
                    float(r.get("discounted_score", float("-inf"))),
                    -int(r.get("calls", 0)),
                ),
            )
        best_by_key[key] = best

    out: List[Dict[str, Any]] = []
    for _, r in sorted(best_by_key.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][5])):
        method = str(r.get("method", "ocats"))
        scope = str(r.get("comparison_scope", "unknown"))
        method_label = f"{method}[OCaTS:{scope}]"
        out.append(
            {
                "trial": int(r.get("trial", -1)),
                "seed": int(r.get("seed", -1)),
                "method": method_label,
                "region_key": str(r.get("region_key", "unknown")),
                "tau_mode": str(r.get("tau_mode", "unknown")),
                "tau": float("nan"),
                "tau_mean": float("nan"),
                "micro_tpr": float(r.get("tpr", float("nan"))),
                "micro_fpr": float(r.get("fpr", float("nan"))),
                "train_tpr": float("nan"),
                "train_fpr": float("nan"),
                "macro_tpr": float(r.get("tpr", float("nan"))),
                "macro_fpr": float(r.get("fpr", float("nan"))),
                "ok_regions": int(max(1, shared_regions)),
                "shared_regions": int(max(1, shared_regions)),
                "dropped_regions_for_comparability": int(max(0, dropped_regions_for_comparability)),
                "input_space": "embedding",
                "time_ms": float("nan"),
            }
        )
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Region-local threshold benchmark with train/calib protocol."
    )
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--region_key", type=str, required=True)
    ap.add_argument(
        "--sem_bucket_k",
        type=int,
        default=64,
        help="When --region_key=sem_bucket and sem_bucket is absent in NPZ, build semantic buckets on-the-fly with this cluster count.",
    )
    ap.add_argument(
        "--sem_bucket_source_key",
        type=str,
        default="global_cluster",
        help="Source region-id key used to build on-the-fly semantic buckets.",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tie_mode", type=str, default="ge", choices=["ge", "gt"])
    ap.add_argument(
        "--tau_mode",
        type=str,
        default="local",
        choices=["local", "global", "shrink_local", "cluster_local"],
    )
    ap.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "ocats_knn", "ocats_mlp"],
        help="Optional method selector for OCATS baselines. Default keeps existing behavior.",
    )
    ap.add_argument(
        "--local_fit_mode",
        type=str,
        default="pooled",
        choices=["pooled", "per_region"],
        help="Local-only control: pooled fits once across regions, per_region refits each method per region.",
    )
    ap.add_argument(
        "--tau_cluster_k",
        type=int,
        default=8,
        help="Number of anchor clusters for tau_mode=cluster_local.",
    )
    ap.add_argument("--tau_shrink", action="store_true", default=False)
    ap.add_argument("--tau_shrink_m", type=float, default=500.0)
    ap.add_argument(
        "--shrink_k",
        type=float,
        default=200.0,
        help="Shrink strength for tau_mode=shrink_local; lambda_global = k/(k+n_calib_h0_region).",
    )
    ap.add_argument(
        "--tau_guardrail",
        type=str,
        default="none",
        choices=["none", "clopper_pearson", "wilson", "beta_ucb"],
    )
    ap.add_argument("--tau_guardrail_delta", type=float, default=0.01)
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_calib", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--min_h0_eval", type=int, default=20)
    ap.add_argument("--min_h1_eval", type=int, default=20)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument(
        "--normalize_data",
        action="store_true",
        default=False,
        help="L2-normalize each row of main features before splitting/training.",
    )
    ap.add_argument(
        "--hadamard_preprocess",
        action="store_true",
        default=False,
        help="Replace X_main with Hadamard(x, anchor(region)) before splitting/training.",
    )
    ap.add_argument(
        "--use_abs_diff",
        action="store_true",
        default=False,
        help=(
            "Use abs(x-anchor) residual features. "
            "With --hadamard_preprocess, concatenate abs(x-anchor) to Hadamard features; "
            "without --hadamard_preprocess, abs(x-anchor) becomes the only X_main feature."
        ),
    )
    ap.add_argument(
        "--use_delta_vec",
        action="store_true",
        default=False,
        help="When used with --hadamard_preprocess, concatenate signed residual (x-anchor) with Hadamard features.",
    )
    ap.add_argument(
        "--hadamard_anchor_strategy",
        type=str,
        default="centroid_nearest",
        choices=["centroid_nearest", "random"],
        help="Anchor selection strategy per region for --hadamard_preprocess.",
    )
    ap.add_argument(
        "--filter_policy",
        type=str,
        default="none",
        choices=["none", "ambiguous_only"],
        help="Shared split filter policy applied equally to train/calib/eval for all methods.",
    )
    ap.add_argument("--ambiguous_cos_min", type=float, default=0.7)
    ap.add_argument("--ambiguous_cos_max", type=float, default=0.9)

    # OCaTS-style teacher-student comparison baselines (disabled by default)
    ap.add_argument("--enable_ocats_baselines", action="store_true", default=False)
    ap.add_argument(
        "--ocats_methods",
        type=str,
        default="ocats_knn,ocats_mlp",
        help="Comma-separated OCATS methods to run when enabled.",
    )
    ap.add_argument("--cache_k", type=int, default=8)
    ap.add_argument("--e_thresh", type=float, default=0.25)
    ap.add_argument("--d_thresh", type=float, default=0.5)
    ap.add_argument("--knn_weight_power", type=float, default=2.0)
    ap.add_argument("--tune_ocats", action="store_true", default=False)
    ap.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.01,0.05",
        help="Comma-separated lambda values for discounted objective: acc - lambda*calls/N.",
    )
    ap.add_argument(
        "--e_thresh_grid",
        type=str,
        default="",
        help="Optional comma-separated entropy grid for OCATS tuning.",
    )
    ap.add_argument(
        "--d_thresh_grid",
        type=str,
        default="",
        help="Optional comma-separated distance grid for OCATS tuning.",
    )
    ap.add_argument("--mlp_hidden_dim", type=int, default=64)
    ap.add_argument("--mlp_dropout", type=float, default=0.1)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_epochs", type=int, default=40)
    ap.add_argument("--mlp_batch_size", type=int, default=128)
    ap.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    ap.add_argument(
        "--online_retrain_interval",
        type=int,
        default=0,
        help="For ocats_mlp: retrain every N teacher calls (0 disables online retraining).",
    )
    ap.add_argument(
        "--online_retrain_last_p",
        type=int,
        default=128,
        help="For ocats_mlp retraining: number of latest cache additions to use.",
    )
    ap.add_argument("--ocats_record_curve", action="store_true", default=False)
    ap.add_argument(
        "--include_ocats_in_comparison",
        action="store_true",
        default=False,
        help="Include OCATS rows in trial_summary/ranking as additional comparison methods.",
    )
    ap.add_argument(
        "--ocats_progress_every",
        type=int,
        default=1000,
        help="Print OCATS stream progress every N samples (0 disables).",
    )

    # StabilizedWhitenedCosine parameters
    ap.add_argument("--swc_k", type=int, default=64,
                    help="PCA dimensionality for StabilizedWhitenedCosine")
    ap.add_argument("--swc_shrinkage", type=float, default=0.1,
                    help="Covariance shrinkage coefficient (used when sklearn unavailable)")
    ap.add_argument("--swc_min_samples", type=int, default=200,
                    help="Minimum samples to attempt whitening per region")
    ap.add_argument("--swc_eps", type=float, default=1e-6,
                    help="Eigenvalue floor for numerical stability")
    ap.add_argument("--swc_fallback", type=lambda v: v.lower() in ('true', '1', 'yes'),
                    default=True,
                    help="Fall back to Cosine when whitening is unstable")
    ap.add_argument("--swc_verbose", action="store_true", default=False,
                    help="Print SWC diagnostics (k_eff, eigenvalues, fallback)")
    ap.add_argument("--swc_mode", type=str, default="global", choices=["global", "region", "cluster"])
    ap.add_argument("--swc_cluster_n_clusters", type=int, default=64)

    # Cosine score local calibration head
    ap.add_argument("--cos_affine_calib", action="store_true", default=False)
    ap.add_argument(
        "--precomputed_cosine",
        action="store_true",
        default=False,
        help="Add explicit scalar-score baseline using precomputed cosine_to_anchor.",
    )
    ap.add_argument(
        "--cos_affine_grouping",
        type=str,
        default="region",
        choices=["region", "cluster"],
    )
    ap.add_argument("--cos_affine_n_clusters", type=int, default=64)
    ap.add_argument(
        "--regional_weighted_ensemble",
        action="store_true",
        default=False,
        help="Add shrunk region-conditional weighted ensemble as a separate method.",
    )
    ap.add_argument("--rwe_k_shrink", type=float, default=200.0)
    ap.add_argument("--rwe_min_region_h0", type=int, default=30)
    ap.add_argument("--rwe_min_region_h1", type=int, default=30)

    # Optional cross-encoder reranker baseline
    ap.add_argument(
        "--enable_bge_reranker",
        action="store_true",
        default=False,
        help="Enable BGE cross-encoder reranker baseline (requires text pairs).",
    )
    ap.add_argument(
        "--bge_model_name",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="HuggingFace model id for BGE reranker.",
    )
    ap.add_argument("--bge_batch_size", type=int, default=32)
    ap.add_argument("--bge_max_length", type=int, default=512)
    ap.add_argument(
        "--bge_backend",
        type=str,
        default="auto",
        choices=["auto", "cross", "bi"],
        help="Backend mode for BGE method: cross-encoder, bi-encoder, or auto.",
    )
    ap.add_argument(
        "--bge_normalize_scores",
        action="store_true",
        default=False,
        help="Apply sigmoid to reranker logits before NP thresholding.",
    )
    ap.add_argument(
        "--text_pair_keys",
        type=str,
        default=None,
        help="Comma-separated NPZ keys for text pairs, e.g. query_text,anchor_text.",
    )
    ap.add_argument(
        "--text_source_pkl",
        type=str,
        default=None,
        help="Optional PKL used to map qid/anchor_qid -> text when NPZ has no text fields.",
    )

    # Online stopping controls
    ap.add_argument("--enable_online_stopping", action="store_true", default=False)
    ap.add_argument("--stop_check_every", type=int, default=5)
    ap.add_argument("--stop_window", type=int, default=5)
    ap.add_argument("--stop_patience", type=int, default=3)
    ap.add_argument("--stop_eps_tpr", type=float, default=1e-3)
    ap.add_argument("--stop_eps_fpr", type=float, default=1e-3)
    ap.add_argument("--stop_eps_tau", type=float, default=1e-4)
    ap.add_argument("--stop_fpr_margin", type=float, default=0.005)
    ap.add_argument("--n_monitor_h0", type=int, default=200)
    ap.add_argument("--n_monitor_h1", type=int, default=200)

    # Online update controls (used only when stopping is enabled)
    ap.add_argument("--online_batch_size", type=int, default=64)
    ap.add_argument("--online_mem_cap", type=int, default=2000)
    ap.add_argument("--online_update_mode", type=str, default="refit", choices=["refit", "hill_climb"])
    ap.add_argument("--online_hill_lr", type=float, default=0.1)
    ap.add_argument("--online_init_h0", type=int, default=50)
    ap.add_argument("--online_init_h1", type=int, default=50)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.local_fit_mode == "per_region":
        if args.tau_mode != "local":
            raise ValueError("--local_fit_mode per_region currently requires --tau_mode local")
        if args.tau_shrink:
            raise ValueError("--local_fit_mode per_region currently does not support --tau_shrink")

    selected_ocats_methods = _resolve_ocats_methods(args.method, args.ocats_methods)
    run_ocats_baselines = bool(
        args.enable_ocats_baselines or args.method in {"ocats_knn", "ocats_mlp"}
    )
    ocats_only_mode = args.method in {"ocats_knn", "ocats_mlp"}
    include_ocats_in_comparison = bool(args.include_ocats_in_comparison or ocats_only_mode)
    lambda_values = _parse_float_csv(args.lambdas)
    if run_ocats_baselines and not lambda_values:
        raise ValueError("--lambdas must provide at least one value when OCATS baselines are enabled")

    npz_path = resolve_npz_path(args.data)
    ds = load_npz(npz_path)

    if "label" not in ds:
        raise ValueError(f"{npz_path} missing required arrays: ['label']; have={sorted(ds.keys())}")
    y = ds["label"].astype(np.int32, copy=False)

    feat_key, X_main, X_cos = resolve_features(ds)
    X_anchor_source = np.asarray(X_main, dtype=np.float32)

    region_key = args.region_key
    sem_bucket_source_key_used: Optional[str] = None

    if region_key == "sem_bucket" and region_key not in ds:
        src_key = str(args.sem_bucket_source_key)
        if src_key not in ds:
            raise ValueError(
                "On-the-fly sem_bucket generation requires a valid source region key; "
                f"missing '{src_key}' in {npz_path}. Available keys={sorted(ds.keys())}"
            )
        source_region_id = ds[src_key].astype(np.int64, copy=False)
        region_id, rid_to_bucket = _build_semantic_buckets_from_regions(
            X_anchor_source,
            source_region_id,
            n_buckets=max(1, int(args.sem_bucket_k)),
            seed=int(args.seed),
            anchor_strategy=args.hadamard_anchor_strategy,
        )
        sem_bucket_source_key_used = src_key
        region_key = "__computed_sem_bucket"
        print(
            "[INFO] Built sem_bucket on-the-fly "
            f"from source_key='{src_key}' source_regions={len(rid_to_bucket)} "
            f"target_buckets={len(np.unique(region_id))} requested_k={int(args.sem_bucket_k)}"
        )
    elif region_key not in ds:
        alias_key = REGION_KEY_ALIASES.get(region_key)
        if alias_key in ds:
            print(
                f"[INFO] region_key='{region_key}' not found; using alias key='{alias_key}'"
            )
            region_key = alias_key

    if region_key in ds:
        region_id = ds[region_key].astype(np.int64, copy=False)
    elif region_key == "__computed_sem_bucket":
        pass
    elif args.tau_mode == "global":
        # Global tau mode does not require region partitioning for splitting/calibration;
        # synthesize a single region so datasets without region arrays can still run.
        region_key_req = args.region_key
        region_key = "__synthetic_global_region"
        region_id = np.zeros(y.shape[0], dtype=np.int64)
        print(
            f"[WARN] region_key='{region_key_req}' is unavailable in {npz_path}; "
            "using a synthetic single-region id for tau_mode='global'"
        )
    else:
        raise ValueError(
            f"{npz_path} missing required region array: ['{args.region_key}']; "
            f"have={sorted(ds.keys())}"
        )

    text_pair_keys = None
    if args.text_pair_keys is not None:
        parts = [p.strip() for p in args.text_pair_keys.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--text_pair_keys must contain exactly 2 comma-separated keys")
        text_pair_keys = (parts[0], parts[1])

    text_key, X_text = resolve_text_pairs(
        ds,
        text_pair_keys=text_pair_keys,
        text_source_pkl=args.text_source_pkl,
        region_id=region_id,
        anchor_features=X_main,
        anchor_strategy=args.hadamard_anchor_strategy,
        seed=args.seed,
    )

    abs_diff_only = bool(args.use_abs_diff and not args.hadamard_preprocess)
    if abs_diff_only:
        print(
            "[INFO] --use_abs_diff enabled without --hadamard_preprocess: "
            "using abs(x-anchor) as the only X_main feature"
        )
    if args.use_delta_vec and not args.hadamard_preprocess:
        print("[WARN] --use_delta_vec has no effect unless --hadamard_preprocess is enabled")
    if args.use_delta_vec and args.use_abs_diff:
        raise ValueError(
            "--use_delta_vec and --use_abs_diff are mutually exclusive experimental residual options"
        )

    if args.hadamard_preprocess or abs_diff_only:
        if X_main.ndim != 2 or X_main.shape[1] <= 1:
            raise ValueError(
                f"--hadamard_preprocess requires embedding-like X_main with shape (N,D), D>1; got {X_main.shape}"
            )
        X_main, X_cos_had = _build_hadamard_features(
            X_main,
            region_id,
            strategy=args.hadamard_anchor_strategy,
            seed=args.seed,
            use_delta_vec=args.use_delta_vec,
            use_abs_diff=args.use_abs_diff,
            abs_diff_only=abs_diff_only,
        )
        X_cos = X_cos_had
        if abs_diff_only:
            feat_key = f"{feat_key}+absdiff_only"
            print(
                "[INFO] Applied abs-diff-only preprocessing "
                f"(strategy={args.hadamard_anchor_strategy}) to X_main"
            )
        else:
            hadamard_suffix_parts: List[str] = ["hadamard"]
            if args.use_delta_vec:
                hadamard_suffix_parts.append("delta")
            elif args.use_abs_diff:
                hadamard_suffix_parts.append("absdiff")
            feat_key = f"{feat_key}+{'_'.join(hadamard_suffix_parts)}"
            print(
                "[INFO] Applied Hadamard preprocessing "
                f"(strategy={args.hadamard_anchor_strategy}, use_delta_vec={bool(args.use_delta_vec)}, "
                f"use_abs_diff={bool(args.use_abs_diff)}) to X_main"
            )

    if args.normalize_data:
        # Optional global preprocessing: normalize each sample vector to unit L2 norm.
        # This is applied once before splitting so every method sees the same input space.
        X_main = _l2_normalize_rows(X_main)
        print("[INFO] Applied row-wise L2 normalization to X_main")

    if X_main.shape[0] != y.shape[0] or X_main.shape[0] != region_id.shape[0]:
        raise ValueError(
            f"Row mismatch: X={X_main.shape[0]} y={y.shape[0]} region_id={region_id.shape[0]}"
        )
    if X_cos is not None and X_cos.shape[0] != X_main.shape[0]:
        raise ValueError(f"Row mismatch: X_cos={X_cos.shape[0]} X_main={X_main.shape[0]}")
    if X_text is not None and X_text.shape[0] != X_main.shape[0]:
        raise ValueError(f"Row mismatch: X_text={X_text.shape[0]} X_main={X_main.shape[0]}")

    run_dir = make_run_dir(base_dir=str(OUT_BASE), run_name=args.run_name)

    print("\n=== Region Threshold Benchmark ===")
    print(f"dataset_npz={npz_path}")
    print(f"region_key={args.region_key} (resolved={region_key}, unique={len(np.unique(region_id))})")
    if sem_bucket_source_key_used is not None:
        print(
            "sem_bucket_on_the_fly="
            f"yes source_key={sem_bucket_source_key_used} "
            f"k={int(args.sem_bucket_k)}"
        )
    print(f"features={feat_key} rows={X_main.shape[0]} dim={X_main.shape[1]}")
    print(f"text_pairs={text_key if text_key is not None else 'none'}")
    print(f"alpha={args.alpha} tie_mode={args.tie_mode} trials={args.n_trials} base_seed={args.seed}")
    print(f"tau_mode={args.tau_mode}")
    if args.tau_mode == "cluster_local":
        print(f"tau_cluster_k={int(args.tau_cluster_k)}")
    print(f"hadamard_preprocess={bool(args.hadamard_preprocess)}")
    print(f"abs_diff_only_preprocess={bool(abs_diff_only)}")
    print(f"use_delta_vec={bool(args.use_delta_vec)}")
    print(f"use_abs_diff={bool(args.use_abs_diff)}")
    print(f"normalize_data={bool(args.normalize_data)}")
    print(f"filter_policy={args.filter_policy}")
    if args.filter_policy == "ambiguous_only":
        print(f"ambiguous_range=[{args.ambiguous_cos_min}, {args.ambiguous_cos_max}]")
    print(f"caps: train={args.n_train} calib={args.n_calib} eval={args.n_eval}")
    print(f"mins: min_h0_eval={args.min_h0_eval} min_h1_eval={args.min_h1_eval}")
    print(f"run_dir={run_dir}")
    if args.enable_online_stopping:
        print(
            "online_stopping: "
            f"check_every={args.stop_check_every} window={args.stop_window} "
            f"patience={args.stop_patience} eps_tpr={args.stop_eps_tpr} "
            f"eps_fpr={args.stop_eps_fpr} eps_tau={args.stop_eps_tau} "
            f"fpr_margin={args.stop_fpr_margin} monitor(h0={args.n_monitor_h0}, h1={args.n_monitor_h1})"
        )
    if run_ocats_baselines:
        print(
            "ocats_baselines: "
            f"methods={selected_ocats_methods} tune={bool(args.tune_ocats)} "
            f"lambdas={lambda_values} cache_k={int(args.cache_k)} "
            f"e_thresh={float(args.e_thresh):.4f} d_thresh={float(args.d_thresh):.4f} "
            f"include_in_comparison={include_ocats_in_comparison} "
            f"progress_every={int(args.ocats_progress_every)}"
        )

    has_xgb = "XGBoost" in build_methods()

    trial_summary_rows: List[Dict[str, Any]] = []
    matched_global_trial_rows: List[Dict[str, Any]] = []
    local_vs_global_matched_rows: List[Dict[str, Any]] = []
    shrink_local_region_rows: List[Dict[str, Any]] = []
    cluster_local_region_rows: List[Dict[str, Any]] = []
    local_region_status_rows: List[Dict[str, Any]] = []
    failures: Dict[str, List[str]] = defaultdict(list)
    configured_methods_last: List[str] = []
    weighted_ensemble_meta_rows: List[Dict[str, Any]] = []
    online_stopping_history_rows: List[Dict[str, Any]] = []
    online_stopping_summary_rows: List[Dict[str, Any]] = []
    ocats_trial_rows: List[Dict[str, Any]] = []
    ocats_tuning_rows: List[Dict[str, Any]] = []
    ocats_curve_rows: List[Dict[str, Any]] = []

    for trial in range(args.n_trials):
        seed = args.seed + trial

        if args.tau_mode == "global":
            # ── TRUE GLOBAL: single stratified split, no region gating ──
            gs, gs_stats = split_global(
                y=y,
                n_train_cap=args.n_train,
                n_calib_cap=args.n_calib,
                n_eval_cap=args.n_eval,
                seed=seed,
            )

            filter_stats_global: Dict[str, int] = {}
            if args.filter_policy == "ambiguous_only":
                if X_cos is None:
                    raise RuntimeError(
                        "filter_policy='ambiguous_only' requires cosine_to_anchor (X_cos), but it is unavailable"
                    )
                gs, filter_stats_global = filter_global_split_by_score_range(
                    gs,
                    score=X_cos[:, 0],
                    score_min=args.ambiguous_cos_min,
                    score_max=args.ambiguous_cos_max,
                )

            h0_monitor_idx = np.array([], dtype=np.int64)
            h1_monitor_idx = np.array([], dtype=np.int64)
            monitor_stats: Dict[str, int] = {}
            if args.enable_online_stopping:
                gs, h0_monitor_idx, h1_monitor_idx, monitor_stats = _sample_monitor_from_global_eval(
                    gs,
                    n_monitor_h0=args.n_monitor_h0,
                    n_monitor_h1=args.n_monitor_h1,
                    seed=seed,
                )

            H0_train = X_main[gs.H0_train]
            H1_train = X_main[gs.H1_train]
            H0_calib_pure = X_main[gs.H0_calib]
            H1_calib_pure = X_main[gs.H1_calib]
            H0_calib_eff = X_main[np.concatenate([gs.H0_train, gs.H0_calib])] \
                if gs.H0_train.size > 0 else X_main[gs.H0_calib]
            H1_calib_eff = X_main[np.concatenate([gs.H1_train, gs.H1_calib])] \
                if gs.H1_train.size > 0 else X_main[gs.H1_calib]

            # Extract X_cos splits if available
            H0_train_cos = X_cos[gs.H0_train] if X_cos is not None else None
            H1_train_cos = X_cos[gs.H1_train] if X_cos is not None else None
            if X_cos is not None:
                H0_calib_pure_cos = X_cos[gs.H0_calib]
                H1_calib_pure_cos = X_cos[gs.H1_calib]
                H0_calib_eff_cos = X_cos[np.concatenate([gs.H0_train, gs.H0_calib])] \
                    if gs.H0_train.size > 0 else X_cos[gs.H0_calib]
                H1_calib_eff_cos = X_cos[np.concatenate([gs.H1_train, gs.H1_calib])] \
                    if gs.H1_train.size > 0 else X_cos[gs.H1_calib]
            else:
                H0_calib_pure_cos = None
                H1_calib_pure_cos = None
                H0_calib_eff_cos = None
                H1_calib_eff_cos = None

            n_unique_regions = int(len(np.unique(region_id)))
            print(f"\n[trial={trial} seed={seed}] GLOBAL mode — total_samples={y.size}")
            print(f"  total: h0={gs_stats['total_h0']} h1={gs_stats['total_h1']}")
            print(f"  pooled_train:  n0={gs_stats['h0_train']} n1={gs_stats['h1_train']}")
            print(f"  pooled_calib:  n0={gs_stats['h0_calib']} n1={gs_stats['h1_calib']}")
            print(f"  pooled_eval:   n0={gs_stats['h0_eval']} n1={gs_stats['h1_eval']}")
            if monitor_stats:
                print(
                    "  monitor_split: "
                    f"n0={monitor_stats['monitor_h0']} n1={monitor_stats['monitor_h1']} "
                    f"eval_remaining(n0={monitor_stats['eval_h0_remaining']}, n1={monitor_stats['eval_h1_remaining']})"
                )
            if filter_stats_global:
                print(
                    "  post_filter_global: "
                    f"train(n0={filter_stats_global['h0_train']},n1={filter_stats_global['h1_train']}) "
                    f"calib(n0={filter_stats_global['h0_calib']},n1={filter_stats_global['h1_calib']}) "
                    f"eval(n0={filter_stats_global['h0_eval']},n1={filter_stats_global['h1_eval']})"
                )
            print(f"  regions (for macro stats only): {n_unique_regions}")

            if H0_calib_eff.shape[0] == 0 or H1_calib_eff.shape[0] == 0:
                raise RuntimeError("No data available for fitting/calibration.")

            if ocats_only_mode:
                train_idx_global = np.concatenate([gs.H0_train, gs.H1_train])
                calib_idx_global = np.concatenate([gs.H0_calib, gs.H1_calib])
                eval_idx_global = np.concatenate([gs.H0_eval, gs.H1_eval])

                oc_out = _run_ocats_for_split(
                    X_main=X_main,
                    y=y,
                    train_idx=train_idx_global,
                    calib_idx=calib_idx_global,
                    eval_idx=eval_idx_global,
                    trial=trial,
                    seed=seed,
                    region_key=args.region_key,
                    tau_mode="global",
                    comparison_scope="global_split",
                    args=args,
                    selected_methods=selected_ocats_methods,
                    lambda_values=lambda_values,
                )
                ocats_trial_rows.extend(oc_out["trial_rows"])
                ocats_tuning_rows.extend(oc_out["tuning_rows"])
                ocats_curve_rows.extend(oc_out["curve_rows"])
                if oc_out["trial_rows"]:
                    print(f"  ocats_methods_evaluated={len(oc_out['trial_rows'])}")

                trial_rows = (
                    _build_ocats_comparison_rows(
                        oc_out["trial_rows"],
                        shared_regions=1,
                        dropped_regions_for_comparability=0,
                        alpha=float(args.alpha),
                    )
                    if include_ocats_in_comparison
                    else []
                )
                trial_summary_rows.extend(trial_rows)
                print_trial_table(trial_rows, alpha=float(args.alpha))
                continue

            # Weights for feature-based methods
            v0 = np.var(H0_calib_eff, axis=0)
            v1 = np.var(H1_calib_eff, axis=0)
            weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)

            methods = build_methods()
            if args.hadamard_preprocess and X_cos is not None:
                try:
                    from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
                    # In hadamard mode, direct sum(hadamard row) is the cosine-to-anchor signal.
                    # Route the "Cosine" baseline to this scalar path to avoid refitting prototype cosine.
                    methods["Cosine"] = PrecomputedCosineMethod()
                    for ens_name in ["WeightedEnsemble", "RegionalWeightedEnsemble"]:
                        if ens_name in methods and hasattr(methods[ens_name], "judges"):
                            ens = methods[ens_name]
                            new_judges = []
                            for j in list(getattr(ens, "judges", [])):
                                if str(getattr(j, "name", "")) == "Cosine":
                                    new_judges.append(PrecomputedCosineMethod())
                                else:
                                    new_judges.append(j)
                            ens.judges = new_judges
                    print("[INFO] Cosine baseline routed to direct Hadamard sum (PrecomputedCosine)")
                    if "WeightedEnsemble" in methods:
                        print("[INFO] WeightedEnsemble Cosine judge routed to direct Hadamard sum")
                    if "RegionalWeightedEnsemble" in methods:
                        print("[INFO] RegionalWeightedEnsemble Cosine judge routed to direct Hadamard sum")
                except Exception as exc:
                    print(f"[WARN] Could not route Cosine baseline to PrecomputedCosine: {exc}")
            # Add SWC with CLI params (fresh instance per trial)
            try:
                from np_bench.methods.stabilized_whitened_cosine import StabilizedWhitenedCosineMethod
                methods["StabilizedWhitenedCosine"] = StabilizedWhitenedCosineMethod(
                    k=args.swc_k, shrinkage=args.swc_shrinkage,
                    eps=args.swc_eps, min_samples=args.swc_min_samples,
                    fallback=args.swc_fallback, verbose=args.swc_verbose,
                )
            except Exception:
                pass

            if args.cos_affine_calib:
                try:
                    from np_bench.methods.cosine_affine_calib import CosineAffineCalibMethod
                    methods["CosineAffineCalib"] = CosineAffineCalibMethod()
                except Exception as exc:
                    print(f"[WARN] Could not load CosineAffineCalib: {exc}")

            if args.precomputed_cosine:
                try:
                    from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
                    methods["PrecomputedCosine"] = PrecomputedCosineMethod()
                except Exception as exc:
                    print(f"[WARN] Could not load PrecomputedCosine: {exc}")

            if args.regional_weighted_ensemble:
                try:
                    from np_bench.methods.regional_weighted_ensemble import (
                        RegionalEnsembleConfig,
                        RegionalWeightedEnsembleMethod,
                    )
                    judges = []
                    if "WeightedEnsemble" in methods and hasattr(methods["WeightedEnsemble"], "judges"):
                        judges = list(getattr(methods["WeightedEnsemble"], "judges", []))
                    if judges:
                        methods["RegionalWeightedEnsemble"] = RegionalWeightedEnsembleMethod(
                            judges=judges,
                            config=RegionalEnsembleConfig(
                                alpha=float(args.alpha),
                                k_shrink=float(args.rwe_k_shrink),
                                min_region_h0=int(args.rwe_min_region_h0),
                                min_region_h1=int(args.rwe_min_region_h1),
                            ),
                        )
                    else:
                        print("[WARN] Could not initialize RegionalWeightedEnsemble: no base judges found")
                except Exception as exc:
                    print(f"[WARN] Could not load RegionalWeightedEnsemble: {exc}")

            if args.enable_bge_reranker:
                if X_text is None:
                    print("[WARN] --enable_bge_reranker set but no text pairs were resolved; skipping method")
                else:
                    try:
                        from np_bench.methods.bge_reranker import BGERerankerMethod
                        methods["BGE Reranker"] = BGERerankerMethod(
                            model_name=args.bge_model_name,
                            batch_size=args.bge_batch_size,
                            max_length=args.bge_max_length,
                            normalize_scores=args.bge_normalize_scores,
                            backend=args.bge_backend,
                        )
                    except Exception as exc:
                        print(f"[WARN] Could not load BGE Reranker method: {exc}")

            method_names = list(methods.keys())
            configured_methods_last = method_names[:]

            online_method = None
            online_history_rows: List[Dict[str, Any]] = []
            online_summary: Dict[str, Any] = {}
            if args.enable_online_stopping:
                online_method, online_history_rows, online_summary = _run_online_stopping(
                    X_main,
                    y,
                    h0_train_idx=gs.H0_train,
                    h1_train_idx=gs.H1_train,
                    h0_calib_idx=gs.H0_calib,
                    h0_monitor_idx=h0_monitor_idx,
                    h1_monitor_idx=h1_monitor_idx,
                    alpha=args.alpha,
                    tie_mode=args.tie_mode,
                    tau_guardrail=args.tau_guardrail,
                    tau_guardrail_delta=args.tau_guardrail_delta,
                    seed=seed,
                    args=args,
                )

            fit_all_methods(
                methods,
                H0_train=H0_train,
                H1_train=H1_train,
                H0_calib_eff=H0_calib_eff,
                H1_calib_eff=H1_calib_eff,
                H0_calib_pure=H0_calib_pure,
                H1_calib_pure=H1_calib_pure,
                H0_train_cos=H0_train_cos,
                H1_train_cos=H1_train_cos,
                H0_calib_eff_cos=H0_calib_eff_cos,
                H1_calib_eff_cos=H1_calib_eff_cos,
                H0_train_text=X_text[gs.H0_train] if X_text is not None else None,
                H1_train_text=X_text[gs.H1_train] if X_text is not None else None,
                H0_calib_eff_text=(
                    X_text[np.concatenate([gs.H0_train, gs.H0_calib])]
                    if X_text is not None and gs.H0_train.size > 0
                    else (X_text[gs.H0_calib] if X_text is not None else None)
                ),
                H1_calib_eff_text=(
                    X_text[np.concatenate([gs.H1_train, gs.H1_calib])]
                    if X_text is not None and gs.H1_train.size > 0
                    else (X_text[gs.H1_calib] if X_text is not None else None)
                ),
                H0_calib_pure_cos=H0_calib_pure_cos,
                H1_calib_pure_cos=H1_calib_pure_cos,
                H0_calib_region_ids=region_id[gs.H0_calib],
                H1_calib_region_ids=region_id[gs.H1_calib],
                tie_mode=args.tie_mode,
                tau_guardrail=args.tau_guardrail,
                tau_guardrail_delta=args.tau_guardrail_delta,
                weights=weights,
                seed=seed,
                alpha=args.alpha,
                trial=trial,
                failures=failures,
                fit_context="global",
            )

            if args.enable_online_stopping:
                if online_method is not None:
                    methods["Online(refit)"] = online_method
                    method_names.append("Online(refit)")
                    configured_methods_last = method_names[:]
                    for r in online_history_rows:
                        r["trial"] = int(trial)
                        r["seed"] = int(seed)
                    online_stopping_history_rows.extend(online_history_rows)
                    online_stopping_summary_rows.append(
                        {
                            "trial": int(trial),
                            "seed": int(seed),
                            **online_summary,
                        }
                    )
                    if online_history_rows:
                        last = online_history_rows[-1]
                        print(
                            "  online_stopping: "
                            f"checks={online_summary.get('history_len', 0)} "
                            f"stopped={online_summary.get('stopped')} reason={online_summary.get('reason')} "
                            f"tpr={float(last.get('tpr_monitor', float('nan'))):.4f} "
                            f"fpr={float(last.get('fpr_monitor', float('nan'))):.4f} "
                            f"tau={float(last.get('tau', float('nan'))):.4f} "
                            f"samples_used={online_summary.get('samples_total_used', 0)}"
                        )
                else:
                    online_stopping_summary_rows.append(
                        {
                            "trial": int(trial),
                            "seed": int(seed),
                            **online_summary,
                        }
                    )
                    print(
                        "  [WARN] online_stopping skipped: "
                        f"reason={online_summary.get('reason', 'unknown')}"
                    )

            if "BGE Reranker" in methods:
                bge_m = methods["BGE Reranker"]
                if bool(getattr(bge_m, "using_fallback", False)):
                    reason = str(getattr(bge_m, "fallback_reason", "model load failed"))
                    reason_one_line = reason.splitlines()[0][:240]
                    print(f"[WARN] BGE Reranker running in fallback mode: {reason_one_line}")

            # Persist trial-wise ensemble weights for auditability.
            if "WeightedEnsemble" in methods and hasattr(methods["WeightedEnsemble"], "meta_w"):
                we = methods["WeightedEnsemble"]
                w = getattr(we, "meta_w", None)
                judges = getattr(we, "judges", [])
                if w is not None and judges:
                    for j, wj in zip(judges, np.asarray(w).reshape(-1)):
                        weighted_ensemble_meta_rows.append(
                            {
                                "trial": int(trial),
                                "seed": int(seed),
                                "judge": str(getattr(j, "name", type(j).__name__)),
                                "weight": float(wj),
                            }
                        )

            trial_rows = evaluate_methods_global(
                methods,
                method_names,
                gs,
                X_main=X_main,
                X_cos=X_cos,
                X_text=X_text,
                alpha=args.alpha,
                tie_mode=args.tie_mode,
                tau_guardrail=args.tau_guardrail,
                tau_guardrail_delta=args.tau_guardrail_delta,
                trial=trial,
                seed=seed,
                region_key=args.region_key,
                region_id=region_id,
                failures=failures,
            )

            if run_ocats_baselines:
                train_idx_global = np.concatenate([gs.H0_train, gs.H1_train])
                calib_idx_global = np.concatenate([gs.H0_calib, gs.H1_calib])
                eval_idx_global = np.concatenate([gs.H0_eval, gs.H1_eval])

                oc_out = _run_ocats_for_split(
                    X_main=X_main,
                    y=y,
                    train_idx=train_idx_global,
                    calib_idx=calib_idx_global,
                    eval_idx=eval_idx_global,
                    trial=trial,
                    seed=seed,
                    region_key=args.region_key,
                    tau_mode="global",
                    comparison_scope="global_split",
                    args=args,
                    selected_methods=selected_ocats_methods,
                    lambda_values=lambda_values,
                )
                ocats_trial_rows.extend(oc_out["trial_rows"])
                ocats_tuning_rows.extend(oc_out["tuning_rows"])
                ocats_curve_rows.extend(oc_out["curve_rows"])
                if oc_out["trial_rows"]:
                    print(f"  ocats_methods_evaluated={len(oc_out['trial_rows'])}")
                    if include_ocats_in_comparison:
                        trial_rows.extend(
                            _build_ocats_comparison_rows(
                                oc_out["trial_rows"],
                                shared_regions=1,
                                dropped_regions_for_comparability=0,
                                alpha=float(args.alpha),
                            )
                        )

        else:
            # ── LOCAL: per-region splits with min gating (unchanged) ──
            splits, split_stats, region_status = split_indices_per_region_detailed(
                region_id=region_id,
                y=y,
                n_train_cap=args.n_train,
                n_calib_cap=args.n_calib,
                n_eval_cap=args.n_eval,
                seed=seed,
                min_h0_eval=args.min_h0_eval,
                min_h1_eval=args.min_h1_eval,
            )

            filter_stats_local: Dict[str, int] = {}
            if args.filter_policy == "ambiguous_only":
                if X_cos is None:
                    raise RuntimeError(
                        "filter_policy='ambiguous_only' requires cosine_to_anchor (X_cos), but it is unavailable"
                    )
                splits, filter_stats_local, filter_status_updates = filter_region_splits_by_score_range_detailed(
                    splits,
                    score=X_cos[:, 0],
                    score_min=args.ambiguous_cos_min,
                    score_max=args.ambiguous_cos_max,
                    min_h0_eval=args.min_h0_eval,
                    min_h1_eval=args.min_h1_eval,
                )
                for rid, st in filter_status_updates.items():
                    region_status[int(rid)] = st

            h0_monitor_idx = np.array([], dtype=np.int64)
            h1_monitor_idx = np.array([], dtype=np.int64)
            monitor_stats = {}
            if args.enable_online_stopping:
                splits, h0_monitor_idx, h1_monitor_idx, monitor_stats = _sample_monitor_from_region_splits(
                    splits,
                    n_monitor_h0=args.n_monitor_h0,
                    n_monitor_h1=args.n_monitor_h1,
                    min_h0_eval=args.min_h0_eval,
                    min_h1_eval=args.min_h1_eval,
                    seed=seed,
                )

            if len(splits) == 0:
                raise RuntimeError(
                    "No regions satisfied eval mins. Lower mins, increase caps, or change regioning."
                )

            tau_cluster_id_by_rid: Optional[Dict[int, int]] = None
            if args.tau_mode == "cluster_local":
                if X_anchor_source.ndim != 2 or X_anchor_source.shape[1] <= 1:
                    raise ValueError(
                        "tau_mode='cluster_local' requires embedding-like source features with shape (N,D), D>1"
                    )
                tau_cluster_id_by_rid = _build_tau_cluster_map(
                    splits=splits,
                    X_anchor_source=X_anchor_source,
                    region_id=region_id,
                    anchor_strategy=args.hadamard_anchor_strategy,
                    n_clusters=max(1, int(args.tau_cluster_k)),
                    seed=seed,
                )
                if not tau_cluster_id_by_rid:
                    raise RuntimeError("tau_mode='cluster_local' produced an empty region-to-cluster map")
                n_clusters_used = int(len(set(tau_cluster_id_by_rid.values())))
                print(
                    "  tau_cluster_local: "
                    f"regions={len(tau_cluster_id_by_rid)} "
                    f"clusters={n_clusters_used} "
                    f"k={int(args.tau_cluster_k)} "
                    f"anchor_strategy={args.hadamard_anchor_strategy}"
                )

            # Pool indices across regions
            h0_train_idx_list = [s.H0_train for s in splits if s.H0_train.size > 0]
            h1_train_idx_list = [s.H1_train for s in splits if s.H1_train.size > 0]
            H0_train = X_main[np.concatenate(h0_train_idx_list)] if h0_train_idx_list else X_main[:0]
            H1_train = X_main[np.concatenate(h1_train_idx_list)] if h1_train_idx_list else X_main[:0]

            h0_calib_list = [s.H0_calib for s in splits if s.H0_calib.size > 0]
            h1_calib_list = [s.H1_calib for s in splits if s.H1_calib.size > 0]
            H0_calib = X_main[np.concatenate(h0_calib_list)] if h0_calib_list else X_main[:0]
            H1_calib = X_main[np.concatenate(h1_calib_list)] if h1_calib_list else X_main[:0]

            H0_calib_eff = np.concatenate([H0_train, H0_calib], axis=0) if H0_train.shape[0] > 0 else H0_calib
            H1_calib_eff = np.concatenate([H1_train, H1_calib], axis=0) if H1_train.shape[0] > 0 else H1_calib

            # Extract X_cos splits if available
            if X_cos is not None:
                H0_train_cos = X_cos[np.concatenate(h0_train_idx_list)] if h0_train_idx_list else X_cos[:0]
                H1_train_cos = X_cos[np.concatenate(h1_train_idx_list)] if h1_train_idx_list else X_cos[:0]
                H0_calib_cos = X_cos[np.concatenate(h0_calib_list)] if h0_calib_list else X_cos[:0]
                H1_calib_cos = X_cos[np.concatenate(h1_calib_list)] if h1_calib_list else X_cos[:0]
                H0_calib_eff_cos = np.concatenate([H0_train_cos, H0_calib_cos], axis=0) if H0_train_cos.shape[0] > 0 else H0_calib_cos
                H1_calib_eff_cos = np.concatenate([H1_train_cos, H1_calib_cos], axis=0) if H1_train_cos.shape[0] > 0 else H1_calib_cos
            else:
                H0_train_cos = None
                H1_train_cos = None
                H0_calib_cos = None
                H1_calib_cos = None
                H0_calib_eff_cos = None
                H1_calib_eff_cos = None

            if X_text is not None:
                H0_train_text = X_text[np.concatenate(h0_train_idx_list)] if h0_train_idx_list else X_text[:0]
                H1_train_text = X_text[np.concatenate(h1_train_idx_list)] if h1_train_idx_list else X_text[:0]
                H0_calib_text = X_text[np.concatenate(h0_calib_list)] if h0_calib_list else X_text[:0]
                H1_calib_text = X_text[np.concatenate(h1_calib_list)] if h1_calib_list else X_text[:0]
                H0_calib_eff_text = np.concatenate([H0_train_text, H0_calib_text], axis=0) if H0_train_text.shape[0] > 0 else H0_calib_text
                H1_calib_eff_text = np.concatenate([H1_train_text, H1_calib_text], axis=0) if H1_train_text.shape[0] > 0 else H1_calib_text
            else:
                H0_train_text = None
                H1_train_text = None
                H0_calib_eff_text = None
                H1_calib_eff_text = None

            print(f"\n[trial={trial} seed={seed}] used_regions={len(splits)} split_stats={split_stats}")
            if filter_stats_local:
                print(f"  post_filter_stats={filter_stats_local}")
            if len(splits) < 5:
                print(f"  [WARN] Only {len(splits)} region(s) evaluated. Results are high-variance.")
            print(f"  local_fit_mode: {args.local_fit_mode}")
            print(f"  pooled_train: n0={H0_train.shape[0]} n1={H1_train.shape[0]}")
            print(f"  pooled_calib_eff: n0={H0_calib_eff.shape[0]} n1={H1_calib_eff.shape[0]}")
            if monitor_stats:
                print(
                    "  monitor_split: "
                    f"n0={monitor_stats['monitor_h0']} n1={monitor_stats['monitor_h1']} "
                    f"eval_remaining(n0={monitor_stats['eval_h0_remaining']}, n1={monitor_stats['eval_h1_remaining']})"
                )

            if H0_calib_eff.shape[0] == 0 or H1_calib_eff.shape[0] == 0:
                raise RuntimeError("No data available for fitting/calibration.")

            if ocats_only_mode:
                ocats_rids = [int(s.rid) for s in splits]
                gs_ocats = _build_global_split_from_local_regions(splits, ocats_rids)

                if (
                    gs_ocats.H0_train.size > 0
                    and gs_ocats.H1_train.size > 0
                    and gs_ocats.H0_eval.size > 0
                    and gs_ocats.H1_eval.size > 0
                ):
                    train_idx_local = np.concatenate([gs_ocats.H0_train, gs_ocats.H1_train])
                    calib_idx_local = np.concatenate([gs_ocats.H0_calib, gs_ocats.H1_calib])
                    eval_idx_local = np.concatenate([gs_ocats.H0_eval, gs_ocats.H1_eval])

                    oc_out = _run_ocats_for_split(
                        X_main=X_main,
                        y=y,
                        train_idx=train_idx_local,
                        calib_idx=calib_idx_local,
                        eval_idx=eval_idx_local,
                        trial=trial,
                        seed=seed,
                        region_key=args.region_key,
                        tau_mode=args.tau_mode,
                        comparison_scope="local_eligible_regions",
                        args=args,
                        selected_methods=selected_ocats_methods,
                        lambda_values=lambda_values,
                    )
                    ocats_trial_rows.extend(oc_out["trial_rows"])
                    ocats_tuning_rows.extend(oc_out["tuning_rows"])
                    ocats_curve_rows.extend(oc_out["curve_rows"])
                    if oc_out["trial_rows"]:
                        print(f"  ocats_methods_evaluated={len(oc_out['trial_rows'])}")

                    trial_rows = (
                        _build_ocats_comparison_rows(
                            oc_out["trial_rows"],
                            shared_regions=len(splits),
                            dropped_regions_for_comparability=0,
                            alpha=float(args.alpha),
                        )
                        if include_ocats_in_comparison
                        else []
                    )
                else:
                    failures["ocats_eval"].append(
                        "trial={} : OCATS skipped due to empty pooled train/eval in scope=local_eligible_regions".format(
                            trial
                        )
                    )
                    trial_rows = []

                trial_summary_rows.extend(trial_rows)
                if trial_rows and args.tau_mode != "global":
                    shared_counts = {int(r.get("shared_regions", r.get("ok_regions", 0))) for r in trial_rows}
                    if len(shared_counts) != 1:
                        raise RuntimeError(
                            f"trial={trial}: comparability validation failed (inconsistent shared region counts): {sorted(shared_counts)}"
                        )
                    if next(iter(shared_counts)) <= 0:
                        raise RuntimeError(f"trial={trial}: comparability validation failed (no shared regions)")
                print_trial_table(trial_rows, alpha=float(args.alpha))
                continue

            # Weights for feature-based methods
            v0 = np.var(H0_calib_eff, axis=0)
            v1 = np.var(H1_calib_eff, axis=0)
            weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)

            methods = build_methods()
            if args.hadamard_preprocess and X_cos is not None:
                try:
                    from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
                    # In hadamard mode, direct sum(hadamard row) is the cosine-to-anchor signal.
                    # Route the "Cosine" baseline to this scalar path to avoid refitting prototype cosine.
                    methods["Cosine"] = PrecomputedCosineMethod()
                    for ens_name in ["WeightedEnsemble", "RegionalWeightedEnsemble"]:
                        if ens_name in methods and hasattr(methods[ens_name], "judges"):
                            ens = methods[ens_name]
                            new_judges = []
                            for j in list(getattr(ens, "judges", [])):
                                if str(getattr(j, "name", "")) == "Cosine":
                                    new_judges.append(PrecomputedCosineMethod())
                                else:
                                    new_judges.append(j)
                            ens.judges = new_judges
                    print("[INFO] Cosine baseline routed to direct Hadamard sum (PrecomputedCosine)")
                    if "WeightedEnsemble" in methods:
                        print("[INFO] WeightedEnsemble Cosine judge routed to direct Hadamard sum")
                    if "RegionalWeightedEnsemble" in methods:
                        print("[INFO] RegionalWeightedEnsemble Cosine judge routed to direct Hadamard sum")
                except Exception as exc:
                    print(f"[WARN] Could not route Cosine baseline to PrecomputedCosine: {exc}")
            # Add SWC with CLI params (fresh instance per trial)
            try:
                from np_bench.methods.stabilized_whitened_cosine import StabilizedWhitenedCosineMethod
                methods["StabilizedWhitenedCosine"] = StabilizedWhitenedCosineMethod(
                    k=args.swc_k, shrinkage=args.swc_shrinkage,
                    eps=args.swc_eps, min_samples=args.swc_min_samples,
                    fallback=args.swc_fallback, verbose=args.swc_verbose,
                )
            except Exception:
                pass

            if args.cos_affine_calib:
                try:
                    from np_bench.methods.cosine_affine_calib import CosineAffineCalibMethod
                    methods["CosineAffineCalib"] = CosineAffineCalibMethod()
                except Exception as exc:
                    print(f"[WARN] Could not load CosineAffineCalib: {exc}")

            if args.precomputed_cosine:
                try:
                    from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
                    methods["PrecomputedCosine"] = PrecomputedCosineMethod()
                except Exception as exc:
                    print(f"[WARN] Could not load PrecomputedCosine: {exc}")

            if args.regional_weighted_ensemble:
                try:
                    from np_bench.methods.regional_weighted_ensemble import (
                        RegionalEnsembleConfig,
                        RegionalWeightedEnsembleMethod,
                    )
                    judges = []
                    if "WeightedEnsemble" in methods and hasattr(methods["WeightedEnsemble"], "judges"):
                        judges = list(getattr(methods["WeightedEnsemble"], "judges", []))
                    if judges:
                        methods["RegionalWeightedEnsemble"] = RegionalWeightedEnsembleMethod(
                            judges=judges,
                            config=RegionalEnsembleConfig(
                                alpha=float(args.alpha),
                                k_shrink=float(args.rwe_k_shrink),
                                min_region_h0=int(args.rwe_min_region_h0),
                                min_region_h1=int(args.rwe_min_region_h1),
                            ),
                        )
                    else:
                        print("[WARN] Could not initialize RegionalWeightedEnsemble: no base judges found")
                except Exception as exc:
                    print(f"[WARN] Could not load RegionalWeightedEnsemble: {exc}")

            if args.enable_bge_reranker:
                if X_text is None:
                    print("[WARN] --enable_bge_reranker set but no text pairs were resolved; skipping method")
                else:
                    try:
                        from np_bench.methods.bge_reranker import BGERerankerMethod
                        methods["BGE Reranker"] = BGERerankerMethod(
                            model_name=args.bge_model_name,
                            batch_size=args.bge_batch_size,
                            max_length=args.bge_max_length,
                            normalize_scores=args.bge_normalize_scores,
                            backend=args.bge_backend,
                        )
                    except Exception as exc:
                        print(f"[WARN] Could not load BGE Reranker method: {exc}")

            method_names = list(methods.keys())
            configured_methods_last = method_names[:]

            online_method = None
            online_history_rows: List[Dict[str, Any]] = []
            online_summary: Dict[str, Any] = {}
            if args.enable_online_stopping:
                online_method, online_history_rows, online_summary = _run_online_stopping(
                    X_main,
                    y,
                    h0_train_idx=_concat_indices(h0_train_idx_list),
                    h1_train_idx=_concat_indices(h1_train_idx_list),
                    h0_calib_idx=_concat_indices(h0_calib_list),
                    h0_monitor_idx=h0_monitor_idx,
                    h1_monitor_idx=h1_monitor_idx,
                    alpha=args.alpha,
                    tie_mode=args.tie_mode,
                    tau_guardrail=args.tau_guardrail,
                    tau_guardrail_delta=args.tau_guardrail_delta,
                    seed=seed,
                    args=args,
                )

            # Fit methods
            if args.local_fit_mode == "pooled":
                fit_all_methods(
                    methods,
                    H0_train=H0_train,
                    H1_train=H1_train,
                    H0_calib_eff=H0_calib_eff,
                    H1_calib_eff=H1_calib_eff,
                    H0_calib_pure=H0_calib,
                    H1_calib_pure=H1_calib,
                    H0_train_cos=H0_train_cos,
                    H1_train_cos=H1_train_cos,
                    H0_calib_eff_cos=H0_calib_eff_cos,
                    H1_calib_eff_cos=H1_calib_eff_cos,
                    H0_calib_pure_cos=H0_calib_cos,
                    H1_calib_pure_cos=H1_calib_cos,
                    H0_calib_region_ids=(
                        region_id[np.concatenate(h0_calib_list)] if h0_calib_list else np.array([], dtype=np.int64)
                    ),
                    H1_calib_region_ids=(
                        region_id[np.concatenate(h1_calib_list)] if h1_calib_list else np.array([], dtype=np.int64)
                    ),
                    H0_train_text=H0_train_text,
                    H1_train_text=H1_train_text,
                    H0_calib_eff_text=H0_calib_eff_text,
                    H1_calib_eff_text=H1_calib_eff_text,
                    tie_mode=args.tie_mode,
                    tau_guardrail=args.tau_guardrail,
                    tau_guardrail_delta=args.tau_guardrail_delta,
                    weights=weights,
                    seed=seed,
                    alpha=args.alpha,
                    trial=trial,
                    failures=failures,
                    require_pure_calib_for_ensemble=True,
                    fit_context="local",
                )
            else:
                print("  local_fit_mode=per_region: fitting deferred to per-region evaluation")

            if args.enable_online_stopping:
                if online_method is not None:
                    methods["Online(refit)"] = online_method
                    method_names.append("Online(refit)")
                    configured_methods_last = method_names[:]
                    for r in online_history_rows:
                        r["trial"] = int(trial)
                        r["seed"] = int(seed)
                    online_stopping_history_rows.extend(online_history_rows)
                    online_stopping_summary_rows.append(
                        {
                            "trial": int(trial),
                            "seed": int(seed),
                            **online_summary,
                        }
                    )
                    if online_history_rows:
                        last = online_history_rows[-1]
                        print(
                            "  online_stopping: "
                            f"checks={online_summary.get('history_len', 0)} "
                            f"stopped={online_summary.get('stopped')} reason={online_summary.get('reason')} "
                            f"tpr={float(last.get('tpr_monitor', float('nan'))):.4f} "
                            f"fpr={float(last.get('fpr_monitor', float('nan'))):.4f} "
                            f"tau={float(last.get('tau', float('nan'))):.4f} "
                            f"samples_used={online_summary.get('samples_total_used', 0)}"
                        )
                else:
                    online_stopping_summary_rows.append(
                        {
                            "trial": int(trial),
                            "seed": int(seed),
                            **online_summary,
                        }
                    )
                    print(
                        "  [WARN] online_stopping skipped: "
                        f"reason={online_summary.get('reason', 'unknown')}"
                    )

            if "BGE Reranker" in methods:
                bge_m = methods["BGE Reranker"]
                if bool(getattr(bge_m, "using_fallback", False)):
                    reason = str(getattr(bge_m, "fallback_reason", "model load failed"))
                    reason_one_line = reason.splitlines()[0][:240]
                    print(f"[WARN] BGE Reranker running in fallback mode: {reason_one_line}")

            # Evaluate methods
            local_eval_meta: Dict[str, Any] = {}
            trial_rows = evaluate_methods(
                methods,
                method_names,
                splits,
                X_main=X_main,
                X_cos=X_cos,
                X_text=X_text,
                alpha=args.alpha,
                tau_mode=args.tau_mode,
                tie_mode=args.tie_mode,
                tau_shrink=args.tau_shrink,
                tau_shrink_m=args.tau_shrink_m,
                shrink_k=args.shrink_k,
                tau_guardrail=args.tau_guardrail,
                tau_guardrail_delta=args.tau_guardrail_delta,
                swc_mode=args.swc_mode,
                swc_cluster_n_clusters=args.swc_cluster_n_clusters,
                cos_affine_grouping=args.cos_affine_grouping,
                cos_affine_n_clusters=args.cos_affine_n_clusters,
                local_fit_mode=args.local_fit_mode,
                trial=trial,
                seed=seed,
                region_key=args.region_key,
                h0_train_idx_list=h0_train_idx_list,
                h1_train_idx_list=h1_train_idx_list,
                h0_calib_list=h0_calib_list,
                h1_calib_list=h1_calib_list,
                H0_calib_eff=H0_calib_eff,
                failures=failures,
                tau_cluster_id_by_rid=tau_cluster_id_by_rid,
                trial_meta=local_eval_meta,
            )

            if args.tau_mode == "cluster_local":
                cluster_rows_trial = [
                    dict(r)
                    for r in local_eval_meta.get("cluster_local_rows", [])
                    if int(r.get("trial", -1)) == int(trial)
                ]
                cluster_local_region_rows.extend(cluster_rows_trial)
                if cluster_rows_trial:
                    n_clusters_trial = len({int(r["cluster_id"]) for r in cluster_rows_trial})
                    print(
                        "  cluster_local: "
                        f"rows={len(cluster_rows_trial)} clusters={n_clusters_trial}"
                    )

            if args.tau_mode == "shrink_local":
                shrink_rows_trial = [
                    dict(r)
                    for r in local_eval_meta.get("shrink_local_rows", [])
                    if int(r.get("trial", -1)) == int(trial)
                ]
                shrink_local_region_rows.extend(shrink_rows_trial)
                if shrink_rows_trial:
                    lambdas = [float(r["lambda_global"]) for r in shrink_rows_trial]
                    print(
                        "  shrink_local: "
                        f"rows={len(shrink_rows_trial)} "
                        f"lambda_global(mean={float(np.mean(lambdas)):.4f}, "
                        f"min={float(np.min(lambdas)):.4f}, max={float(np.max(lambdas)):.4f})"
                    )

            tested_region_ids = [int(r) for r in local_eval_meta.get("tested_region_ids", [])]
            tested_set = set(tested_region_ids)

            for rid in sorted(region_status.keys()):
                st = region_status.get(rid, {"status": "unknown", "reason": "unknown"})
                status = str(st.get("status", "unknown"))
                reason = str(st.get("reason", ""))
                if rid in tested_set:
                    status = "evaluated_local"
                    reason = "used_in_local_metrics"
                elif status in {"eligible_after_split", "eligible_after_filter"}:
                    status = "excluded_from_local_metrics"
                    reason = "not_in_shared_tested_regions"
                local_region_status_rows.append(
                    {
                        "trial": int(trial),
                        "seed": int(seed),
                        "rid": int(rid),
                        "status": status,
                        "reason": reason,
                        "is_tested": bool(rid in tested_set),
                    }
                )

            if tested_region_ids:
                gs_matched = _build_global_split_from_local_regions(splits, tested_region_ids)
                if gs_matched.H0_calib.size > 0 and gs_matched.H0_eval.size > 0 and gs_matched.H1_eval.size > 0:
                    # Rebuild and refit a fresh method set to avoid state carry-over
                    # from local per-region evaluation (e.g., methods with fit_region state).
                    methods_matched = build_methods()
                    if args.hadamard_preprocess and X_cos is not None:
                        try:
                            from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
                            methods_matched["Cosine"] = PrecomputedCosineMethod()
                            if "WeightedEnsemble" in methods_matched and hasattr(methods_matched["WeightedEnsemble"], "judges"):
                                we_m = methods_matched["WeightedEnsemble"]
                                new_judges_m = []
                                for j in list(getattr(we_m, "judges", [])):
                                    if str(getattr(j, "name", "")) == "Cosine":
                                        new_judges_m.append(PrecomputedCosineMethod())
                                    else:
                                        new_judges_m.append(j)
                                we_m.judges = new_judges_m
                        except Exception:
                            pass

                    try:
                        from np_bench.methods.stabilized_whitened_cosine import StabilizedWhitenedCosineMethod
                        methods_matched["StabilizedWhitenedCosine"] = StabilizedWhitenedCosineMethod(
                            k=args.swc_k,
                            shrinkage=args.swc_shrinkage,
                            eps=args.swc_eps,
                            min_samples=args.swc_min_samples,
                            fallback=args.swc_fallback,
                            verbose=args.swc_verbose,
                        )
                    except Exception:
                        pass

                    if args.cos_affine_calib:
                        try:
                            from np_bench.methods.cosine_affine_calib import CosineAffineCalibMethod
                            methods_matched["CosineAffineCalib"] = CosineAffineCalibMethod()
                        except Exception:
                            pass

                    if args.precomputed_cosine:
                        try:
                            from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
                            methods_matched["PrecomputedCosine"] = PrecomputedCosineMethod()
                        except Exception:
                            pass

                    if args.regional_weighted_ensemble:
                        try:
                            from np_bench.methods.regional_weighted_ensemble import (
                                RegionalEnsembleConfig,
                                RegionalWeightedEnsembleMethod,
                            )
                            judges_m = []
                            if "WeightedEnsemble" in methods_matched and hasattr(methods_matched["WeightedEnsemble"], "judges"):
                                judges_m = list(getattr(methods_matched["WeightedEnsemble"], "judges", []))
                            if judges_m:
                                methods_matched["RegionalWeightedEnsemble"] = RegionalWeightedEnsembleMethod(
                                    judges=judges_m,
                                    config=RegionalEnsembleConfig(
                                        alpha=float(args.alpha),
                                        k_shrink=float(args.rwe_k_shrink),
                                        min_region_h0=int(args.rwe_min_region_h0),
                                        min_region_h1=int(args.rwe_min_region_h1),
                                    ),
                                )
                        except Exception:
                            pass

                    if args.enable_bge_reranker and X_text is not None:
                        try:
                            from np_bench.methods.bge_reranker import BGERerankerMethod
                            methods_matched["BGE Reranker"] = BGERerankerMethod(
                                model_name=args.bge_model_name,
                                batch_size=args.bge_batch_size,
                                max_length=args.bge_max_length,
                                normalize_scores=args.bge_normalize_scores,
                                backend=args.bge_backend,
                            )
                        except Exception:
                            pass

                    method_names_matched = list(methods_matched.keys())
                    fit_all_methods(
                        methods_matched,
                        H0_train=H0_train,
                        H1_train=H1_train,
                        H0_calib_eff=H0_calib_eff,
                        H1_calib_eff=H1_calib_eff,
                        H0_calib_pure=H0_calib,
                        H1_calib_pure=H1_calib,
                        H0_train_cos=H0_train_cos,
                        H1_train_cos=H1_train_cos,
                        H0_calib_eff_cos=H0_calib_eff_cos,
                        H1_calib_eff_cos=H1_calib_eff_cos,
                        H0_calib_pure_cos=H0_calib_cos,
                        H1_calib_pure_cos=H1_calib_cos,
                        H0_calib_region_ids=(
                            region_id[np.concatenate(h0_calib_list)] if h0_calib_list else np.array([], dtype=np.int64)
                        ),
                        H1_calib_region_ids=(
                            region_id[np.concatenate(h1_calib_list)] if h1_calib_list else np.array([], dtype=np.int64)
                        ),
                        H0_train_text=H0_train_text,
                        H1_train_text=H1_train_text,
                        H0_calib_eff_text=H0_calib_eff_text,
                        H1_calib_eff_text=H1_calib_eff_text,
                        tie_mode=args.tie_mode,
                        tau_guardrail=args.tau_guardrail,
                        tau_guardrail_delta=args.tau_guardrail_delta,
                        weights=weights,
                        seed=seed,
                        alpha=args.alpha,
                        trial=trial,
                        failures=failures,
                        require_pure_calib_for_ensemble=True,
                        fit_context="matched_global_on_local",
                    )

                    matched_rows = evaluate_methods_global(
                        methods_matched,
                        method_names_matched,
                        gs_matched,
                        X_main=X_main,
                        X_cos=X_cos,
                        X_text=X_text,
                        alpha=args.alpha,
                        tie_mode=args.tie_mode,
                        tau_guardrail=args.tau_guardrail,
                        tau_guardrail_delta=args.tau_guardrail_delta,
                        trial=trial,
                        seed=seed,
                        region_key=args.region_key,
                        region_id=region_id,
                        failures=failures,
                    )
                    for mr in matched_rows:
                        mr["tau_mode"] = "global_on_local_regions"
                        mr["comparison_scope"] = "local_tested_regions"
                        mr["tested_regions"] = int(len(tested_region_ids))
                    matched_global_trial_rows.extend(matched_rows)

                    local_by_method = {str(r.get("method")): r for r in trial_rows}
                    global_by_method = {str(r.get("method")): r for r in matched_rows}
                    trial_comp_rows: List[Dict[str, Any]] = []
                    for m in sorted(set(local_by_method.keys()) & set(global_by_method.keys())):
                        lr = local_by_method[m]
                        gr = global_by_method[m]
                        comp = {
                            "trial": int(trial),
                            "seed": int(seed),
                            "method": m,
                            "tested_regions": int(len(tested_region_ids)),
                            "local_micro_tpr": float(lr.get("micro_tpr", float("nan"))),
                            "local_micro_fpr": float(lr.get("micro_fpr", float("nan"))),
                            "local_macro_tpr": float(lr.get("macro_tpr", float("nan"))),
                            "local_macro_fpr": float(lr.get("macro_fpr", float("nan"))),
                            "global_micro_tpr": float(gr.get("micro_tpr", float("nan"))),
                            "global_micro_fpr": float(gr.get("micro_fpr", float("nan"))),
                            "global_macro_tpr": float(gr.get("macro_tpr", float("nan"))),
                            "global_macro_fpr": float(gr.get("macro_fpr", float("nan"))),
                            "delta_micro_tpr": float(gr.get("micro_tpr", float("nan")) - lr.get("micro_tpr", float("nan"))),
                            "delta_micro_fpr": float(gr.get("micro_fpr", float("nan")) - lr.get("micro_fpr", float("nan"))),
                        }
                        local_vs_global_matched_rows.append(comp)
                        trial_comp_rows.append(comp)
                    _print_matched_comparison(trial_comp_rows)
                else:
                    failures["matched_global_eval"].append(
                        f"trial={trial}: matched global skipped due to empty calib/eval pools on tested regions"
                    )
            else:
                failures["matched_global_eval"].append(
                    f"trial={trial}: matched global skipped because tested_region_ids is empty"
                )

            if run_ocats_baselines:
                ocats_rids = tested_region_ids if tested_region_ids else [int(s.rid) for s in splits]
                gs_ocats = _build_global_split_from_local_regions(splits, ocats_rids)

                if (
                    gs_ocats.H0_train.size > 0
                    and gs_ocats.H1_train.size > 0
                    and gs_ocats.H0_eval.size > 0
                    and gs_ocats.H1_eval.size > 0
                ):
                    train_idx_local = np.concatenate([gs_ocats.H0_train, gs_ocats.H1_train])
                    calib_idx_local = np.concatenate([gs_ocats.H0_calib, gs_ocats.H1_calib])
                    eval_idx_local = np.concatenate([gs_ocats.H0_eval, gs_ocats.H1_eval])

                    scope = "local_tested_regions" if tested_region_ids else "local_eligible_regions"
                    oc_out = _run_ocats_for_split(
                        X_main=X_main,
                        y=y,
                        train_idx=train_idx_local,
                        calib_idx=calib_idx_local,
                        eval_idx=eval_idx_local,
                        trial=trial,
                        seed=seed,
                        region_key=args.region_key,
                        tau_mode=args.tau_mode,
                        comparison_scope=scope,
                        args=args,
                        selected_methods=selected_ocats_methods,
                        lambda_values=lambda_values,
                    )
                    ocats_trial_rows.extend(oc_out["trial_rows"])
                    ocats_tuning_rows.extend(oc_out["tuning_rows"])
                    ocats_curve_rows.extend(oc_out["curve_rows"])
                    if oc_out["trial_rows"]:
                        print(f"  ocats_methods_evaluated={len(oc_out['trial_rows'])}")
                        if include_ocats_in_comparison:
                            shared_regions_local = int(
                                trial_rows[0].get("shared_regions", trial_rows[0].get("ok_regions", len(ocats_rids)))
                            ) if trial_rows else int(max(1, len(ocats_rids)))
                            dropped_regions_local = int(
                                trial_rows[0].get("dropped_regions_for_comparability", 0)
                            ) if trial_rows else 0
                            trial_rows.extend(
                                _build_ocats_comparison_rows(
                                    oc_out["trial_rows"],
                                    shared_regions=shared_regions_local,
                                    dropped_regions_for_comparability=dropped_regions_local,
                                    alpha=float(args.alpha),
                                )
                            )
                else:
                    failures["ocats_eval"].append(
                        f"trial={trial}: OCATS skipped due to empty pooled train/eval in scope={('local_tested_regions' if tested_region_ids else 'local_eligible_regions')}"
                    )

        trial_summary_rows.extend(trial_rows)
        if trial_rows and args.tau_mode != "global":
            shared_counts = {int(r.get("shared_regions", r.get("ok_regions", 0))) for r in trial_rows}
            if len(shared_counts) != 1:
                raise RuntimeError(
                    f"trial={trial}: comparability validation failed (inconsistent shared region counts): {sorted(shared_counts)}"
                )
            if next(iter(shared_counts)) <= 0:
                raise RuntimeError(f"trial={trial}: comparability validation failed (no shared regions)")
        print_trial_table(trial_rows, alpha=float(args.alpha))

    # Aggregate ranking
    ranking = aggregate_ranking(trial_summary_rows)
    constrained = print_ranking(ranking, alpha=float(args.alpha))

    # Save outputs
    save_csv_rows(
        run_dir / "trial_summary.csv",
        trial_summary_rows,
        fieldnames=[
            "trial", "seed", "method", "region_key", "tau_mode", "tau", "tau_mean",
            "micro_tpr", "micro_fpr", "train_tpr", "train_fpr",
            "macro_tpr", "macro_fpr",
            "ok_regions", "shared_regions", "dropped_regions_for_comparability",
            "input_space", "time_ms",
        ],
    )
    save_json(run_dir / "ranking.json", {"ranking": ranking})
    if matched_global_trial_rows:
        save_csv_rows(
            run_dir / "matched_global_trial_summary.csv",
            matched_global_trial_rows,
            fieldnames=[
                "trial", "seed", "method", "region_key", "tau_mode", "comparison_scope",
                "tau", "tau_mean", "micro_tpr", "micro_fpr", "train_tpr", "train_fpr",
                "macro_tpr", "macro_fpr", "ok_regions", "tested_regions", "input_space", "time_ms",
            ],
        )
    if local_vs_global_matched_rows:
        save_csv_rows(
            run_dir / "local_vs_global_matched.csv",
            local_vs_global_matched_rows,
            fieldnames=[
                "trial", "seed", "method", "tested_regions",
                "local_micro_tpr", "local_micro_fpr", "local_macro_tpr", "local_macro_fpr",
                "global_micro_tpr", "global_micro_fpr", "global_macro_tpr", "global_macro_fpr",
                "delta_micro_tpr", "delta_micro_fpr",
            ],
        )
        save_json(
            run_dir / "local_vs_global_matched.json",
            {"rows": local_vs_global_matched_rows},
        )
    if local_region_status_rows:
        save_csv_rows(
            run_dir / "local_region_status.csv",
            local_region_status_rows,
            fieldnames=["trial", "seed", "rid", "status", "reason", "is_tested"],
        )
        save_json(
            run_dir / "local_region_status.json",
            {"rows": local_region_status_rows},
        )
    if shrink_local_region_rows:
        save_csv_rows(
            run_dir / "shrink_local_region_thresholds.csv",
            shrink_local_region_rows,
            fieldnames=[
                "trial", "seed", "method", "rid",
                "n_calib_h0", "lambda_global",
                "tau_local", "tau_global", "tau_shrink",
            ],
        )
        save_json(
            run_dir / "shrink_local_region_thresholds.json",
            {"rows": shrink_local_region_rows},
        )
    if cluster_local_region_rows:
        save_csv_rows(
            run_dir / "cluster_local_region_thresholds.csv",
            cluster_local_region_rows,
            fieldnames=[
                "trial", "seed", "method", "rid", "cluster_id",
                "n_calib_h0_region", "n_calib_h0_cluster", "tau_cluster",
            ],
        )
        save_json(
            run_dir / "cluster_local_region_thresholds.json",
            {"rows": cluster_local_region_rows},
        )
    if online_stopping_history_rows:
        save_csv_rows(
            run_dir / "online_stopping_history.csv",
            online_stopping_history_rows,
            fieldnames=[
                "trial", "seed", "checkpoint",
                "tpr_monitor", "fpr_monitor", "tau",
                "slope_tpr", "slope_fpr", "slope_tau",
                "condition_passed", "stop_streak", "should_stop",
            ],
        )
    if online_stopping_summary_rows:
        save_json(
            run_dir / "online_stopping_summary.json",
            {"rows": online_stopping_summary_rows},
        )
    if ocats_trial_rows:
        save_csv_rows(
            run_dir / "ocats_trial_summary.csv",
            ocats_trial_rows,
            fieldnames=[
                "trial", "seed", "method", "lambda", "region_key", "tau_mode", "comparison_scope",
                "n_stream", "accuracy", "calls", "call_rate", "discounted_score",
                "tpr", "fpr", "precision", "recall",
                "e_thresh", "d_thresh", "cache_k", "online_retrains", "tune_ocats",
            ],
        )
        save_json(
            run_dir / "ocats_trial_summary.json",
            {"rows": ocats_trial_rows},
        )
    if ocats_tuning_rows:
        save_csv_rows(
            run_dir / "ocats_tuning.csv",
            ocats_tuning_rows,
            fieldnames=[
                "trial", "seed", "method", "lambda", "region_key", "tau_mode", "comparison_scope",
                "e_thresh", "d_thresh", "accuracy", "calls", "call_rate", "discounted_score",
            ],
        )
        save_json(
            run_dir / "ocats_tuning.json",
            {"rows": ocats_tuning_rows},
        )
    if ocats_curve_rows:
        save_csv_rows(
            run_dir / "ocats_curve.csv",
            ocats_curve_rows,
            fieldnames=[
                "trial", "seed", "method", "lambda", "region_key", "tau_mode", "comparison_scope",
                "step", "cum_accuracy", "teacher_calls", "call_rate",
                "entropy", "is_near", "used_teacher", "pred", "label",
                "e_thresh", "d_thresh",
            ],
        )
    save_csv_rows(
        run_dir / "weighted_ensemble_meta_weights.csv",
        weighted_ensemble_meta_rows,
        fieldnames=["trial", "seed", "judge", "weight"],
    )
    save_json(
        run_dir / "notes.json",
        {
            "experiment": "region_local_threshold",
            "data": str(npz_path),
            "region_key": args.region_key,
            "region_key_resolved": str(region_key),
            "features": feat_key,
            "alpha": float(args.alpha),
            "tie_mode": args.tie_mode,
            "tau_mode": args.tau_mode,
            "local_fit_mode": args.local_fit_mode,
            "method_flag": args.method,
            "tau_cluster_k": int(args.tau_cluster_k),
            "hadamard_preprocess": bool(args.hadamard_preprocess),
            "abs_diff_only_preprocess": bool(abs_diff_only),
            "use_delta_vec": bool(args.use_delta_vec),
            "use_abs_diff": bool(args.use_abs_diff),
            "hadamard_anchor_strategy": args.hadamard_anchor_strategy,
            "hadamard_cosine_direct": bool(args.hadamard_preprocess),
            "normalize_data": bool(args.normalize_data),
            "tau_shrink": bool(args.tau_shrink),
            "tau_shrink_m": float(args.tau_shrink_m),
            "shrink_k": float(args.shrink_k),
            "tau_guardrail": args.tau_guardrail,
            "tau_guardrail_delta": float(args.tau_guardrail_delta),
            "filter_policy": args.filter_policy,
            "ambiguous_cos_min": float(args.ambiguous_cos_min),
            "ambiguous_cos_max": float(args.ambiguous_cos_max),
            "cos_affine_calib": bool(args.cos_affine_calib),
            "precomputed_cosine": bool(args.precomputed_cosine),
            "cos_affine_grouping": args.cos_affine_grouping,
            "cos_affine_n_clusters": int(args.cos_affine_n_clusters),
            "regional_weighted_ensemble": bool(args.regional_weighted_ensemble),
            "rwe_k_shrink": float(args.rwe_k_shrink),
            "rwe_min_region_h0": int(args.rwe_min_region_h0),
            "rwe_min_region_h1": int(args.rwe_min_region_h1),
            "sem_bucket_k": int(args.sem_bucket_k),
            "sem_bucket_source_key": str(args.sem_bucket_source_key),
            "sem_bucket_source_key_used": sem_bucket_source_key_used,
            "swc_mode": args.swc_mode,
            "swc_cluster_n_clusters": int(args.swc_cluster_n_clusters),
            "n_trials": int(args.n_trials),
            "seed": int(args.seed),
            "caps": {
                "n_train": int(args.n_train),
                "n_calib": int(args.n_calib),
                "n_eval": int(args.n_eval),
            },
            "mins": {
                "min_h0_eval": int(args.min_h0_eval),
                "min_h1_eval": int(args.min_h1_eval),
            },
            "methods": configured_methods_last,
            "methods_configured": configured_methods_last,
            "method_input_spaces": {
                str(r.get("method")): str(r.get("input_space", "unknown"))
                for r in trial_summary_rows
                if r.get("method")
            },
            "methods_evaluated": sorted({str(r.get("method", "")) for r in trial_summary_rows if r.get("method")}),
            "x_cos_used": bool(
                X_cos is not None and any(str(r.get("input_space", "")) in {"scalar_score", "mixed"} for r in trial_summary_rows)
            ),
            "final_shared_region_count": int(
                min(
                    [int(r.get("shared_regions", r.get("ok_regions", 0))) for r in trial_summary_rows]
                ) if trial_summary_rows else 0
            ),
            "regions_dropped_for_comparability": int(
                max(
                    [int(r.get("dropped_regions_for_comparability", 0)) for r in trial_summary_rows]
                ) if trial_summary_rows else 0
            ),
            "xgboost_available": bool(has_xgb),
            "cosine_feature_available": bool(X_cos is not None),
            "text_pair_feature_available": bool(X_text is not None),
            "text_pair_source": text_key,
            "enable_bge_reranker": bool(args.enable_bge_reranker),
            "bge_model_name": args.bge_model_name,
            "bge_batch_size": int(args.bge_batch_size),
            "bge_max_length": int(args.bge_max_length),
            "bge_backend": args.bge_backend,
            "bge_normalize_scores": bool(args.bge_normalize_scores),
            "text_pair_keys": args.text_pair_keys,
            "text_source_pkl": args.text_source_pkl,
            "weighted_ensemble_meta_weights_logged": bool(len(weighted_ensemble_meta_rows) > 0),
            "ocats": {
                "enabled": bool(run_ocats_baselines),
                "ocats_only_mode": bool(ocats_only_mode),
                "include_in_comparison": bool(include_ocats_in_comparison),
                "selected_methods": selected_ocats_methods,
                "tune_ocats": bool(args.tune_ocats),
                "lambdas": lambda_values,
                "cache_k": int(args.cache_k),
                "e_thresh": float(args.e_thresh),
                "d_thresh": float(args.d_thresh),
                "e_thresh_grid": _parse_float_csv(args.e_thresh_grid),
                "d_thresh_grid": _parse_float_csv(args.d_thresh_grid),
                "knn_weight_power": float(args.knn_weight_power),
                "mlp_hidden_dim": int(args.mlp_hidden_dim),
                "mlp_dropout": float(args.mlp_dropout),
                "mlp_lr": float(args.mlp_lr),
                "mlp_epochs": int(args.mlp_epochs),
                "mlp_batch_size": int(args.mlp_batch_size),
                "mlp_weight_decay": float(args.mlp_weight_decay),
                "online_retrain_interval": int(args.online_retrain_interval),
                "online_retrain_last_p": int(args.online_retrain_last_p),
                "record_curve": bool(args.ocats_record_curve),
                "progress_every": int(args.ocats_progress_every),
                "trial_rows": int(len(ocats_trial_rows)),
                "tuning_rows": int(len(ocats_tuning_rows)),
                "curve_rows": int(len(ocats_curve_rows)),
            },
            "matched_global_eval_enabled": bool(args.tau_mode == "local"),
            "matched_global_trial_rows": int(len(matched_global_trial_rows)),
            "local_vs_global_matched_rows": int(len(local_vs_global_matched_rows)),
            "local_region_status_rows": int(len(local_region_status_rows)),
            "shrink_local_region_rows": int(len(shrink_local_region_rows)),
            "cluster_local_region_rows": int(len(cluster_local_region_rows)),
            "online_stopping": {
                "enabled": bool(args.enable_online_stopping),
                "stop_check_every": int(args.stop_check_every),
                "stop_window": int(args.stop_window),
                "stop_patience": int(args.stop_patience),
                "stop_eps_tpr": float(args.stop_eps_tpr),
                "stop_eps_fpr": float(args.stop_eps_fpr),
                "stop_eps_tau": float(args.stop_eps_tau),
                "stop_fpr_margin": float(args.stop_fpr_margin),
                "n_monitor_h0": int(args.n_monitor_h0),
                "n_monitor_h1": int(args.n_monitor_h1),
                "online_batch_size": int(args.online_batch_size),
                "online_mem_cap": int(args.online_mem_cap),
                "online_update_mode": args.online_update_mode,
                "online_hill_lr": float(args.online_hill_lr),
                "online_init_h0": int(args.online_init_h0),
                "online_init_h1": int(args.online_init_h1),
                "history_rows": int(len(online_stopping_history_rows)),
                "summary_rows": int(len(online_stopping_summary_rows)),
            },
            "failures": failures,
        },
    )

    if constrained:
        print(f"\nBest (constrained FPR <= alpha): {constrained[0]['method']}")
    elif ranking:
        print(f"\nBest (unconstrained fallback; none met FPR <= alpha): {ranking[0]['method']}")
    print(f"[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
