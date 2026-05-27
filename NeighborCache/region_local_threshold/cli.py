"""CLI entry point: argument parsing and main experiment orchestration."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
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
from .io_helpers import resolve_text_pairs, resolve_train_pairwise_cosine
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
from NeighborCache.optimization.optuna_search import (
    apply_params_to_namespace,
    run_optuna_search,
    split_global_eval_for_validation,
    split_region_eval_for_validation,
)

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


def _class1_ratio(n0: int, n1: int) -> float:
    total = int(n0) + int(n1)
    if total <= 0:
        return float("nan")
    return float(int(n1) / total)


def _repeat_train_indices_by_hardness(
    idx: np.ndarray,
    *,
    pair_cosine: np.ndarray,
    for_h0: bool,
    gamma: float,
    extra_repeats: int,
) -> tuple[np.ndarray, Dict[str, float]]:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx, {
            "mean_hardness": float("nan"),
            "mean_repeat": float("nan"),
        }

    cos = np.asarray(pair_cosine[idx], dtype=np.float32).reshape(-1)
    cos01 = np.clip((cos + 1.0) * 0.5, 0.0, 1.0)
    hardness = cos01 if for_h0 else (1.0 - cos01)

    scaled = np.power(hardness.astype(np.float64), float(gamma))
    repeats = 1 + np.rint(float(extra_repeats) * scaled).astype(np.int64)
    repeats = np.maximum(repeats, 1)

    out = np.repeat(idx, repeats)
    return out.astype(np.int64, copy=False), {
        "mean_hardness": float(np.mean(hardness)),
        "mean_repeat": float(np.mean(repeats)),
    }


def _apply_train_cosine_policy_to_indices(
    h0_idx: np.ndarray,
    h1_idx: np.ndarray,
    *,
    pair_cosine: np.ndarray,
    cosine_min: Optional[float],
    cosine_max: Optional[float],
    use_hardness_weighting: bool,
    hardness_gamma: float,
    hardness_extra_repeats: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    h0 = np.asarray(h0_idx, dtype=np.int64).reshape(-1)
    h1 = np.asarray(h1_idx, dtype=np.int64).reshape(-1)

    cos = np.asarray(pair_cosine, dtype=np.float32).reshape(-1)
    all_idx = _concat_indices([h0, h1])
    if all_idx.size > 0 and int(np.max(all_idx)) >= cos.shape[0]:
        raise ValueError(
            "train cosine policy index out of range: "
            f"max_train_idx={int(np.max(all_idx))} pair_cos_rows={int(cos.shape[0])}"
        )

    band_active = cosine_min is not None or cosine_max is not None
    if use_hardness_weighting and band_active:
        raise ValueError(
            "Use either train cosine-band filtering or hardness weighting, not both simultaneously."
        )

    cmin = -1.0 if cosine_min is None else float(cosine_min)
    cmax = 1.0 if cosine_max is None else float(cosine_max)
    if cmin > cmax:
        raise ValueError(f"Invalid cosine band: min={cmin} > max={cmax}")

    h0_kept = h0
    h1_kept = h1
    if band_active:
        keep0 = (cos[h0] >= cmin) & (cos[h0] <= cmax) if h0.size > 0 else np.zeros(0, dtype=bool)
        keep1 = (cos[h1] >= cmin) & (cos[h1] <= cmax) if h1.size > 0 else np.zeros(0, dtype=bool)
        h0_kept = h0[keep0]
        h1_kept = h1[keep1]

    h0_stats = {"mean_hardness": float("nan"), "mean_repeat": float("nan")}
    h1_stats = {"mean_hardness": float("nan"), "mean_repeat": float("nan")}
    if use_hardness_weighting:
        h0_out, h0_stats = _repeat_train_indices_by_hardness(
            h0_kept,
            pair_cosine=cos,
            for_h0=True,
            gamma=hardness_gamma,
            extra_repeats=hardness_extra_repeats,
        )
        h1_out, h1_stats = _repeat_train_indices_by_hardness(
            h1_kept,
            pair_cosine=cos,
            for_h0=False,
            gamma=hardness_gamma,
            extra_repeats=hardness_extra_repeats,
        )
        mode = "hardness_weighting"
    else:
        h0_out = h0_kept
        h1_out = h1_kept
        mode = "band_filter" if band_active else "none"

    h0_before = int(h0.size)
    h1_before = int(h1.size)
    h0_kept_n = int(h0_kept.size)
    h1_kept_n = int(h1_kept.size)
    h0_eff_n = int(h0_out.size)
    h1_eff_n = int(h1_out.size)

    stats: Dict[str, Any] = {
        "mode": mode,
        "cosine_min": float(cmin) if band_active else None,
        "cosine_max": float(cmax) if band_active else None,
        "h0_before": h0_before,
        "h1_before": h1_before,
        "h0_kept_unique": h0_kept_n,
        "h1_kept_unique": h1_kept_n,
        "h0_after_effective": h0_eff_n,
        "h1_after_effective": h1_eff_n,
        "class1_ratio_before": _class1_ratio(h0_before, h1_before),
        "class1_ratio_kept": _class1_ratio(h0_kept_n, h1_kept_n),
        "class1_ratio_effective": _class1_ratio(h0_eff_n, h1_eff_n),
        "h0_keep_rate": float(h0_kept_n / h0_before) if h0_before > 0 else float("nan"),
        "h1_keep_rate": float(h1_kept_n / h1_before) if h1_before > 0 else float("nan"),
        "h0_mean_hardness": float(h0_stats["mean_hardness"]),
        "h1_mean_hardness": float(h1_stats["mean_hardness"]),
        "h0_mean_repeat": float(h0_stats["mean_repeat"]),
        "h1_mean_repeat": float(h1_stats["mean_repeat"]),
        "hardness_gamma": float(hardness_gamma) if use_hardness_weighting else None,
        "hardness_extra_repeats": int(hardness_extra_repeats) if use_hardness_weighting else None,
    }
    return h0_out.astype(np.int64, copy=False), h1_out.astype(np.int64, copy=False), stats


def _apply_train_cosine_policy_global(
    gs: GlobalSplit,
    *,
    pair_cosine: np.ndarray,
    cosine_min: Optional[float],
    cosine_max: Optional[float],
    use_hardness_weighting: bool,
    hardness_gamma: float,
    hardness_extra_repeats: int,
) -> tuple[GlobalSplit, Dict[str, Any]]:
    h0_train, h1_train, stats = _apply_train_cosine_policy_to_indices(
        gs.H0_train,
        gs.H1_train,
        pair_cosine=pair_cosine,
        cosine_min=cosine_min,
        cosine_max=cosine_max,
        use_hardness_weighting=use_hardness_weighting,
        hardness_gamma=hardness_gamma,
        hardness_extra_repeats=hardness_extra_repeats,
    )
    out = GlobalSplit(
        H0_train=h0_train,
        H1_train=h1_train,
        H0_calib=gs.H0_calib,
        H1_calib=gs.H1_calib,
        H0_eval=gs.H0_eval,
        H1_eval=gs.H1_eval,
    )
    return out, stats


def _apply_train_cosine_policy_regions(
    splits: List[Any],
    *,
    pair_cosine: np.ndarray,
    cosine_min: Optional[float],
    cosine_max: Optional[float],
    use_hardness_weighting: bool,
    hardness_gamma: float,
    hardness_extra_repeats: int,
) -> tuple[List[Any], Dict[str, Any]]:
    out: List[Any] = []
    agg = {
        "h0_before": 0,
        "h1_before": 0,
        "h0_kept_unique": 0,
        "h1_kept_unique": 0,
        "h0_after_effective": 0,
        "h1_after_effective": 0,
    }

    h0_hard_num = 0.0
    h0_hard_den = 0
    h1_hard_num = 0.0
    h1_hard_den = 0
    h0_repeat_num = 0.0
    h0_repeat_den = 0
    h1_repeat_num = 0.0
    h1_repeat_den = 0

    for s in splits:
        h0_train_new, h1_train_new, stats = _apply_train_cosine_policy_to_indices(
            s.H0_train,
            s.H1_train,
            pair_cosine=pair_cosine,
            cosine_min=cosine_min,
            cosine_max=cosine_max,
            use_hardness_weighting=use_hardness_weighting,
            hardness_gamma=hardness_gamma,
            hardness_extra_repeats=hardness_extra_repeats,
        )

        agg["h0_before"] += int(stats["h0_before"])
        agg["h1_before"] += int(stats["h1_before"])
        agg["h0_kept_unique"] += int(stats["h0_kept_unique"])
        agg["h1_kept_unique"] += int(stats["h1_kept_unique"])
        agg["h0_after_effective"] += int(stats["h0_after_effective"])
        agg["h1_after_effective"] += int(stats["h1_after_effective"])

        n0 = int(stats["h0_kept_unique"])
        n1 = int(stats["h1_kept_unique"])
        if np.isfinite(float(stats["h0_mean_hardness"])) and n0 > 0:
            h0_hard_num += float(stats["h0_mean_hardness"]) * n0
            h0_hard_den += n0
        if np.isfinite(float(stats["h1_mean_hardness"])) and n1 > 0:
            h1_hard_num += float(stats["h1_mean_hardness"]) * n1
            h1_hard_den += n1
        if np.isfinite(float(stats["h0_mean_repeat"])) and n0 > 0:
            h0_repeat_num += float(stats["h0_mean_repeat"]) * n0
            h0_repeat_den += n0
        if np.isfinite(float(stats["h1_mean_repeat"])) and n1 > 0:
            h1_repeat_num += float(stats["h1_mean_repeat"]) * n1
            h1_repeat_den += n1

        out.append(
            type(s)(
                rid=int(s.rid),
                H0_train=h0_train_new,
                H1_train=h1_train_new,
                H0_calib=s.H0_calib,
                H1_calib=s.H1_calib,
                H0_eval=s.H0_eval,
                H1_eval=s.H1_eval,
            )
        )

    band_active = cosine_min is not None or cosine_max is not None
    mode = "hardness_weighting" if use_hardness_weighting else ("band_filter" if band_active else "none")
    stats_out: Dict[str, Any] = {
        "mode": mode,
        "cosine_min": float(cosine_min) if cosine_min is not None else (None if not band_active else -1.0),
        "cosine_max": float(cosine_max) if cosine_max is not None else (None if not band_active else 1.0),
        "h0_before": int(agg["h0_before"]),
        "h1_before": int(agg["h1_before"]),
        "h0_kept_unique": int(agg["h0_kept_unique"]),
        "h1_kept_unique": int(agg["h1_kept_unique"]),
        "h0_after_effective": int(agg["h0_after_effective"]),
        "h1_after_effective": int(agg["h1_after_effective"]),
        "class1_ratio_before": _class1_ratio(int(agg["h0_before"]), int(agg["h1_before"])),
        "class1_ratio_kept": _class1_ratio(int(agg["h0_kept_unique"]), int(agg["h1_kept_unique"])),
        "class1_ratio_effective": _class1_ratio(int(agg["h0_after_effective"]), int(agg["h1_after_effective"])),
        "h0_keep_rate": float(agg["h0_kept_unique"] / agg["h0_before"]) if agg["h0_before"] > 0 else float("nan"),
        "h1_keep_rate": float(agg["h1_kept_unique"] / agg["h1_before"]) if agg["h1_before"] > 0 else float("nan"),
        "h0_mean_hardness": (h0_hard_num / h0_hard_den) if h0_hard_den > 0 else float("nan"),
        "h1_mean_hardness": (h1_hard_num / h1_hard_den) if h1_hard_den > 0 else float("nan"),
        "h0_mean_repeat": (h0_repeat_num / h0_repeat_den) if h0_repeat_den > 0 else float("nan"),
        "h1_mean_repeat": (h1_repeat_num / h1_repeat_den) if h1_repeat_den > 0 else float("nan"),
        "hardness_gamma": float(hardness_gamma) if use_hardness_weighting else None,
        "hardness_extra_repeats": int(hardness_extra_repeats) if use_hardness_weighting else None,
    }
    return out, stats_out


def _log_train_cosine_policy_stats(stats: Dict[str, Any]) -> None:
    mode = str(stats.get("mode", "none"))
    if mode == "none":
        return

    print(
        "  train_cosine_policy: "
        f"mode={mode} "
        f"kept_unique(n0={int(stats.get('h0_kept_unique', 0))}, n1={int(stats.get('h1_kept_unique', 0))}, "
        f"h1_ratio={float(stats.get('class1_ratio_kept', float('nan'))):.4f}) "
        f"effective_train(n0={int(stats.get('h0_after_effective', 0))}, n1={int(stats.get('h1_after_effective', 0))}, "
        f"h1_ratio={float(stats.get('class1_ratio_effective', float('nan'))):.4f})"
    )

    if mode == "band_filter":
        print(
            "    cosine_band: "
            f"[{float(stats.get('cosine_min', -1.0)):.4f}, {float(stats.get('cosine_max', 1.0)):.4f}] "
            f"keep_rate(n0={float(stats.get('h0_keep_rate', float('nan'))):.4f}, "
            f"n1={float(stats.get('h1_keep_rate', float('nan'))):.4f})"
        )

    if mode == "hardness_weighting":
        print(
            "    hardness_weighting: "
            f"gamma={float(stats.get('hardness_gamma', 1.0)):.3f} "
            f"extra_repeats={int(stats.get('hardness_extra_repeats', 0))} "
            f"mean_repeat(n0={float(stats.get('h0_mean_repeat', float('nan'))):.3f}, "
            f"n1={float(stats.get('h1_mean_repeat', float('nan'))):.3f})"
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


def _tiny_mlp_hidden_layers(args: argparse.Namespace) -> tuple[int, ...]:
    hidden_dim = int(max(1, getattr(args, "tiny_mlp_hidden_dim", 16)))
    n_layers = int(max(1, getattr(args, "tiny_mlp_n_layers", 1)))
    return tuple(hidden_dim for _ in range(n_layers))


def _ensemble_config_from_args(args: argparse.Namespace) -> Any:
    from np_bench.methods.weighted_ensemble import EnsembleConfig

    return EnsembleConfig(
        alpha=float(args.alpha),
        ridge=float(getattr(args, "ensemble_ridge", 1e-3)),
        standardize=bool(getattr(args, "ensemble_standardize", False)),
        nonneg_simplex=bool(getattr(args, "ensemble_nonneg_simplex", True)),
        meta_frac=float(getattr(args, "ensemble_meta_frac", 0.30)),
        tpr_tie_tol=float(getattr(args, "ensemble_tpr_tie_tol", 1e-4)),
        tie_break_entropy=bool(getattr(args, "ensemble_tie_break_entropy", True)),
    )


def _build_ensemble_judges(args: argparse.Namespace, *, use_precomputed_cosine: bool, has_xgb: bool) -> List[Any]:
    from np_bench.methods.cosine import CosineMethod
    from np_bench.methods.lda import LDAMethod
    from np_bench.methods.tiny_mlp import TinyMLPMethod

    judges: List[Any] = []
    if use_precomputed_cosine:
        from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
        judges.append(PrecomputedCosineMethod())
    else:
        judges.append(CosineMethod())

    # try:
    #     from np_bench.methods.whitened_cosine import WhitenedCosineMethod
    #     judges.append(
    #         WhitenedCosineMethod(
    #             abs_eps=float(getattr(args, "pca_whiten_abs_eps", 1e-6)),
    #             rel_eps=float(getattr(args, "pca_whiten_rel_eps", 1e-6)),
    #             max_rank=getattr(args, "pca_whiten_max_rank", 128),
    #             rank_mode=str(getattr(args, "pca_whiten_rank_mode", "explained_variance")),
    #             explained_variance=float(getattr(args, "pca_whiten_explained_variance", 0.99)),
    #             norm_eps=float(getattr(args, "pca_whiten_norm_eps", 1e-12)),
    #         )
    #     )
    # except Exception:
    #     pass

    if has_xgb:
        try:
            from np_bench.methods.xgboost import XGBoostLightMethod
            judges.append(
                XGBoostLightMethod(
                    n_estimators=int(getattr(args, "xgb_n_estimators", 30)),
                    max_depth=int(getattr(args, "xgb_max_depth", 3)),
                    learning_rate=float(getattr(args, "xgb_learning_rate", 0.1)),
                    subsample=float(getattr(args, "xgb_subsample", 1.0)),
                    colsample_bytree=float(getattr(args, "xgb_colsample_bytree", 1.0)),
                    min_child_weight=float(getattr(args, "xgb_min_child_weight", 1.0)),
                    gamma=float(getattr(args, "xgb_gamma", 0.0)),
                    reg_alpha=float(getattr(args, "xgb_reg_alpha", 0.0)),
                    reg_lambda=float(getattr(args, "xgb_reg_lambda", 1.0)),
                )
            )
        except Exception:
            pass

    # judges.append(
    #     TinyMLPMethod(
    #         hidden_layer_sizes=_tiny_mlp_hidden_layers(args),
    #         activation=str(getattr(args, "tiny_mlp_activation", "relu")),
    #         alpha=float(getattr(args, "tiny_mlp_alpha", 0.001)),
    #         learning_rate_init=float(getattr(args, "tiny_mlp_learning_rate_init", 0.001)),
    #         max_iter=int(getattr(args, "tiny_mlp_max_iter", 800)),
    #         early_stopping=bool(getattr(args, "tiny_mlp_early_stopping", False)),
    #     )
    # )
    judges.append(
        LDAMethod(
            solver=str(getattr(args, "lda_solver", "lsqr")),
            shrinkage=getattr(args, "lda_shrinkage", "auto"),
            tol=float(getattr(args, "lda_tol", 1e-4)),
        )
    )
    return judges


def _build_configured_methods(
    args: argparse.Namespace,
    *,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
    quiet: bool = False,
) -> Dict[str, Any]:
    methods = build_methods()
    has_xgb = "XGBoost" in methods
    use_precomputed_cosine = bool(args.hadamard_preprocess and X_cos is not None)

    if use_precomputed_cosine:
        try:
            from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
            methods["Cosine"] = PrecomputedCosineMethod()
            if not quiet:
                print("[INFO] Cosine baseline routed to direct Hadamard sum (PrecomputedCosine)")
        except Exception as exc:
            if not quiet:
                print(f"[WARN] Could not route Cosine baseline to PrecomputedCosine: {exc}")

    try:
        from np_bench.methods.whitened_cosine import WhitenedCosineMethod
        methods["PCAWhitenedCosine"] = WhitenedCosineMethod(
            abs_eps=float(getattr(args, "pca_whiten_abs_eps", 1e-6)),
            rel_eps=float(getattr(args, "pca_whiten_rel_eps", 1e-6)),
            max_rank=getattr(args, "pca_whiten_max_rank", 128),
            rank_mode=str(getattr(args, "pca_whiten_rank_mode", "explained_variance")),
            explained_variance=float(getattr(args, "pca_whiten_explained_variance", 0.99)),
            norm_eps=float(getattr(args, "pca_whiten_norm_eps", 1e-12)),
        )
    except Exception:
        pass

    if has_xgb:
        try:
            from np_bench.methods.xgboost import XGBoostLightMethod
            methods["XGBoost"] = XGBoostLightMethod(
                n_estimators=int(getattr(args, "xgb_n_estimators", 30)),
                max_depth=int(getattr(args, "xgb_max_depth", 3)),
                learning_rate=float(getattr(args, "xgb_learning_rate", 0.1)),
                subsample=float(getattr(args, "xgb_subsample", 1.0)),
                colsample_bytree=float(getattr(args, "xgb_colsample_bytree", 1.0)),
                min_child_weight=float(getattr(args, "xgb_min_child_weight", 1.0)),
                gamma=float(getattr(args, "xgb_gamma", 0.0)),
                reg_alpha=float(getattr(args, "xgb_reg_alpha", 0.0)),
                reg_lambda=float(getattr(args, "xgb_reg_lambda", 1.0)),
            )
        except Exception as exc:
            if not quiet:
                print(f"[WARN] Could not configure XGBoost: {exc}")

    try:
        from np_bench.methods.tiny_mlp import TinyMLPMethod
        methods["Tiny MLP"] = TinyMLPMethod(
            hidden_layer_sizes=_tiny_mlp_hidden_layers(args),
            activation=str(getattr(args, "tiny_mlp_activation", "relu")),
            alpha=float(getattr(args, "tiny_mlp_alpha", 0.001)),
            learning_rate_init=float(getattr(args, "tiny_mlp_learning_rate_init", 0.001)),
            max_iter=int(getattr(args, "tiny_mlp_max_iter", 800)),
            early_stopping=bool(getattr(args, "tiny_mlp_early_stopping", False)),
        )
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not configure Tiny MLP: {exc}")

    try:
        from np_bench.methods.lda import LDAMethod
        methods["LDA"] = LDAMethod(
            solver=str(getattr(args, "lda_solver", "lsqr")),
            shrinkage=getattr(args, "lda_shrinkage", "auto"),
            tol=float(getattr(args, "lda_tol", 1e-4)),
        )
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not configure LDA: {exc}")

    try:
        from np_bench.methods.weighted_ensemble import WeightedEnsembleMethod
        methods["WeightedEnsemble"] = WeightedEnsembleMethod(
            judges=_build_ensemble_judges(
                args,
                use_precomputed_cosine=use_precomputed_cosine,
                has_xgb=has_xgb,
            ),
            config=_ensemble_config_from_args(args),
        )
        if use_precomputed_cosine and not quiet:
            print("[INFO] WeightedEnsemble Cosine judge routed to direct Hadamard sum")
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not configure WeightedEnsemble: {exc}")

    try:
        from np_bench.methods.stabilized_whitened_cosine import StabilizedWhitenedCosineMethod
        methods["StabilizedWhitenedCosine"] = StabilizedWhitenedCosineMethod(
            k=int(getattr(args, "swc_k", 64)),
            shrinkage=float(getattr(args, "swc_shrinkage", 0.1)),
            eps=float(getattr(args, "swc_eps", 1e-6)),
            min_samples=int(getattr(args, "swc_min_samples", 200)),
            fallback=bool(getattr(args, "swc_fallback", True)),
            verbose=bool(getattr(args, "swc_verbose", False)),
        )
    except Exception:
        pass

    if args.cos_affine_calib:
        try:
            from np_bench.methods.cosine_affine_calib import CosineAffineCalibMethod
            methods["CosineAffineCalib"] = CosineAffineCalibMethod()
        except Exception as exc:
            if not quiet:
                print(f"[WARN] Could not load CosineAffineCalib: {exc}")

    if args.precomputed_cosine:
        try:
            from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
            methods["PrecomputedCosine"] = PrecomputedCosineMethod()
        except Exception as exc:
            if not quiet:
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
                        ridge=float(getattr(args, "ensemble_ridge", 1e-3)),
                        standardize=bool(getattr(args, "ensemble_standardize", False)),
                        nonneg_simplex=bool(getattr(args, "ensemble_nonneg_simplex", True)),
                        meta_frac=float(getattr(args, "ensemble_meta_frac", 0.30)),
                        tpr_tie_tol=float(getattr(args, "ensemble_tpr_tie_tol", 1e-4)),
                        tie_break_entropy=bool(getattr(args, "ensemble_tie_break_entropy", True)),
                        k_shrink=float(args.rwe_k_shrink),
                        min_region_h0=int(args.rwe_min_region_h0),
                        min_region_h1=int(args.rwe_min_region_h1),
                    ),
                )
                if use_precomputed_cosine and not quiet:
                    print("[INFO] RegionalWeightedEnsemble Cosine judge routed to direct Hadamard sum")
            elif not quiet:
                print("[WARN] Could not initialize RegionalWeightedEnsemble: no base judges found")
        except Exception as exc:
            if not quiet:
                print(f"[WARN] Could not load RegionalWeightedEnsemble: {exc}")

    if args.enable_bge_reranker:
        if X_text is None:
            if not quiet:
                print("[WARN] --enable_bge_reranker set but no text pairs were resolved; skipping method")
        else:
            try:
                from np_bench.methods.bge_reranker import BGERerankerMethod
                methods["BGE Reranker"] = BGERerankerMethod(
                    model_name=args.bge_model_name,
                    batch_size=int(args.bge_batch_size),
                    max_length=int(args.bge_max_length),
                    normalize_scores=bool(args.bge_normalize_scores),
                    backend=args.bge_backend,
                )
            except Exception as exc:
                if not quiet:
                    print(f"[WARN] Could not load BGE Reranker method: {exc}")

    return methods


def _evaluate_global_candidate(
    candidate_args: argparse.Namespace,
    *,
    gs: GlobalSplit,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
    y: np.ndarray,
    region_id: np.ndarray,
    h0_monitor_idx: np.ndarray,
    h1_monitor_idx: np.ndarray,
    trial: int,
    seed: int,
    region_key: str,
    selected_ocats_methods: List[str],
    lambda_values: List[float],
    run_ocats_baselines: bool,
    include_ocats_in_comparison: bool,
    suppress_output: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    failures_local: Dict[str, List[str]] = defaultdict(list)

    h0_train_idx_fit = np.asarray(gs.H0_train, dtype=np.int64).reshape(-1)
    h1_train_idx_fit = np.asarray(gs.H1_train, dtype=np.int64).reshape(-1)
    h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
    h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

    H0_train = X_main[h0_train_idx_fit]
    H1_train = X_main[h1_train_idx_fit]
    H0_calib_pure = X_main[gs.H0_calib]
    H1_calib_pure = X_main[gs.H1_calib]
    H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
        if h0_train_idx_unique.size > 0 else X_main[gs.H0_calib]
    H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
        if h1_train_idx_unique.size > 0 else X_main[gs.H1_calib]

    H0_train_cos = X_cos[h0_train_idx_fit] if X_cos is not None else None
    H1_train_cos = X_cos[h1_train_idx_fit] if X_cos is not None else None
    if X_cos is not None:
        H0_calib_pure_cos = X_cos[gs.H0_calib]
        H1_calib_pure_cos = X_cos[gs.H1_calib]
        H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
            if h0_train_idx_unique.size > 0 else X_cos[gs.H0_calib]
        H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
            if h1_train_idx_unique.size > 0 else X_cos[gs.H1_calib]
    else:
        H0_calib_pure_cos = None
        H1_calib_pure_cos = None
        H0_calib_eff_cos = None
        H1_calib_eff_cos = None

    if X_text is not None:
        H0_train_text = X_text[h0_train_idx_fit]
        H1_train_text = X_text[h1_train_idx_fit]
        H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
            if h0_train_idx_unique.size > 0 else X_text[gs.H0_calib]
        H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
            if h1_train_idx_unique.size > 0 else X_text[gs.H1_calib]
    else:
        H0_train_text = None
        H1_train_text = None
        H0_calib_eff_text = None
        H1_calib_eff_text = None

    if H0_calib_eff.shape[0] == 0 or H1_calib_eff.shape[0] == 0:
        return [], {"reason": "empty_calib_eff"}

    v0 = np.var(H0_calib_eff, axis=0)
    v1 = np.var(H1_calib_eff, axis=0)
    weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)

    stream = io.StringIO()
    cm = contextlib.redirect_stdout(stream) if suppress_output else contextlib.nullcontext()
    with cm:
        methods = _build_configured_methods(candidate_args, X_cos=X_cos, X_text=X_text, quiet=True)
        method_names = list(methods.keys())

        online_method = None
        if candidate_args.enable_online_stopping:
            online_method, _, _ = _run_online_stopping(
                X_main,
                y,
                h0_train_idx=gs.H0_train,
                h1_train_idx=gs.H1_train,
                h0_calib_idx=gs.H0_calib,
                h0_monitor_idx=h0_monitor_idx,
                h1_monitor_idx=h1_monitor_idx,
                alpha=candidate_args.alpha,
                tie_mode=candidate_args.tie_mode,
                tau_guardrail=candidate_args.tau_guardrail,
                tau_guardrail_delta=candidate_args.tau_guardrail_delta,
                seed=seed,
                args=candidate_args,
            )
            if online_method is not None:
                methods["Online(refit)"] = online_method
                method_names.append("Online(refit)")

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
            H0_train_text=H0_train_text,
            H1_train_text=H1_train_text,
            H0_calib_eff_text=H0_calib_eff_text,
            H1_calib_eff_text=H1_calib_eff_text,
            H0_calib_pure_cos=H0_calib_pure_cos,
            H1_calib_pure_cos=H1_calib_pure_cos,
            H0_calib_region_ids=region_id[gs.H0_calib],
            H1_calib_region_ids=region_id[gs.H1_calib],
            tie_mode=candidate_args.tie_mode,
            tau_guardrail=candidate_args.tau_guardrail,
            tau_guardrail_delta=candidate_args.tau_guardrail_delta,
            weights=weights,
            seed=seed,
            alpha=candidate_args.alpha,
            trial=trial,
            failures=failures_local,
            fit_context="optuna_global_validation",
        )

        rows = evaluate_methods_global(
            methods,
            method_names,
            gs,
            X_main=X_main,
            X_cos=X_cos,
            X_text=X_text,
            alpha=candidate_args.alpha,
            tie_mode=candidate_args.tie_mode,
            tau_guardrail=candidate_args.tau_guardrail,
            tau_guardrail_delta=candidate_args.tau_guardrail_delta,
            trial=trial,
            seed=seed,
            region_key=region_key,
            region_id=region_id,
            failures=failures_local,
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
                region_key=region_key,
                tau_mode="global",
                comparison_scope="optuna_validation",
                args=candidate_args,
                selected_methods=selected_ocats_methods,
                lambda_values=lambda_values,
            )
            if include_ocats_in_comparison:
                rows.extend(
                    _build_ocats_comparison_rows(
                        oc_out["trial_rows"],
                        shared_regions=1,
                        dropped_regions_for_comparability=0,
                        alpha=float(candidate_args.alpha),
                    )
                )

    return rows, {"failures": sum(len(v) for v in failures_local.values())}


def _evaluate_local_candidate(
    candidate_args: argparse.Namespace,
    *,
    splits: List[RegionSplit],
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
    y: np.ndarray,
    region_id: np.ndarray,
    X_anchor_source: np.ndarray,
    h0_monitor_idx: np.ndarray,
    h1_monitor_idx: np.ndarray,
    trial: int,
    seed: int,
    region_key: str,
    selected_ocats_methods: List[str],
    lambda_values: List[float],
    run_ocats_baselines: bool,
    include_ocats_in_comparison: bool,
    suppress_output: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    failures_local: Dict[str, List[str]] = defaultdict(list)

    tau_cluster_id_by_rid: Optional[Dict[int, int]] = None
    if candidate_args.tau_mode == "cluster_local":
        tau_cluster_id_by_rid = _build_tau_cluster_map(
            splits=splits,
            X_anchor_source=X_anchor_source,
            region_id=region_id,
            anchor_strategy=candidate_args.hadamard_anchor_strategy,
            n_clusters=max(1, int(candidate_args.tau_cluster_k)),
            seed=seed,
        )
        if not tau_cluster_id_by_rid:
            return [], {"reason": "empty_tau_cluster_map"}

    h0_train_idx_list = [s.H0_train for s in splits if s.H0_train.size > 0]
    h1_train_idx_list = [s.H1_train for s in splits if s.H1_train.size > 0]
    h0_train_idx_fit = _concat_indices(h0_train_idx_list)
    h1_train_idx_fit = _concat_indices(h1_train_idx_list)
    h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
    h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

    H0_train = X_main[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_main[:0]
    H1_train = X_main[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_main[:0]

    h0_calib_list = [s.H0_calib for s in splits if s.H0_calib.size > 0]
    h1_calib_list = [s.H1_calib for s in splits if s.H1_calib.size > 0]
    h0_calib_idx = _concat_indices(h0_calib_list)
    h1_calib_idx = _concat_indices(h1_calib_list)
    H0_calib = X_main[h0_calib_idx] if h0_calib_idx.size > 0 else X_main[:0]
    H1_calib = X_main[h1_calib_idx] if h1_calib_idx.size > 0 else X_main[:0]
    H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
        if h0_train_idx_unique.size > 0 else H0_calib
    H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
        if h1_train_idx_unique.size > 0 else H1_calib

    if X_cos is not None:
        H0_train_cos = X_cos[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_cos[:0]
        H1_train_cos = X_cos[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_cos[:0]
        H0_calib_cos = X_cos[h0_calib_idx] if h0_calib_idx.size > 0 else X_cos[:0]
        H1_calib_cos = X_cos[h1_calib_idx] if h1_calib_idx.size > 0 else X_cos[:0]
        H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
            if h0_train_idx_unique.size > 0 else H0_calib_cos
        H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
            if h1_train_idx_unique.size > 0 else H1_calib_cos
    else:
        H0_train_cos = None
        H1_train_cos = None
        H0_calib_cos = None
        H1_calib_cos = None
        H0_calib_eff_cos = None
        H1_calib_eff_cos = None

    if X_text is not None:
        H0_train_text = X_text[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_text[:0]
        H1_train_text = X_text[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_text[:0]
        H0_calib_text = X_text[h0_calib_idx] if h0_calib_idx.size > 0 else X_text[:0]
        H1_calib_text = X_text[h1_calib_idx] if h1_calib_idx.size > 0 else X_text[:0]
        H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
            if h0_train_idx_unique.size > 0 else H0_calib_text
        H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
            if h1_train_idx_unique.size > 0 else H1_calib_text
    else:
        H0_train_text = None
        H1_train_text = None
        H0_calib_eff_text = None
        H1_calib_eff_text = None

    if H0_calib_eff.shape[0] == 0 or H1_calib_eff.shape[0] == 0:
        return [], {"reason": "empty_calib_eff"}

    v0 = np.var(H0_calib_eff, axis=0)
    v1 = np.var(H1_calib_eff, axis=0)
    weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)

    stream = io.StringIO()
    cm = contextlib.redirect_stdout(stream) if suppress_output else contextlib.nullcontext()
    with cm:
        methods = _build_configured_methods(candidate_args, X_cos=X_cos, X_text=X_text, quiet=True)
        method_names = list(methods.keys())

        online_method = None
        if candidate_args.enable_online_stopping:
            online_method, _, _ = _run_online_stopping(
                X_main,
                y,
                h0_train_idx=_concat_indices(h0_train_idx_list),
                h1_train_idx=_concat_indices(h1_train_idx_list),
                h0_calib_idx=_concat_indices(h0_calib_list),
                h0_monitor_idx=h0_monitor_idx,
                h1_monitor_idx=h1_monitor_idx,
                alpha=candidate_args.alpha,
                tie_mode=candidate_args.tie_mode,
                tau_guardrail=candidate_args.tau_guardrail,
                tau_guardrail_delta=candidate_args.tau_guardrail_delta,
                seed=seed,
                args=candidate_args,
            )
            if online_method is not None:
                methods["Online(refit)"] = online_method
                method_names.append("Online(refit)")

        if candidate_args.local_fit_mode == "pooled":
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
                tie_mode=candidate_args.tie_mode,
                tau_guardrail=candidate_args.tau_guardrail,
                tau_guardrail_delta=candidate_args.tau_guardrail_delta,
                weights=weights,
                seed=seed,
                alpha=candidate_args.alpha,
                trial=trial,
                failures=failures_local,
                require_pure_calib_for_ensemble=True,
                fit_context="optuna_local_validation",
            )

        local_eval_meta: Dict[str, Any] = {}
        rows = evaluate_methods(
            methods,
            method_names,
            splits,
            X_main=X_main,
            X_cos=X_cos,
            X_text=X_text,
            alpha=candidate_args.alpha,
            tau_mode=candidate_args.tau_mode,
            tie_mode=candidate_args.tie_mode,
            tau_shrink=bool(candidate_args.tau_shrink),
            tau_shrink_m=float(candidate_args.tau_shrink_m),
            shrink_k=float(candidate_args.shrink_k),
            tau_guardrail=candidate_args.tau_guardrail,
            tau_guardrail_delta=float(candidate_args.tau_guardrail_delta),
            swc_mode=candidate_args.swc_mode,
            swc_cluster_n_clusters=int(candidate_args.swc_cluster_n_clusters),
            cos_affine_grouping=candidate_args.cos_affine_grouping,
            cos_affine_n_clusters=int(candidate_args.cos_affine_n_clusters),
            local_fit_mode=candidate_args.local_fit_mode,
            trial=trial,
            seed=seed,
            region_key=region_key,
            h0_train_idx_list=h0_train_idx_list,
            h1_train_idx_list=h1_train_idx_list,
            h0_calib_list=h0_calib_list,
            h1_calib_list=h1_calib_list,
            H0_calib_eff=H0_calib_eff,
            failures=failures_local,
            tau_cluster_id_by_rid=tau_cluster_id_by_rid,
            trial_meta=local_eval_meta,
        )

        if run_ocats_baselines:
            tested_region_ids = [int(r) for r in local_eval_meta.get("tested_region_ids", [])]
            ocats_rids = tested_region_ids if tested_region_ids else [int(s.rid) for s in splits]
            gs_ocats = _build_global_split_from_local_regions(splits, ocats_rids)
            if (
                gs_ocats.H0_train.size > 0
                and gs_ocats.H1_train.size > 0
                and gs_ocats.H0_eval.size > 0
                and gs_ocats.H1_eval.size > 0
            ):
                oc_out = _run_ocats_for_split(
                    X_main=X_main,
                    y=y,
                    train_idx=np.concatenate([gs_ocats.H0_train, gs_ocats.H1_train]),
                    calib_idx=np.concatenate([gs_ocats.H0_calib, gs_ocats.H1_calib]),
                    eval_idx=np.concatenate([gs_ocats.H0_eval, gs_ocats.H1_eval]),
                    trial=trial,
                    seed=seed,
                    region_key=region_key,
                    tau_mode=candidate_args.tau_mode,
                    comparison_scope="optuna_validation",
                    args=candidate_args,
                    selected_methods=selected_ocats_methods,
                    lambda_values=lambda_values,
                )
                if include_ocats_in_comparison:
                    shared_regions = int(rows[0].get("shared_regions", rows[0].get("ok_regions", len(ocats_rids)))) if rows else int(max(1, len(ocats_rids)))
                    dropped_regions = int(rows[0].get("dropped_regions_for_comparability", 0)) if rows else 0
                    rows.extend(
                        _build_ocats_comparison_rows(
                            oc_out["trial_rows"],
                            shared_regions=shared_regions,
                            dropped_regions_for_comparability=dropped_regions,
                            alpha=float(candidate_args.alpha),
                        )
                    )

    return rows, {"failures": sum(len(v) for v in failures_local.values())}


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

    # Train-only cosine policy (from raw pair embeddings x/y)
    ap.add_argument(
        "--train_cosine_min",
        type=float,
        default=None,
        help="Optional train-only lower bound on raw pairwise cosine(x,y).",
    )
    ap.add_argument(
        "--train_cosine_max",
        type=float,
        default=None,
        help="Optional train-only upper bound on raw pairwise cosine(x,y).",
    )
    ap.add_argument(
        "--train_hardness_weighting",
        action="store_true",
        default=False,
        help=(
            "Prefer non-destructive train weighting by repeating hard examples: "
            "H0 high-cosine and H1 low-cosine receive larger effective weight."
        ),
    )
    ap.add_argument(
        "--train_hardness_gamma",
        type=float,
        default=1.0,
        help="Hardness exponent for train weighting (larger means stronger emphasis on hard samples).",
    )
    ap.add_argument(
        "--train_hardness_extra_repeats",
        type=int,
        default=2,
        help="Maximum additional repeats per training sample in hardness-weighting mode.",
    )
    ap.add_argument(
        "--train_pair_x_key",
        type=str,
        default=None,
        help="Optional NPZ key for raw x embeddings used in train cosine policy.",
    )
    ap.add_argument(
        "--train_pair_y_key",
        type=str,
        default=None,
        help="Optional NPZ key for raw y embeddings used in train cosine policy.",
    )

    # Static scorer hyperparameters. Defaults preserve the previous hard-coded behavior.
    ap.add_argument("--pca_whiten_abs_eps", type=float, default=1e-6)
    ap.add_argument("--pca_whiten_rel_eps", type=float, default=1e-6)
    ap.add_argument("--pca_whiten_max_rank", type=int, default=128)
    ap.add_argument(
        "--pca_whiten_rank_mode",
        type=str,
        default="explained_variance",
        choices=["fixed", "explained_variance", "threshold"],
    )
    ap.add_argument("--pca_whiten_explained_variance", type=float, default=0.99)
    ap.add_argument("--pca_whiten_norm_eps", type=float, default=1e-12)

    ap.add_argument("--xgb_n_estimators", type=int, default=30)
    ap.add_argument("--xgb_max_depth", type=int, default=3)
    ap.add_argument("--xgb_learning_rate", type=float, default=0.1)
    ap.add_argument("--xgb_subsample", type=float, default=1.0)
    ap.add_argument("--xgb_colsample_bytree", type=float, default=1.0)
    ap.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    ap.add_argument("--xgb_gamma", type=float, default=0.0)
    ap.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    ap.add_argument("--xgb_reg_lambda", type=float, default=1.0)

    ap.add_argument("--tiny_mlp_hidden_dim", type=int, default=16)
    ap.add_argument("--tiny_mlp_n_layers", type=int, default=1)
    ap.add_argument("--tiny_mlp_alpha", type=float, default=0.001)
    ap.add_argument("--tiny_mlp_learning_rate_init", type=float, default=0.001)
    ap.add_argument("--tiny_mlp_max_iter", type=int, default=800)
    ap.add_argument("--tiny_mlp_activation", type=str, default="relu", choices=["relu", "tanh", "logistic"])
    ap.add_argument("--tiny_mlp_early_stopping", action="store_true", default=False)

    ap.add_argument("--lda_solver", type=str, default="lsqr", choices=["svd", "lsqr", "eigen"])
    ap.add_argument("--lda_shrinkage", type=str, default="auto")
    ap.add_argument("--lda_tol", type=float, default=1e-4)

    ap.add_argument("--ensemble_ridge", type=float, default=1e-3)
    ap.add_argument("--ensemble_standardize", action="store_true", default=False)
    ap.add_argument("--ensemble_nonneg_simplex", type=lambda v: v.lower() in ("true", "1", "yes"), default=True)
    ap.add_argument("--ensemble_meta_frac", type=float, default=0.30)
    ap.add_argument("--ensemble_tpr_tie_tol", type=float, default=1e-4)
    ap.add_argument("--ensemble_tie_break_entropy", type=lambda v: v.lower() in ("true", "1", "yes"), default=True)

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

    # Optuna validation-only hyperparameter search.
    ap.add_argument("--enable_optuna", action="store_true", default=False)
    ap.add_argument("--optuna_trials", type=int, default=50)
    ap.add_argument("--optuna_timeout", type=float, default=None)
    ap.add_argument("--optuna_seed", type=int, default=None)
    ap.add_argument("--optuna_val_frac", type=float, default=0.5)
    ap.add_argument(
        "--optuna_target_method",
        type=str,
        default="best_feasible",
        help="Exact method name to optimize, or best_feasible/all to select the best validation row.",
    )
    ap.add_argument(
        "--optuna_metric",
        type=str,
        default="micro_tpr",
        choices=["micro_tpr", "macro_tpr", "utility"],
    )
    ap.add_argument("--optuna_fpr_penalty", type=float, default=5.0)
    ap.add_argument("--optuna_storage", type=str, default=None)
    ap.add_argument("--optuna_study_name", type=str, default=None)
    ap.add_argument(
        "--optuna_apply_best",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=True,
        help="Apply the best validation params before final test evaluation.",
    )

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    train_cosine_band_active = args.train_cosine_min is not None or args.train_cosine_max is not None
    train_cosine_policy_active = bool(args.train_hardness_weighting or train_cosine_band_active)

    if args.train_hardness_weighting and train_cosine_band_active:
        raise ValueError(
            "Use either hardness weighting (--train_hardness_weighting) or cosine-band filtering "
            "(--train_cosine_min/--train_cosine_max), not both."
        )
    if args.train_hardness_gamma <= 0.0:
        raise ValueError("--train_hardness_gamma must be > 0")
    if args.train_hardness_extra_repeats < 0:
        raise ValueError("--train_hardness_extra_repeats must be >= 0")
    if (args.train_pair_x_key is None) != (args.train_pair_y_key is None):
        raise ValueError("Provide both --train_pair_x_key and --train_pair_y_key together")
    if args.train_cosine_min is not None and not (-1.0 <= float(args.train_cosine_min) <= 1.0):
        raise ValueError("--train_cosine_min must lie in [-1, 1]")
    if args.train_cosine_max is not None and not (-1.0 <= float(args.train_cosine_max) <= 1.0):
        raise ValueError("--train_cosine_max must lie in [-1, 1]")
    if train_cosine_band_active:
        cmin = -1.0 if args.train_cosine_min is None else float(args.train_cosine_min)
        cmax = 1.0 if args.train_cosine_max is None else float(args.train_cosine_max)
        if cmin > cmax:
            raise ValueError(f"Invalid train cosine band: min={cmin} > max={cmax}")
    if args.enable_optuna:
        if int(args.optuna_trials) <= 0:
            raise ValueError("--optuna_trials must be > 0 when --enable_optuna is set")
        if not (0.05 <= float(args.optuna_val_frac) <= 0.95):
            raise ValueError("--optuna_val_frac must lie in [0.05, 0.95]")

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

    train_pair_cosine_source: Optional[str] = None
    train_pair_cosine: Optional[np.ndarray] = None
    if train_cosine_policy_active:
        train_pair_cosine_source, train_pair_cosine = resolve_train_pairwise_cosine(
            ds,
            x_key=args.train_pair_x_key,
            y_key=args.train_pair_y_key,
        )
        if train_pair_cosine is None:
            raise ValueError(
                "Train cosine policy requested but raw pair embeddings were not found. "
                "Provide --train_pair_x_key/--train_pair_y_key or add supported x/y keys to the NPZ."
            )
        if int(train_pair_cosine.shape[0]) != int(y.shape[0]):
            raise ValueError(
                "train pair cosine rows mismatch labels: "
                f"pair_cos={int(train_pair_cosine.shape[0])} labels={int(y.shape[0])}"
            )

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
    print(f"train_cosine_policy_active={train_cosine_policy_active}")
    if train_cosine_policy_active:
        mode = "hardness_weighting" if args.train_hardness_weighting else "band_filter"
        print(f"train_cosine_policy_mode={mode}")
        print(f"train_pair_cosine_source={train_pair_cosine_source}")
        if args.train_hardness_weighting:
            print(
                "train_hardness: "
                f"gamma={float(args.train_hardness_gamma):.3f} "
                f"extra_repeats={int(args.train_hardness_extra_repeats)}"
            )
        else:
            cmin = -1.0 if args.train_cosine_min is None else float(args.train_cosine_min)
            cmax = 1.0 if args.train_cosine_max is None else float(args.train_cosine_max)
            print(f"train_cosine_band=[{cmin:.4f}, {cmax:.4f}]")
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
    if args.enable_optuna:
        print(
            "optuna: "
            f"trials={int(args.optuna_trials)} val_frac={float(args.optuna_val_frac):.3f} "
            f"target={args.optuna_target_method} metric={args.optuna_metric} "
            f"apply_best={bool(args.optuna_apply_best)}"
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
    optuna_trial_rows: List[Dict[str, Any]] = []
    optuna_candidate_rows: List[Dict[str, Any]] = []
    optuna_best_rows: List[Dict[str, Any]] = []

    base_args = args

    for trial in range(args.n_trials):
        args = base_args
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

            train_cos_stats_global: Optional[Dict[str, Any]] = None
            if train_cosine_policy_active:
                if train_pair_cosine is None:
                    raise RuntimeError("Train cosine policy active but train_pair_cosine is unavailable")
                gs, train_cos_stats_global = _apply_train_cosine_policy_global(
                    gs,
                    pair_cosine=train_pair_cosine,
                    cosine_min=args.train_cosine_min,
                    cosine_max=args.train_cosine_max,
                    use_hardness_weighting=bool(args.train_hardness_weighting),
                    hardness_gamma=float(args.train_hardness_gamma),
                    hardness_extra_repeats=int(args.train_hardness_extra_repeats),
                )

            optuna_split_stats: Dict[str, int] = {}
            if args.enable_optuna:
                gs_val, gs_test, optuna_split_stats = split_global_eval_for_validation(
                    gs,
                    val_frac=float(args.optuna_val_frac),
                    seed=seed + 100_000,
                    min_test_h0=1,
                    min_test_h1=1,
                )
                if gs_val.H0_eval.size == 0 or gs_val.H1_eval.size == 0:
                    raise RuntimeError("Optuna validation split is empty; increase --n_eval or adjust --optuna_val_frac")
                if gs_test.H0_eval.size == 0 or gs_test.H1_eval.size == 0:
                    raise RuntimeError("Optuna final test split is empty; increase --n_eval or adjust --optuna_val_frac")

                def _eval_candidate_global(candidate_args: argparse.Namespace, params: Dict[str, Any], optuna_trial: Any) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
                    del params
                    return _evaluate_global_candidate(
                        candidate_args,
                        gs=gs_val,
                        X_main=X_main,
                        X_cos=X_cos,
                        X_text=X_text,
                        y=y,
                        region_id=region_id,
                        h0_monitor_idx=h0_monitor_idx,
                        h1_monitor_idx=h1_monitor_idx,
                        trial=int(optuna_trial.number),
                        seed=seed,
                        region_key=args.region_key,
                        selected_ocats_methods=selected_ocats_methods,
                        lambda_values=lambda_values,
                        run_ocats_baselines=run_ocats_baselines,
                        include_ocats_in_comparison=include_ocats_in_comparison,
                    )

                print(
                    "  optuna_split: "
                    f"val(n0={optuna_split_stats['val_h0']}, n1={optuna_split_stats['val_h1']}) "
                    f"test(n0={optuna_split_stats['test_h0']}, n1={optuna_split_stats['test_h1']})"
                )
                search_result = run_optuna_search(
                    args=args,
                    scope="global",
                    evaluate_candidate=_eval_candidate_global,
                    has_xgb=has_xgb,
                    has_text_pairs=X_text is not None,
                    has_x_cos=X_cos is not None,
                    include_ocats=run_ocats_baselines,
                    include_online=bool(args.enable_online_stopping),
                    n_trials=int(args.optuna_trials),
                    timeout=args.optuna_timeout,
                    seed=int(args.optuna_seed if args.optuna_seed is not None else seed),
                    target_method=(
                        f"{selected_ocats_methods[0]}[OCaTS:optuna_validation]"
                        if ocats_only_mode and selected_ocats_methods
                        else str(args.optuna_target_method)
                    ),
                    metric=str(args.optuna_metric),
                    fpr_penalty=float(args.optuna_fpr_penalty),
                    storage=args.optuna_storage,
                    study_name=args.optuna_study_name,
                )
                optuna_trial_rows.extend(
                    {
                        "trial": int(trial),
                        "seed": int(seed),
                        "scope": "global",
                        "optuna_trial": int(r["optuna_trial"]),
                        "value": float(r["value"]),
                        "selected_method": str(r.get("selected_method", "")),
                        "selected_tpr": float(r.get("selected_tpr", float("nan"))),
                        "selected_fpr": float(r.get("selected_fpr", float("nan"))),
                        "failed": bool(r.get("failed", False)),
                        "params_json": json.dumps(r.get("params", {}), sort_keys=True),
                    }
                    for r in search_result.trials
                )
                optuna_candidate_rows.extend(
                    {**dict(r), "outer_trial": int(trial), "outer_seed": int(seed), "scope": "global"}
                    for r in search_result.candidate_rows
                )
                optuna_best_rows.append(
                    {
                        "trial": int(trial),
                        "seed": int(seed),
                        "scope": "global",
                        "best_optuna_trial": int(search_result.best_trial_number),
                        "best_value": float(search_result.best_value),
                        "best_params": dict(search_result.best_params),
                    }
                )
                if args.optuna_apply_best:
                    args = apply_params_to_namespace(args, search_result.best_params)
                print(
                    "  optuna_best: "
                    f"trial={search_result.best_trial_number} value={search_result.best_value:.6f} "
                    f"tau_mode={getattr(args, 'tau_mode', 'global')}"
                )
                gs = gs_test

            h0_train_idx_fit = np.asarray(gs.H0_train, dtype=np.int64).reshape(-1)
            h1_train_idx_fit = np.asarray(gs.H1_train, dtype=np.int64).reshape(-1)
            h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
            h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

            H0_train = X_main[h0_train_idx_fit]
            H1_train = X_main[h1_train_idx_fit]
            H0_calib_pure = X_main[gs.H0_calib]
            H1_calib_pure = X_main[gs.H1_calib]
            H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                if h0_train_idx_unique.size > 0 else X_main[gs.H0_calib]
            H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                if h1_train_idx_unique.size > 0 else X_main[gs.H1_calib]

            # Extract X_cos splits if available
            H0_train_cos = X_cos[h0_train_idx_fit] if X_cos is not None else None
            H1_train_cos = X_cos[h1_train_idx_fit] if X_cos is not None else None
            if X_cos is not None:
                H0_calib_pure_cos = X_cos[gs.H0_calib]
                H1_calib_pure_cos = X_cos[gs.H1_calib]
                H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                    if h0_train_idx_unique.size > 0 else X_cos[gs.H0_calib]
                H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                    if h1_train_idx_unique.size > 0 else X_cos[gs.H1_calib]
            else:
                H0_calib_pure_cos = None
                H1_calib_pure_cos = None
                H0_calib_eff_cos = None
                H1_calib_eff_cos = None

            if X_text is not None:
                H0_train_text = X_text[h0_train_idx_fit]
                H1_train_text = X_text[h1_train_idx_fit]
                H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                    if h0_train_idx_unique.size > 0 else X_text[gs.H0_calib]
                H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                    if h1_train_idx_unique.size > 0 else X_text[gs.H1_calib]
            else:
                H0_train_text = None
                H1_train_text = None
                H0_calib_eff_text = None
                H1_calib_eff_text = None

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
            if train_cos_stats_global is not None:
                _log_train_cosine_policy_stats(train_cos_stats_global)
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

            methods = _build_configured_methods(args, X_cos=X_cos, X_text=X_text, quiet=True)
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

            if args.regional_weighted_ensemble and "RegionalWeightedEnsemble" not in methods:
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
                H0_train_text=H0_train_text,
                H1_train_text=H1_train_text,
                H0_calib_eff_text=H0_calib_eff_text,
                H1_calib_eff_text=H1_calib_eff_text,
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

            train_cos_stats_local: Optional[Dict[str, Any]] = None
            if train_cosine_policy_active:
                if train_pair_cosine is None:
                    raise RuntimeError("Train cosine policy active but train_pair_cosine is unavailable")
                splits, train_cos_stats_local = _apply_train_cosine_policy_regions(
                    splits,
                    pair_cosine=train_pair_cosine,
                    cosine_min=args.train_cosine_min,
                    cosine_max=args.train_cosine_max,
                    use_hardness_weighting=bool(args.train_hardness_weighting),
                    hardness_gamma=float(args.train_hardness_gamma),
                    hardness_extra_repeats=int(args.train_hardness_extra_repeats),
                )

            if len(splits) == 0:
                raise RuntimeError(
                    "No regions satisfied eval mins. Lower mins, increase caps, or change regioning."
                )

            optuna_split_stats: Dict[str, int] = {}
            if args.enable_optuna:
                splits_val, splits_test, optuna_split_stats = split_region_eval_for_validation(
                    splits,
                    val_frac=float(args.optuna_val_frac),
                    seed=seed + 100_000,
                    min_test_h0=1,
                    min_test_h1=1,
                )
                if not splits_val or not splits_test:
                    raise RuntimeError("Optuna validation/test region split is empty; increase --n_eval or adjust --optuna_val_frac")

                def _eval_candidate_local(candidate_args: argparse.Namespace, params: Dict[str, Any], optuna_trial: Any) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
                    del params
                    return _evaluate_local_candidate(
                        candidate_args,
                        splits=splits_val,
                        X_main=X_main,
                        X_cos=X_cos,
                        X_text=X_text,
                        y=y,
                        region_id=region_id,
                        X_anchor_source=X_anchor_source,
                        h0_monitor_idx=h0_monitor_idx,
                        h1_monitor_idx=h1_monitor_idx,
                        trial=int(optuna_trial.number),
                        seed=seed,
                        region_key=args.region_key,
                        selected_ocats_methods=selected_ocats_methods,
                        lambda_values=lambda_values,
                        run_ocats_baselines=run_ocats_baselines,
                        include_ocats_in_comparison=include_ocats_in_comparison,
                    )

                print(
                    "  optuna_split: "
                    f"val_regions={optuna_split_stats['val_regions']} "
                    f"test_regions={optuna_split_stats['test_regions']} "
                    f"val(n0={optuna_split_stats['val_h0']}, n1={optuna_split_stats['val_h1']}) "
                    f"test(n0={optuna_split_stats['test_h0']}, n1={optuna_split_stats['test_h1']}) "
                    f"dropped_regions={optuna_split_stats['dropped_regions']}"
                )
                search_result = run_optuna_search(
                    args=args,
                    scope="local",
                    evaluate_candidate=_eval_candidate_local,
                    has_xgb=has_xgb,
                    has_text_pairs=X_text is not None,
                    has_x_cos=X_cos is not None,
                    include_ocats=run_ocats_baselines,
                    include_online=bool(args.enable_online_stopping),
                    n_trials=int(args.optuna_trials),
                    timeout=args.optuna_timeout,
                    seed=int(args.optuna_seed if args.optuna_seed is not None else seed),
                    target_method=(
                        f"{selected_ocats_methods[0]}[OCaTS:optuna_validation]"
                        if ocats_only_mode and selected_ocats_methods
                        else str(args.optuna_target_method)
                    ),
                    metric=str(args.optuna_metric),
                    fpr_penalty=float(args.optuna_fpr_penalty),
                    storage=args.optuna_storage,
                    study_name=args.optuna_study_name,
                )
                optuna_trial_rows.extend(
                    {
                        "trial": int(trial),
                        "seed": int(seed),
                        "scope": "local",
                        "optuna_trial": int(r["optuna_trial"]),
                        "value": float(r["value"]),
                        "selected_method": str(r.get("selected_method", "")),
                        "selected_tpr": float(r.get("selected_tpr", float("nan"))),
                        "selected_fpr": float(r.get("selected_fpr", float("nan"))),
                        "failed": bool(r.get("failed", False)),
                        "params_json": json.dumps(r.get("params", {}), sort_keys=True),
                    }
                    for r in search_result.trials
                )
                optuna_candidate_rows.extend(
                    {**dict(r), "outer_trial": int(trial), "outer_seed": int(seed), "scope": "local"}
                    for r in search_result.candidate_rows
                )
                optuna_best_rows.append(
                    {
                        "trial": int(trial),
                        "seed": int(seed),
                        "scope": "local",
                        "best_optuna_trial": int(search_result.best_trial_number),
                        "best_value": float(search_result.best_value),
                        "best_params": dict(search_result.best_params),
                    }
                )
                if args.optuna_apply_best:
                    args = apply_params_to_namespace(args, search_result.best_params)
                print(
                    "  optuna_best: "
                    f"trial={search_result.best_trial_number} value={search_result.best_value:.6f} "
                    f"tau_mode={getattr(args, 'tau_mode', 'local')}"
                )
                splits = splits_test

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
            h0_train_idx_fit = _concat_indices(h0_train_idx_list)
            h1_train_idx_fit = _concat_indices(h1_train_idx_list)
            h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
            h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

            H0_train = X_main[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_main[:0]
            H1_train = X_main[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_main[:0]

            h0_calib_list = [s.H0_calib for s in splits if s.H0_calib.size > 0]
            h1_calib_list = [s.H1_calib for s in splits if s.H1_calib.size > 0]
            h0_calib_idx = _concat_indices(h0_calib_list)
            h1_calib_idx = _concat_indices(h1_calib_list)
            H0_calib = X_main[h0_calib_idx] if h0_calib_idx.size > 0 else X_main[:0]
            H1_calib = X_main[h1_calib_idx] if h1_calib_idx.size > 0 else X_main[:0]

            H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                if h0_train_idx_unique.size > 0 else H0_calib
            H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                if h1_train_idx_unique.size > 0 else H1_calib

            # Extract X_cos splits if available
            if X_cos is not None:
                H0_train_cos = X_cos[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_cos[:0]
                H1_train_cos = X_cos[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_cos[:0]
                H0_calib_cos = X_cos[h0_calib_idx] if h0_calib_idx.size > 0 else X_cos[:0]
                H1_calib_cos = X_cos[h1_calib_idx] if h1_calib_idx.size > 0 else X_cos[:0]
                H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                    if h0_train_idx_unique.size > 0 else H0_calib_cos
                H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                    if h1_train_idx_unique.size > 0 else H1_calib_cos
            else:
                H0_train_cos = None
                H1_train_cos = None
                H0_calib_cos = None
                H1_calib_cos = None
                H0_calib_eff_cos = None
                H1_calib_eff_cos = None

            if X_text is not None:
                H0_train_text = X_text[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_text[:0]
                H1_train_text = X_text[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_text[:0]
                H0_calib_text = X_text[h0_calib_idx] if h0_calib_idx.size > 0 else X_text[:0]
                H1_calib_text = X_text[h1_calib_idx] if h1_calib_idx.size > 0 else X_text[:0]
                H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                    if h0_train_idx_unique.size > 0 else H0_calib_text
                H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                    if h1_train_idx_unique.size > 0 else H1_calib_text
            else:
                H0_train_text = None
                H1_train_text = None
                H0_calib_eff_text = None
                H1_calib_eff_text = None

            print(f"\n[trial={trial} seed={seed}] used_regions={len(splits)} split_stats={split_stats}")
            if filter_stats_local:
                print(f"  post_filter_stats={filter_stats_local}")
            if train_cos_stats_local is not None:
                _log_train_cosine_policy_stats(train_cos_stats_local)
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

            methods = _build_configured_methods(args, X_cos=X_cos, X_text=X_text, quiet=True)
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

            if args.regional_weighted_ensemble and "RegionalWeightedEnsemble" not in methods:
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
                    methods_matched = _build_configured_methods(args, X_cos=X_cos, X_text=X_text, quiet=True)
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

                    if args.regional_weighted_ensemble and "RegionalWeightedEnsemble" not in methods_matched:
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
    if optuna_trial_rows:
        save_csv_rows(
            run_dir / "optuna_trials.csv",
            optuna_trial_rows,
            fieldnames=[
                "trial", "seed", "scope", "optuna_trial", "value",
                "selected_method", "selected_tpr", "selected_fpr",
                "failed", "params_json",
            ],
        )
        save_json(
            run_dir / "optuna_trials.json",
            {"rows": optuna_trial_rows},
        )
    if optuna_candidate_rows:
        save_json(
            run_dir / "optuna_validation_rows.json",
            {"rows": optuna_candidate_rows},
        )
    if optuna_best_rows:
        save_json(
            run_dir / "optuna_best.json",
            {"rows": optuna_best_rows},
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
            "train_cosine_policy_active": bool(train_cosine_policy_active),
            "train_cosine_policy_mode": (
                "hardness_weighting"
                if args.train_hardness_weighting
                else ("band_filter" if train_cosine_band_active else "none")
            ),
            "train_cosine_min": (float(args.train_cosine_min) if args.train_cosine_min is not None else None),
            "train_cosine_max": (float(args.train_cosine_max) if args.train_cosine_max is not None else None),
            "train_hardness_weighting": bool(args.train_hardness_weighting),
            "train_hardness_gamma": float(args.train_hardness_gamma),
            "train_hardness_extra_repeats": int(args.train_hardness_extra_repeats),
            "train_pair_x_key": args.train_pair_x_key,
            "train_pair_y_key": args.train_pair_y_key,
            "train_pair_cosine_source": train_pair_cosine_source,
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
            "optuna": {
                "enabled": bool(base_args.enable_optuna),
                "trials": int(base_args.optuna_trials),
                "timeout": base_args.optuna_timeout,
                "seed": base_args.optuna_seed,
                "val_frac": float(base_args.optuna_val_frac),
                "target_method": str(base_args.optuna_target_method),
                "metric": str(base_args.optuna_metric),
                "fpr_penalty": float(base_args.optuna_fpr_penalty),
                "storage": base_args.optuna_storage,
                "study_name": base_args.optuna_study_name,
                "apply_best": bool(base_args.optuna_apply_best),
                "trial_rows": int(len(optuna_trial_rows)),
                "validation_rows": int(len(optuna_candidate_rows)),
                "best_rows": int(len(optuna_best_rows)),
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
