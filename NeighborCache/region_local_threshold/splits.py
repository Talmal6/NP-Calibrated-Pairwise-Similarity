"""Region splitting: RegionSplit dataclass and splitting logic."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RegionSplit:
    rid: int
    H0_train: np.ndarray
    H1_train: np.ndarray
    H0_calib: np.ndarray
    H1_calib: np.ndarray
    H0_eval: np.ndarray
    H1_eval: np.ndarray


def _take_split_train_calib_eval(
    idx: np.ndarray,
    rng: np.random.Generator,
    n_train_cap: int,
    n_calib_cap: int,
    n_eval_cap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train / calib / eval, respecting caps.

    When n < sum(caps), each slice is shrunk proportionally so that
    no slice is starved (especially eval).
    """
    idx = np.asarray(idx, dtype=np.int64).copy()
    rng.shuffle(idx)

    n = idx.size
    total_requested = n_train_cap + n_calib_cap + n_eval_cap

    if total_requested <= n:
        # Enough data — honour caps exactly
        n_train_eff = n_train_cap
        n_calib_eff = n_calib_cap
        n_eval_eff = n_eval_cap
    else:
        # Not enough data — scale proportionally
        if total_requested > 0:
            scale = n / total_requested
            n_train_eff = int(round(n_train_cap * scale))
            n_calib_eff = int(round(n_calib_cap * scale))
            n_eval_eff = n - n_train_eff - n_calib_eff
            # Ensure none go negative from rounding
            if n_eval_eff < 0:
                n_calib_eff += n_eval_eff
                n_eval_eff = 0
        else:
            n_train_eff = n_calib_eff = n_eval_eff = 0

    tr = idx[:n_train_eff]
    ca = idx[n_train_eff:n_train_eff + n_calib_eff]
    ev = idx[n_train_eff + n_calib_eff:n_train_eff + n_calib_eff + n_eval_eff]

    return tr, ca, ev


def split_indices_per_region(
    region_id: np.ndarray,
    y: np.ndarray,
    *,
    n_train_cap: int,
    n_calib_cap: int,
    n_eval_cap: int,
    seed: int,
    min_h0_eval: int,
    min_h1_eval: int,
) -> Tuple[List[RegionSplit], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    regions = np.unique(region_id.astype(np.int64))

    splits: List[RegionSplit] = []
    stats = defaultdict(int)

    for rid in regions:
        idx_r = np.flatnonzero(region_id == rid)
        if idx_r.size == 0:
            continue

        idx0 = idx_r[y[idx_r] == 0]
        idx1 = idx_r[y[idx_r] == 1]

        if idx0.size < min_h0_eval or idx1.size < min_h1_eval:
            stats["skipped_region_insufficient_eval_mins"] += 1
            continue

        h0_tr, h0_ca, h0_ev = _take_split_train_calib_eval(idx0, rng, n_train_cap, n_calib_cap, n_eval_cap)
        h1_tr, h1_ca, h1_ev = _take_split_train_calib_eval(idx1, rng, n_train_cap, n_calib_cap, n_eval_cap)

        if h0_ev.size < min_h0_eval:
            stats["skipped_region_min_h0_eval_after_sampling"] += 1
            continue
        if h1_ev.size < min_h1_eval:
            stats["skipped_region_min_h1_eval_after_sampling"] += 1
            continue

        splits.append(
            RegionSplit(
                rid=int(rid),
                H0_train=h0_tr.astype(np.int64),
                H1_train=h1_tr.astype(np.int64),
                H0_calib=h0_ca.astype(np.int64),
                H1_calib=h1_ca.astype(np.int64),
                H0_eval=h0_ev.astype(np.int64),
                H1_eval=h1_ev.astype(np.int64),
            )
        )
        stats["used_regions"] += 1

    return splits, dict(stats)


@dataclass
class GlobalSplit:
    """A single global train/calib/eval split (no region gating)."""
    H0_train: np.ndarray
    H1_train: np.ndarray
    H0_calib: np.ndarray
    H1_calib: np.ndarray
    H0_eval: np.ndarray
    H1_eval: np.ndarray


def filter_region_splits_by_score_range(
    splits: List[RegionSplit],
    *,
    score: np.ndarray,
    score_min: float,
    score_max: float,
    min_h0_eval: int,
    min_h1_eval: int,
) -> Tuple[List[RegionSplit], Dict[str, int]]:
    """Apply one shared score-range filter to train/calib/eval for all regions.

    The same predicate is applied to each split partition and each class.
    Regions that no longer satisfy eval mins are dropped globally.
    """
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    keep = (s >= float(score_min)) & (s <= float(score_max))

    out: List[RegionSplit] = []
    stats = defaultdict(int)

    def _f(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return idx
        return idx[keep[idx]]

    for r in splits:
        h0_tr = _f(r.H0_train)
        h1_tr = _f(r.H1_train)
        h0_ca = _f(r.H0_calib)
        h1_ca = _f(r.H1_calib)
        h0_ev = _f(r.H0_eval)
        h1_ev = _f(r.H1_eval)

        if h0_ev.size < int(min_h0_eval):
            stats["dropped_region_min_h0_eval_after_filter"] += 1
            continue
        if h1_ev.size < int(min_h1_eval):
            stats["dropped_region_min_h1_eval_after_filter"] += 1
            continue
        if h0_ca.size == 0:
            stats["dropped_region_empty_h0_calib_after_filter"] += 1
            continue

        out.append(
            RegionSplit(
                rid=int(r.rid),
                H0_train=h0_tr.astype(np.int64, copy=False),
                H1_train=h1_tr.astype(np.int64, copy=False),
                H0_calib=h0_ca.astype(np.int64, copy=False),
                H1_calib=h1_ca.astype(np.int64, copy=False),
                H0_eval=h0_ev.astype(np.int64, copy=False),
                H1_eval=h1_ev.astype(np.int64, copy=False),
            )
        )

    stats["used_regions"] = int(len(out))
    return out, dict(stats)


def filter_global_split_by_score_range(
    gs: GlobalSplit,
    *,
    score: np.ndarray,
    score_min: float,
    score_max: float,
) -> Tuple[GlobalSplit, Dict[str, int]]:
    """Apply one shared score-range filter to train/calib/eval for a global split."""
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    keep = (s >= float(score_min)) & (s <= float(score_max))

    def _f(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return idx
        return idx[keep[idx]].astype(np.int64, copy=False)

    out = GlobalSplit(
        H0_train=_f(gs.H0_train),
        H1_train=_f(gs.H1_train),
        H0_calib=_f(gs.H0_calib),
        H1_calib=_f(gs.H1_calib),
        H0_eval=_f(gs.H0_eval),
        H1_eval=_f(gs.H1_eval),
    )
    stats = {
        "h0_train": int(out.H0_train.size),
        "h1_train": int(out.H1_train.size),
        "h0_calib": int(out.H0_calib.size),
        "h1_calib": int(out.H1_calib.size),
        "h0_eval": int(out.H0_eval.size),
        "h1_eval": int(out.H1_eval.size),
    }
    return out, stats


def split_global(
    y: np.ndarray,
    *,
    n_train_cap: int,
    n_calib_cap: int,
    n_eval_cap: int,
    seed: int,
) -> Tuple[GlobalSplit, Dict[str, int]]:
    """Stratified global split: pool all H0/H1, shuffle, take train/calib/eval.

    No region filtering — every sample is eligible.
    Caps are applied per-class; excess data is silently dropped.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int32)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    if idx0.size == 0:
        raise ValueError("Global split: no H0 (label==0) samples in dataset.")
    if idx1.size == 0:
        raise ValueError("Global split: no H1 (label==1) samples in dataset.")

    h0_tr, h0_ca, h0_ev = _take_split_train_calib_eval(idx0, rng, n_train_cap, n_calib_cap, n_eval_cap)
    h1_tr, h1_ca, h1_ev = _take_split_train_calib_eval(idx1, rng, n_train_cap, n_calib_cap, n_eval_cap)

    stats = {
        "total_h0": int(idx0.size),
        "total_h1": int(idx1.size),
        "h0_train": int(h0_tr.size),
        "h1_train": int(h1_tr.size),
        "h0_calib": int(h0_ca.size),
        "h1_calib": int(h1_ca.size),
        "h0_eval": int(h0_ev.size),
        "h1_eval": int(h1_ev.size),
    }

    gs = GlobalSplit(
        H0_train=h0_tr.astype(np.int64),
        H1_train=h1_tr.astype(np.int64),
        H0_calib=h0_ca.astype(np.int64),
        H1_calib=h1_ca.astype(np.int64),
        H0_eval=h0_ev.astype(np.int64),
        H1_eval=h1_ev.astype(np.int64),
    )
    return gs, stats
