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
