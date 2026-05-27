"""Shared preprocessing helpers for region-local threshold experiments."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


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
