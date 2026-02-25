"""Threshold application and per-region evaluation loop."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .methods import method_train_required, needs_weights, needs_seed, try_fit_method
from .splits import RegionSplit, GlobalSplit


def apply_threshold(scores: np.ndarray, tau: float, tie_mode: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if tie_mode == "gt":
        return (s > tau).astype(np.int32)
    return (s >= tau).astype(np.int32)


def _beta_ppf(q: float, a: float, b: float) -> float:
    try:
        from scipy.stats import beta as scipy_beta  # type: ignore

        return float(scipy_beta.ppf(q, a, b))
    except Exception:
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return 1.0
        return float(q)


def _normal_ppf(q: float) -> float:
    try:
        from scipy.stats import norm  # type: ignore

        return float(norm.ppf(q))
    except Exception:
        if q >= 0.995:
            return 2.5758
        if q >= 0.99:
            return 2.3263
        if q >= 0.975:
            return 1.9600
        if q >= 0.95:
            return 1.6449
        return 1.2816


def _fpr_ucb(k: int, n: int, *, method: str, delta: float) -> float:
    if n <= 0:
        return 1.0
    k = int(max(0, min(k, n)))
    delta = float(np.clip(delta, 1e-12, 0.5))

    if method == "clopper_pearson":
        if k >= n:
            return 1.0
        return _beta_ppf(1.0 - delta, k + 1.0, n - k)

    if method == "beta_ucb":
        return _beta_ppf(1.0 - delta, k + 1.0, n - k + 1.0)

    if method == "wilson":
        phat = k / n
        z = _normal_ppf(1.0 - delta)
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (phat + z2 / (2.0 * n)) / denom
        radius = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
        return float(min(1.0, max(0.0, center + radius)))

    return float(k / n)


def _select_tau(
    scores: np.ndarray,
    *,
    alpha: float,
    tie_mode: str,
    guardrail: str,
    guardrail_delta: float,
) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("empty calibration scores")

    if guardrail == "none":
        return float(np.quantile(s, 1.0 - alpha))

    uniq, counts = np.unique(s, return_counts=True)
    n = int(s.size)
    cumsum = np.cumsum(counts)

    for i, tau in enumerate(uniq):
        if tie_mode == "gt":
            k = int(n - cumsum[i])
        else:
            k = int(n - (cumsum[i - 1] if i > 0 else 0))
        ucb = _fpr_ucb(k, n, method=guardrail, delta=guardrail_delta)
        if ucb <= alpha:
            return float(tau)

    return float("inf")


def _l2_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


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


def _build_region_cluster_map(
    *,
    splits: List[RegionSplit],
    X_proto: np.ndarray,
    region_to_idx: Dict[int, np.ndarray],
    n_clusters: int,
    seed: int,
) -> Dict[int, int]:
    rids = [int(s.rid) for s in splits]
    if not rids:
        return {}

    d = X_proto.shape[1]
    protos = np.zeros((len(rids), d), dtype=np.float64)
    for i, rid in enumerate(rids):
        idx = region_to_idx.get(rid)
        if idx is not None and idx.size > 0:
            protos[i] = np.mean(X_proto[idx], axis=0)

    protos = _l2_rows(protos)
    labels = _kmeans_numpy(protos, n_clusters=n_clusters, seed=seed)
    return {rid: int(labels[i]) for i, rid in enumerate(rids)}


def _score_method_with_routing(
    method: Any,
    method_name: str,
    X_main_slice: np.ndarray,
    X_cos_slice: Optional[np.ndarray],
) -> np.ndarray:
    """Score samples using the appropriate input routing.

    Args:
        method: The method instance to score with
        method_name: Name of the method (to check for WeightedEnsemble)
        X_main_slice: Main feature matrix slice (embeddings)
        X_cos_slice: Cosine feature matrix slice (Hadamard vectors), or None

    Returns:
        Score array for the input samples
    """
    if method_name == "WeightedEnsemble":
        # WeightedEnsemble needs both matrices for per-judge routing
        return method.score(X_main_slice, X_alt=X_cos_slice)
    else:
        # Regular methods use single matrix determined by their type
        use_cos = method_name in {"Cosine", "CosineAffineCalib"}
        X_use = X_cos_slice if (use_cos and X_cos_slice is not None) else X_main_slice
        return method.score(X_use)


def _compute_cosine_scores(
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Compute cosine similarity scores for all samples.
    
    Args:
        X_main: Main embeddings (N, D) - not used if X_cos is available
        X_cos: Optional Hadamard product vectors (N, D) where sum = cosine
        
    Returns:
        Array of cosine scores (N,) if X_cos is available, None otherwise
    """
    if X_cos is not None:
        # Hadamard vectors: sum across features gives cosine
        # This matches CosineMethod.score() implementation
        scores = np.sum(X_cos.astype(np.float64, copy=False), axis=1)
        scores = np.clip(scores, -1.0, 1.0)
        scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)
        return scores.astype(np.float32)
    else:
        # Cannot reliably compute cosine without Hadamard vectors
        return None


def _filter_ambiguous_region(
    H0: np.ndarray,
    H1: np.ndarray,
    X_cos_H0: Optional[np.ndarray],
    X_cos_H1: Optional[np.ndarray],
    cos_min: float = 0.7,
    cos_max: float = 0.9,
    min_samples_per_class: int = 200,
    trial: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Filter samples to ambiguous cosine similarity region.
    
    Args:
        H0: H0 embeddings (N0, D)
        H1: H1 embeddings (N1, D)
        X_cos_H0: Optional Hadamard vectors for H0 (N0, D)
        X_cos_H1: Optional Hadamard vectors for H1 (N1, D)
        cos_min: Minimum cosine threshold (default 0.7)
        cos_max: Maximum cosine threshold (default 0.9)
        min_samples_per_class: Minimum samples required after filtering (default 200)
        trial: Trial number for logging
        
    Returns:
        Tuple of (H0_filtered, H1_filtered, X_cos_H0_filtered, X_cos_H1_filtered, used_filtering)
    """
    n0_orig = H0.shape[0]
    n1_orig = H1.shape[0]
    
    # Compute cosine scores (requires X_cos Hadamard vectors)
    cos_H0 = _compute_cosine_scores(H0, X_cos_H0)
    cos_H1 = _compute_cosine_scores(H1, X_cos_H1)
    
    # If we can't compute cosine (no X_cos available), skip filtering
    if cos_H0 is None or cos_H1 is None:
        print(f"  [trial={trial}] AMBIGUOUS FILTER: X_cos not available, using FULL dataset")
        return H0, H1, X_cos_H0, X_cos_H1, False
    
    # Filter to ambiguous region
    mask_H0 = (cos_H0 >= cos_min) & (cos_H0 <= cos_max)
    mask_H1 = (cos_H1 >= cos_min) & (cos_H1 <= cos_max)
    
    n0_filtered = np.sum(mask_H0)
    n1_filtered = np.sum(mask_H1)
    
    # Check if we have enough samples
    if n0_filtered < min_samples_per_class or n1_filtered < min_samples_per_class:
        print(f"  [trial={trial}] AMBIGUOUS FILTER: Insufficient samples after filtering")
        print(f"    H0: {n0_orig} → {n0_filtered} (need >={min_samples_per_class})")
        print(f"    H1: {n1_orig} → {n1_filtered} (need >={min_samples_per_class})")
        print(f"    → Using FULL dataset (no filtering)")
        return H0, H1, X_cos_H0, X_cos_H1, False
    
    # Apply filtering
    H0_filt = H0[mask_H0]
    H1_filt = H1[mask_H1]
    X_cos_H0_filt = X_cos_H0[mask_H0] if X_cos_H0 is not None else None
    X_cos_H1_filt = X_cos_H1[mask_H1] if X_cos_H1 is not None else None
    
    # Log statistics
    cos_H0_filt = cos_H0[mask_H0]
    cos_H1_filt = cos_H1[mask_H1]
    cos_all_filt = np.concatenate([cos_H0_filt, cos_H1_filt])
    
    print(f"  [trial={trial}] AMBIGUOUS FILTER: {cos_min} <= cosine <= {cos_max}")
    print(f"    H0: {n0_orig} → {n0_filtered} ({100.0*n0_filtered/max(1,n0_orig):.1f}% retained)")
    print(f"    H1: {n1_orig} → {n1_filtered} ({100.0*n1_filtered/max(1,n1_orig):.1f}% retained)")
    print(f"    Cosine stats in filtered region: mean={np.mean(cos_all_filt):.3f} std={np.std(cos_all_filt):.3f}")
    
    return H0_filt, H1_filt, X_cos_H0_filt, X_cos_H1_filt, True


def fit_all_methods(
    methods: Dict[str, Any],
    *,
    H0_train: np.ndarray,
    H1_train: np.ndarray,
    H0_calib_eff: np.ndarray,
    H1_calib_eff: np.ndarray,
    H0_train_cos: Optional[np.ndarray],
    H1_train_cos: Optional[np.ndarray],
    H0_calib_eff_cos: Optional[np.ndarray],
    H1_calib_eff_cos: Optional[np.ndarray],
    weights: np.ndarray,
    seed: int,
    alpha: float,
    tie_mode: str,
    tau_guardrail: str,
    tau_guardrail_delta: float,
    trial: int,
    failures: Dict[str, List[str]],
) -> None:
    """Fit all methods in-place (modifies *methods* dict on failure).
    
    Applies ambiguous-region filtering (0.7 <= cosine <= 0.9) to train/calib data
    for all learned methods and ensemble. CosineMethod uses full data.
    """
    # Apply ambiguous region filtering for learned methods
    # CosineMethod and related methods use full data
    COSINE_METHODS = {"Cosine", "CosineAffineCalib"}
    
    # Filter train data to ambiguous region (0.7 <= cosine <= 0.9)
    H0_train_filt, H1_train_filt, H0_train_cos_filt, H1_train_cos_filt, used_train_filter = \
        _filter_ambiguous_region(
            H0_train, H1_train,
            H0_train_cos, H1_train_cos,
            cos_min=0.7, cos_max=0.9,
            min_samples_per_class=50,  # Reduced to allow filtering with smaller datasets
            trial=trial,
        )
    
    # Filter calib data to ambiguous region
    H0_calib_filt, H1_calib_filt, H0_calib_cos_filt, H1_calib_cos_filt, used_calib_filter = \
        _filter_ambiguous_region(
            H0_calib_eff, H1_calib_eff,
            H0_calib_eff_cos, H1_calib_eff_cos,
            cos_min=0.7, cos_max=0.9,
            min_samples_per_class=50,  # Reduced to allow filtering with smaller datasets
            trial=trial,
        )
    
    for name, method in list(methods.items()):
        if not hasattr(method, "fit"):
            continue
        if name == "CosineAffineCalib":
            continue
        
        # Determine which data to use: filtered (for learned methods) or full (for Cosine)
        use_filtered = name not in COSINE_METHODS
        
        if use_filtered:
            H0_train_use = H0_train_filt
            H1_train_use = H1_train_filt
            H0_calib_use = H0_calib_filt
            H1_calib_use = H1_calib_filt
            H0_train_cos_use = H0_train_cos_filt
            H1_train_cos_use = H1_train_cos_filt
            H0_calib_cos_use = H0_calib_cos_filt
            H1_calib_cos_use = H1_calib_cos_filt
        else:
            H0_train_use = H0_train
            H1_train_use = H1_train
            H0_calib_use = H0_calib_eff
            H1_calib_use = H1_calib_eff
            H0_train_cos_use = H0_train_cos
            H1_train_cos_use = H1_train_cos
            H0_calib_cos_use = H0_calib_eff_cos
            H1_calib_cos_use = H1_calib_eff_cos
        
        try:
            w = weights if needs_weights(method) else None
            s_for_method = seed if needs_seed(method) else seed

            if method_train_required(method):
                if H0_train_use.shape[0] == 0 or H1_train_use.shape[0] == 0:
                    failures[name].append(f"trial={trial}: fit required but pooled train empty")
                    continue
                
                # Special handling for WeightedEnsemble with alt matrices
                if name == "WeightedEnsemble":
                    fit_kwargs = {
                        "weights": w,
                        "seed": s_for_method,
                        "alpha": alpha,
                        "H0_calib": H0_calib_use,
                        "H1_calib": H1_calib_use,
                        "H0_train_alt": H0_train_cos_use,
                        "H1_train_alt": H1_train_cos_use,
                        "H0_calib_alt": H0_calib_cos_use,
                        "H1_calib_alt": H1_calib_cos_use,
                        "judge_input": {"Cosine": "alt"},  # Route Cosine to alt (Hadamard) matrices
                        "tie_mode": tie_mode,
                        "guardrail": tau_guardrail,
                        "guardrail_delta": tau_guardrail_delta,
                    }
                    try:
                        method.fit(H0_train_use, H1_train_use, **fit_kwargs)
                    except TypeError as e:
                        # Fallback if method doesn't support these params
                        failures[name].append(f"trial={trial}: fit with alt matrices failed: {e}")
                        try_fit_method(method, H0_train_use, H1_train_use, weights=w, seed=s_for_method, alpha=alpha)
                else:
                    # Try passing external calib data for other NP-safe methods
                    fit_kwargs = {"weights": w, "seed": s_for_method, "alpha": alpha}
                    fit_kwargs_with_calib = {**fit_kwargs,
                                             "H0_calib": H0_calib_use,
                                             "H1_calib": H1_calib_use}
                    try:
                        method.fit(H0_train_use, H1_train_use, **fit_kwargs_with_calib)
                    except TypeError:
                        # Method doesn't support external calib, use try_fit_method fallback
                        try_fit_method(method, H0_train_use, H1_train_use, weights=w, seed=s_for_method, alpha=alpha)
            else:
                try_fit_method(method, H0_calib_use, H1_calib_use, weights=w, seed=s_for_method, alpha=alpha)

        except Exception as exc:
            failures[name].append(f"trial={trial}: fit failed: {exc}")

def evaluate_methods(
    methods: Dict[str, Any],
    method_names: List[str],
    splits: List[RegionSplit],
    *,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    alpha: float,
    tau_mode: str,
    tie_mode: str,
    trial: int,
    seed: int,
    region_key: str,
    h0_train_idx_list: List[np.ndarray],
    h1_train_idx_list: List[np.ndarray],
    h0_calib_list: List[np.ndarray],
    h1_calib_list: List[np.ndarray],
    H0_calib_eff: np.ndarray,
    tau_shrink: bool,
    tau_shrink_m: float,
    tau_guardrail: str,
    tau_guardrail_delta: float,
    swc_mode: str,
    swc_cluster_n_clusters: int,
    cos_affine_grouping: str,
    cos_affine_n_clusters: int,
    failures: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Evaluate all methods across regions for one trial. Returns per-method rows."""
    trial_rows: List[Dict[str, Any]] = []

    for name in method_names:
        if name not in methods:
            continue
        method = methods[name]

        micro_tp = micro_fp = micro_tn = micro_fn = 0
        train_tp = train_fp = train_tn = train_fn = 0
        macro_tprs: List[float] = []
        macro_fprs: List[float] = []
        tau_values: List[float] = []

        t_method_start = time.perf_counter()
        ok_regions = 0

        # choose feature matrix for this method
        use_cos = (name in {"Cosine", "CosineAffineCalib"} and X_cos is not None)
        if name == "CosineAffineCalib" and X_cos is None:
            failures[name].append(f"trial={trial}: CosineAffineCalib requires X_cos")
            continue
        X_use = X_cos if use_cos else X_main

        h0_pool_idx = np.concatenate(h0_train_idx_list + h0_calib_list) \
            if (h0_train_idx_list or h0_calib_list) else np.array([], dtype=np.int64)
        h1_pool_idx = np.concatenate(h1_train_idx_list + h1_calib_list) \
            if (h1_train_idx_list or h1_calib_list) else np.array([], dtype=np.int64)

        if name == "CosineAffineCalib":
            try:
                method.fit(
                    X_use[h0_pool_idx] if h0_pool_idx.size > 0 else X_use[:0],
                    X_use[h1_pool_idx] if h1_pool_idx.size > 0 else X_use[:0],
                    seed=seed,
                    alpha=alpha,
                )
            except Exception as exc:
                failures[name].append(f"trial={trial}: CosineAffineCalib global fit failed: {exc}")
                continue

        # Build per-region calibration/effective indices
        h0_idx_by_rid: Dict[int, np.ndarray] = {}
        h1_idx_by_rid: Dict[int, np.ndarray] = {}
        region_idx_for_proto: Dict[int, np.ndarray] = {}
        for s in splits:
            if method_train_required(method):
                h0_idx = s.H0_calib
                h1_idx = s.H1_calib
            else:
                h0_idx = np.concatenate([s.H0_train, s.H0_calib]) if s.H0_train.size > 0 else s.H0_calib
                h1_idx = np.concatenate([s.H1_train, s.H1_calib]) if s.H1_train.size > 0 else s.H1_calib
            h0_idx_by_rid[int(s.rid)] = h0_idx
            h1_idx_by_rid[int(s.rid)] = h1_idx
            region_idx_for_proto[int(s.rid)] = np.concatenate([h0_idx, h1_idx]) if (h0_idx.size + h1_idx.size) > 0 else np.array([], dtype=np.int64)

        cos_affine_gid_by_rid: Dict[int, int] = {}
        if name == "CosineAffineCalib":
            if cos_affine_grouping == "cluster":
                cos_affine_gid_by_rid = _build_region_cluster_map(
                    splits=splits,
                    X_proto=X_use,
                    region_to_idx=region_idx_for_proto,
                    n_clusters=cos_affine_n_clusters,
                    seed=seed,
                )
            else:
                cos_affine_gid_by_rid = {int(s.rid): int(s.rid) for s in splits}

            scores_all: List[np.ndarray] = []
            y_all: List[np.ndarray] = []
            gid_all: List[np.ndarray] = []
            for s in splits:
                rid = int(s.rid)
                gid = int(cos_affine_gid_by_rid.get(rid, rid))
                idx0 = h0_idx_by_rid[rid]
                idx1 = h1_idx_by_rid[rid]
                if idx0.size > 0:
                    sc0 = np.asarray(np.sum(X_use[idx0], axis=1), dtype=np.float32).reshape(-1)
                    scores_all.append(sc0)
                    y_all.append(np.zeros(sc0.shape[0], dtype=np.int32))
                    gid_all.append(np.full(sc0.shape[0], gid, dtype=np.int64))
                if idx1.size > 0:
                    sc1 = np.asarray(np.sum(X_use[idx1], axis=1), dtype=np.float32).reshape(-1)
                    scores_all.append(sc1)
                    y_all.append(np.ones(sc1.shape[0], dtype=np.int32))
                    gid_all.append(np.full(sc1.shape[0], gid, dtype=np.int64))

            if scores_all:
                try:
                    method.fit_group_calibrators(
                        np.concatenate(scores_all, axis=0),
                        np.concatenate(y_all, axis=0),
                        np.concatenate(gid_all, axis=0),
                    )
                except Exception as exc:
                    failures[name].append(f"trial={trial}: CosineAffineCalib group fit failed: {exc}")

        swc_gid_by_rid: Dict[int, int] = {}
        swc_fit_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if name == "StabilizedWhitenedCosine" and swc_mode == "cluster":
            swc_gid_by_rid = _build_region_cluster_map(
                splits=splits,
                X_proto=X_main,
                region_to_idx=region_idx_for_proto,
                n_clusters=swc_cluster_n_clusters,
                seed=seed,
            )
            for s in splits:
                rid = int(s.rid)
                gid = int(swc_gid_by_rid.get(rid, rid))
                if gid not in swc_fit_cache:
                    swc_fit_cache[gid] = (
                        np.array([], dtype=np.int64),
                        np.array([], dtype=np.int64),
                    )
                h0_prev, h1_prev = swc_fit_cache[gid]
                swc_fit_cache[gid] = (
                    np.concatenate([h0_prev, h0_idx_by_rid[rid]]) if h0_idx_by_rid[rid].size > 0 else h0_prev,
                    np.concatenate([h1_prev, h1_idx_by_rid[rid]]) if h1_idx_by_rid[rid].size > 0 else h1_prev,
                )

        tau_global: Optional[float] = None
        if tau_mode == "global" or tau_shrink:
            try:
                if use_cos:
                    if name == "CosineAffineCalib" and hasattr(method, "set_active_group"):
                        method.set_active_group(None)
                    sc0_cal = np.asarray(
                        _score_method_with_routing(
                            method, name,
                            X_main[h0_pool_idx],
                            X_cos[h0_pool_idx] if X_cos is not None else None,
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                else:
                    sc0_cal = np.asarray(
                        _score_method_with_routing(
                            method, name,
                            H0_calib_eff,
                            X_cos[np.concatenate(h0_train_idx_list + h0_calib_list)] if X_cos is not None and (h0_train_idx_list or h0_calib_list) else None,
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                tau_global = _select_tau(
                    sc0_cal,
                    alpha=alpha,
                    tie_mode=tie_mode,
                    guardrail=tau_guardrail,
                    guardrail_delta=tau_guardrail_delta,
                )
                if not np.isfinite(tau_global):
                    tau_global = float(np.quantile(sc0_cal, 1.0 - alpha))
                    failures[name].append(
                        f"trial={trial}: guardrail infeasible for global tau (n0={int(sc0_cal.size)}, alpha={alpha}, delta={tau_guardrail_delta}, method={tau_guardrail}) — fell back to empirical quantile tau={tau_global:.4f}"
                    )
            except Exception as exc:
                failures[name].append(f"trial={trial}: global tau failed: {exc}")
                continue

        for s in splits:
            rid = int(s.rid)
            h0_cal_idx = h0_idx_by_rid[rid]
            h1_cal_idx = h1_idx_by_rid[rid]

            H0_cal_r = X_use[h0_cal_idx] if h0_cal_idx.size > 0 else X_use[:0]
            H1_cal_r = X_use[h1_cal_idx] if h1_cal_idx.size > 0 else X_use[:0]
            H0_ev = X_use[s.H0_eval]
            H1_ev = X_use[s.H1_eval]

            try:
                if name == "CosineAffineCalib" and hasattr(method, "set_active_group"):
                    gid = int(cos_affine_gid_by_rid.get(rid, rid))
                    method.set_active_group(gid)

                # Per-region fitting for methods that support it (e.g. StabilizedWhitenedCosine)
                if getattr(method, "supports_local_fit", False) and name == "StabilizedWhitenedCosine":
                    try:
                        if swc_mode == "region":
                            method.fit_region(H0_cal_r, H1_cal_r)
                        elif swc_mode == "cluster":
                            gid = int(swc_gid_by_rid.get(rid, rid))
                            h0_gid_idx, h1_gid_idx = swc_fit_cache.get(
                                gid,
                                (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
                            )
                            method.fit_region(
                                X_main[h0_gid_idx] if h0_gid_idx.size > 0 else X_main[:0],
                                X_main[h1_gid_idx] if h1_gid_idx.size > 0 else X_main[:0],
                            )
                    except Exception as exc_lr:
                        failures[name].append(
                            f"trial={trial} region={rid}: fit_region failed: {exc_lr}"
                        )

                if tau_mode == "local":
                    sc0_cal_r = np.asarray(
                        _score_method_with_routing(
                            method, name,
                            H0_cal_r,
                            X_cos[h0_cal_idx] if X_cos is not None and h0_cal_idx.size > 0 else None,
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    if sc0_cal_r.size == 0:
                        failures[name].append(f"trial={trial} region={rid}: empty calib for tau")
                        continue
                    tau_local = _select_tau(
                        sc0_cal_r,
                        alpha=alpha,
                        tie_mode=tie_mode,
                        guardrail=tau_guardrail,
                        guardrail_delta=tau_guardrail_delta,
                    )
                    if not np.isfinite(tau_local):
                        tau_local = float(np.quantile(sc0_cal_r, 1.0 - alpha))
                        failures[name].append(
                            f"trial={trial} region={rid}: guardrail infeasible (n0={int(sc0_cal_r.size)}, alpha={alpha}, delta={tau_guardrail_delta}, method={tau_guardrail}) — fell back to empirical tau={tau_local:.4f}"
                        )
                    if tau_shrink:
                        n0_r = float(sc0_cal_r.size)
                        lam = n0_r / (n0_r + float(max(tau_shrink_m, 1e-9)))
                        tau_r = (1.0 - lam) * float(tau_global) + lam * float(tau_local)
                    else:
                        tau_r = float(tau_local)
                else:
                    tau_r = float(tau_global)  # type: ignore[arg-type]

                sc0_ev = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H0_ev,
                        X_cos[s.H0_eval] if X_cos is not None else None,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                sc1_ev = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H1_ev,
                        X_cos[s.H1_eval] if X_cos is not None else None,
                    ),
                    dtype=np.float32,
                ).reshape(-1)

                p0 = apply_threshold(sc0_ev, tau_r, tie_mode)
                p1 = apply_threshold(sc1_ev, tau_r, tie_mode)

                fpr_r = float(np.mean(p0 == 1))
                tpr_r = float(np.mean(p1 == 1))

                micro_fp += int(np.sum(p0 == 1))
                micro_tn += int(np.sum(p0 == 0))
                micro_tp += int(np.sum(p1 == 1))
                micro_fn += int(np.sum(p1 == 0))

                # Train/calib metrics: score calibration data against the same tau
                sc0_cal_full = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H0_cal_r,
                        X_cos[h0_cal_idx] if X_cos is not None and h0_cal_idx.size > 0 else None,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                sc1_cal_full = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H1_cal_r,
                        X_cos[h1_cal_idx] if X_cos is not None and h1_cal_idx.size > 0 else None,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                p0_tr = apply_threshold(sc0_cal_full, tau_r, tie_mode)
                p1_tr = apply_threshold(sc1_cal_full, tau_r, tie_mode)
                train_fp += int(np.sum(p0_tr == 1))
                train_tn += int(np.sum(p0_tr == 0))
                train_tp += int(np.sum(p1_tr == 1))
                train_fn += int(np.sum(p1_tr == 0))

                macro_fprs.append(fpr_r)
                macro_tprs.append(tpr_r)
                tau_values.append(tau_r)
                ok_regions += 1

            except Exception as exc:
                failures[name].append(f"trial={trial} region={rid}: score failed: {exc}")
                continue

        t_method_ms = (time.perf_counter() - t_method_start) * 1000.0
        if ok_regions == 0:
            failures[name].append(f"trial={trial}: no regions evaluated")
            continue

        micro_tpr = float(micro_tp / max(1, (micro_tp + micro_fn)))
        micro_fpr = float(micro_fp / max(1, (micro_fp + micro_tn)))
        train_tpr = float(train_tp / max(1, (train_tp + train_fn)))
        train_fpr = float(train_fp / max(1, (train_fp + train_tn)))
        macro_tpr = float(np.mean(macro_tprs)) if macro_tprs else float("nan")
        macro_fpr = float(np.mean(macro_fprs)) if macro_fprs else float("nan")
        tau_mean = float(np.mean(tau_values)) if tau_values else float("nan")
        tau_out = float(tau_global) if tau_mode == "global" and tau_global is not None else tau_mean

        row = {
            "trial": trial,
            "seed": seed,
            "method": name,
            "region_key": region_key,
            "tau_mode": tau_mode,
            "tau": tau_out,
            "tau_mean": tau_mean,
            "micro_tpr": micro_tpr,
            "micro_fpr": micro_fpr,
            "train_tpr": train_tpr,
            "train_fpr": train_fpr,
            "macro_tpr": macro_tpr,
            "macro_fpr": macro_fpr,
            "ok_regions": int(ok_regions),
            "time_ms": float(t_method_ms),
        }
        trial_rows.append(row)

    return trial_rows


def aggregate_ranking(
    trial_summary_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-trial rows into a ranking sorted by mean micro TPR."""
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in trial_summary_rows:
        agg[r["method"]]["micro_tpr"].append(float(r["micro_tpr"]))
        agg[r["method"]]["micro_fpr"].append(float(r["micro_fpr"]))
        agg[r["method"]]["macro_tpr"].append(float(r["macro_tpr"]))
        agg[r["method"]]["macro_fpr"].append(float(r["macro_fpr"]))

    ranking = []
    for m, d in agg.items():
        ranking.append(
            {
                "method": m,
                "mean_micro_tpr": float(np.mean(d["micro_tpr"])),
                "std_micro_tpr": float(np.std(d["micro_tpr"])),
                "mean_micro_fpr": float(np.mean(d["micro_fpr"])),
                "std_micro_fpr": float(np.std(d["micro_fpr"])),
                "mean_macro_tpr": float(np.mean(d["macro_tpr"])),
                "std_macro_tpr": float(np.std(d["macro_tpr"])),
                "mean_macro_fpr": float(np.mean(d["macro_fpr"])),
                "std_macro_fpr": float(np.std(d["macro_fpr"])),
            }
        )
    ranking.sort(key=lambda r: r["mean_micro_tpr"], reverse=True)
    return ranking


def evaluate_methods_global(
    methods: Dict[str, Any],
    method_names: List[str],
    gs: GlobalSplit,
    *,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    alpha: float,
    tie_mode: str,
    tau_guardrail: str,
    tau_guardrail_delta: float,
    trial: int,
    seed: int,
    region_key: str,
    region_id: Optional[np.ndarray],
    failures: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Evaluate all methods with a single global tau on pooled eval data.

    Macro TPR/FPR are optionally computed by grouping eval samples via
    *region_id* (metadata only — no sample is discarded).
    """
    trial_rows: List[Dict[str, Any]] = []

    for name in method_names:
        if name not in methods:
            continue
        method = methods[name]

        use_cos = (name in {"Cosine", "CosineAffineCalib"} and X_cos is not None)
        if name == "CosineAffineCalib" and X_cos is None:
            failures[name].append(f"trial={trial}: CosineAffineCalib requires X_cos")
            continue
        X_use = X_cos if use_cos else X_main

        t_start = time.perf_counter()

        # Determine whether this method needs separate train/calib splits
        requires_train_split = method_train_required(method)

        # Build calibration indices for tau computation
        # - For train-required methods (e.g., WeightedEnsemble): use CALIB only
        # - For others: use pooled TRAIN+CALIB
        if requires_train_split:
            h0_calib_eff_idx = gs.H0_calib
            h1_calib_eff_idx = gs.H1_calib
            h0_train_idx = gs.H0_train
            h1_train_idx = gs.H1_train
        else:
            h0_calib_eff_idx = np.concatenate([gs.H0_train, gs.H0_calib]) \
                if gs.H0_train.size > 0 else gs.H0_calib
            h1_calib_eff_idx = np.concatenate([gs.H1_train, gs.H1_calib]) \
                if gs.H1_train.size > 0 else gs.H1_calib

        # Fit method (only if not already fitted by fit_all_methods)
        if name == "CosineAffineCalib":
            try:
                method.fit(X_use[h0_calib_eff_idx], X_use[h1_calib_eff_idx], seed=seed, alpha=alpha)
                if hasattr(method, "set_active_group"):
                    method.set_active_group(None)
            except Exception as exc:
                failures[name].append(f"trial={trial}: CosineAffineCalib global fit failed: {exc}")
                continue
        # For train-required methods, fitting is already done by fit_all_methods()
        # No need to refit here

        # --- calibrate single global tau ---
        try:
            sc0_cal = np.asarray(
                _score_method_with_routing(
                    method, name,
                    X_main[h0_calib_eff_idx],
                    X_cos[h0_calib_eff_idx] if X_cos is not None else None,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if sc0_cal.size == 0:
                failures[name].append(f"trial={trial}: empty calib for global tau")
                continue
            tau = _select_tau(
                sc0_cal,
                alpha=alpha,
                tie_mode=tie_mode,
                guardrail=tau_guardrail,
                guardrail_delta=tau_guardrail_delta,
            )
            if not np.isfinite(tau):
                tau = float(np.quantile(sc0_cal, 1.0 - alpha))
                failures[name].append(
                    f"trial={trial}: guardrail infeasible (n0={int(sc0_cal.size)}, alpha={alpha}, delta={tau_guardrail_delta}, method={tau_guardrail}) — fell back to empirical tau={tau:.4f}"
                )
        except Exception as exc:
            failures[name].append(f"trial={trial}: global tau failed: {exc}")
            continue

        # --- score pooled eval ---
        try:
            sc0_ev = np.asarray(
                _score_method_with_routing(
                    method, name,
                    X_main[gs.H0_eval],
                    X_cos[gs.H0_eval] if X_cos is not None else None,
                ),
                dtype=np.float32,
            ).reshape(-1)
            sc1_ev = np.asarray(
                _score_method_with_routing(
                    method, name,
                    X_main[gs.H1_eval],
                    X_cos[gs.H1_eval] if X_cos is not None else None,
                ),
                dtype=np.float32,
            ).reshape(-1)

            p0 = apply_threshold(sc0_ev, tau, tie_mode)
            p1 = apply_threshold(sc1_ev, tau, tie_mode)
        except Exception as exc:
            failures[name].append(f"trial={trial}: global eval scoring failed: {exc}")
            continue

        micro_fp = int(np.sum(p0 == 1))
        micro_tn = int(np.sum(p0 == 0))
        micro_tp = int(np.sum(p1 == 1))
        micro_fn = int(np.sum(p1 == 0))
        micro_tpr = float(micro_tp / max(1, micro_tp + micro_fn))
        micro_fpr = float(micro_fp / max(1, micro_fp + micro_tn))

        # --- train/calib metrics ---
        try:
            sc0_tr = np.asarray(
                _score_method_with_routing(
                    method, name,
                    X_main[h0_calib_eff_idx],
                    X_cos[h0_calib_eff_idx] if X_cos is not None else None,
                ),
                dtype=np.float32,
            ).reshape(-1)
            sc1_tr = np.asarray(
                _score_method_with_routing(
                    method, name,
                    X_main[h1_calib_eff_idx],
                    X_cos[h1_calib_eff_idx] if X_cos is not None else None,
                ),
                dtype=np.float32,
            ).reshape(-1)
            p0_tr = apply_threshold(sc0_tr, tau, tie_mode)
            p1_tr = apply_threshold(sc1_tr, tau, tie_mode)
            train_tpr = float(np.sum(p1_tr == 1) / max(1, p1_tr.size))
            train_fpr = float(np.sum(p0_tr == 1) / max(1, p0_tr.size))
        except Exception:
            train_tpr = float("nan")
            train_fpr = float("nan")

        # --- optional macro stats by region (metadata only, no gating) ---
        macro_tpr = float("nan")
        macro_fpr = float("nan")
        n_macro_regions = 0
        if region_id is not None:
            rid_h0 = region_id[gs.H0_eval]
            rid_h1 = region_id[gs.H1_eval]
            all_rids = np.unique(np.concatenate([rid_h0, rid_h1]))
            region_tprs: List[float] = []
            region_fprs: List[float] = []
            for rid in all_rids:
                mask0 = rid_h0 == rid
                mask1 = rid_h1 == rid
                if mask0.sum() > 0:
                    region_fprs.append(float(np.mean(p0[mask0] == 1)))
                if mask1.sum() > 0:
                    region_tprs.append(float(np.mean(p1[mask1] == 1)))
            macro_tpr = float(np.mean(region_tprs)) if region_tprs else float("nan")
            macro_fpr = float(np.mean(region_fprs)) if region_fprs else float("nan")
            n_macro_regions = int(all_rids.size)

        t_ms = (time.perf_counter() - t_start) * 1000.0

        row = {
            "trial": trial,
            "seed": seed,
            "method": name,
            "region_key": region_key,
            "tau_mode": "global",
            "tau": float(tau),
            "tau_mean": float(tau),
            "micro_tpr": micro_tpr,
            "micro_fpr": micro_fpr,
            "train_tpr": train_tpr,
            "train_fpr": train_fpr,
            "macro_tpr": macro_tpr,
            "macro_fpr": macro_fpr,
            "ok_regions": n_macro_regions,
            "time_ms": float(t_ms),
        }
        trial_rows.append(row)

    return trial_rows
