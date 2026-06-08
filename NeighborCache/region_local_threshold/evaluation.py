"""Threshold application and per-region evaluation loop."""
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from np_bench.thresholding import select_np_threshold

from .methods import method_input_space, needs_weights, needs_seed, try_fit_method
from .splits import RegionSplit, GlobalSplit


DEBUG_ABLATION_SCORES = False
DEBUG_ABLATION_COMPARE_METHODS = (
    "PCAWhitenedCosine",
    "ablation:pca_whitened_hadamard_linear",
    "ablation:raw_hadamard_linear",
    "ablation:diag_whitened_hadamard_linear",
)


def _debug_enabled() -> bool:
    return bool(DEBUG_ABLATION_SCORES)


def _method_debug_mode(method: Any) -> str:
    for attr in ("resolved_ablation", "ablation", "score_mode", "mode"):
        value = getattr(method, attr, None)
        if value is not None:
            return f"{attr}={value}"
    return "mode=none"


def _input_route_label(space: str) -> str:
    if space == "mixed":
        return "preprocessed_feature_matrix+aux"
    if space == "scalar_score":
        return "scalar_score_matrix"
    if space == "text_pair":
        return "text_pair_matrix"
    return "preprocessed_feature_matrix"


def _score_stats(scores: np.ndarray) -> tuple[float, float, float, float, list[float]]:
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), []
    return (
        float(np.mean(flat)),
        float(np.std(flat)),
        float(np.min(flat)),
        float(np.max(flat)),
        [float(x) for x in flat[:5]],
    )


def _debug_print_fit(name: str, method: Any, space: str, fit_H0: np.ndarray, fit_H1: np.ndarray) -> None:
    if not _debug_enabled():
        return
    print(
        "[DEBUG_ABLATION_FIT] "
        f"method={name} class={type(method).__name__} {_method_debug_mode(method)} "
        f"input_route={_input_route_label(space)} "
        f"train_shapes=H0{tuple(fit_H0.shape)} H1{tuple(fit_H1.shape)}"
    )


def _debug_print_scores(
    name: str,
    method: Any,
    space: str,
    eval_H0_shape: tuple[int, ...],
    eval_H1_shape: tuple[int, ...],
    scores: np.ndarray,
) -> None:
    if not _debug_enabled():
        return
    mean, std, min_val, max_val, first5 = _score_stats(scores)
    print(
        "[DEBUG_ABLATION_SCORE] "
        f"method={name} class={type(method).__name__} {_method_debug_mode(method)} "
        f"input_route={_input_route_label(space)} "
        f"eval_shapes=H0{tuple(eval_H0_shape)} H1{tuple(eval_H1_shape)} "
        f"score_mean={mean:.6f} score_std={std:.6f} score_min={min_val:.6f} score_max={max_val:.6f} "
        f"first5={first5}"
    )


def _debug_print_pairwise_comparisons(score_vectors: Dict[str, np.ndarray]) -> None:
    if not _debug_enabled():
        return

    names = [name for name in DEBUG_ABLATION_COMPARE_METHODS if name in score_vectors]
    if len(names) < 2:
        return

    for left_name, right_name in itertools.combinations(names, 2):
        left = np.asarray(score_vectors[left_name], dtype=np.float64).reshape(-1)
        right = np.asarray(score_vectors[right_name], dtype=np.float64).reshape(-1)
        n = min(left.size, right.size)
        if n == 0:
            corr = float("nan")
            max_abs_diff = float("nan")
            mean_abs_diff = float("nan")
            allclose = False
        else:
            left = left[:n]
            right = right[:n]
            diff = np.abs(left - right)
            max_abs_diff = float(np.max(diff))
            mean_abs_diff = float(np.mean(diff))
            if np.std(left) > 0.0 and np.std(right) > 0.0:
                corr = float(np.corrcoef(left, right)[0, 1])
            else:
                corr = float("nan")
            allclose = bool(np.allclose(left, right))
        print(
            f"[DEBUG_COMPARE] {left_name} vs {right_name}:\n"
            f"  max_abs_diff={max_abs_diff:.6e}\n"
            f"  mean_abs_diff={mean_abs_diff:.6e}\n"
            f"  corr={corr:.6f}\n"
            f"  allclose={allclose}"
        )


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
    return select_np_threshold(
        scores,
        alpha=alpha,
        tie_mode=tie_mode,
        guardrail=guardrail,
        guardrail_delta=guardrail_delta,
    )


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
    X_text_slice: Optional[np.ndarray],
    region_ids_slice: Optional[np.ndarray] = None,
    row_indices_slice: Optional[np.ndarray] = None,
) -> np.ndarray:
    score_with_indices = getattr(method, "score_with_indices", None)
    if callable(score_with_indices):
        if row_indices_slice is None:
            raise ValueError(f"method={method_name} requires original row indices for paired scoring")
        return score_with_indices(np.asarray(row_indices_slice, dtype=np.int64))

    space = method_input_space(method)
    needs_region_ids = bool(getattr(method, "requires_region_ids", False))

    if space == "mixed":
        if needs_region_ids:
            return method.score(X_main_slice, X_alt=X_cos_slice, region_ids=region_ids_slice)
        return method.score(X_main_slice, X_alt=X_cos_slice)

    if space == "scalar_score":
        if X_cos_slice is None:
            raise ValueError(f"method={method_name} requires scalar_score input but X_cos is None")
        x = np.asarray(X_cos_slice)
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(
                f"method={method_name} expects scalar_score matrix with shape (N,1), got {x.shape}"
            )
        return method.score(x)

    if space == "text_pair":
        if X_text_slice is None:
            raise ValueError(f"method={method_name} requires text_pair input but X_text is None")
        x = np.asarray(X_text_slice, dtype=object)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(
                f"method={method_name} expects text_pair matrix with shape (N,2), got {x.shape}"
            )
        return method.score(x)

    x = np.asarray(X_main_slice)
    if x.ndim != 2 or x.shape[1] <= 1:
        raise ValueError(
            f"method={method_name} expects embedding input with shape (N,D), D>1, got {x.shape}"
        )
    return method.score(x)


def fit_all_methods(
    methods: Dict[str, Any],
    *,
    H0_train: np.ndarray,
    H1_train: np.ndarray,
    H0_calib_eff: np.ndarray,
    H1_calib_eff: np.ndarray,
    H0_calib_pure: Optional[np.ndarray] = None,
    H1_calib_pure: Optional[np.ndarray] = None,
    H0_train_cos: Optional[np.ndarray],
    H1_train_cos: Optional[np.ndarray],
    H0_calib_eff_cos: Optional[np.ndarray],
    H1_calib_eff_cos: Optional[np.ndarray],
    H0_train_text: Optional[np.ndarray],
    H1_train_text: Optional[np.ndarray],
    H0_calib_eff_text: Optional[np.ndarray],
    H1_calib_eff_text: Optional[np.ndarray],
    H0_calib_pure_cos: Optional[np.ndarray] = None,
    H1_calib_pure_cos: Optional[np.ndarray] = None,
    H0_calib_region_ids: Optional[np.ndarray] = None,
    H1_calib_region_ids: Optional[np.ndarray] = None,
    weights: np.ndarray,
    seed: int,
    alpha: float,
    tie_mode: str,
    tau_guardrail: str,
    tau_guardrail_delta: float,
    trial: int,
    failures: Dict[str, List[str]],
    require_pure_calib_for_ensemble: bool = False,
    fit_context: str = "unknown",
) -> None:
    """Fit all methods in-place (modifies *methods* dict on failure).

    Input routing is driven strictly by each method's declared input_space.
    """
    for name, method in list(methods.items()):
        if not hasattr(method, "fit"):
            continue

        try:
            w = weights if needs_weights(method) else None
            s_for_method = seed if needs_seed(method) else None
            space = method_input_space(method)

            if space == "embedding":
                if H0_train.shape[1] <= 1 or H1_train.shape[1] <= 1:
                    failures[name].append(
                        f"trial={trial}: method requires embedding input with dim>1"
                    )
                    methods.pop(name, None)
                    continue

            if space == "scalar_score":
                if H0_train_cos is None or H1_train_cos is None or H0_calib_eff_cos is None or H1_calib_eff_cos is None:
                    failures[name].append(
                        f"trial={trial}: method requires scalar_score input but X_cos is unavailable"
                    )
                    methods.pop(name, None)
                    continue
                H0_train_use = H0_train_cos
                H1_train_use = H1_train_cos
                H0_calib_use = H0_calib_eff_cos
                H1_calib_use = H1_calib_eff_cos
            elif space == "text_pair":
                if H0_train_text is None or H1_train_text is None or H0_calib_eff_text is None or H1_calib_eff_text is None:
                    failures[name].append(
                        f"trial={trial}: method requires text_pair input but X_text is unavailable"
                    )
                    methods.pop(name, None)
                    continue
                H0_train_use = H0_train_text
                H1_train_use = H1_train_text
                H0_calib_use = H0_calib_eff_text
                H1_calib_use = H1_calib_eff_text
            else:
                H0_train_use = H0_train
                H1_train_use = H1_train
                H0_calib_use = H0_calib_eff
                H1_calib_use = H1_calib_eff

            # Strict protocol: fit on TRAIN when available; reserve CALIB for tau.
            fit_H0 = H0_train_use if H0_train_use.shape[0] > 0 else H0_calib_use
            fit_H1 = H1_train_use if H1_train_use.shape[0] > 0 else H1_calib_use
            if fit_H0.shape[0] == 0 or fit_H1.shape[0] == 0:
                failures[name].append(f"trial={trial}: fit failed due to empty class in fit split")
                methods.pop(name, None)
                continue

            # Special handling for ensemble methods with per-judge routing.
            if name in {
                "WeightedEnsemble",
                "RandomForestEnsemble",
                "WeightedEnsembleNoPCAWhitenedCosine",
                "RandomForestEnsembleNoPCAWhitenedCosine",
                "RegionalWeightedEnsemble",
            }:
                # In local contexts we require pure calib inputs for external meta-calibration.
                has_pure_calib = H0_calib_pure is not None and H1_calib_pure is not None
                if require_pure_calib_for_ensemble and not has_pure_calib:
                    msg = (
                        f"trial={trial}: {name} requires pure calib in fit_context={fit_context}, "
                        "but H*_calib_pure is unavailable; skipping method"
                    )
                    failures[name].append(msg)
                    print(f"[WARN] {msg}")
                    methods.pop(name, None)
                    continue

                H0_cal_for_ens = H0_calib_pure if has_pure_calib else H0_calib_use
                H1_cal_for_ens = H1_calib_pure if has_pure_calib else H1_calib_use

                has_pure_calib_alt = H0_calib_pure_cos is not None and H1_calib_pure_cos is not None
                H0_cal_for_ens_alt = H0_calib_pure_cos if has_pure_calib_alt else H0_calib_eff_cos
                H1_cal_for_ens_alt = H1_calib_pure_cos if has_pure_calib_alt else H1_calib_eff_cos

                if fit_context in {"local", "matched_global_on_local"}:
                    print(
                        "[DEBUG][WeightedEnsemble][fit_all_methods] "
                        f"method={name} context={fit_context} "
                        f"using_pure_calib={bool(has_pure_calib)} "
                        f"H0_train={int(H0_train_use.shape[0])} H1_train={int(H1_train_use.shape[0])} "
                        f"H0_calib_pure={int(H0_calib_pure.shape[0]) if H0_calib_pure is not None else -1} "
                        f"H1_calib_pure={int(H1_calib_pure.shape[0]) if H1_calib_pure is not None else -1} "
                        f"H0_calib_eff={int(H0_calib_eff.shape[0])} H1_calib_eff={int(H1_calib_eff.shape[0])}"
                    )

                judge_input: Dict[str, str] = {}
                for judge in getattr(method, "judges", []):
                    j_name = str(getattr(judge, "name", type(judge).__name__))
                    if method_input_space(judge) == "scalar_score":
                        judge_input[j_name] = "alt"
                fit_kwargs = {
                    "weights": w,
                    "seed": s_for_method,
                    "alpha": alpha,
                    "H0_calib": H0_cal_for_ens,
                    "H1_calib": H1_cal_for_ens,
                    "H0_train_alt": H0_train_cos,
                    "H1_train_alt": H1_train_cos,
                    "H0_calib_alt": H0_cal_for_ens_alt,
                    "H1_calib_alt": H1_cal_for_ens_alt,
                    "judge_input": judge_input,
                    "tie_mode": tie_mode,
                    "guardrail": tau_guardrail,
                    "guardrail_delta": tau_guardrail_delta,
                    "fit_context": fit_context,
                }
                if name == "RegionalWeightedEnsemble":
                    fit_kwargs["H0_calib_region_ids"] = H0_calib_region_ids
                    fit_kwargs["H1_calib_region_ids"] = H1_calib_region_ids
                try:
                    method.fit(fit_H0, fit_H1, **fit_kwargs)
                except TypeError as e:
                    failures[name].append(f"trial={trial}: fit with routed matrices failed: {e}")
                    try_fit_method(method, fit_H0, fit_H1, weights=w, seed=s_for_method, alpha=alpha)
            else:
                fit_kwargs = {"weights": w, "seed": s_for_method, "alpha": alpha}
                fit_kwargs_with_calib = {
                    **fit_kwargs,
                    "H0_calib": H0_calib_use,
                    "H1_calib": H1_calib_use,
                }
                try:
                    method.fit(fit_H0, fit_H1, **fit_kwargs_with_calib)
                except TypeError:
                    try_fit_method(method, fit_H0, fit_H1, weights=w, seed=s_for_method, alpha=alpha)

            _debug_print_fit(name, method, space, fit_H0, fit_H1)

        except Exception as exc:
            failures[name].append(f"trial={trial}: fit failed: {exc}")
            methods.pop(name, None)

def evaluate_methods(
    methods: Dict[str, Any],
    method_names: List[str],
    splits: List[RegionSplit],
    *,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
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
    shrink_k: float,
    tau_guardrail: str,
    tau_guardrail_delta: float,
    swc_mode: str,
    swc_cluster_n_clusters: int,
    cos_affine_grouping: str,
    cos_affine_n_clusters: int,
    local_fit_mode: str,
    failures: Dict[str, List[str]],
    tau_cluster_id_by_rid: Optional[Dict[int, int]] = None,
    trial_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all methods across regions for one trial. Returns per-method rows.

    local_fit_mode:
      - "pooled": use methods pre-fitted on pooled local train/calib data.
      - "per_region": refit each method on each region's local split before scoring.
    """
    if local_fit_mode not in {"pooled", "per_region"}:
        raise ValueError(f"invalid local_fit_mode={local_fit_mode!r}; expected 'pooled' or 'per_region'")

    trial_rows: List[Dict[str, Any]] = []
    method_region_stats: Dict[str, Dict[int, Dict[str, Any]]] = {}
    method_time_ms: Dict[str, float] = {}
    method_space: Dict[str, str] = {}
    debug_score_vectors: Dict[str, np.ndarray] = {} if _debug_enabled() else {}

    for name in method_names:
        if name not in methods:
            continue
        method = methods[name]

        method_space[name] = method_input_space(method)
        if method_space[name] == "scalar_score" and X_cos is None:
            failures[name].append(f"trial={trial}: method requires scalar_score input but X_cos is unavailable")
            continue
        if method_space[name] == "text_pair" and X_text is None:
            failures[name].append(f"trial={trial}: method requires text_pair input but X_text is unavailable")
            continue
        if method_space[name] == "embedding" and X_main.shape[1] <= 1:
            failures[name].append(f"trial={trial}: method requires embedding input with dim>1")
            continue
        per_region: Dict[int, Dict[str, Any]] = {}
        debug_scores: List[np.ndarray] = []
        debug_h0_count = 0
        debug_h1_count = 0

        t_method_start = time.perf_counter()

        # Build per-region indices
        h0_train_idx_by_rid: Dict[int, np.ndarray] = {}
        h1_train_idx_by_rid: Dict[int, np.ndarray] = {}
        h0_idx_by_rid: Dict[int, np.ndarray] = {}
        h1_idx_by_rid: Dict[int, np.ndarray] = {}
        region_idx_for_proto: Dict[int, np.ndarray] = {}
        for s in splits:
            h0_tr = s.H0_train
            h1_tr = s.H1_train
            h0_idx = s.H0_calib
            h1_idx = s.H1_calib
            h0_train_idx_by_rid[int(s.rid)] = h0_tr
            h1_train_idx_by_rid[int(s.rid)] = h1_tr
            h0_idx_by_rid[int(s.rid)] = h0_idx
            h1_idx_by_rid[int(s.rid)] = h1_idx
            region_idx_for_proto[int(s.rid)] = np.concatenate([h0_idx, h1_idx]) if (h0_idx.size + h1_idx.size) > 0 else np.array([], dtype=np.int64)

        cos_affine_gid_by_rid: Dict[int, int] = {}
        if local_fit_mode == "pooled" and name == "CosineAffineCalib":
            if cos_affine_grouping == "cluster":
                cos_affine_gid_by_rid = _build_region_cluster_map(
                    splits=splits,
                    X_proto=X_main,
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
                    sc0 = np.asarray(method._raw_cosine(X_main[idx0]), dtype=np.float32).reshape(-1)
                    scores_all.append(sc0)
                    y_all.append(np.zeros(sc0.shape[0], dtype=np.int32))
                    gid_all.append(np.full(sc0.shape[0], gid, dtype=np.int64))
                if idx1.size > 0:
                    sc1 = np.asarray(method._raw_cosine(X_main[idx1]), dtype=np.float32).reshape(-1)
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
        if local_fit_mode == "pooled" and name == "StabilizedWhitenedCosine" and swc_mode == "cluster":
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
        need_global_tau = (tau_mode in {"global", "shrink_local"}) or tau_shrink
        if need_global_tau:
            try:
                h0_calib_idx_all = np.concatenate(h0_calib_list) if h0_calib_list else np.array([], dtype=np.int64)
                if h0_calib_idx_all.size == 0:
                    failures[name].append(f"trial={trial}: empty calibration H0 pool for global tau")
                    continue
                if name == "CosineAffineCalib" and hasattr(method, "set_active_group"):
                    method.set_active_group(None)
                sc0_cal = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        X_main[h0_calib_idx_all],
                        X_cos[h0_calib_idx_all] if X_cos is not None else None,
                        X_text[h0_calib_idx_all] if X_text is not None else None,
                        region_id[h0_calib_idx_all] if region_id is not None else None,
                        row_indices_slice=h0_calib_idx_all,
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

        tau_cluster_by_gid: Dict[int, float] = {}
        n_calib_h0_cluster: Dict[int, int] = {}
        n_calib_h0_region: Dict[int, int] = {}
        if tau_mode == "cluster_local":
            if not tau_cluster_id_by_rid:
                failures[name].append(
                    f"trial={trial}: tau_mode=cluster_local requires tau_cluster_id_by_rid"
                )
                continue

            missing_rids = [
                int(s.rid) for s in splits if int(s.rid) not in tau_cluster_id_by_rid
            ]
            if missing_rids:
                failures[name].append(
                    f"trial={trial}: missing cluster ids for regions={missing_rids[:20]}"
                )
                continue

            h0_scores_by_gid: Dict[int, List[np.ndarray]] = defaultdict(list)
            for s in splits:
                rid = int(s.rid)
                gid = int(tau_cluster_id_by_rid[rid])
                h0_cal_idx = h0_idx_by_rid[rid]
                h1_cal_idx = h1_idx_by_rid[rid]
                n_calib_h0_region[rid] = int(h0_cal_idx.size)

                H0_cal_r = X_main[h0_cal_idx] if h0_cal_idx.size > 0 else X_main[:0]
                H1_cal_r = X_main[h1_cal_idx] if h1_cal_idx.size > 0 else X_main[:0]

                try:
                    if name == "CosineAffineCalib" and hasattr(method, "set_active_group"):
                        gid_aff = int(cos_affine_gid_by_rid.get(rid, rid))
                        method.set_active_group(gid_aff)

                    if local_fit_mode == "pooled" and getattr(method, "supports_local_fit", False) and name == "StabilizedWhitenedCosine":
                        try:
                            if swc_mode == "region":
                                method.fit_region(H0_cal_r, H1_cal_r)
                            elif swc_mode == "cluster":
                                swc_gid = int(swc_gid_by_rid.get(rid, rid))
                                h0_gid_idx, h1_gid_idx = swc_fit_cache.get(
                                    swc_gid,
                                    (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
                                )
                                method.fit_region(
                                    X_main[h0_gid_idx] if h0_gid_idx.size > 0 else X_main[:0],
                                    X_main[h1_gid_idx] if h1_gid_idx.size > 0 else X_main[:0],
                                )
                        except Exception as exc_lr:
                            failures[name].append(
                                f"trial={trial} region={rid}: fit_region failed during cluster_local prep: {exc_lr}"
                            )

                    sc0_cal_r = np.asarray(
                        _score_method_with_routing(
                            method,
                            name,
                            H0_cal_r,
                            X_cos[h0_cal_idx] if X_cos is not None and h0_cal_idx.size > 0 else None,
                            X_text[h0_cal_idx] if X_text is not None and h0_cal_idx.size > 0 else None,
                            row_indices_slice=h0_cal_idx,
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    if sc0_cal_r.size == 0:
                        failures[name].append(f"trial={trial} region={rid}: empty calib for cluster_local tau")
                        continue

                    n_calib_h0_region[rid] = int(sc0_cal_r.size)
                    h0_scores_by_gid[gid].append(sc0_cal_r)

                except Exception as exc:
                    failures[name].append(
                        f"trial={trial} region={rid}: cluster_local prep failed: {exc}"
                    )

            for gid, parts in h0_scores_by_gid.items():
                if not parts:
                    continue
                sc0_cal_cluster = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
                n_calib_h0_cluster[int(gid)] = int(sc0_cal_cluster.size)
                tau_cluster = _select_tau(
                    sc0_cal_cluster,
                    alpha=alpha,
                    tie_mode=tie_mode,
                    guardrail=tau_guardrail,
                    guardrail_delta=tau_guardrail_delta,
                )
                if not np.isfinite(tau_cluster):
                    tau_cluster = float(np.quantile(sc0_cal_cluster, 1.0 - alpha))
                    failures[name].append(
                        f"trial={trial} cluster={int(gid)}: guardrail infeasible (n0={int(sc0_cal_cluster.size)}, alpha={alpha}, delta={tau_guardrail_delta}, method={tau_guardrail}) — fell back to empirical tau={tau_cluster:.4f}"
                    )
                tau_cluster_by_gid[int(gid)] = float(tau_cluster)

            if trial_meta is not None:
                crows = trial_meta.setdefault("cluster_local_rows", [])
                for s in splits:
                    rid = int(s.rid)
                    gid = int(tau_cluster_id_by_rid[rid])
                    tau_cluster = tau_cluster_by_gid.get(gid)
                    if tau_cluster is None:
                        continue
                    crows.append(
                        {
                            "trial": int(trial),
                            "seed": int(seed),
                            "method": str(name),
                            "rid": int(rid),
                            "cluster_id": int(gid),
                            "n_calib_h0_region": int(n_calib_h0_region[rid]),
                            "n_calib_h0_cluster": int(n_calib_h0_cluster.get(gid, 0)),
                            "tau_cluster": float(tau_cluster),
                        }
                    )

            if not tau_cluster_by_gid:
                failures[name].append(f"trial={trial}: cluster_local produced no cluster taus")
                continue

        for s in splits:
            rid = int(s.rid)
            h0_train_idx = h0_train_idx_by_rid[rid]
            h1_train_idx = h1_train_idx_by_rid[rid]
            h0_cal_idx = h0_idx_by_rid[rid]
            h1_cal_idx = h1_idx_by_rid[rid]

            H0_train_r = X_main[h0_train_idx] if h0_train_idx.size > 0 else X_main[:0]
            H1_train_r = X_main[h1_train_idx] if h1_train_idx.size > 0 else X_main[:0]
            H0_cal_r = X_main[h0_cal_idx] if h0_cal_idx.size > 0 else X_main[:0]
            H1_cal_r = X_main[h1_cal_idx] if h1_cal_idx.size > 0 else X_main[:0]
            H0_calib_eff_r = np.concatenate([H0_train_r, H0_cal_r], axis=0) if H0_train_r.shape[0] > 0 else H0_cal_r
            H1_calib_eff_r = np.concatenate([H1_train_r, H1_cal_r], axis=0) if H1_train_r.shape[0] > 0 else H1_cal_r
            H0_ev = X_main[s.H0_eval]
            H1_ev = X_main[s.H1_eval]

            if X_cos is not None:
                H0_train_cos_r = X_cos[h0_train_idx] if h0_train_idx.size > 0 else X_cos[:0]
                H1_train_cos_r = X_cos[h1_train_idx] if h1_train_idx.size > 0 else X_cos[:0]
                H0_calib_cos_r = X_cos[h0_cal_idx] if h0_cal_idx.size > 0 else X_cos[:0]
                H1_calib_cos_r = X_cos[h1_cal_idx] if h1_cal_idx.size > 0 else X_cos[:0]
                H0_calib_eff_cos_r = np.concatenate([H0_train_cos_r, H0_calib_cos_r], axis=0) if H0_train_cos_r.shape[0] > 0 else H0_calib_cos_r
                H1_calib_eff_cos_r = np.concatenate([H1_train_cos_r, H1_calib_cos_r], axis=0) if H1_train_cos_r.shape[0] > 0 else H1_calib_cos_r
            else:
                H0_train_cos_r = None
                H1_train_cos_r = None
                H0_calib_cos_r = None
                H1_calib_cos_r = None
                H0_calib_eff_cos_r = None
                H1_calib_eff_cos_r = None

            if X_text is not None:
                H0_train_text_r = X_text[h0_train_idx] if h0_train_idx.size > 0 else X_text[:0]
                H1_train_text_r = X_text[h1_train_idx] if h1_train_idx.size > 0 else X_text[:0]
                H0_calib_text_r = X_text[h0_cal_idx] if h0_cal_idx.size > 0 else X_text[:0]
                H1_calib_text_r = X_text[h1_cal_idx] if h1_cal_idx.size > 0 else X_text[:0]
                H0_calib_eff_text_r = np.concatenate([H0_train_text_r, H0_calib_text_r], axis=0) if H0_train_text_r.shape[0] > 0 else H0_calib_text_r
                H1_calib_eff_text_r = np.concatenate([H1_train_text_r, H1_calib_text_r], axis=0) if H1_train_text_r.shape[0] > 0 else H1_calib_text_r
            else:
                H0_train_text_r = None
                H1_train_text_r = None
                H0_calib_eff_text_r = None
                H1_calib_eff_text_r = None

            try:
                if local_fit_mode == "per_region":
                    if tau_mode != "local" or tau_shrink:
                        failures[name].append(
                            f"trial={trial} region={rid}: local_fit_mode=per_region currently supports tau_mode='local' without tau_shrink"
                        )
                        continue
                    if H0_calib_eff_r.shape[0] == 0 or H1_calib_eff_r.shape[0] == 0:
                        failures[name].append(
                            f"trial={trial} region={rid}: empty local fit/calib pool"
                        )
                        continue

                    v0_r = np.var(H0_calib_eff_r, axis=0)
                    v1_r = np.var(H1_calib_eff_r, axis=0)
                    weights_r = (v1_r / (v0_r + 1e-12)).astype(np.float32, copy=False)

                    fit_one = {name: method}
                    fit_all_methods(
                        fit_one,
                        H0_train=H0_train_r,
                        H1_train=H1_train_r,
                        H0_calib_eff=H0_calib_eff_r,
                        H1_calib_eff=H1_calib_eff_r,
                        H0_calib_pure=H0_cal_r,
                        H1_calib_pure=H1_cal_r,
                        H0_train_cos=H0_train_cos_r,
                        H1_train_cos=H1_train_cos_r,
                        H0_calib_eff_cos=H0_calib_eff_cos_r,
                        H1_calib_eff_cos=H1_calib_eff_cos_r,
                        H0_calib_pure_cos=H0_calib_cos_r,
                        H1_calib_pure_cos=H1_calib_cos_r,
                        H0_calib_region_ids=np.full(H0_cal_r.shape[0], int(rid), dtype=np.int64),
                        H1_calib_region_ids=np.full(H1_cal_r.shape[0], int(rid), dtype=np.int64),
                        H0_train_text=H0_train_text_r,
                        H1_train_text=H1_train_text_r,
                        H0_calib_eff_text=H0_calib_eff_text_r,
                        H1_calib_eff_text=H1_calib_eff_text_r,
                        tie_mode=tie_mode,
                        tau_guardrail=tau_guardrail,
                        tau_guardrail_delta=tau_guardrail_delta,
                        weights=weights_r,
                        seed=seed,
                        alpha=alpha,
                        trial=trial,
                        failures=failures,
                        require_pure_calib_for_ensemble=True,
                        fit_context="local_per_region",
                    )
                    if name not in fit_one:
                        failures[name].append(f"trial={trial} region={rid}: per-region fit failed")
                        continue
                    method = fit_one[name]

                if name == "CosineAffineCalib" and hasattr(method, "set_active_group"):
                    gid = int(cos_affine_gid_by_rid.get(rid, rid))
                    method.set_active_group(gid)

                # Per-region fitting for methods that support it (e.g. StabilizedWhitenedCosine)
                if local_fit_mode == "pooled" and getattr(method, "supports_local_fit", False) and name == "StabilizedWhitenedCosine":
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

                if tau_mode in {"local", "shrink_local"}:
                    if bool(getattr(method, "uses_internal_thresholds", False)):
                        tau_r = float(getattr(method, "get_region_tau")(int(rid)))
                    else:
                        sc0_cal_r = np.asarray(
                            _score_method_with_routing(
                                method, name,
                                H0_cal_r,
                                X_cos[h0_cal_idx] if X_cos is not None and h0_cal_idx.size > 0 else None,
                                X_text[h0_cal_idx] if X_text is not None and h0_cal_idx.size > 0 else None,
                                np.full(H0_cal_r.shape[0], int(rid), dtype=np.int64),
                                row_indices_slice=h0_cal_idx,
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

                        if tau_mode == "shrink_local":
                            if tau_global is None or not np.isfinite(tau_global):
                                failures[name].append(
                                    f"trial={trial} region={rid}: shrink_local requires finite tau_global"
                                )
                                continue
                            n0_r = float(sc0_cal_r.size)
                            k = float(max(shrink_k, 1e-9))
                            lambda_global = k / (k + n0_r)
                            tau_r = lambda_global * float(tau_global) + (1.0 - lambda_global) * float(tau_local)
                            if trial_meta is not None:
                                srows = trial_meta.setdefault("shrink_local_rows", [])
                                srows.append(
                                    {
                                        "trial": int(trial),
                                        "seed": int(seed),
                                        "method": str(name),
                                        "rid": int(rid),
                                        "n_calib_h0": int(sc0_cal_r.size),
                                        "lambda_global": float(lambda_global),
                                        "tau_local": float(tau_local),
                                        "tau_global": float(tau_global),
                                        "tau_shrink": float(tau_r),
                                    }
                                )
                        elif tau_shrink:
                            n0_r = float(sc0_cal_r.size)
                            lam = n0_r / (n0_r + float(max(tau_shrink_m, 1e-9)))
                            tau_r = (1.0 - lam) * float(tau_global) + lam * float(tau_local)
                        else:
                            tau_r = float(tau_local)
                elif tau_mode == "cluster_local":
                    gid = int(tau_cluster_id_by_rid.get(rid, -1)) if tau_cluster_id_by_rid else -1
                    tau_cluster = tau_cluster_by_gid.get(gid)
                    if tau_cluster is None or not np.isfinite(tau_cluster):
                        failures[name].append(
                            f"trial={trial} region={rid}: missing finite cluster-local tau for cluster={gid}"
                        )
                        continue

                    if tau_shrink:
                        if tau_global is None or not np.isfinite(tau_global):
                            failures[name].append(
                                f"trial={trial} region={rid}: tau_shrink requires finite tau_global"
                            )
                            continue
                        n0_r = float(n_calib_h0_region.get(rid, h0_cal_idx.size))
                        lam = n0_r / (n0_r + float(max(tau_shrink_m, 1e-9)))
                        tau_r = (1.0 - lam) * float(tau_global) + lam * float(tau_cluster)
                    else:
                        tau_r = float(tau_cluster)
                else:
                    tau_r = float(tau_global)  # type: ignore[arg-type]

                sc0_ev = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H0_ev,
                        X_cos[s.H0_eval] if X_cos is not None else None,
                        X_text[s.H0_eval] if X_text is not None else None,
                        np.full(H0_ev.shape[0], int(rid), dtype=np.int64),
                        row_indices_slice=s.H0_eval,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                sc1_ev = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H1_ev,
                        X_cos[s.H1_eval] if X_cos is not None else None,
                        X_text[s.H1_eval] if X_text is not None else None,
                        np.full(H1_ev.shape[0], int(rid), dtype=np.int64),
                        row_indices_slice=s.H1_eval,
                    ),
                    dtype=np.float32,
                ).reshape(-1)

                if _debug_enabled():
                    debug_scores.append(np.concatenate([sc0_ev, sc1_ev], axis=0))
                    debug_h0_count += int(H0_ev.shape[0])
                    debug_h1_count += int(H1_ev.shape[0])

                p0 = apply_threshold(sc0_ev, tau_r, tie_mode)
                p1 = apply_threshold(sc1_ev, tau_r, tie_mode)

                fpr_r = float(np.mean(p0 == 1))
                tpr_r = float(np.mean(p1 == 1))

                # Train/calib metrics: score calibration data against the same tau
                sc0_cal_full = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H0_cal_r,
                        X_cos[h0_cal_idx] if X_cos is not None and h0_cal_idx.size > 0 else None,
                        X_text[h0_cal_idx] if X_text is not None and h0_cal_idx.size > 0 else None,
                        np.full(H0_cal_r.shape[0], int(rid), dtype=np.int64),
                        row_indices_slice=h0_cal_idx,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                sc1_cal_full = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        H1_cal_r,
                        X_cos[h1_cal_idx] if X_cos is not None and h1_cal_idx.size > 0 else None,
                        X_text[h1_cal_idx] if X_text is not None and h1_cal_idx.size > 0 else None,
                        np.full(H1_cal_r.shape[0], int(rid), dtype=np.int64),
                        row_indices_slice=h1_cal_idx,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                p0_tr = apply_threshold(sc0_cal_full, tau_r, tie_mode)
                p1_tr = apply_threshold(sc1_cal_full, tau_r, tie_mode)
                per_region[rid] = {
                    "fp": int(np.sum(p0 == 1)),
                    "tn": int(np.sum(p0 == 0)),
                    "tp": int(np.sum(p1 == 1)),
                    "fn": int(np.sum(p1 == 0)),
                    "train_fp": int(np.sum(p0_tr == 1)),
                    "train_tn": int(np.sum(p0_tr == 0)),
                    "train_tp": int(np.sum(p1_tr == 1)),
                    "train_fn": int(np.sum(p1_tr == 0)),
                    "fpr": fpr_r,
                    "tpr": tpr_r,
                    "tau": float(tau_r),
                }

            except Exception as exc:
                failures[name].append(f"trial={trial} region={rid}: score failed: {exc}")
                continue

        t_method_ms = (time.perf_counter() - t_method_start) * 1000.0
        if not per_region:
            failures[name].append(f"trial={trial}: no regions evaluated")
            continue

        if _debug_enabled():
            if debug_scores:
                pooled_scores = np.concatenate(debug_scores, axis=0)
            else:
                pooled_scores = np.array([], dtype=np.float32)
            debug_score_vectors[name] = pooled_scores
            _debug_print_scores(
                name,
                method,
                method_space[name],
                (debug_h0_count, int(X_main.shape[1])),
                (debug_h1_count, int(X_main.shape[1])),
                pooled_scores,
            )

        method_time_ms[name] = float(t_method_ms)
        method_region_stats[name] = per_region

    if not method_region_stats:
        return trial_rows

    shared_rids: Optional[set[int]] = None
    all_rids: set[int] = set()
    for m_name, stats_by_rid in method_region_stats.items():
        rid_set = set(int(r) for r in stats_by_rid.keys())
        all_rids |= rid_set
        shared_rids = rid_set if shared_rids is None else (shared_rids & rid_set)

    shared = sorted(shared_rids) if shared_rids is not None else []
    if len(shared) == 0:
        raise RuntimeError(
            f"trial={trial}: comparability failure, no shared valid regions across methods"
        )

    if trial_meta is not None:
        trial_meta["tested_region_ids"] = [int(r) for r in shared]
        trial_meta["all_region_ids"] = sorted(int(r) for r in all_rids)

    for name, per_region in method_region_stats.items():
        missing = sorted(int(r) for r in (all_rids - set(per_region.keys())))
        if missing:
            failures[name].append(
                f"trial={trial}: dropped {len(missing)} region(s) for comparability; missing={missing[:20]}"
            )

        micro_tp = micro_fp = micro_tn = micro_fn = 0
        train_tp = train_fp = train_tn = train_fn = 0
        macro_tprs: List[float] = []
        macro_fprs: List[float] = []
        tau_values: List[float] = []
        for rid in shared:
            rr = per_region[rid]
            micro_fp += int(rr["fp"])
            micro_tn += int(rr["tn"])
            micro_tp += int(rr["tp"])
            micro_fn += int(rr["fn"])
            train_fp += int(rr["train_fp"])
            train_tn += int(rr["train_tn"])
            train_tp += int(rr["train_tp"])
            train_fn += int(rr["train_fn"])
            macro_fprs.append(float(rr["fpr"]))
            macro_tprs.append(float(rr["tpr"]))
            tau_values.append(float(rr["tau"]))

        micro_tpr = float(micro_tp / max(1, (micro_tp + micro_fn)))
        micro_fpr = float(micro_fp / max(1, (micro_fp + micro_tn)))
        train_tpr = float(train_tp / max(1, (train_tp + train_fn)))
        train_fpr = float(train_fp / max(1, (train_fp + train_tn)))
        macro_tpr = float(np.mean(macro_tprs)) if macro_tprs else float("nan")
        macro_fpr = float(np.mean(macro_fprs)) if macro_fprs else float("nan")
        tau_mean = float(np.mean(tau_values)) if tau_values else float("nan")
        tau_out = tau_mean

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
            "ok_regions": int(len(shared)),
            "shared_regions": int(len(shared)),
            "dropped_regions_for_comparability": int(len(all_rids) - len(shared)),
            "input_space": method_space[name],
            "time_ms": float(method_time_ms.get(name, 0.0)),
        }
        trial_rows.append(row)

    _debug_print_pairwise_comparisons(debug_score_vectors)

    return trial_rows


def aggregate_ranking(
    trial_summary_rows: List[Dict[str, Any]],
    *,
    alpha: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Aggregate per-trial rows into a safety-first ranking."""
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in trial_summary_rows:
        agg[r["method"]]["micro_tpr"].append(float(r["micro_tpr"]))
        agg[r["method"]]["micro_fpr"].append(float(r["micro_fpr"]))
        agg[r["method"]]["macro_tpr"].append(float(r["macro_tpr"]))
        agg[r["method"]]["macro_fpr"].append(float(r["macro_fpr"]))
        train_n = r.get("train_samples_needed", r.get("train_total_samples"))
        if train_n not in (None, ""):
            agg[r["method"]]["train_n"].append(float(train_n))

    ranking = []
    for m, d in agg.items():
        fprs = np.asarray(d["micro_fpr"], dtype=np.float64)
        tprs = np.asarray(d["micro_tpr"], dtype=np.float64)
        train_counts = np.asarray(d.get("train_n", []), dtype=np.float64)
        alpha_val = float(alpha) if alpha is not None else float("nan")
        valid_rate = (
            float(np.mean(fprs <= alpha_val + 1e-12))
            if alpha is not None and fprs.size > 0
            else float("nan")
        )
        mean_train_n = float(np.mean(train_counts)) if train_counts.size > 0 else float("nan")
        min_train_n = int(np.min(train_counts)) if train_counts.size > 0 else None
        max_train_n = int(np.max(train_counts)) if train_counts.size > 0 else None
        ranking.append(
            {
                "method": m,
                "mean_micro_tpr": float(np.mean(tprs)),
                "std_micro_tpr": float(np.std(tprs)),
                "mean_micro_fpr": float(np.mean(fprs)),
                "std_micro_fpr": float(np.std(fprs)),
                "mean_eval_tpr": float(np.mean(tprs)),
                "mean_eval_fpr": float(np.mean(fprs)),
                "max_eval_fpr": float(np.max(fprs)) if fprs.size > 0 else float("nan"),
                "valid_rate": valid_rate,
                "mean_train_n": mean_train_n,
                "min_train_n": min_train_n,
                "max_train_n": max_train_n,
                "mean_macro_tpr": float(np.mean(d["macro_tpr"])),
                "std_macro_tpr": float(np.std(d["macro_tpr"])),
                "mean_macro_fpr": float(np.mean(d["macro_fpr"])),
                "std_macro_fpr": float(np.std(d["macro_fpr"])),
            }
        )

    if alpha is None:
        ranking.sort(key=lambda r: r["mean_micro_tpr"], reverse=True)
    else:
        alpha_val = float(alpha)

        def safety_key(r: Dict[str, Any]) -> tuple[Any, ...]:
            valid_rate = float(r.get("valid_rate", 0.0))
            mean_fpr = float(r.get("mean_eval_fpr", r.get("mean_micro_fpr", float("inf"))))
            max_fpr = float(r.get("max_eval_fpr", float("inf")))
            mean_tpr = float(r.get("mean_eval_tpr", r.get("mean_micro_tpr", float("-inf"))))
            train_n = float(r.get("mean_train_n", float("inf")))
            if not np.isfinite(train_n):
                train_n = float("inf")
            return (
                0 if valid_rate >= 1.0 - 1e-12 else 1,
                0 if mean_fpr <= alpha_val + 1e-12 else 1,
                0 if max_fpr <= alpha_val + 1e-12 else 1,
                -mean_tpr,
                train_n,
                -valid_rate,
            )

        ranking.sort(key=safety_key)
    for i, row in enumerate(ranking, start=1):
        row["primary_safety_rank"] = int(i)
    return ranking


def evaluate_methods_global(
    methods: Dict[str, Any],
    method_names: List[str],
    gs: GlobalSplit,
    *,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
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
    debug_score_vectors: Dict[str, np.ndarray] = {} if _debug_enabled() else {}

    for name in method_names:
        if name not in methods:
            continue
        method = methods[name]
        space = method_input_space(method)
        if space == "scalar_score" and X_cos is None:
            failures[name].append(f"trial={trial}: method requires scalar_score input but X_cos is unavailable")
            continue
        if space == "text_pair" and X_text is None:
            failures[name].append(f"trial={trial}: method requires text_pair input but X_text is unavailable")
            continue
        if space == "embedding" and X_main.shape[1] <= 1:
            failures[name].append(f"trial={trial}: method requires embedding input with dim>1")
            continue

        t_start = time.perf_counter()

        # Calibration indices for tau computation: CALIB only for all methods.
        h0_calib_eff_idx = gs.H0_calib
        h1_calib_eff_idx = gs.H1_calib

        if name == "CosineAffineCalib" and hasattr(method, "set_active_group"):
            method.set_active_group(None)

        # --- calibrate single global tau ---
        internal_tau = bool(getattr(method, "uses_internal_thresholds", False))
        try:
            if internal_tau:
                tau = float(getattr(method, "global_tau", 0.0))
            else:
                sc0_cal = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        X_main[h0_calib_eff_idx],
                        X_cos[h0_calib_eff_idx] if X_cos is not None else None,
                        X_text[h0_calib_eff_idx] if X_text is not None else None,
                        region_id[h0_calib_eff_idx] if region_id is not None else None,
                        row_indices_slice=h0_calib_eff_idx,
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
            if internal_tau:
                p0 = np.asarray(
                    method.predict(
                        X_main[gs.H0_eval],
                        X_alt=(X_cos[gs.H0_eval] if X_cos is not None else None),
                        region_ids=(region_id[gs.H0_eval] if region_id is not None else None),
                        tie_mode=tie_mode,
                    ),
                    dtype=np.int32,
                ).reshape(-1)
                p1 = np.asarray(
                    method.predict(
                        X_main[gs.H1_eval],
                        X_alt=(X_cos[gs.H1_eval] if X_cos is not None else None),
                        region_ids=(region_id[gs.H1_eval] if region_id is not None else None),
                        tie_mode=tie_mode,
                    ),
                    dtype=np.int32,
                ).reshape(-1)
            else:
                sc0_ev = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        X_main[gs.H0_eval],
                        X_cos[gs.H0_eval] if X_cos is not None else None,
                        X_text[gs.H0_eval] if X_text is not None else None,
                        region_id[gs.H0_eval] if region_id is not None else None,
                        row_indices_slice=gs.H0_eval,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                sc1_ev = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        X_main[gs.H1_eval],
                        X_cos[gs.H1_eval] if X_cos is not None else None,
                        X_text[gs.H1_eval] if X_text is not None else None,
                        region_id[gs.H1_eval] if region_id is not None else None,
                        row_indices_slice=gs.H1_eval,
                    ),
                    dtype=np.float32,
                ).reshape(-1)

                pooled_scores = np.concatenate([sc0_ev, sc1_ev], axis=0)
                _debug_print_scores(
                    name,
                    method,
                    space,
                    tuple(X_main[gs.H0_eval].shape),
                    tuple(X_main[gs.H1_eval].shape),
                    pooled_scores,
                )
                if _debug_enabled():
                    debug_score_vectors[name] = pooled_scores

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
            if internal_tau:
                p0_tr = np.asarray(
                    method.predict(
                        X_main[h0_calib_eff_idx],
                        X_alt=(X_cos[h0_calib_eff_idx] if X_cos is not None else None),
                        region_ids=(region_id[h0_calib_eff_idx] if region_id is not None else None),
                        tie_mode=tie_mode,
                    ),
                    dtype=np.int32,
                ).reshape(-1)
                p1_tr = np.asarray(
                    method.predict(
                        X_main[h1_calib_eff_idx],
                        X_alt=(X_cos[h1_calib_eff_idx] if X_cos is not None else None),
                        region_ids=(region_id[h1_calib_eff_idx] if region_id is not None else None),
                        tie_mode=tie_mode,
                    ),
                    dtype=np.int32,
                ).reshape(-1)
            else:
                sc0_tr = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        X_main[h0_calib_eff_idx],
                        X_cos[h0_calib_eff_idx] if X_cos is not None else None,
                        X_text[h0_calib_eff_idx] if X_text is not None else None,
                        region_id[h0_calib_eff_idx] if region_id is not None else None,
                        row_indices_slice=h0_calib_eff_idx,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                sc1_tr = np.asarray(
                    _score_method_with_routing(
                        method, name,
                        X_main[h1_calib_eff_idx],
                        X_cos[h1_calib_eff_idx] if X_cos is not None else None,
                        X_text[h1_calib_eff_idx] if X_text is not None else None,
                        region_id[h1_calib_eff_idx] if region_id is not None else None,
                        row_indices_slice=h1_calib_eff_idx,
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
            "input_space": space,
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

    _debug_print_pairwise_comparisons(debug_score_vectors)

    return trial_rows
