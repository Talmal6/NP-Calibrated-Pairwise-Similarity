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
from np_bench.methods.faiss_backed import (
    FAISS_SUFFIX,
    PairContext,
    build_faiss_variant,
    is_faiss_variant_name,
    source_name_for_faiss,
)
from .io_helpers import resolve_npz_path, load_npz, resolve_features
from .io_helpers import resolve_text_pairs, resolve_train_pairwise_cosine
from .methods import build_methods, needs_weights
from .splits import (
    GlobalSplit,
    filter_global_split_by_score_range,
    filter_region_splits_by_score_range_detailed,
    split_indices_per_region_detailed,
    split_global,
)
from .evaluation import fit_all_methods, evaluate_methods, evaluate_methods_global, aggregate_ranking
from . import evaluation as evaluation_mod
from .display import print_trial_table, print_ranking
from .online_stopping_eval import (
    _run_online_stopping,
    _sample_monitor_from_global_eval,
    _sample_monitor_from_region_splits,
)
from .preprocessing import (
    _build_hadamard_features,
    _build_semantic_buckets_from_regions,
    _build_tau_cluster_map,
    _l2_normalize_rows,
)
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


def _concat_indices(parts: List[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(p, dtype=np.int64) for p in parts if p is not None and p.size > 0]
    if not valid:
        return np.array([], dtype=np.int64)
    return np.concatenate(valid).astype(np.int64, copy=False)


def _npz_float_matrix(ds: Dict[str, Any], key: str, n_rows: int) -> Optional[np.ndarray]:
    if key not in ds:
        return None
    try:
        arr = np.asarray(ds[key], dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] != int(n_rows) or arr.shape[1] <= 0:
        return None
    return arr


def _optional_row_ids(ds: Dict[str, Any], n_rows: int) -> Optional[np.ndarray]:
    for key in (
        "anchor_id",
        "anchor_ids",
        "anchor_qid",
        "anchor_doc_id",
        "candidate_id",
        "doc_id",
        "region_id",
    ):
        if key not in ds:
            continue
        arr = np.asarray(ds[key])
        if arr.ndim >= 1 and arr.shape[0] == int(n_rows):
            return arr.reshape(-1)
    return None


def _sampled_hadamard_match(
    X_main: np.ndarray,
    query: np.ndarray,
    anchor: np.ndarray,
    *,
    max_rows: int = 4096,
) -> bool:
    X = np.asarray(X_main)
    q = np.asarray(query)
    a = np.asarray(anchor)
    if X.ndim != 2 or q.ndim != 2 or a.ndim != 2:
        return False
    if q.shape != a.shape or X.shape[0] != q.shape[0] or X.shape[1] != q.shape[1]:
        return False

    n = int(X.shape[0])
    if n <= 0:
        return False
    if n > max_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=int(max_rows), replace=False))
    else:
        idx = np.arange(n, dtype=np.int64)
    prod = (q[idx].astype(np.float32, copy=False) * a[idx].astype(np.float32, copy=False))
    return bool(np.allclose(np.asarray(X[idx], dtype=np.float32), prod, rtol=1e-4, atol=1e-4))


def _resolve_faiss_pair_context(
    *,
    ds: Dict[str, Any],
    X_main: np.ndarray,
    region_id: np.ndarray,
    args: argparse.Namespace,
    generated_query: Optional[np.ndarray],
    generated_anchor: Optional[np.ndarray],
    abs_diff_only: bool,
) -> PairContext:
    n_rows = int(X_main.shape[0])
    if generated_query is not None and generated_anchor is not None:
        residual_active = bool(args.use_delta_vec or args.use_abs_diff or abs_diff_only)
        features_clean = bool(args.hadamard_preprocess and not residual_active and not args.normalize_data)
        reason = ""
        if residual_active:
            reason = "residual/abs-diff features break clean q*anchor Hadamard factorization"
        elif args.normalize_data:
            reason = "row-wise normalization changed X_main after Hadamard construction"
        return PairContext(
            query=np.asarray(generated_query, dtype=np.float32),
            anchor=np.asarray(generated_anchor, dtype=np.float32),
            anchor_ids=np.asarray(region_id).reshape(-1),
            features_are_hadamard=features_clean,
            reason=reason,
            source="generated_hadamard_region_anchor",
        )

    query = _npz_float_matrix(ds, "query_emb", n_rows)
    anchor = _npz_float_matrix(ds, "anchor_emb", n_rows)
    if query is None or anchor is None:
        return PairContext(
            query=None,
            anchor=None,
            features_are_hadamard=False,
            reason="missing query_emb/anchor_emb and no generated pair context",
            source="unavailable",
        )

    clean = bool(not args.normalize_data and _sampled_hadamard_match(X_main, query, anchor))
    reason = "" if clean else "current X_main is not the exact query_emb*anchor_emb Hadamard product"
    return PairContext(
        query=query,
        anchor=anchor,
        anchor_ids=_optional_row_ids(ds, n_rows),
        features_are_hadamard=clean,
        reason=reason,
        source="npz_query_emb_anchor_emb",
    )


def _faiss_static_ineligible_reason(name: str, args: argparse.Namespace, scope: str) -> Optional[str]:
    if name.startswith("WeightedEnsemble") or name == "RegionalWeightedEnsemble":
        return "WeightedEnsemble uses per-judge H0-CDF normalization, so it is not a single bilinear/IP score"
    if name.startswith("RandomForestEnsemble"):
        return "RandomForest ensemble is nonlinear and not a single bilinear/IP score"
    if name.startswith("MahalanobisDelta"):
        return "MahalanobisDelta is an L2-style distance and is not handled by this IndexFlatIP pass"
    if "Separation" in name:
        return "separation methods are left ineligible in this pass"
    if name in {"XGBoost", "Tiny MLP", "BGE Reranker", "Online(refit)", "CosineAffineCalib"}:
        return "method is not an exact fitted pair bilinear/IP scorer"
    if name == "StabilizedWhitenedCosine" and str(getattr(args, "swc_mode", "global")) != "global":
        return "StabilizedWhitenedCosine mutates by fit_region in this swc_mode, so a single wrapper would be stale"
    if str(getattr(args, "local_fit_mode", "pooled")) == "per_region" and scope == "local":
        return "local_fit_mode=per_region refits methods inside evaluation; FAISS variants are skipped for this pass"
    return None


def _register_faiss_variants(
    *,
    methods: Dict[str, Any],
    method_names: List[str],
    pair_context: PairContext,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    args: argparse.Namespace,
    trial: int,
    seed: int,
    scope: str,
    report_rows: List[Dict[str, Any]],
) -> None:
    if not bool(getattr(args, "include_faiss_variants", False)):
        return

    original_names = [name for name in method_names if name in methods and not is_faiss_variant_name(name)]
    added = 0
    for name in original_names:
        faiss_name = f"{name}{FAISS_SUFFIX}"
        static_reason = _faiss_static_ineligible_reason(name, args, scope)
        if static_reason is not None:
            eligible = False
            reason = static_reason
            scorer = None
        else:
            scorer, result = build_faiss_variant(
                source_name=name,
                source_method=methods[name],
                pair_context=pair_context,
                X_main=X_main,
                X_cos=X_cos,
            )
            eligible = bool(result.eligible)
            reason = str(result.reason)

        report_rows.append(
            {
                "trial": int(trial),
                "seed": int(seed),
                "scope": str(scope),
                "method": str(name),
                "faiss_method": str(faiss_name),
                "eligible": bool(eligible),
                "reason": reason,
                "pair_context_source": str(pair_context.source),
                "pair_context_reason": str(pair_context.reason),
                "features_are_hadamard": bool(pair_context.features_are_hadamard),
                "pair_context_rows": int(pair_context.n_rows),
            }
        )
        if scorer is None or not eligible:
            continue
        methods[faiss_name] = scorer
        method_names.append(faiss_name)
        added += 1

    if added > 0:
        print(f"  faiss_variants: added={added} source={pair_context.source}")
    else:
        print("  faiss_variants: none eligible")


def _append_faiss_equivalence_rows(
    trial_rows: List[Dict[str, Any]],
    *,
    methods: Dict[str, Any],
    alpha: float,
    scope: str,
    out_rows: List[Dict[str, Any]],
) -> None:
    by_key: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in trial_rows:
        method = str(row.get("method", ""))
        key = (
            int(row.get("trial", -1)),
            int(row.get("seed", -1)),
            method,
            str(row.get("region_key", "")),
            str(row.get("tau_mode", "")),
            str(row.get("comparison_scope", "")),
        )
        by_key[key] = row

    for faiss_row in trial_rows:
        faiss_name = str(faiss_row.get("method", ""))
        if not is_faiss_variant_name(faiss_name):
            continue
        source_name = source_name_for_faiss(faiss_name)
        source_key = (
            int(faiss_row.get("trial", -1)),
            int(faiss_row.get("seed", -1)),
            source_name,
            str(faiss_row.get("region_key", "")),
            str(faiss_row.get("tau_mode", "")),
            str(faiss_row.get("comparison_scope", "")),
        )
        source_row = by_key.get(source_key)
        if source_row is None:
            continue

        scorer = methods.get(faiss_name)
        out_rows.append(
            {
                "trial": int(faiss_row.get("trial", -1)),
                "seed": int(faiss_row.get("seed", -1)),
                "scope": str(scope),
                "alpha": float(alpha),
                "source_method": source_name,
                "faiss_method": faiss_name,
                "tau_mode": str(faiss_row.get("tau_mode", "")),
                "comparison_scope": str(faiss_row.get("comparison_scope", "")),
                "max_score_diff": float(getattr(scorer, "diagnostic_max_abs_diff", float("nan"))),
                "scored_rows": int(getattr(scorer, "diagnostic_count", 0)),
                "rank_agreement": bool(getattr(scorer, "diagnostic_rank_agreement", False)),
                "dTPR": float(faiss_row.get("micro_tpr", float("nan")))
                - float(source_row.get("micro_tpr", float("nan"))),
                "dFPR": float(faiss_row.get("micro_fpr", float("nan")))
                - float(source_row.get("micro_fpr", float("nan"))),
                "dMacroTPR": float(faiss_row.get("macro_tpr", float("nan")))
                - float(source_row.get("macro_tpr", float("nan"))),
                "dMacroFPR": float(faiss_row.get("macro_fpr", float("nan")))
                - float(source_row.get("macro_fpr", float("nan"))),
                "dTrainTPR": float(faiss_row.get("train_tpr", float("nan")))
                - float(source_row.get("train_tpr", float("nan"))),
                "dTrainFPR": float(faiss_row.get("train_fpr", float("nan")))
                - float(source_row.get("train_fpr", float("nan"))),
            }
        )


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


def _online_stopping_as_method(args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "enable_online_stopping", False)):
        return False
    explicit = getattr(args, "online_stopping_as_method", None)
    if explicit is not None:
        return bool(explicit)
    return not bool(getattr(args, "early_stop_train_subset", False))


def _validate_online_used_indices(
    *,
    y: np.ndarray,
    used_h0_train_idx: np.ndarray,
    used_h1_train_idx: np.ndarray,
) -> None:
    h0 = np.asarray(used_h0_train_idx, dtype=np.int64).reshape(-1)
    h1 = np.asarray(used_h1_train_idx, dtype=np.int64).reshape(-1)
    if h0.size == 0 or h1.size == 0:
        raise RuntimeError(
            "early_stop_train_subset requested but online stopping returned an empty used train class"
        )
    if int(np.max(np.concatenate([h0, h1]))) >= int(y.shape[0]):
        raise RuntimeError("early_stop_train_subset returned an out-of-range dataset index")
    if not np.all(np.asarray(y[h0], dtype=np.int32) == 0):
        raise RuntimeError("early_stop_train_subset returned non-H0 labels in used_h0_train_idx")
    if not np.all(np.asarray(y[h1], dtype=np.int32) == 1):
        raise RuntimeError("early_stop_train_subset returned non-H1 labels in used_h1_train_idx")


def _replace_global_train_indices(
    gs: GlobalSplit,
    *,
    used_h0_train_idx: np.ndarray,
    used_h1_train_idx: np.ndarray,
) -> GlobalSplit:
    return GlobalSplit(
        H0_train=np.asarray(used_h0_train_idx, dtype=np.int64).reshape(-1),
        H1_train=np.asarray(used_h1_train_idx, dtype=np.int64).reshape(-1),
        H0_calib=gs.H0_calib,
        H1_calib=gs.H1_calib,
        H0_eval=gs.H0_eval,
        H1_eval=gs.H1_eval,
    )


def _replace_region_train_indices(
    splits: List[Any],
    *,
    used_h0_train_idx: np.ndarray,
    used_h1_train_idx: np.ndarray,
    region_id: np.ndarray,
) -> List[Any]:
    h0 = np.asarray(used_h0_train_idx, dtype=np.int64).reshape(-1)
    h1 = np.asarray(used_h1_train_idx, dtype=np.int64).reshape(-1)
    h0_by_rid: Dict[int, np.ndarray] = {}
    h1_by_rid: Dict[int, np.ndarray] = {}
    h0_rids = np.unique(region_id[h0]) if h0.size > 0 else np.array([], dtype=np.int64)
    h1_rids = np.unique(region_id[h1]) if h1.size > 0 else np.array([], dtype=np.int64)
    for rid in h0_rids:
        h0_by_rid[int(rid)] = h0[region_id[h0] == int(rid)]
    for rid in h1_rids:
        h1_by_rid[int(rid)] = h1[region_id[h1] == int(rid)]

    out: List[Any] = []
    for s in splits:
        rid = int(s.rid)
        out.append(
            type(s)(
                rid=rid,
                H0_train=h0_by_rid.get(rid, np.array([], dtype=np.int64)),
                H1_train=h1_by_rid.get(rid, np.array([], dtype=np.int64)),
                H0_calib=s.H0_calib,
                H1_calib=s.H1_calib,
                H0_eval=s.H0_eval,
                H1_eval=s.H1_eval,
            )
        )
    return out


def _augment_online_summary(
    summary: Dict[str, Any],
    *,
    early_stop_train_subset_active: bool,
    early_stop_train_subset_applied: bool,
    online_stopping_as_method: bool,
) -> Dict[str, Any]:
    out = dict(summary)
    original_total = int(out.get("original_total_train", 0))
    used_total = int(out.get("used_total_train", out.get("samples_total_used", 0)))
    out.update(
        {
            "early_stop_train_subset_active": bool(early_stop_train_subset_active),
            "early_stop_train_subset_applied": bool(early_stop_train_subset_applied),
            "online_stopping_as_method": bool(online_stopping_as_method),
            "online_stopped": bool(out.get("stopped", False)),
            "online_stop_reason": str(out.get("reason", "unknown")),
            "online_updates": int(out.get("updates", 0)),
            "online_samples_streamed": int(out.get("samples_streamed", 0)),
            "train_fraction": (
                float(out.get("train_fraction"))
                if "train_fraction" in out and np.isfinite(float(out.get("train_fraction", float("nan"))))
                else (float(used_total / original_total) if original_total > 0 else float("nan"))
            ),
        }
    )
    return out


def _print_online_stopping_log(
    *,
    history_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    if history_rows:
        last = history_rows[-1]
        print(
            "  online_stopping: "
            f"checks={summary.get('history_len', 0)} "
            f"stopped={summary.get('stopped')} reason={summary.get('reason')} "
            f"tpr={float(last.get('tpr_monitor', float('nan'))):.4f} "
            f"fpr={float(last.get('fpr_monitor', float('nan'))):.4f} "
            f"tau={float(last.get('tau', float('nan'))):.4f} "
            f"samples_used={summary.get('samples_total_used', 0)}"
        )


def _print_early_stop_train_subset_log(summary: Dict[str, Any]) -> None:
    original_total = int(summary.get("original_total_train", 0))
    used_total = int(summary.get("used_total_train", 0))
    train_fraction = float(used_total / original_total) if original_total > 0 else float("nan")
    print("  early_stop_train_subset:")
    print(
        "    "
        f"used_h0_train={int(summary.get('used_h0_train', 0))} "
        f"used_h1_train={int(summary.get('used_h1_train', 0))} "
        f"used_total_train={used_total}"
    )
    print(
        "    "
        f"original_h0_train={int(summary.get('original_h0_train', 0))} "
        f"original_h1_train={int(summary.get('original_h1_train', 0))} "
        f"train_fraction={train_fraction:.4f}"
    )
    print(
        "    "
        f"stopped={summary.get('stopped')} "
        f"reason={summary.get('reason')} "
        f"updates={int(summary.get('updates', 0))} "
        f"samples_streamed={int(summary.get('samples_streamed', 0))}"
    )


def _record_online_stopping_outputs(
    *,
    trial: int,
    seed: int,
    history_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    early_stop_train_subset_active: bool,
    early_stop_train_subset_applied: bool,
    online_stopping_as_method: bool,
    online_stopping_history_rows: List[Dict[str, Any]],
    online_stopping_summary_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary_out = _augment_online_summary(
        summary,
        early_stop_train_subset_active=early_stop_train_subset_active,
        early_stop_train_subset_applied=early_stop_train_subset_applied,
        online_stopping_as_method=online_stopping_as_method,
    )
    if history_rows:
        for r in history_rows:
            r["trial"] = int(trial)
            r["seed"] = int(seed)
        online_stopping_history_rows.extend(history_rows)
        _print_online_stopping_log(history_rows=history_rows, summary=summary_out)
    else:
        print(
            "  [WARN] online_stopping skipped: "
            f"reason={summary_out.get('reason', 'unknown')}"
        )
    if early_stop_train_subset_applied:
        _print_early_stop_train_subset_log(summary_out)
    online_stopping_summary_rows.append(
        {
            "trial": int(trial),
            "seed": int(seed),
            **summary_out,
        }
    )
    return summary_out


def _train_sample_count_meta(
    *,
    h0_train_idx: np.ndarray,
    h1_train_idx: np.ndarray,
    source: str,
) -> Dict[str, Any]:
    n0 = int(np.asarray(h0_train_idx, dtype=np.int64).reshape(-1).size)
    n1 = int(np.asarray(h1_train_idx, dtype=np.int64).reshape(-1).size)
    total = int(n0 + n1)
    return {
        "train_h0_samples": n0,
        "train_h1_samples": n1,
        "train_total_samples": total,
        "train_samples_needed": total,
        "train_sample_source": str(source),
    }


def _online_method_train_count_meta(summary: Dict[str, Any]) -> Dict[str, Any]:
    n0 = int(summary.get("used_h0_train", 0))
    n1 = int(summary.get("used_h1_train", 0))
    total = int(summary.get("used_total_train", n0 + n1))
    return {
        "train_h0_samples": n0,
        "train_h1_samples": n1,
        "train_total_samples": total,
        "train_samples_needed": total,
        "train_sample_source": "online_stopping_method",
    }


def _annotate_train_sample_counts(
    rows: List[Dict[str, Any]],
    *,
    h0_train_idx: np.ndarray,
    h1_train_idx: np.ndarray,
    source: str,
    method_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    default_meta = _train_sample_count_meta(
        h0_train_idx=h0_train_idx,
        h1_train_idx=h1_train_idx,
        source=source,
    )
    overrides = method_overrides or {}
    for row in rows:
        meta = overrides.get(str(row.get("method", "")), default_meta)
        row.update(meta)


def _attach_train_sample_counts_to_ranking(
    ranking: List[Dict[str, Any]],
    trial_summary_rows: List[Dict[str, Any]],
) -> None:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in trial_summary_rows:
        method = str(row.get("method", ""))
        if method:
            by_method[method].append(row)

    for row in ranking:
        method = str(row.get("method", ""))
        method_rows = by_method.get(method, [])
        counts = [
            float(r["train_samples_needed"])
            for r in method_rows
            if r.get("train_samples_needed") not in (None, "")
        ]
        if not counts:
            continue
        h0_counts = [
            float(r["train_h0_samples"])
            for r in method_rows
            if r.get("train_h0_samples") not in (None, "")
        ]
        h1_counts = [
            float(r["train_h1_samples"])
            for r in method_rows
            if r.get("train_h1_samples") not in (None, "")
        ]
        sources = sorted(
            {
                str(r.get("train_sample_source", ""))
                for r in method_rows
                if str(r.get("train_sample_source", ""))
            }
        )
        row.update(
            {
                "mean_train_samples_needed": float(np.mean(counts)),
                "min_train_samples_needed": int(np.min(counts)),
                "max_train_samples_needed": int(np.max(counts)),
                "mean_train_n": float(np.mean(counts)),
                "min_train_n": int(np.min(counts)),
                "max_train_n": int(np.max(counts)),
                "mean_train_h0_samples": float(np.mean(h0_counts)) if h0_counts else float("nan"),
                "mean_train_h1_samples": float(np.mean(h1_counts)) if h1_counts else float("nan"),
                "train_sample_sources": sources,
            }
        )


def _build_cache_efficiency_ranking(
    ranking: List[Dict[str, Any]],
    *,
    alpha: float,
    tpr_tolerance: float,
) -> List[Dict[str, Any]]:
    safe_rows = [
        r for r in ranking
        if float(r.get("valid_rate", 0.0)) >= 1.0 - 1e-12
        and float(r.get("mean_eval_fpr", r.get("mean_micro_fpr", float("inf")))) <= float(alpha) + 1e-12
        and float(r.get("max_eval_fpr", float("inf"))) <= float(alpha) + 1e-12
    ]
    if not safe_rows:
        return []

    best_tpr = max(float(r.get("mean_eval_tpr", r.get("mean_micro_tpr", float("-inf")))) for r in safe_rows)
    near_best = [
        dict(r)
        for r in safe_rows
        if float(r.get("mean_eval_tpr", r.get("mean_micro_tpr", float("-inf")))) >= best_tpr - float(tpr_tolerance)
    ]
    def cache_sort_train_n(row: Dict[str, Any]) -> float:
        out = float(row.get("mean_train_n", row.get("mean_train_samples_needed", float("inf"))))
        return out if np.isfinite(out) else float("inf")

    near_best.sort(
        key=lambda r: (
            cache_sort_train_n(r),
            -float(r.get("mean_eval_tpr", r.get("mean_micro_tpr", float("-inf")))),
            float(r.get("mean_eval_fpr", r.get("mean_micro_fpr", float("inf")))),
        )
    )
    for i, row in enumerate(near_best, start=1):
        row["cache_efficiency_rank"] = int(i)
        row["cache_efficiency_best_valid_tpr"] = float(best_tpr)
        row["cache_efficiency_tpr_tolerance"] = float(tpr_tolerance)
    return near_best


def _parse_int_csv(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    txt = str(raw).strip()
    if not txt:
        return []

    out: List[int] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _build_train_dosage_grid(
    h0_train_idx: np.ndarray,
    h1_train_idx: np.ndarray,
    *,
    raw_grid: Optional[str],
    no_auto_full: bool = False,
    max_train: Optional[int] = None,
) -> List[int]:
    full_total = int(np.asarray(h0_train_idx).size + np.asarray(h1_train_idx).size)
    if full_total <= 0:
        return []

    parsed = _parse_int_csv(raw_grid)
    if parsed:
        grid = []
        for v in parsed:
            vv = int(v)
            if vv <= 1:
                continue
            if vv > full_total:
                if no_auto_full:
                    continue
                vv = full_total
            grid.append(vv)
        if not no_auto_full:
            grid.append(full_total)
    else:
        start = 2 * max(1, min(int(np.asarray(h0_train_idx).size), int(np.asarray(h1_train_idx).size), 32))
        grid = []
        n = int(start)
        while n < full_total:
            grid.append(n)
            n *= 2
        if not no_auto_full:
            grid.append(full_total)

    grid = [int(min(max(2, v), full_total)) for v in grid]
    if max_train is not None:
        cap = int(max_train)
        grid = [v for v in grid if v <= cap]

    grid = sorted({int(v) for v in grid})
    if not no_auto_full and max_train is None and full_total not in grid:
        grid.append(full_total)
    return grid


def _subset_train_indices_for_dosage(
    h0_train_idx: np.ndarray,
    h1_train_idx: np.ndarray,
    *,
    total_train_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    h0 = np.asarray(h0_train_idx, dtype=np.int64).reshape(-1)
    h1 = np.asarray(h1_train_idx, dtype=np.int64).reshape(-1)
    full_total = int(h0.size + h1.size)
    target = int(min(max(2, total_train_samples), full_total))
    if target >= full_total:
        return h0, h1

    if h0.size == 0 or h1.size == 0:
        return h0[:0], h1[:0]

    n0 = int(round(target * (h0.size / max(1, full_total))))
    n0 = int(min(max(1, n0), h0.size))
    n1 = int(target - n0)
    if n1 < 1:
        n1 = 1
        n0 = int(min(h0.size, target - n1))
    if n1 > h1.size:
        n1 = int(h1.size)
        n0 = int(min(h0.size, target - n1))
    if n0 < 1:
        n0 = 1
        n1 = int(min(h1.size, target - n0))

    return h0[:n0], h1[:n1]


def _fit_one_method_on_global_indices(
    *,
    method_name: str,
    method: Any,
    h0_train_idx: np.ndarray,
    h1_train_idx: np.ndarray,
    h0_calib_idx: np.ndarray,
    h1_calib_idx: np.ndarray,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
    region_id: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    trial: int,
    failures: Dict[str, List[str]],
    fit_context: str,
) -> Optional[Any]:
    h0_train_idx = np.asarray(h0_train_idx, dtype=np.int64).reshape(-1)
    h1_train_idx = np.asarray(h1_train_idx, dtype=np.int64).reshape(-1)
    h0_train_unique = np.unique(h0_train_idx) if h0_train_idx.size > 0 else h0_train_idx
    h1_train_unique = np.unique(h1_train_idx) if h1_train_idx.size > 0 else h1_train_idx

    H0_train = X_main[h0_train_idx]
    H1_train = X_main[h1_train_idx]
    H0_calib_pure = X_main[h0_calib_idx]
    H1_calib_pure = X_main[h1_calib_idx]
    H0_calib_eff = X_main[np.concatenate([h0_train_unique, h0_calib_idx])] \
        if h0_train_unique.size > 0 else H0_calib_pure
    H1_calib_eff = X_main[np.concatenate([h1_train_unique, h1_calib_idx])] \
        if h1_train_unique.size > 0 else H1_calib_pure

    H0_train_cos = X_cos[h0_train_idx] if X_cos is not None else None
    H1_train_cos = X_cos[h1_train_idx] if X_cos is not None else None
    if X_cos is not None:
        H0_calib_pure_cos = X_cos[h0_calib_idx]
        H1_calib_pure_cos = X_cos[h1_calib_idx]
        H0_calib_eff_cos = X_cos[np.concatenate([h0_train_unique, h0_calib_idx])] \
            if h0_train_unique.size > 0 else H0_calib_pure_cos
        H1_calib_eff_cos = X_cos[np.concatenate([h1_train_unique, h1_calib_idx])] \
            if h1_train_unique.size > 0 else H1_calib_pure_cos
    else:
        H0_calib_pure_cos = None
        H1_calib_pure_cos = None
        H0_calib_eff_cos = None
        H1_calib_eff_cos = None

    if X_text is not None:
        H0_train_text = X_text[h0_train_idx]
        H1_train_text = X_text[h1_train_idx]
        H0_calib_eff_text = X_text[np.concatenate([h0_train_unique, h0_calib_idx])] \
            if h0_train_unique.size > 0 else X_text[h0_calib_idx]
        H1_calib_eff_text = X_text[np.concatenate([h1_train_unique, h1_calib_idx])] \
            if h1_train_unique.size > 0 else X_text[h1_calib_idx]
    else:
        H0_train_text = None
        H1_train_text = None
        H0_calib_eff_text = None
        H1_calib_eff_text = None

    if H0_calib_eff.shape[0] == 0 or H1_calib_eff.shape[0] == 0:
        failures[method_name].append(f"trial={trial}: dosage fit skipped due to empty effective calibration")
        return None

    v0 = np.var(H0_calib_eff, axis=0)
    v1 = np.var(H1_calib_eff, axis=0)
    weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)

    fitted = {method_name: method}
    fit_all_methods(
        fitted,
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
        H0_calib_region_ids=region_id[h0_calib_idx],
        H1_calib_region_ids=region_id[h1_calib_idx],
        tie_mode=args.tie_mode,
        tau_guardrail=args.tau_guardrail,
        tau_guardrail_delta=args.tau_guardrail_delta,
        weights=weights,
        seed=seed,
        alpha=args.alpha,
        trial=trial,
        failures=failures,
        fit_context=fit_context,
    )
    return fitted.get(method_name)


def _run_global_train_dosage_search(
    *,
    method_names: List[str],
    gs: GlobalSplit,
    h0_monitor_idx: np.ndarray,
    h1_monitor_idx: np.ndarray,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
    region_id: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    trial: int,
    failures: Dict[str, List[str]],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    h0_monitor_idx = np.asarray(h0_monitor_idx, dtype=np.int64).reshape(-1)
    h1_monitor_idx = np.asarray(h1_monitor_idx, dtype=np.int64).reshape(-1)
    if h0_monitor_idx.size == 0 or h1_monitor_idx.size == 0:
        raise RuntimeError(
            "train dosage search requires a non-empty monitor split; "
            "increase --n_monitor_h0/--n_monitor_h1 and --n_eval"
        )

    grid = _build_train_dosage_grid(
        gs.H0_train,
        gs.H1_train,
        raw_grid=args.train_dosage_grid,
        no_auto_full=bool(getattr(args, "train_dosage_no_auto_full", False)),
        max_train=getattr(args, "train_dosage_max_train", None),
    )
    if not grid:
        raise RuntimeError(
            "train dosage search grid is empty after applying "
            "--train_dosage_no_auto_full/--train_dosage_max_train; provide at least one valid dosage"
        )
    monitor_gs = GlobalSplit(
        H0_train=gs.H0_train,
        H1_train=gs.H1_train,
        H0_calib=gs.H0_calib,
        H1_calib=gs.H1_calib,
        H0_eval=h0_monitor_idx,
        H1_eval=h1_monitor_idx,
    )

    selected_methods: Dict[str, Any] = {}
    selected_meta: Dict[str, Dict[str, Any]] = {}
    search_rows: List[Dict[str, Any]] = []
    tol = float(getattr(args, "train_dosage_tpr_tolerance", 0.01))
    fpr_margin = float(getattr(args, "train_dosage_fpr_margin", 0.0))
    fpr_limit = float(args.alpha) - fpr_margin
    requested_grid = _parse_int_csv(args.train_dosage_grid)
    max_train_arg = getattr(args, "train_dosage_max_train", None)
    no_auto_full = bool(getattr(args, "train_dosage_no_auto_full", False))

    print("  train_dosage_search:")
    print(f"    requested_grid={requested_grid if requested_grid else 'auto'}")
    print(f"    actual_grid={grid}")
    print(f"    no_auto_full={no_auto_full}")
    print(f"    max_train={max_train_arg if max_train_arg is not None else 'none'}")
    print(f"    monitor(h0={int(h0_monitor_idx.size)}, h1={int(h1_monitor_idx.size)})")
    print(f"    tpr_tolerance={tol:.4f}")
    print(f"    fpr_margin={fpr_margin:.4f}")
    print(f"    effective_monitor_fpr_limit={fpr_limit:.4f}")

    for method_name in method_names:
        candidate_rows: List[Dict[str, Any]] = []
        candidate_methods: Dict[int, Any] = {}

        for requested_total in grid:
            h0_dose, h1_dose = _subset_train_indices_for_dosage(
                gs.H0_train,
                gs.H1_train,
                total_train_samples=int(requested_total),
            )
            effective_total = int(h0_dose.size + h1_dose.size)
            row_base: Dict[str, Any] = {
                "trial": int(trial),
                "seed": int(seed),
                "method": str(method_name),
                "requested_train_total": int(requested_total),
                "train_dosage_total": int(effective_total),
                "train_h0": int(h0_dose.size),
                "train_h1": int(h1_dose.size),
                "train_h0_samples": int(h0_dose.size),
                "train_h1_samples": int(h1_dose.size),
                "train_total_samples": int(effective_total),
                "alpha": float(args.alpha),
                "train_dosage_fpr_margin": float(fpr_margin),
                "effective_monitor_fpr_limit": float(fpr_limit),
                "fpr_feasible": False,
                "selected": False,
                "feasible": False,
                "fit_failed": False,
                "selection_reason": "",
                "best_feasible_monitor_tpr": float("nan"),
                "tpr_gap_from_best_feasible": float("nan"),
                "train_dosage_tpr_tolerance": float(tol),
                "train_dosage_max_train": int(max_train_arg) if max_train_arg is not None else None,
                "train_dosage_no_auto_full": bool(no_auto_full),
            }

            fresh = _build_configured_methods(args, X_cos=X_cos, X_text=X_text, quiet=True)
            if method_name not in fresh:
                row = {
                    **row_base,
                    "fit_failed": True,
                    "failure_reason": "method_not_configured",
                }
                candidate_rows.append(row)
                search_rows.append(row)
                continue

            candidate_failures: Dict[str, List[str]] = defaultdict(list)
            fit_cm = (
                contextlib.nullcontext()
                if bool(getattr(args, "debug_ablation_scores", False))
                else contextlib.redirect_stdout(io.StringIO())
            )
            with fit_cm:
                fitted_method = _fit_one_method_on_global_indices(
                    method_name=method_name,
                    method=fresh[method_name],
                    h0_train_idx=h0_dose,
                    h1_train_idx=h1_dose,
                    h0_calib_idx=gs.H0_calib,
                    h1_calib_idx=gs.H1_calib,
                    X_main=X_main,
                    X_cos=X_cos,
                    X_text=X_text,
                    region_id=region_id,
                    args=args,
                    seed=seed,
                    trial=trial,
                    failures=candidate_failures,
                    fit_context="train_dosage_search",
                )
            if fitted_method is None:
                reason = "; ".join(candidate_failures.get(method_name, [])) or "fit_failed"
                row = {
                    **row_base,
                    "fit_failed": True,
                    "failure_reason": reason[:500],
                }
                candidate_rows.append(row)
                search_rows.append(row)
                continue

            monitor_failures: Dict[str, List[str]] = defaultdict(list)
            eval_cm = (
                contextlib.nullcontext()
                if bool(getattr(args, "debug_ablation_scores", False))
                else contextlib.redirect_stdout(io.StringIO())
            )
            with eval_cm:
                monitor_rows = evaluate_methods_global(
                    {method_name: fitted_method},
                    [method_name],
                    monitor_gs,
                    X_main=X_main,
                    X_cos=X_cos,
                    X_text=X_text,
                    alpha=float(args.alpha),
                    tie_mode=args.tie_mode,
                    tau_guardrail=args.tau_guardrail,
                    tau_guardrail_delta=args.tau_guardrail_delta,
                    trial=trial,
                    seed=seed,
                    region_key=args.region_key,
                    region_id=region_id,
                    failures=monitor_failures,
                )
            if not monitor_rows:
                reason = "; ".join(monitor_failures.get(method_name, [])) or "monitor_eval_failed"
                row = {
                    **row_base,
                    "fit_failed": True,
                    "failure_reason": reason[:500],
                }
                candidate_rows.append(row)
                search_rows.append(row)
                continue

            mr = monitor_rows[0]
            monitor_fpr = float(mr.get("micro_fpr", float("inf")))
            monitor_tpr = float(mr.get("micro_tpr", float("nan")))
            fpr_feasible = bool(np.isfinite(monitor_fpr) and monitor_fpr <= fpr_limit + 1e-12)
            row = {
                **row_base,
                "monitor_tpr": monitor_tpr,
                "monitor_fpr": monitor_fpr,
                "monitor_train_tpr": float(mr.get("train_tpr", float("nan"))),
                "monitor_train_fpr": float(mr.get("train_fpr", float("nan"))),
                "monitor_tau": float(mr.get("tau", float("nan"))),
                "input_space": str(mr.get("input_space", "unknown")),
                "time_ms": float(mr.get("time_ms", float("nan"))),
                "fpr_feasible": fpr_feasible,
                "feasible": fpr_feasible,
                "failure_reason": "",
            }
            candidate_rows.append(row)
            search_rows.append(row)
            candidate_methods[effective_total] = fitted_method

        valid_rows = [r for r in candidate_rows if not bool(r.get("fit_failed", False))]
        if not valid_rows:
            failures[method_name].append(f"trial={trial}: all train dosage candidates failed")
            continue

        feasible_rows = [
            r for r in valid_rows
            if bool(r.get("fpr_feasible", False)) and np.isfinite(float(r.get("monitor_tpr", float("nan"))))
        ]
        if feasible_rows:
            best_tpr = max(float(r["monitor_tpr"]) for r in feasible_rows)
            near_best = [
                r for r in feasible_rows
                if float(r["monitor_tpr"]) >= best_tpr - tol
            ]
            selected = min(
                near_best,
                key=lambda r: (
                    int(r["train_total_samples"]),
                    -float(r["monitor_tpr"]),
                    float(r["monitor_fpr"]),
                ),
            )
            reason = "smallest_fpr_safe_near_best_tpr"
        else:
            best_tpr = float("nan")
            selected = min(
                valid_rows,
                key=lambda r: (
                    float(r.get("monitor_fpr", float("inf"))),
                    -float(r.get("monitor_tpr", float("-inf"))),
                    int(r["train_total_samples"]),
                ),
            )
            reason = "fallback_lowest_monitor_fpr_no_fpr_safe_candidate"

        for r in candidate_rows:
            r["best_feasible_monitor_tpr"] = float(best_tpr)
            monitor_tpr = float(r.get("monitor_tpr", float("nan")))
            r["tpr_gap_from_best_feasible"] = (
                float(best_tpr - monitor_tpr)
                if np.isfinite(best_tpr) and np.isfinite(monitor_tpr)
                else float("nan")
            )
            r["selection_reason"] = reason

        selected_total = int(selected["train_total_samples"])
        selected["selected"] = True
        selected_method = candidate_methods.get(selected_total)
        if selected_method is None:
            h0_dose, h1_dose = _subset_train_indices_for_dosage(
                gs.H0_train,
                gs.H1_train,
                total_train_samples=selected_total,
            )
            fresh = _build_configured_methods(args, X_cos=X_cos, X_text=X_text, quiet=True)
            selected_method = _fit_one_method_on_global_indices(
                method_name=method_name,
                method=fresh[method_name],
                h0_train_idx=h0_dose,
                h1_train_idx=h1_dose,
                h0_calib_idx=gs.H0_calib,
                h1_calib_idx=gs.H1_calib,
                X_main=X_main,
                X_cos=X_cos,
                X_text=X_text,
                region_id=region_id,
                args=args,
                seed=seed,
                trial=trial,
                failures=failures,
                fit_context="train_dosage_selected",
            )
        if selected_method is None:
            failures[method_name].append(f"trial={trial}: selected train dosage refit failed")
            continue

        selected_methods[method_name] = selected_method
        selected_meta[method_name] = {
            "train_h0_samples": int(selected["train_h0_samples"]),
            "train_h1_samples": int(selected["train_h1_samples"]),
            "train_total_samples": int(selected["train_total_samples"]),
            "train_samples_needed": int(selected["train_total_samples"]),
            "train_sample_source": "train_dosage_search",
            "train_dosage_search_active": True,
            "train_dosage_monitor_tpr": float(selected.get("monitor_tpr", float("nan"))),
            "train_dosage_monitor_fpr": float(selected.get("monitor_fpr", float("nan"))),
            "train_dosage_fpr_margin": float(fpr_margin),
            "train_dosage_fpr_limit": float(fpr_limit),
            "train_dosage_fpr_feasible": bool(selected.get("fpr_feasible", False)),
            "train_dosage_tpr_tolerance": float(tol),
            "train_dosage_best_feasible_monitor_tpr": float(
                selected.get("best_feasible_monitor_tpr", float("nan"))
            ),
            "train_dosage_tpr_gap_from_best_feasible": float(
                selected.get("tpr_gap_from_best_feasible", float("nan"))
            ),
            "train_dosage_max_train": int(max_train_arg) if max_train_arg is not None else None,
            "train_dosage_no_auto_full": bool(no_auto_full),
            "train_dosage_selection_reason": reason,
        }
        print(f"    {method_name}:")
        print(
            "      "
            f"selected_train={int(selected['train_total_samples'])} "
            f"h0={int(selected['train_h0_samples'])} "
            f"h1={int(selected['train_h1_samples'])} "
            f"monitor_tpr={float(selected.get('monitor_tpr', float('nan'))):.4f} "
            f"monitor_fpr={float(selected.get('monitor_fpr', float('nan'))):.4f} "
            f"fpr_limit={fpr_limit:.4f} "
            f"reason={reason}"
        )

    return selected_methods, selected_meta, search_rows


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


def _tiny_mlp_batch_size(args: argparse.Namespace) -> int | str:
    batch_size = getattr(args, "tiny_mlp_batch_size", 128)
    if isinstance(batch_size, str):
        if batch_size.strip().lower() == "auto":
            return "auto"
        batch_size = int(batch_size)
    return int(max(1, batch_size))


def _xgboost_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "xgb_n_estimators": int,
        "xgb_max_depth": int,
        "xgb_learning_rate": float,
        "xgb_subsample": float,
        "xgb_colsample_bytree": float,
        "xgb_min_child_weight": float,
        "xgb_gamma": float,
        "xgb_reg_alpha": float,
        "xgb_reg_lambda": float,
    }

    out: Dict[str, Any] = {}
    for name, cast in mapping.items():
        value = getattr(args, name, None)
        if value is not None:
            out[name] = cast(value)
    return out


def _whitened_cosine_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "pca_whiten_abs_eps": float,
        "pca_whiten_rel_eps": float,
        "pca_whiten_max_rank": int,
        "pca_whiten_rank_mode": str,
        "pca_whiten_explained_variance": float,
        "pca_whiten_norm_eps": float,
    }

    out: Dict[str, Any] = {}
    for name, cast in mapping.items():
        value = getattr(args, name, None)
        if value is not None:
            out[name] = cast(value)
    return out


WHITENED_COSINE_VARIANTS = {
    "PCAWhitenedCosine": "zca",
    "PCAWhitenedCosine_PCA": "pca",
    "PCAWhitenedCosine_ZCAcor": "zca_cor",
    "PCAWhitenedCosine_PCAcor": "pca_cor",
}


def _configured_whitened_cosine_methods_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    from np_bench.methods.whitened_cosine import WhitenedCosineMethod

    kwargs = _whitened_cosine_kwargs_from_args(args)
    return {
        name: WhitenedCosineMethod(
            name=name,
            whitening_type=whitening_type,  # type: ignore[arg-type]
            **kwargs,
        )
        for name, whitening_type in WHITENED_COSINE_VARIANTS.items()
    }


def _tiny_mlp_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "tiny_mlp_activation": str,
        "tiny_mlp_alpha": float,
        "tiny_mlp_batch_size": str,
        "tiny_mlp_early_stopping": bool,
        "tiny_mlp_hidden_dim": int,
        "tiny_mlp_learning_rate_init": float,
        "tiny_mlp_max_iter": int,
        "tiny_mlp_n_layers": int,
        "tiny_mlp_validation_fraction": float,
    }

    out: Dict[str, Any] = {}
    for name, cast in mapping.items():
        value = getattr(args, name, None)
        if value is not None:
            out[name] = cast(value)
    return out


def _lda_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "lda_solver": str,
        "lda_shrinkage": str,
        "lda_tol": float,
    }

    out: Dict[str, Any] = {}
    for name, cast in mapping.items():
        value = getattr(args, name, None)
        if value is not None:
            out[name] = cast(value)
    return out


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


def _build_ensemble_judges(
    args: argparse.Namespace,
    *,
    use_precomputed_cosine: bool,
    has_xgb: bool,
    include_whitened: bool = True,
) -> List[Any]:
    from np_bench.methods.cosine import CosineMethod
    from np_bench.methods.lda import LDAMethod
    from np_bench.methods.tiny_mlp import TinyMLPMethod

    judges: List[Any] = []
    if use_precomputed_cosine:
        from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
        judges.append(PrecomputedCosineMethod())
    else:
        judges.append(CosineMethod())

    if include_whitened:
        try:
            from np_bench.methods.whitened_cosine import WhitenedCosineMethod
            judges.append(WhitenedCosineMethod(**_whitened_cosine_kwargs_from_args(args)))
        except Exception:
            pass

    if has_xgb:
        try:
            from np_bench.methods.xgboost import XGBoostLightMethod
            judges.append(XGBoostLightMethod(**_xgboost_kwargs_from_args(args)))
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
        LDAMethod(**_lda_kwargs_from_args(args))
    )
    return judges


def _build_configured_methods(
    args: argparse.Namespace,
    *,
    X_cos: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
    quiet: bool = False,
) -> Dict[str, Any]:
    methods = build_methods(include_streaming=bool(getattr(args, "include_streaming_whitening", False)))
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
        methods.update(_configured_whitened_cosine_methods_from_args(args))
    except Exception:
        pass

    if has_xgb:
        try:
            from np_bench.methods.xgboost import XGBoostLightMethod
            methods["XGBoost"] = XGBoostLightMethod(**_xgboost_kwargs_from_args(args))
        except Exception as exc:
            if not quiet:
                print(f"[WARN] Could not configure XGBoost: {exc}")

    try:
        from np_bench.methods.tiny_mlp import TinyMLPMethod
        methods["Tiny MLP"] = TinyMLPMethod(**_tiny_mlp_kwargs_from_args(args))
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not configure Tiny MLP: {exc}")

    try:
        from np_bench.methods.lda import LDAMethod
        methods["LDA"] = LDAMethod(**_lda_kwargs_from_args(args))
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
        from np_bench.methods.weighted_ensemble import WeightedEnsembleMethod
        methods["WeightedEnsembleNoPCAWhitenedCosine"] = WeightedEnsembleMethod(
            judges=_build_ensemble_judges(
                args,
                use_precomputed_cosine=use_precomputed_cosine,
                has_xgb=has_xgb,
                include_whitened=False,
            ),
            config=_ensemble_config_from_args(args),
        )
        methods["WeightedEnsembleNoPCAWhitenedCosine"].name = "WeightedEnsembleNoPCAWhitenedCosine"
        if use_precomputed_cosine and not quiet:
            print("[INFO] WeightedEnsembleNoPCAWhitenedCosine Cosine judge routed to direct Hadamard sum")
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not configure WeightedEnsembleNoPCAWhitenedCosine: {exc}")

    try:
        from np_bench.methods.random_forest_ensemble import RandomForestEnsembleMethod
        methods["RandomForestEnsemble"] = RandomForestEnsembleMethod(
            judges=_build_ensemble_judges(
                args,
                use_precomputed_cosine=use_precomputed_cosine,
                has_xgb=has_xgb,
            ),
            config=_ensemble_config_from_args(args),
        )
        if use_precomputed_cosine and not quiet:
            print("[INFO] RandomForestEnsemble Cosine judge routed to direct Hadamard sum")
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not configure RandomForestEnsemble: {exc}")

    try:
        from np_bench.methods.random_forest_ensemble import RandomForestEnsembleMethod
        methods["RandomForestEnsembleNoPCAWhitenedCosine"] = RandomForestEnsembleMethod(
            judges=_build_ensemble_judges(
                args,
                use_precomputed_cosine=use_precomputed_cosine,
                has_xgb=has_xgb,
                include_whitened=False,
            ),
            config=_ensemble_config_from_args(args),
        )
        methods["RandomForestEnsembleNoPCAWhitenedCosine"].name = "RandomForestEnsembleNoPCAWhitenedCosine"
        if use_precomputed_cosine and not quiet:
            print("[INFO] RandomForestEnsembleNoPCAWhitenedCosine Cosine judge routed to direct Hadamard sum")
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not configure RandomForestEnsembleNoPCAWhitenedCosine: {exc}")

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

    if args.include_vcache_baseline:
        if X_cos is None:
            if not quiet:
                print("[WARN] --include_vcache_baseline set but cosine_to_anchor is unavailable; skipping vCache(original)")
        else:
            try:
                from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod
                vcache_method = PrecomputedCosineMethod()
                vcache_method.name = "vCache(original)"
                methods["vCache(original)"] = vcache_method
                if not quiet:
                    print("[INFO] Added vCache(original) competitor using precomputed nearest-neighbor cosine")
            except Exception as exc:
                if not quiet:
                    print(f"[WARN] Could not load vCache(original) baseline: {exc}")

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
        online_summary: Dict[str, Any] = {}
        if candidate_args.enable_online_stopping:
            online_method, _, online_summary, used_h0_train_idx, used_h1_train_idx = _run_online_stopping(
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
            if bool(getattr(candidate_args, "early_stop_train_subset", False)):
                if online_method is None:
                    return [], {
                        "reason": f"online_stopping_skipped:{online_summary.get('reason', 'unknown')}"
                    }
                _validate_online_used_indices(
                    y=y,
                    used_h0_train_idx=used_h0_train_idx,
                    used_h1_train_idx=used_h1_train_idx,
                )
                gs = _replace_global_train_indices(
                    gs,
                    used_h0_train_idx=used_h0_train_idx,
                    used_h1_train_idx=used_h1_train_idx,
                )
                h0_train_idx_fit = np.asarray(gs.H0_train, dtype=np.int64).reshape(-1)
                h1_train_idx_fit = np.asarray(gs.H1_train, dtype=np.int64).reshape(-1)
                h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
                h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

                H0_train = X_main[h0_train_idx_fit]
                H1_train = X_main[h1_train_idx_fit]
                H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                    if h0_train_idx_unique.size > 0 else X_main[gs.H0_calib]
                H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                    if h1_train_idx_unique.size > 0 else X_main[gs.H1_calib]
                H0_train_cos = X_cos[h0_train_idx_fit] if X_cos is not None else None
                H1_train_cos = X_cos[h1_train_idx_fit] if X_cos is not None else None
                if X_cos is not None:
                    H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                        if h0_train_idx_unique.size > 0 else X_cos[gs.H0_calib]
                    H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                        if h1_train_idx_unique.size > 0 else X_cos[gs.H1_calib]
                if X_text is not None:
                    H0_train_text = X_text[h0_train_idx_fit]
                    H1_train_text = X_text[h1_train_idx_fit]
                    H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                        if h0_train_idx_unique.size > 0 else X_text[gs.H0_calib]
                    H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                        if h1_train_idx_unique.size > 0 else X_text[gs.H1_calib]
                v0 = np.var(H0_calib_eff, axis=0)
                v1 = np.var(H1_calib_eff, axis=0)
                weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)
            if online_method is not None and _online_stopping_as_method(candidate_args):
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

        source = (
            "early_stop_train_subset"
            if bool(getattr(candidate_args, "early_stop_train_subset", False))
            else "full_train_split"
        )
        overrides = (
            {"Online(refit)": _online_method_train_count_meta(online_summary)}
            if online_method is not None and _online_stopping_as_method(candidate_args)
            else None
        )
        _annotate_train_sample_counts(
            rows,
            h0_train_idx=gs.H0_train,
            h1_train_idx=gs.H1_train,
            source=source,
            method_overrides=overrides,
        )

    return rows, {"failures": sum(len(v) for v in failures_local.values())}


def _evaluate_local_candidate(
    candidate_args: argparse.Namespace,
    *,
    splits: List[Any],
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
        online_summary: Dict[str, Any] = {}
        if candidate_args.enable_online_stopping:
            online_method, _, online_summary, used_h0_train_idx, used_h1_train_idx = _run_online_stopping(
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
            if bool(getattr(candidate_args, "early_stop_train_subset", False)):
                if online_method is None:
                    return [], {
                        "reason": f"online_stopping_skipped:{online_summary.get('reason', 'unknown')}"
                    }
                _validate_online_used_indices(
                    y=y,
                    used_h0_train_idx=used_h0_train_idx,
                    used_h1_train_idx=used_h1_train_idx,
                )
                splits = _replace_region_train_indices(
                    splits,
                    used_h0_train_idx=used_h0_train_idx,
                    used_h1_train_idx=used_h1_train_idx,
                    region_id=region_id,
                )
                h0_train_idx_list = [s.H0_train for s in splits if s.H0_train.size > 0]
                h1_train_idx_list = [s.H1_train for s in splits if s.H1_train.size > 0]
                h0_train_idx_fit = _concat_indices(h0_train_idx_list)
                h1_train_idx_fit = _concat_indices(h1_train_idx_list)
                h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
                h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

                H0_train = X_main[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_main[:0]
                H1_train = X_main[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_main[:0]
                H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                    if h0_train_idx_unique.size > 0 else H0_calib
                H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                    if h1_train_idx_unique.size > 0 else H1_calib

                if X_cos is not None:
                    H0_train_cos = X_cos[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_cos[:0]
                    H1_train_cos = X_cos[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_cos[:0]
                    H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                        if h0_train_idx_unique.size > 0 else H0_calib_cos
                    H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                        if h1_train_idx_unique.size > 0 else H1_calib_cos
                if X_text is not None:
                    H0_train_text = X_text[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_text[:0]
                    H1_train_text = X_text[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_text[:0]
                    H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                        if h0_train_idx_unique.size > 0 else H0_calib_text
                    H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                        if h1_train_idx_unique.size > 0 else H1_calib_text
                v0 = np.var(H0_calib_eff, axis=0)
                v1 = np.var(H1_calib_eff, axis=0)
                weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)
            if online_method is not None and _online_stopping_as_method(candidate_args):
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

        source = (
            "early_stop_train_subset"
            if bool(getattr(candidate_args, "early_stop_train_subset", False))
            else "local_train_split"
        )
        overrides = (
            {"Online(refit)": _online_method_train_count_meta(online_summary)}
            if online_method is not None and _online_stopping_as_method(candidate_args)
            else None
        )
        _annotate_train_sample_counts(
            rows,
            h0_train_idx=_concat_indices(h0_train_idx_list),
            h1_train_idx=_concat_indices(h1_train_idx_list),
            source=source,
            method_overrides=overrides,
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
        "--include_streaming_whitening",
        action="store_true",
        default=False,
        help="Opt in to StreamingWhitenedCosineMethod variants in the offline benchmark method list.",
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

    # Static scorer hyperparameters. None means use each method class default.
    ap.add_argument("--pca_whiten_abs_eps", type=float, default=None)
    ap.add_argument("--pca_whiten_rel_eps", type=float, default=None)
    ap.add_argument("--pca_whiten_max_rank", type=int, default=None)
    ap.add_argument(
        "--pca_whiten_rank_mode",
        type=str,
        default=None,
        choices=["fixed", "explained_variance", "threshold"],
    )
    ap.add_argument("--pca_whiten_explained_variance", type=float, default=None)
    ap.add_argument("--pca_whiten_norm_eps", type=float, default=None)

    ap.add_argument("--xgb_n_estimators", type=int, default=None)
    ap.add_argument("--xgb_max_depth", type=int, default=None)
    ap.add_argument("--xgb_learning_rate", type=float, default=None)
    ap.add_argument("--xgb_subsample", type=float, default=None)
    ap.add_argument("--xgb_colsample_bytree", type=float, default=None)
    ap.add_argument("--xgb_min_child_weight", type=float, default=None)
    ap.add_argument("--xgb_gamma", type=float, default=None)
    ap.add_argument("--xgb_reg_alpha", type=float, default=None)
    ap.add_argument("--xgb_reg_lambda", type=float, default=None)

    ap.add_argument("--tiny_mlp_hidden_dim", type=int, default=None)
    ap.add_argument("--tiny_mlp_n_layers", type=int, default=None)
    ap.add_argument("--tiny_mlp_alpha", type=float, default=None)
    ap.add_argument("--tiny_mlp_learning_rate_init", type=float, default=None)
    ap.add_argument("--tiny_mlp_max_iter", type=int, default=None)
    ap.add_argument("--tiny_mlp_batch_size", type=str, default=None)
    ap.add_argument(
        "--tiny_mlp_activation",
        type=str,
        default=None,
        choices=["identity", "relu", "tanh", "logistic"],
    )
    ap.add_argument("--tiny_mlp_early_stopping", dest="tiny_mlp_early_stopping", action="store_true", default=None)
    ap.add_argument("--no_tiny_mlp_early_stopping", dest="tiny_mlp_early_stopping", action="store_false")
    ap.add_argument("--tiny_mlp_validation_fraction", type=float, default=None)

    ap.add_argument("--lda_solver", type=str, default=None, choices=["svd", "lsqr", "eigen"])
    ap.add_argument("--lda_shrinkage", type=str, default=None)
    ap.add_argument("--lda_tol", type=float, default=None)

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
        "--include_vcache_baseline",
        action="store_true",
        default=False,
        help=(
            "Add vCache(original) as a competitor row using the original vCache nearest-neighbor "
            "cosine decision score, NP-calibrated on this benchmark split."
        ),
    )
    ap.add_argument(
        "--include_faiss_variants",
        action="store_true",
        default=False,
        help="Add exact factorized FAISS IndexFlatIP variants for eligible fitted pair scorers.",
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
    ap.add_argument(
        "--early_stop_train_subset",
        action="store_true",
        default=False,
        help="Use online stopping as a train-subset selector for all methods instead of training them on the full train split.",
    )
    ap.add_argument(
        "--online_stopping_as_method",
        action="store_true",
        default=None,
        help=(
            "Also add Online(refit) as a method row. If omitted, legacy mode adds it when "
            "--enable_online_stopping is set; --early_stop_train_subset suppresses it unless this flag is set."
        ),
    )
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
    ap.add_argument("--online_update_mode", type=str, default="refit", choices=["refit", "hill_climb", "reservoir"])
    ap.add_argument("--online_hill_lr", type=float, default=0.1)
    ap.add_argument("--online_init_h0", type=int, default=50)
    ap.add_argument("--online_init_h1", type=int, default=50)

    # Per-method train dosage search (validation on held-out monitor split)
    ap.add_argument(
        "--enable_train_dosage_search",
        action="store_true",
        default=False,
        help=(
            "For tau_mode=global, search a per-method train sample count using the monitor split, "
            "then fit/evaluate each method at its selected dosage."
        ),
    )
    ap.add_argument(
        "--train_dosage_grid",
        type=str,
        default="",
        help=(
            "Comma-separated total train sample counts to try per method. "
            "The full available train size is included unless --train_dosage_no_auto_full is set; "
            "if empty, uses a doubling grid."
        ),
    )
    ap.add_argument(
        "--train_dosage_no_auto_full",
        action="store_true",
        default=False,
        help="Do not automatically append the full available train size to --train_dosage_grid.",
    )
    ap.add_argument(
        "--train_dosage_max_train",
        type=int,
        default=None,
        help="Optional hard cap on train dosage candidates; candidates above this total are removed.",
    )
    ap.add_argument(
        "--train_dosage_tpr_tolerance",
        type=float,
        default=0.01,
        help=(
            "Select the smallest feasible dosage whose monitor TPR is within this absolute tolerance "
            "of the best feasible monitor TPR."
        ),
    )
    ap.add_argument(
        "--train_dosage_fpr_margin",
        type=float,
        default=0.0,
        help="Require monitor_fpr <= alpha - margin for train dosage FPR feasibility.",
    )

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
    ap.add_argument(
        "--debug_ablation_scores",
        action="store_true",
        default=False,
        help="Print ablation score diagnostics and pairwise score comparisons.",
    )

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    evaluation_mod.DEBUG_ABLATION_SCORES = bool(args.debug_ablation_scores)

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
    if args.early_stop_train_subset and not args.enable_online_stopping:
        raise ValueError("--early_stop_train_subset requires --enable_online_stopping")
    if args.online_stopping_as_method and not args.enable_online_stopping:
        raise ValueError("--online_stopping_as_method requires --enable_online_stopping")
    if args.enable_train_dosage_search and args.tau_mode != "global":
        raise ValueError("--enable_train_dosage_search currently supports --tau_mode global")
    if args.train_dosage_tpr_tolerance < 0.0:
        raise ValueError("--train_dosage_tpr_tolerance must be >= 0")
    if args.train_dosage_fpr_margin < 0.0:
        raise ValueError("--train_dosage_fpr_margin must be >= 0")
    if args.train_dosage_max_train is not None and int(args.train_dosage_max_train) <= 1:
        raise ValueError("--train_dosage_max_train must be > 1 when provided")
    if args.enable_train_dosage_search:
        for n in _parse_int_csv(args.train_dosage_grid):
            if n <= 1:
                raise ValueError("--train_dosage_grid values must be > 1 total samples")
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
    generated_pair_query: Optional[np.ndarray] = None
    generated_pair_anchor: Optional[np.ndarray] = None
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
        had_out = _build_hadamard_features(
            X_main,
            region_id,
            strategy=args.hadamard_anchor_strategy,
            seed=args.seed,
            use_delta_vec=args.use_delta_vec,
            use_abs_diff=args.use_abs_diff,
            abs_diff_only=abs_diff_only,
            return_pair_matrices=True,
        )
        X_main, X_cos_had, generated_pair_query, generated_pair_anchor = had_out
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

    faiss_pair_context = _resolve_faiss_pair_context(
        ds=ds,
        X_main=X_main,
        region_id=region_id,
        args=args,
        generated_query=generated_pair_query,
        generated_anchor=generated_pair_anchor,
        abs_diff_only=abs_diff_only,
    )

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
    print(f"include_faiss_variants={bool(args.include_faiss_variants)}")
    if args.include_faiss_variants:
        print(
            "faiss_pair_context: "
            f"source={faiss_pair_context.source} "
            f"available={bool(faiss_pair_context.available)} "
            f"features_are_hadamard={bool(faiss_pair_context.features_are_hadamard)} "
            f"reason={faiss_pair_context.reason or 'ok'}"
        )
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
        print(
            "online_stopping_policy: "
            f"early_stop_train_subset={bool(args.early_stop_train_subset)} "
            f"online_stopping_as_method={bool(_online_stopping_as_method(args))}"
        )
    if args.enable_train_dosage_search:
        print(
            "train_dosage_search: "
            f"grid={_parse_int_csv(args.train_dosage_grid) or 'auto'} "
            f"no_auto_full={bool(args.train_dosage_no_auto_full)} "
            f"max_train={args.train_dosage_max_train if args.train_dosage_max_train is not None else 'none'} "
            f"monitor(h0={args.n_monitor_h0}, h1={args.n_monitor_h1}) "
            f"tpr_tolerance={float(args.train_dosage_tpr_tolerance):.4f} "
            f"fpr_margin={float(args.train_dosage_fpr_margin):.4f}"
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
    train_dosage_search_rows: List[Dict[str, Any]] = []
    optuna_trial_rows: List[Dict[str, Any]] = []
    optuna_candidate_rows: List[Dict[str, Any]] = []
    optuna_best_rows: List[Dict[str, Any]] = []
    faiss_eligibility_rows: List[Dict[str, Any]] = []
    faiss_equivalence_rows: List[Dict[str, Any]] = []

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
            if args.enable_online_stopping or args.enable_train_dosage_search:
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

            online_method = None
            online_history_rows: List[Dict[str, Any]] = []
            online_summary: Dict[str, Any] = {}
            early_stop_train_subset_applied = False
            online_as_method = _online_stopping_as_method(args)
            if args.enable_online_stopping:
                (
                    online_method,
                    online_history_rows,
                    online_summary,
                    used_h0_train_idx,
                    used_h1_train_idx,
                ) = _run_online_stopping(
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
                if args.early_stop_train_subset:
                    if online_method is None:
                        raise RuntimeError(
                            "early_stop_train_subset requested but online stopping did not run: "
                            f"reason={online_summary.get('reason', 'unknown')}"
                        )
                    _validate_online_used_indices(
                        y=y,
                        used_h0_train_idx=used_h0_train_idx,
                        used_h1_train_idx=used_h1_train_idx,
                    )
                    gs = _replace_global_train_indices(
                        gs,
                        used_h0_train_idx=used_h0_train_idx,
                        used_h1_train_idx=used_h1_train_idx,
                    )
                    early_stop_train_subset_applied = True

                    h0_train_idx_fit = np.asarray(gs.H0_train, dtype=np.int64).reshape(-1)
                    h1_train_idx_fit = np.asarray(gs.H1_train, dtype=np.int64).reshape(-1)
                    h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
                    h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

                    H0_train = X_main[h0_train_idx_fit]
                    H1_train = X_main[h1_train_idx_fit]
                    H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                        if h0_train_idx_unique.size > 0 else X_main[gs.H0_calib]
                    H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                        if h1_train_idx_unique.size > 0 else X_main[gs.H1_calib]

                    H0_train_cos = X_cos[h0_train_idx_fit] if X_cos is not None else None
                    H1_train_cos = X_cos[h1_train_idx_fit] if X_cos is not None else None
                    if X_cos is not None:
                        H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                            if h0_train_idx_unique.size > 0 else X_cos[gs.H0_calib]
                        H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                            if h1_train_idx_unique.size > 0 else X_cos[gs.H1_calib]
                    if X_text is not None:
                        H0_train_text = X_text[h0_train_idx_fit]
                        H1_train_text = X_text[h1_train_idx_fit]
                        H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, gs.H0_calib])] \
                            if h0_train_idx_unique.size > 0 else X_text[gs.H0_calib]
                        H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, gs.H1_calib])] \
                            if h1_train_idx_unique.size > 0 else X_text[gs.H1_calib]

                online_summary = _record_online_stopping_outputs(
                    trial=trial,
                    seed=seed,
                    history_rows=online_history_rows,
                    summary=online_summary,
                    early_stop_train_subset_active=bool(args.early_stop_train_subset),
                    early_stop_train_subset_applied=bool(early_stop_train_subset_applied),
                    online_stopping_as_method=bool(online_as_method),
                    online_stopping_history_rows=online_stopping_history_rows,
                    online_stopping_summary_rows=online_stopping_summary_rows,
                )

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
                _annotate_train_sample_counts(
                    trial_rows,
                    h0_train_idx=gs.H0_train,
                    h1_train_idx=gs.H1_train,
                    source="early_stop_train_subset" if early_stop_train_subset_applied else "global_split",
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
                    for ens_name in [
                        "WeightedEnsemble",
                        "RandomForestEnsemble",
                        "WeightedEnsembleNoPCAWhitenedCosine",
                        "RandomForestEnsembleNoPCAWhitenedCosine",
                        "RegionalWeightedEnsemble",
                    ]:
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
                    if "RandomForestEnsemble" in methods:
                        print("[INFO] RandomForestEnsemble Cosine judge routed to direct Hadamard sum")
                    if "WeightedEnsembleNoPCAWhitenedCosine" in methods:
                        print("[INFO] WeightedEnsembleNoPCAWhitenedCosine Cosine judge routed to direct Hadamard sum")
                    if "RandomForestEnsembleNoPCAWhitenedCosine" in methods:
                        print("[INFO] RandomForestEnsembleNoPCAWhitenedCosine Cosine judge routed to direct Hadamard sum")
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

            train_dosage_method_overrides: Dict[str, Dict[str, Any]] = {}
            if args.enable_train_dosage_search:
                methods, train_dosage_method_overrides, dosage_rows_trial = _run_global_train_dosage_search(
                    method_names=method_names,
                    gs=gs,
                    h0_monitor_idx=h0_monitor_idx,
                    h1_monitor_idx=h1_monitor_idx,
                    X_main=X_main,
                    X_cos=X_cos,
                    X_text=X_text,
                    region_id=region_id,
                    args=args,
                    seed=seed,
                    trial=trial,
                    failures=failures,
                )
                train_dosage_search_rows.extend(dosage_rows_trial)
                method_names = [m for m in method_names if m in methods]
                configured_methods_last = method_names[:]
            else:
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

            if args.enable_online_stopping and online_as_method:
                if online_method is not None:
                    methods["Online(refit)"] = online_method
                    method_names.append("Online(refit)")
                    configured_methods_last = method_names[:]

            if "BGE Reranker" in methods:
                bge_m = methods["BGE Reranker"]
                if bool(getattr(bge_m, "using_fallback", False)):
                    reason = str(getattr(bge_m, "fallback_reason", "model load failed"))
                    reason_one_line = reason.splitlines()[0][:240]
                    print(f"[WARN] BGE Reranker running in fallback mode: {reason_one_line}")

            _register_faiss_variants(
                methods=methods,
                method_names=method_names,
                pair_context=faiss_pair_context,
                X_main=X_main,
                X_cos=X_cos,
                args=args,
                trial=trial,
                seed=seed,
                scope="global",
                report_rows=faiss_eligibility_rows,
            )
            configured_methods_last = method_names[:]

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
            _append_faiss_equivalence_rows(
                trial_rows,
                methods=methods,
                alpha=float(args.alpha),
                scope="global",
                out_rows=faiss_equivalence_rows,
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

            train_sample_source = "early_stop_train_subset" if early_stop_train_subset_applied else "full_train_split"
            method_overrides = dict(train_dosage_method_overrides)
            if args.enable_online_stopping and online_as_method and online_method is not None:
                method_overrides["Online(refit)"] = _online_method_train_count_meta(online_summary)
            _annotate_train_sample_counts(
                trial_rows,
                h0_train_idx=gs.H0_train,
                h1_train_idx=gs.H1_train,
                source=train_sample_source,
                method_overrides=method_overrides or None,
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

            online_method = None
            online_history_rows: List[Dict[str, Any]] = []
            online_summary: Dict[str, Any] = {}
            early_stop_train_subset_applied = False
            online_as_method = _online_stopping_as_method(args)
            if args.enable_online_stopping:
                (
                    online_method,
                    online_history_rows,
                    online_summary,
                    used_h0_train_idx,
                    used_h1_train_idx,
                ) = _run_online_stopping(
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
                if args.early_stop_train_subset:
                    if online_method is None:
                        raise RuntimeError(
                            "early_stop_train_subset requested but online stopping did not run: "
                            f"reason={online_summary.get('reason', 'unknown')}"
                        )
                    _validate_online_used_indices(
                        y=y,
                        used_h0_train_idx=used_h0_train_idx,
                        used_h1_train_idx=used_h1_train_idx,
                    )
                    splits = _replace_region_train_indices(
                        splits,
                        used_h0_train_idx=used_h0_train_idx,
                        used_h1_train_idx=used_h1_train_idx,
                        region_id=region_id,
                    )
                    early_stop_train_subset_applied = True

                    h0_train_idx_list = [s.H0_train for s in splits if s.H0_train.size > 0]
                    h1_train_idx_list = [s.H1_train for s in splits if s.H1_train.size > 0]
                    h0_train_idx_fit = _concat_indices(h0_train_idx_list)
                    h1_train_idx_fit = _concat_indices(h1_train_idx_list)
                    h0_train_idx_unique = np.unique(h0_train_idx_fit) if h0_train_idx_fit.size > 0 else h0_train_idx_fit
                    h1_train_idx_unique = np.unique(h1_train_idx_fit) if h1_train_idx_fit.size > 0 else h1_train_idx_fit

                    H0_train = X_main[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_main[:0]
                    H1_train = X_main[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_main[:0]
                    H0_calib_eff = X_main[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                        if h0_train_idx_unique.size > 0 else H0_calib
                    H1_calib_eff = X_main[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                        if h1_train_idx_unique.size > 0 else H1_calib

                    if X_cos is not None:
                        H0_train_cos = X_cos[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_cos[:0]
                        H1_train_cos = X_cos[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_cos[:0]
                        H0_calib_eff_cos = X_cos[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                            if h0_train_idx_unique.size > 0 else H0_calib_cos
                        H1_calib_eff_cos = X_cos[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                            if h1_train_idx_unique.size > 0 else H1_calib_cos
                    if X_text is not None:
                        H0_train_text = X_text[h0_train_idx_fit] if h0_train_idx_fit.size > 0 else X_text[:0]
                        H1_train_text = X_text[h1_train_idx_fit] if h1_train_idx_fit.size > 0 else X_text[:0]
                        H0_calib_eff_text = X_text[np.concatenate([h0_train_idx_unique, h0_calib_idx])] \
                            if h0_train_idx_unique.size > 0 else H0_calib_text
                        H1_calib_eff_text = X_text[np.concatenate([h1_train_idx_unique, h1_calib_idx])] \
                            if h1_train_idx_unique.size > 0 else H1_calib_text

                online_summary = _record_online_stopping_outputs(
                    trial=trial,
                    seed=seed,
                    history_rows=online_history_rows,
                    summary=online_summary,
                    early_stop_train_subset_active=bool(args.early_stop_train_subset),
                    early_stop_train_subset_applied=bool(early_stop_train_subset_applied),
                    online_stopping_as_method=bool(online_as_method),
                    online_stopping_history_rows=online_stopping_history_rows,
                    online_stopping_summary_rows=online_stopping_summary_rows,
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

                _annotate_train_sample_counts(
                    trial_rows,
                    h0_train_idx=gs_ocats.H0_train,
                    h1_train_idx=gs_ocats.H1_train,
                    source=(
                        "early_stop_train_subset"
                        if early_stop_train_subset_applied
                        else "local_eligible_regions_train_split"
                    ),
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
                    for ens_name in [
                        "WeightedEnsemble",
                        "RandomForestEnsemble",
                        "WeightedEnsembleNoPCAWhitenedCosine",
                        "RandomForestEnsembleNoPCAWhitenedCosine",
                        "RegionalWeightedEnsemble",
                    ]:
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
                    if "RandomForestEnsemble" in methods:
                        print("[INFO] RandomForestEnsemble Cosine judge routed to direct Hadamard sum")
                    if "WeightedEnsembleNoPCAWhitenedCosine" in methods:
                        print("[INFO] WeightedEnsembleNoPCAWhitenedCosine Cosine judge routed to direct Hadamard sum")
                    if "RandomForestEnsembleNoPCAWhitenedCosine" in methods:
                        print("[INFO] RandomForestEnsembleNoPCAWhitenedCosine Cosine judge routed to direct Hadamard sum")
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

            if args.enable_online_stopping and online_as_method:
                if online_method is not None:
                    methods["Online(refit)"] = online_method
                    method_names.append("Online(refit)")
                    configured_methods_last = method_names[:]

            if "BGE Reranker" in methods:
                bge_m = methods["BGE Reranker"]
                if bool(getattr(bge_m, "using_fallback", False)):
                    reason = str(getattr(bge_m, "fallback_reason", "model load failed"))
                    reason_one_line = reason.splitlines()[0][:240]
                    print(f"[WARN] BGE Reranker running in fallback mode: {reason_one_line}")

            _register_faiss_variants(
                methods=methods,
                method_names=method_names,
                pair_context=faiss_pair_context,
                X_main=X_main,
                X_cos=X_cos,
                args=args,
                trial=trial,
                seed=seed,
                scope="local",
                report_rows=faiss_eligibility_rows,
            )
            configured_methods_last = method_names[:]

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
            _append_faiss_equivalence_rows(
                trial_rows,
                methods=methods,
                alpha=float(args.alpha),
                scope="local",
                out_rows=faiss_equivalence_rows,
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
                            for ens_name in [
                                "WeightedEnsemble",
                                "RandomForestEnsemble",
                                "WeightedEnsembleNoPCAWhitenedCosine",
                                "RandomForestEnsembleNoPCAWhitenedCosine",
                            ]:
                                if ens_name in methods_matched and hasattr(methods_matched[ens_name], "judges"):
                                    ens_m = methods_matched[ens_name]
                                    new_judges_m = []
                                    for j in list(getattr(ens_m, "judges", [])):
                                        if str(getattr(j, "name", "")) == "Cosine":
                                            new_judges_m.append(PrecomputedCosineMethod())
                                        else:
                                            new_judges_m.append(j)
                                    ens_m.judges = new_judges_m
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
                    _annotate_train_sample_counts(
                        matched_rows,
                        h0_train_idx=gs_matched.H0_train,
                        h1_train_idx=gs_matched.H1_train,
                        source=(
                            "early_stop_train_subset_matched_local_regions"
                            if early_stop_train_subset_applied
                            else "matched_local_regions_train_split"
                        ),
                    )
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

            train_sample_source = "early_stop_train_subset" if early_stop_train_subset_applied else "local_train_split"
            method_overrides = (
                {"Online(refit)": _online_method_train_count_meta(online_summary)}
                if args.enable_online_stopping and online_as_method and online_method is not None
                else None
            )
            _annotate_train_sample_counts(
                trial_rows,
                h0_train_idx=_concat_indices(h0_train_idx_list),
                h1_train_idx=_concat_indices(h1_train_idx_list),
                source=train_sample_source,
                method_overrides=method_overrides,
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
    ranking = aggregate_ranking(trial_summary_rows, alpha=float(args.alpha))
    _attach_train_sample_counts_to_ranking(ranking, trial_summary_rows)
    for i, row in enumerate(ranking, start=1):
        row["primary_safety_rank"] = int(i)
    cache_efficiency_ranking = _build_cache_efficiency_ranking(
        ranking,
        alpha=float(args.alpha),
        tpr_tolerance=float(args.train_dosage_tpr_tolerance),
    )
    cache_rank_by_method = {
        str(r.get("method", "")): int(r.get("cache_efficiency_rank", 0))
        for r in cache_efficiency_ranking
    }
    for row in ranking:
        method = str(row.get("method", ""))
        if method in cache_rank_by_method:
            row["cache_efficiency_rank"] = int(cache_rank_by_method[method])
    constrained = print_ranking(
        ranking,
        alpha=float(args.alpha),
        tpr_tolerance=float(args.train_dosage_tpr_tolerance),
    )

    # Save outputs
    save_csv_rows(
        run_dir / "trial_summary.csv",
        trial_summary_rows,
        fieldnames=[
            "trial", "seed", "method", "region_key", "tau_mode", "tau", "tau_mean",
            "train_h0_samples", "train_h1_samples", "train_total_samples",
            "train_samples_needed", "train_sample_source",
            "train_dosage_search_active", "train_dosage_monitor_tpr", "train_dosage_monitor_fpr",
            "train_dosage_fpr_margin", "train_dosage_fpr_limit", "train_dosage_fpr_feasible",
            "train_dosage_tpr_tolerance", "train_dosage_best_feasible_monitor_tpr",
            "train_dosage_tpr_gap_from_best_feasible", "train_dosage_max_train",
            "train_dosage_no_auto_full", "train_dosage_selection_reason",
            "micro_tpr", "micro_fpr", "train_tpr", "train_fpr",
            "macro_tpr", "macro_fpr",
            "ok_regions", "shared_regions", "dropped_regions_for_comparability",
            "input_space", "time_ms",
        ],
    )
    save_json(
        run_dir / "ranking.json",
        {
            "ranking_objective": "primary_safety_then_tpr_then_train_n",
            "alpha": float(args.alpha),
            "cache_efficiency_tpr_tolerance": float(args.train_dosage_tpr_tolerance),
            "ranking": ranking,
            "cache_efficiency_ranking": cache_efficiency_ranking,
        },
    )
    if matched_global_trial_rows:
        save_csv_rows(
            run_dir / "matched_global_trial_summary.csv",
            matched_global_trial_rows,
            fieldnames=[
                "trial", "seed", "method", "region_key", "tau_mode", "comparison_scope",
                "train_h0_samples", "train_h1_samples", "train_total_samples",
                "train_samples_needed", "train_sample_source",
                "train_dosage_search_active", "train_dosage_monitor_tpr", "train_dosage_monitor_fpr",
                "train_dosage_selection_reason",
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
    if train_dosage_search_rows:
        save_csv_rows(
            run_dir / "train_dosage_search.csv",
            train_dosage_search_rows,
            fieldnames=[
                "trial", "seed", "method",
                "requested_train_total", "train_dosage_total", "train_h0", "train_h1",
                "train_h0_samples", "train_h1_samples", "train_total_samples",
                "monitor_tpr", "monitor_fpr", "monitor_train_tpr", "monitor_train_fpr", "monitor_tau",
                "alpha", "train_dosage_fpr_margin", "effective_monitor_fpr_limit",
                "fpr_feasible", "feasible", "selected", "selection_reason",
                "best_feasible_monitor_tpr", "tpr_gap_from_best_feasible",
                "train_dosage_tpr_tolerance", "train_dosage_max_train", "train_dosage_no_auto_full",
                "fit_failed", "failure_reason",
                "input_space", "time_ms",
            ],
        )
        save_json(
            run_dir / "train_dosage_search.json",
            {"rows": train_dosage_search_rows},
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
    if args.include_faiss_variants:
        save_csv_rows(
            run_dir / "faiss_eligibility_report.csv",
            faiss_eligibility_rows,
            fieldnames=[
                "trial", "seed", "scope", "method", "faiss_method",
                "eligible", "reason",
                "pair_context_source", "pair_context_reason",
                "features_are_hadamard", "pair_context_rows",
            ],
        )
        save_csv_rows(
            run_dir / "faiss_equivalence_report.csv",
            faiss_equivalence_rows,
            fieldnames=[
                "trial", "seed", "scope", "alpha",
                "source_method", "faiss_method", "tau_mode", "comparison_scope",
                "max_score_diff", "scored_rows", "rank_agreement",
                "dTPR", "dFPR", "dMacroTPR", "dMacroFPR", "dTrainTPR", "dTrainFPR",
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
            "include_faiss_variants": bool(args.include_faiss_variants),
            "faiss_pair_context_source": str(faiss_pair_context.source),
            "faiss_pair_context_available": bool(faiss_pair_context.available),
            "faiss_pair_context_features_are_hadamard": bool(faiss_pair_context.features_are_hadamard),
            "faiss_pair_context_reason": str(faiss_pair_context.reason),
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
                "early_stop_train_subset_active": bool(getattr(base_args, "early_stop_train_subset", False)),
                "online_stopping_as_method": bool(_online_stopping_as_method(base_args)),
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
            "train_dosage_search": {
                "enabled": bool(base_args.enable_train_dosage_search),
                "grid": _parse_int_csv(base_args.train_dosage_grid),
                "grid_auto": bool(not _parse_int_csv(base_args.train_dosage_grid)),
                "no_auto_full": bool(base_args.train_dosage_no_auto_full),
                "max_train": (
                    int(base_args.train_dosage_max_train)
                    if base_args.train_dosage_max_train is not None
                    else None
                ),
                "tpr_tolerance": float(base_args.train_dosage_tpr_tolerance),
                "fpr_margin": float(base_args.train_dosage_fpr_margin),
                "effective_monitor_fpr_limit": float(base_args.alpha - base_args.train_dosage_fpr_margin),
                "rows": int(len(train_dosage_search_rows)),
                "monitor_h0": int(base_args.n_monitor_h0),
                "monitor_h1": int(base_args.n_monitor_h1),
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
        print(f"\nBest (primary safety ranking): {constrained[0]['method']}")
    elif ranking:
        print(f"\nBest (fallback; none met full safety criteria): {ranking[0]['method']}")
    print(f"[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
