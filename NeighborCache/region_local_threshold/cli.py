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
from .evaluation import fit_all_methods, evaluate_methods, evaluate_methods_global, aggregate_ranking
from .display import print_trial_table, print_ranking

NC_ROOT = ROOT / "NeighborCache"
OUT_BASE = NC_ROOT / "outputs" / "region_local_threshold"

REGION_KEY_ALIASES = {
    "sem_bucket": "global_cluster",
}


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(denom, eps)).astype(np.float32, copy=False)


def _build_hadamard_features(
    X_main: np.ndarray,
    region_id: np.ndarray,
    *,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Hadamard features and cosine-to-anchor from feature vectors.

    Returns:
      hadamard: (N, D) float32
      cosine_to_anchor: (N, 1) float32
    """
    Xn = _l2_normalize_rows(X_main)
    region_id = np.asarray(region_id, dtype=np.int64).reshape(-1)
    N, D = Xn.shape

    anchors = np.zeros((N, D), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for rid in np.unique(region_id):
        idx = np.flatnonzero(region_id == rid)
        if idx.size == 0:
            continue
        Xr = Xn[idx]

        if strategy == "random":
            anchor_global = int(rng.choice(idx))
        else:
            centroid = np.mean(Xr, axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            sims = Xr @ centroid
            anchor_global = int(idx[int(np.argmax(sims))])

        anchors[idx] = Xn[anchor_global]

    had = (Xn * anchors).astype(np.float32, copy=False)
    cos = np.sum(had, axis=1, keepdims=True).astype(np.float32, copy=False)
    return had, cos


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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Region-local threshold benchmark with train/calib protocol."
    )
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--region_key", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tie_mode", type=str, default="ge", choices=["ge", "gt"])
    ap.add_argument("--tau_mode", type=str, default="local", choices=["local", "global"])
    ap.add_argument("--tau_shrink", action="store_true", default=False)
    ap.add_argument("--tau_shrink_m", type=float, default=500.0)
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

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    npz_path = resolve_npz_path(args.data)
    ds = load_npz(npz_path)

    region_key = args.region_key
    if region_key not in ds:
        alias_key = REGION_KEY_ALIASES.get(region_key)
        if alias_key in ds:
            print(
                f"[INFO] region_key='{region_key}' not found; using alias key='{alias_key}'"
            )
            region_key = alias_key

    required = {region_key, "label"}
    miss = sorted(required - set(ds.keys()))
    if miss:
        raise ValueError(f"{npz_path} missing required arrays: {miss}; have={sorted(ds.keys())}")

    region_id = ds[region_key].astype(np.int64, copy=False)
    y = ds["label"].astype(np.int32, copy=False)

    feat_key, X_main, X_cos = resolve_features(ds)

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

    if args.hadamard_preprocess:
        if X_main.ndim != 2 or X_main.shape[1] <= 1:
            raise ValueError(
                f"--hadamard_preprocess requires embedding-like X_main with shape (N,D), D>1; got {X_main.shape}"
            )
        X_main, X_cos_had = _build_hadamard_features(
            X_main,
            region_id,
            strategy=args.hadamard_anchor_strategy,
            seed=args.seed,
        )
        X_cos = X_cos_had
        feat_key = f"{feat_key}+hadamard"
        print(
            "[INFO] Applied Hadamard preprocessing "
            f"(strategy={args.hadamard_anchor_strategy}) to X_main"
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
    print(f"features={feat_key} rows={X_main.shape[0]} dim={X_main.shape[1]}")
    print(f"text_pairs={text_key if text_key is not None else 'none'}")
    print(f"alpha={args.alpha} tie_mode={args.tie_mode} trials={args.n_trials} base_seed={args.seed}")
    print(f"tau_mode={args.tau_mode}")
    print(f"hadamard_preprocess={bool(args.hadamard_preprocess)}")
    print(f"normalize_data={bool(args.normalize_data)}")
    print(f"filter_policy={args.filter_policy}")
    if args.filter_policy == "ambiguous_only":
        print(f"ambiguous_range=[{args.ambiguous_cos_min}, {args.ambiguous_cos_max}]")
    print(f"caps: train={args.n_train} calib={args.n_calib} eval={args.n_eval}")
    print(f"mins: min_h0_eval={args.min_h0_eval} min_h1_eval={args.min_h1_eval}")
    print(f"run_dir={run_dir}")

    has_xgb = "XGBoost" in build_methods()

    trial_summary_rows: List[Dict[str, Any]] = []
    matched_global_trial_rows: List[Dict[str, Any]] = []
    local_vs_global_matched_rows: List[Dict[str, Any]] = []
    local_region_status_rows: List[Dict[str, Any]] = []
    failures: Dict[str, List[str]] = defaultdict(list)
    configured_methods_last: List[str] = []
    weighted_ensemble_meta_rows: List[Dict[str, Any]] = []

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
                    if "WeightedEnsemble" in methods and hasattr(methods["WeightedEnsemble"], "judges"):
                        we = methods["WeightedEnsemble"]
                        new_judges = []
                        replaced = 0
                        for j in list(getattr(we, "judges", [])):
                            if str(getattr(j, "name", "")) == "Cosine":
                                new_judges.append(PrecomputedCosineMethod())
                                replaced += 1
                            else:
                                new_judges.append(j)
                        we.judges = new_judges
                    print("[INFO] Cosine baseline routed to direct Hadamard sum (PrecomputedCosine)")
                    if "WeightedEnsemble" in methods:
                        print("[INFO] WeightedEnsemble Cosine judge routed to direct Hadamard sum")
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
                tie_mode=args.tie_mode,
                tau_guardrail=args.tau_guardrail,
                tau_guardrail_delta=args.tau_guardrail_delta,
                weights=weights,
                seed=seed,
                alpha=args.alpha,
                trial=trial,
                failures=failures,
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

            if len(splits) == 0:
                raise RuntimeError(
                    "No regions satisfied eval mins. Lower mins, increase caps, or change regioning."
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
            print(f"  pooled_train: n0={H0_train.shape[0]} n1={H1_train.shape[0]}")
            print(f"  pooled_calib_eff: n0={H0_calib_eff.shape[0]} n1={H1_calib_eff.shape[0]}")

            if H0_calib_eff.shape[0] == 0 or H1_calib_eff.shape[0] == 0:
                raise RuntimeError("No data available for fitting/calibration.")

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
                    if "WeightedEnsemble" in methods and hasattr(methods["WeightedEnsemble"], "judges"):
                        we = methods["WeightedEnsemble"]
                        new_judges = []
                        replaced = 0
                        for j in list(getattr(we, "judges", [])):
                            if str(getattr(j, "name", "")) == "Cosine":
                                new_judges.append(PrecomputedCosineMethod())
                                replaced += 1
                            else:
                                new_judges.append(j)
                        we.judges = new_judges
                    print("[INFO] Cosine baseline routed to direct Hadamard sum (PrecomputedCosine)")
                    if "WeightedEnsemble" in methods:
                        print("[INFO] WeightedEnsemble Cosine judge routed to direct Hadamard sum")
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
            fit_all_methods(
                methods,
                H0_train=H0_train,
                H1_train=H1_train,
                H0_calib_eff=H0_calib_eff,
                H1_calib_eff=H1_calib_eff,
                H0_train_cos=H0_train_cos,
                H1_train_cos=H1_train_cos,
                H0_calib_eff_cos=H0_calib_eff_cos,
                H1_calib_eff_cos=H1_calib_eff_cos,
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
                tau_guardrail=args.tau_guardrail,
                tau_guardrail_delta=args.tau_guardrail_delta,
                swc_mode=args.swc_mode,
                swc_cluster_n_clusters=args.swc_cluster_n_clusters,
                cos_affine_grouping=args.cos_affine_grouping,
                cos_affine_n_clusters=args.cos_affine_n_clusters,
                trial=trial,
                seed=seed,
                region_key=args.region_key,
                h0_train_idx_list=h0_train_idx_list,
                h1_train_idx_list=h1_train_idx_list,
                h0_calib_list=h0_calib_list,
                h1_calib_list=h1_calib_list,
                H0_calib_eff=H0_calib_eff,
                failures=failures,
                trial_meta=local_eval_meta,
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
                        H0_train_cos=H0_train_cos,
                        H1_train_cos=H1_train_cos,
                        H0_calib_eff_cos=H0_calib_eff_cos,
                        H1_calib_eff_cos=H1_calib_eff_cos,
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
            "features": feat_key,
            "alpha": float(args.alpha),
            "tie_mode": args.tie_mode,
            "tau_mode": args.tau_mode,
            "hadamard_preprocess": bool(args.hadamard_preprocess),
            "hadamard_anchor_strategy": args.hadamard_anchor_strategy,
            "hadamard_cosine_direct": bool(args.hadamard_preprocess),
            "normalize_data": bool(args.normalize_data),
            "tau_shrink": bool(args.tau_shrink),
            "tau_shrink_m": float(args.tau_shrink_m),
            "tau_guardrail": args.tau_guardrail,
            "tau_guardrail_delta": float(args.tau_guardrail_delta),
            "filter_policy": args.filter_policy,
            "ambiguous_cos_min": float(args.ambiguous_cos_min),
            "ambiguous_cos_max": float(args.ambiguous_cos_max),
            "cos_affine_calib": bool(args.cos_affine_calib),
            "precomputed_cosine": bool(args.precomputed_cosine),
            "cos_affine_grouping": args.cos_affine_grouping,
            "cos_affine_n_clusters": int(args.cos_affine_n_clusters),
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
            "matched_global_eval_enabled": bool(args.tau_mode == "local"),
            "matched_global_trial_rows": int(len(matched_global_trial_rows)),
            "local_vs_global_matched_rows": int(len(local_vs_global_matched_rows)),
            "local_region_status_rows": int(len(local_region_status_rows)),
            "failures": failures,
        },
    )

    best = constrained[0]["method"] if constrained else (ranking[0]["method"] if ranking else None)
    if best:
        print(f"\nBest (respecting constraint if possible): {best}")
    print(f"[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
