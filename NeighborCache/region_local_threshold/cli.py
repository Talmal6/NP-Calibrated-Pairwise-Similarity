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
from .methods import build_methods, needs_weights
from .splits import split_indices_per_region, split_global
from .evaluation import fit_all_methods, evaluate_methods, evaluate_methods_global, aggregate_ranking
from .display import print_trial_table, print_ranking

NC_ROOT = ROOT / "NeighborCache"
OUT_BASE = NC_ROOT / "outputs" / "region_local_threshold"


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
        "--cos_affine_grouping",
        type=str,
        default="region",
        choices=["region", "cluster"],
    )
    ap.add_argument("--cos_affine_n_clusters", type=int, default=64)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    npz_path = resolve_npz_path(args.data)
    ds = load_npz(npz_path)

    required = {args.region_key, "label"}
    miss = sorted(required - set(ds.keys()))
    if miss:
        raise ValueError(f"{npz_path} missing required arrays: {miss}; have={sorted(ds.keys())}")

    region_id = ds[args.region_key].astype(np.int64, copy=False)
    y = ds["label"].astype(np.int32, copy=False)

    feat_key, X_main, X_cos = resolve_features(ds)

    if X_main.shape[0] != y.shape[0] or X_main.shape[0] != region_id.shape[0]:
        raise ValueError(
            f"Row mismatch: X={X_main.shape[0]} y={y.shape[0]} region_id={region_id.shape[0]}"
        )
    if X_cos is not None and X_cos.shape[0] != X_main.shape[0]:
        raise ValueError(f"Row mismatch: X_cos={X_cos.shape[0]} X_main={X_main.shape[0]}")

    run_dir = make_run_dir(base_dir=str(OUT_BASE), run_name=args.run_name)

    print("\n=== Region Threshold Benchmark ===")
    print(f"dataset_npz={npz_path}")
    print(f"region_key={args.region_key} (unique={len(np.unique(region_id))})")
    print(f"features={feat_key} rows={X_main.shape[0]} dim={X_main.shape[1]}")
    print(f"alpha={args.alpha} tie_mode={args.tie_mode} trials={args.n_trials} base_seed={args.seed}")
    print(f"tau_mode={args.tau_mode}")
    print(f"caps: train={args.n_train} calib={args.n_calib} eval={args.n_eval}")
    print(f"mins: min_h0_eval={args.min_h0_eval} min_h1_eval={args.min_h1_eval}")
    print(f"run_dir={run_dir}")

    has_xgb = "XGBoost" in build_methods()

    trial_summary_rows: List[Dict[str, Any]] = []
    failures: Dict[str, List[str]] = defaultdict(list)
    configured_methods_last: List[str] = []

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

            H0_train = X_main[gs.H0_train]
            H1_train = X_main[gs.H1_train]
            H0_calib_eff = X_main[np.concatenate([gs.H0_train, gs.H0_calib])] \
                if gs.H0_train.size > 0 else X_main[gs.H0_calib]
            H1_calib_eff = X_main[np.concatenate([gs.H1_train, gs.H1_calib])] \
                if gs.H1_train.size > 0 else X_main[gs.H1_calib]

            # Extract X_cos splits if available
            H0_train_cos = X_cos[gs.H0_train] if X_cos is not None else None
            H1_train_cos = X_cos[gs.H1_train] if X_cos is not None else None
            if X_cos is not None:
                H0_calib_eff_cos = X_cos[np.concatenate([gs.H0_train, gs.H0_calib])] \
                    if gs.H0_train.size > 0 else X_cos[gs.H0_calib]
                H1_calib_eff_cos = X_cos[np.concatenate([gs.H1_train, gs.H1_calib])] \
                    if gs.H1_train.size > 0 else X_cos[gs.H1_calib]
            else:
                H0_calib_eff_cos = None
                H1_calib_eff_cos = None

            n_unique_regions = int(len(np.unique(region_id)))
            print(f"\n[trial={trial} seed={seed}] GLOBAL mode — total_samples={y.size}")
            print(f"  total: h0={gs_stats['total_h0']} h1={gs_stats['total_h1']}")
            print(f"  pooled_train:  n0={gs_stats['h0_train']} n1={gs_stats['h1_train']}")
            print(f"  pooled_calib:  n0={gs_stats['h0_calib']} n1={gs_stats['h1_calib']}")
            print(f"  pooled_eval:   n0={gs_stats['h0_eval']} n1={gs_stats['h1_eval']}")
            print(f"  regions (for macro stats only): {n_unique_regions}")

            if H0_calib_eff.shape[0] == 0 or H1_calib_eff.shape[0] == 0:
                raise RuntimeError("No data available for fitting/calibration.")

            # Weights for feature-based methods
            v0 = np.var(H0_calib_eff, axis=0)
            v1 = np.var(H1_calib_eff, axis=0)
            weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)

            methods = build_methods()
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

            method_names = list(methods.keys())
            configured_methods_last = method_names[:]

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
                tie_mode=args.tie_mode,
                tau_guardrail=args.tau_guardrail,
                tau_guardrail_delta=args.tau_guardrail_delta,
                weights=weights,
                seed=seed,
                alpha=args.alpha,
                trial=trial,
                failures=failures,
            )

            trial_rows = evaluate_methods_global(
                methods,
                method_names,
                gs,
                X_main=X_main,
                X_cos=X_cos,
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
            splits, split_stats = split_indices_per_region(
                region_id=region_id,
                y=y,
                n_train_cap=args.n_train,
                n_calib_cap=args.n_calib,
                n_eval_cap=args.n_eval,
                seed=seed,
                min_h0_eval=args.min_h0_eval,
                min_h1_eval=args.min_h1_eval,
            )

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

            print(f"\n[trial={trial} seed={seed}] used_regions={len(splits)} split_stats={split_stats}")
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
                tie_mode=args.tie_mode,
                tau_guardrail=args.tau_guardrail,
                tau_guardrail_delta=args.tau_guardrail_delta,
                weights=weights,
                seed=seed,
                alpha=args.alpha,
                trial=trial,
                failures=failures,
            )

            # Evaluate methods
            trial_rows = evaluate_methods(
                methods,
                method_names,
                splits,
                X_main=X_main,
                X_cos=X_cos,
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
            )

        trial_summary_rows.extend(trial_rows)
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
            "ok_regions", "time_ms",
        ],
    )
    save_json(run_dir / "ranking.json", {"ranking": ranking})
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
            "tau_shrink": bool(args.tau_shrink),
            "tau_shrink_m": float(args.tau_shrink_m),
            "tau_guardrail": args.tau_guardrail,
            "tau_guardrail_delta": float(args.tau_guardrail_delta),
            "cos_affine_calib": bool(args.cos_affine_calib),
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
            "methods": list(build_methods().keys()),
            "methods_configured": configured_methods_last,
            "methods_evaluated": sorted({str(r.get("method", "")) for r in trial_summary_rows if r.get("method")}),
            "xgboost_available": bool(has_xgb),
            "cosine_feature_available": bool(X_cos is not None),
            "failures": failures,
        },
    )

    best = constrained[0]["method"] if constrained else (ranking[0]["method"] if ranking else None)
    if best:
        print(f"\nBest (respecting constraint if possible): {best}")
    print(f"[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
