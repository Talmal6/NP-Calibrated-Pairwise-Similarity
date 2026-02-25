#!/usr/bin/env python3
"""
Online NP-calibrated region-threshold experiment.

Reuses the same dataset / splits / evaluation logic as experiment.py,
but replaces the static fit-once pipeline with OnlineBaseMethod streaming.

Flow
----
1.  Load dataset, split per region into train / calib / eval  (same as static).
2.  Seed the OnlineBaseMethod with a small initial batch from the first
    ``--n_init_regions`` regions.
3.  Stream the remaining regions' train data as batches.
    After each batch the model refits (or hill-climbs) and recalibrates tau.
4.  Evaluate every method (online + static baselines) per-region exactly as
    in experiment.py.

Usage
-----
    python NeighborCache/online_experiment.py \
        --data NeighborCache/data/bge_full_emb_sem_k100.npz \
        --alpha 0.05 --update_mode refit --n_init_regions 5
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# Paths / imports
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NC_ROOT = ROOT / "NeighborCache"
OUT_BASE = NC_ROOT / "outputs" / "online_experiment"

# Reuse helpers from the static experiment
from NeighborCache.experiment import (
    resolve_npz_path,
    load_npz,
    resolve_features,
    score_quantiles,
    build_methods,
    needs_weights,
    needs_seed,
    try_fit_method,
    split_indices_per_region,
    apply_threshold,
    RegionSplit,
)

from np_bench.methods.base import OnlineBaseMethod
from np_bench.utils import make_run_dir, save_csv_rows, save_json


# ------------------------------------------------------------------
# Online method builder
# ------------------------------------------------------------------

def build_online_method(
    *,
    update_mode: str,
    update_every: int,
    mem_cap: int,
    hill_lr: float,
    seed: int,
) -> OnlineBaseMethod:
    return OnlineBaseMethod(
        mem_cap_H0=mem_cap,
        mem_cap_H1=mem_cap,
        update_mode=update_mode,
        update_every=update_every,
        hill_lr=hill_lr,
        seed=seed,
    )


# ------------------------------------------------------------------
# Streaming helpers
# ------------------------------------------------------------------

def _collect_init_data(
    splits: List[RegionSplit],
    X: np.ndarray,
    n_init_regions: int,
) -> Tuple[np.ndarray, np.ndarray, List[RegionSplit]]:
    """
    Use the first ``n_init_regions`` regions to build the initial seed
    (train memory).  Returns remaining splits.
    """
    init_splits = splits[:n_init_regions]
    rest_splits = splits[n_init_regions:]

    h0_train_parts: List[np.ndarray] = []
    h1_train_parts: List[np.ndarray] = []

    for s in init_splits:
        if s.H0_train.size > 0:
            h0_train_parts.append(X[s.H0_train])
        if s.H1_train.size > 0:
            h1_train_parts.append(X[s.H1_train])

    H0_init = np.vstack(h0_train_parts) if h0_train_parts else np.empty((0, X.shape[1]))
    H1_init = np.vstack(h1_train_parts) if h1_train_parts else np.empty((0, X.shape[1]))

    return H0_init, H1_init, rest_splits


def stream_batches(
    splits: List[RegionSplit],
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Flatten all train indices from remaining splits into a single stream
    and chop into fixed-size batches of (X_batch, y_batch).
    """
    all_idx: List[np.ndarray] = []
    for s in splits:
        if s.H0_train.size > 0:
            all_idx.append(s.H0_train)
        if s.H1_train.size > 0:
            all_idx.append(s.H1_train)

    if not all_idx:
        return []

    idx = np.concatenate(all_idx)
    X_all = X[idx]
    y_all = y[idx]

    batches: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(idx), batch_size):
        end = min(start + batch_size, len(idx))
        batches.append((X_all[start:end], y_all[start:end]))
    return batches


# ------------------------------------------------------------------
# Per-method evaluation (shared between online + static)
# ------------------------------------------------------------------

@dataclass
class EvalResult:
    micro_tpr: float
    micro_fpr: float
    macro_tpr: float
    macro_fpr: float
    ok_regions: int
    tau: float
    time_ms: float
    per_region: List[Dict[str, Any]]


def evaluate_method(
    method: Any,
    method_name: str,
    splits: List[RegionSplit],
    X: np.ndarray,
    H0_train: np.ndarray,
    *,
    alpha: float,
    tie_mode: str,
    region_key: str,
    trial: int,
    seed: int,
) -> Optional[EvalResult]:
    """
    Evaluate a fitted method across all regions.
    Uses a single global tau computed from H0_train.
    """
    t0 = time.perf_counter()

    # Compute single global tau from H0_train
    try:
        sc0_train = np.asarray(method.score(H0_train), dtype=np.float32).ravel()
        tau = float(np.quantile(sc0_train, 1.0 - alpha))
    except Exception:
        return None

    micro_tp = micro_fp = micro_tn = micro_fn = 0
    macro_tprs: List[float] = []
    macro_fprs: List[float] = []
    per_region_rows: List[Dict[str, Any]] = []
    ok_regions = 0

    for s in splits:
        rid = s.rid
        H0_ev = X[s.H0_eval]
        H1_ev = X[s.H1_eval]

        try:
            sc0_ev = np.asarray(method.score(H0_ev), dtype=np.float32).ravel()
            sc1_ev = np.asarray(method.score(H1_ev), dtype=np.float32).ravel()

            p0 = apply_threshold(sc0_ev, tau, tie_mode)
            p1 = apply_threshold(sc1_ev, tau, tie_mode)
            fpr_r = float(np.mean(p0 == 1))
            tpr_r = float(np.mean(p1 == 1))

            micro_fp += int(np.sum(p0 == 1))
            micro_tn += int(np.sum(p0 == 0))
            micro_tp += int(np.sum(p1 == 1))
            micro_fn += int(np.sum(p1 == 0))
            macro_fprs.append(fpr_r)
            macro_tprs.append(tpr_r)
            ok_regions += 1

            per_region_rows.append({
                "trial": trial,
                "seed": seed,
                "method": method_name,
                "region_key": region_key,
                "region_id": int(rid),
                "tau": float(tau),
                "n0_eval": int(H0_ev.shape[0]),
                "n1_eval": int(H1_ev.shape[0]),
                "tpr_r": float(tpr_r),
                "fpr_r": float(fpr_r),
                "H0_eval_q": json.dumps(score_quantiles(sc0_ev)),
                "H1_eval_q": json.dumps(score_quantiles(sc1_ev)),
            })
        except Exception:
            continue

    dt_ms = (time.perf_counter() - t0) * 1000.0

    if ok_regions == 0:
        return None

    micro_tpr = float(micro_tp / max(1, micro_tp + micro_fn))
    micro_fpr = float(micro_fp / max(1, micro_fp + micro_tn))
    macro_tpr = float(np.mean(macro_tprs)) if macro_tprs else float("nan")
    macro_fpr = float(np.mean(macro_fprs)) if macro_fprs else float("nan")

    return EvalResult(
        micro_tpr=micro_tpr,
        micro_fpr=micro_fpr,
        macro_tpr=macro_tpr,
        macro_fpr=macro_fpr,
        ok_regions=ok_regions,
        tau=tau,
        time_ms=dt_ms,
        per_region=per_region_rows,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Online region threshold experiment."
    )
    ap.add_argument("--data", type=str,
                    default="NeighborCache/data/bge_full_emb_sem_k100.npz")
    ap.add_argument("--region_key", type=str, default="bucket_id")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tie_mode", type=str, default="ge", choices=["ge", "gt"])

    # caps per region per class
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=20)

    # hard mins
    ap.add_argument("--min_h0_eval", type=int, default=20)
    ap.add_argument("--min_h1_eval", type=int, default=20)

    # --- online-specific ---
    ap.add_argument("--update_mode", type=str, default="refit",
                    choices=["refit", "hill_climb"],
                    help="Online update strategy.")
    ap.add_argument("--update_every", type=int, default=1,
                    help="Run model update every N batches.")
    ap.add_argument("--mem_cap", type=int, default=2000,
                    help="Max samples per class in training memory.")
    ap.add_argument("--hill_lr", type=float, default=0.1,
                    help="Learning rate for hill_climb mode.")
    ap.add_argument("--n_init_regions", type=int, default=5,
                    help="Number of regions used as initial seed.")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Online batch size for streaming.")
    ap.add_argument("--include_static", action="store_true", default=True,
                    help="Also run static baselines for comparison.")
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    npz_path, stats_path = resolve_npz_path(args.data)
    ds = load_npz(npz_path)

    required = {args.region_key, "label"}
    miss = sorted(required - set(ds.keys()))
    if miss:
        raise ValueError(f"{npz_path} missing: {miss}; have={sorted(ds.keys())}")

    region_id = ds[args.region_key].astype(np.int64, copy=False)
    y = ds["label"].astype(np.int32, copy=False)
    X = resolve_features(ds)

    run_dir = make_run_dir(base_dir=str(OUT_BASE), run_name=args.run_name)

    print("\n=== Online Region Threshold Experiment ===")
    print(f"region_key={args.region_key}")
    print(f"dataset={npz_path}  rows={X.shape[0]}  dim={X.shape[1]}")
    print(f"alpha={args.alpha}  tie_mode={args.tie_mode}  trials={args.n_trials}")
    print(f"online: mode={args.update_mode}  update_every={args.update_every}  "
          f"mem_cap={args.mem_cap}  init_regions={args.n_init_regions}  "
          f"batch={args.batch_size}")
    print(f"run_dir={run_dir}")

    trial_rows: List[Dict[str, Any]] = []
    per_region_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []

    for trial in range(args.n_trials):
        seed = args.seed + trial

        # Split
        splits, split_stats = split_indices_per_region(
            region_id=region_id, y=y,
            n_train_cap=args.n_train,
            n_eval_cap=args.n_eval,
            seed=seed,
            min_h0_eval=args.min_h0_eval,
            min_h1_eval=args.min_h1_eval,
        )
        if not splits:
            raise RuntimeError("No regions satisfied eval mins.")

        print(f"\n[trial={trial} seed={seed}] regions={len(splits)}  "
              f"split_stats={split_stats}")

        # ==============================================================
        # ONLINE METHOD
        # ==============================================================
        H0_init, H1_init, rest_splits = _collect_init_data(
            splits, X, args.n_init_regions,
        )

        if H0_init.shape[0] == 0 or H1_init.shape[0] == 0:
            print("  [WARN] Not enough init data for online method; skipping trial.")
            continue

        online = build_online_method(
            update_mode=args.update_mode,
            update_every=args.update_every,
            mem_cap=args.mem_cap,
            hill_lr=args.hill_lr,
            seed=seed,
        )
        online.initialize(H0_init, H1_init)
        online.recalibrate_threshold(args.alpha)

        # Stream remaining train data
        batches = stream_batches(rest_splits, X, y, args.batch_size)
        t_stream_start = time.perf_counter()

        for bi, (Xb, yb) in enumerate(batches):
            online.update(Xb, yb)
            online.recalibrate_threshold(args.alpha)

            # Periodic snapshot (every 10 batches) for diagnostics
            if (bi + 1) % 10 == 0 or bi == len(batches) - 1:
                snapshot_rows.append({
                    "trial": trial,
                    "batch": bi + 1,
                    "total_batches": len(batches),
                    "mem_H0": online.mem_H0.shape[0] if online.mem_H0 is not None else 0,
                    "mem_H1": online.mem_H1.shape[0] if online.mem_H1 is not None else 0,
                    "tau_np": float(online.tau_np),
                })

        stream_ms = (time.perf_counter() - t_stream_start) * 1000.0
        print(f"  Online streaming: {len(batches)} batches in {stream_ms:.1f}ms  "
              f"mem_H0={online.mem_H0.shape[0] if online.mem_H0 is not None else 0}  "
              f"mem_H1={online.mem_H1.shape[0] if online.mem_H1 is not None else 0}  "
              f"tau={online.tau_np:.4f}")

        # Build pooled H0_train for evaluate_method
        h0_tr_idx = [s.H0_train for s in splits if s.H0_train.size > 0]
        h1_tr_idx = [s.H1_train for s in splits if s.H1_train.size > 0]
        if not h0_tr_idx or not h1_tr_idx:
            print("  [WARN] No pooled train data; skipping evaluation.")
            continue
        H0_train = X[np.concatenate(h0_tr_idx)]
        H1_train = X[np.concatenate(h1_tr_idx)]

        # For online method, use mem_H0 as the H0_train for threshold
        online_H0_train = online.mem_H0 if online.mem_H0 is not None else H0_train

        # Evaluate online method
        ev_online = evaluate_method(
            online, f"Online({args.update_mode})", splits, X,
            online_H0_train,
            alpha=args.alpha,
            tie_mode=args.tie_mode,
            region_key=args.region_key,
            trial=trial, seed=seed,
        )
        if ev_online is not None:
            trial_rows.append({
                "trial": trial, "seed": seed,
                "method": f"Online({args.update_mode})",
                "region_key": args.region_key,
                "tau": ev_online.tau,
                "micro_tpr": ev_online.micro_tpr,
                "micro_fpr": ev_online.micro_fpr,
                "macro_tpr": ev_online.macro_tpr,
                "macro_fpr": ev_online.macro_fpr,
                "ok_regions": ev_online.ok_regions,
                "time_ms": ev_online.time_ms + stream_ms,
            })
            per_region_rows.extend(ev_online.per_region)
            print(f"  {'Online(' + args.update_mode + ')':<20} "
                  f"micro(TPR={ev_online.micro_tpr:.4f}, FPR={ev_online.micro_fpr:.4f}) "
                  f"macro(TPR={ev_online.macro_tpr:.4f}, FPR={ev_online.macro_fpr:.4f}) "
                  f"tau={ev_online.tau:.4f} regions={ev_online.ok_regions}")

        # ==============================================================
        # STATIC BASELINES (optional, for comparison)
        # ==============================================================
        if args.include_static:
            v0 = np.var(H0_train, axis=0)
            v1 = np.var(H1_train, axis=0)
            weights = (v1 / (v0 + 1e-12)).astype(np.float32)

            methods = build_methods()
            for name, method in methods.items():
                try:
                    w = weights if needs_weights(method) else None
                    try_fit_method(method, H0_train, H1_train,
                                  weights=w, seed=seed, alpha=args.alpha)
                except Exception:
                    continue

                ev = evaluate_method(
                    method, name, splits, X, H0_train,
                    alpha=args.alpha,
                    tie_mode=args.tie_mode,
                    region_key=args.region_key,
                    trial=trial, seed=seed,
                )
                if ev is None:
                    continue

                trial_rows.append({
                    "trial": trial, "seed": seed, "method": name,
                    "region_key": args.region_key,
                    "tau": ev.tau,
                    "micro_tpr": ev.micro_tpr, "micro_fpr": ev.micro_fpr,
                    "macro_tpr": ev.macro_tpr, "macro_fpr": ev.macro_fpr,
                    "ok_regions": ev.ok_regions,
                    "time_ms": ev.time_ms,
                })
                per_region_rows.extend(ev.per_region)

                print(f"  {name:<20} "
                      f"micro(TPR={ev.micro_tpr:.4f}, FPR={ev.micro_fpr:.4f}) "
                      f"macro(TPR={ev.macro_tpr:.4f}, FPR={ev.macro_fpr:.4f}) "
                      f"tau={ev.tau:.4f} regions={ev.ok_regions}")

    # ------------------------------------------------------------------
    # Aggregate ranking
    # ------------------------------------------------------------------
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in trial_rows:
        agg[r["method"]]["micro_tpr"].append(float(r["micro_tpr"]))
        agg[r["method"]]["micro_fpr"].append(float(r["micro_fpr"]))
        agg[r["method"]]["macro_tpr"].append(float(r["macro_tpr"]))
        agg[r["method"]]["macro_fpr"].append(float(r["macro_fpr"]))

    ranking = []
    for m, d in agg.items():
        ranking.append({
            "method": m,
            "mean_micro_tpr": float(np.mean(d["micro_tpr"])),
            "std_micro_tpr": float(np.std(d["micro_tpr"])),
            "mean_micro_fpr": float(np.mean(d["micro_fpr"])),
            "std_micro_fpr": float(np.std(d["micro_fpr"])),
            "mean_macro_tpr": float(np.mean(d["macro_tpr"])),
            "std_macro_tpr": float(np.std(d["macro_tpr"])),
            "mean_macro_fpr": float(np.mean(d["macro_fpr"])),
            "std_macro_fpr": float(np.std(d["macro_fpr"])),
        })
    ranking.sort(key=lambda r: r["mean_micro_tpr"], reverse=True)

    print("\n=== Ranking by mean micro TPR ===")
    for i, r in enumerate(ranking, start=1):
        is_online = "Online" in r["method"]
        marker = " ***" if is_online else ""
        print(f"{i:2d}. {r['method']:<20} "
              f"micro_TPR={r['mean_micro_tpr']:.4f}±{r['std_micro_tpr']:.4f}  "
              f"micro_FPR={r['mean_micro_fpr']:.4f}±{r['std_micro_fpr']:.4f}  | "
              f"macro_TPR={r['mean_macro_tpr']:.4f}{marker}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    fieldnames_trial = [
        "trial", "seed", "method", "region_key", "tau",
        "micro_tpr", "micro_fpr", "macro_tpr", "macro_fpr",
        "ok_regions", "time_ms",
    ]
    fieldnames_region = [
        "trial", "seed", "method", "region_key", "region_id",
        "tau", "n0_eval", "n1_eval",
        "tpr_r", "fpr_r",
        "H0_eval_q", "H1_eval_q",
    ]

    save_csv_rows(run_dir / "trial_summary.csv", trial_rows,
                  fieldnames=fieldnames_trial)
    save_csv_rows(run_dir / "per_region_metrics.csv", per_region_rows,
                  fieldnames=fieldnames_region)
    if snapshot_rows:
        save_csv_rows(run_dir / "online_snapshots.csv", snapshot_rows,
                      fieldnames=["trial", "batch", "total_batches",
                                  "mem_H0", "mem_H1", "tau_np"])
    save_json(run_dir / "ranking.json", {"ranking": ranking})
    save_json(run_dir / "notes.json", {
        "experiment": "online_region_threshold",
        "region_key": args.region_key,
        "data_arg": args.data,
        "dataset_npz": str(npz_path),
        "alpha": float(args.alpha),
        "tie_mode": args.tie_mode,
        "n_trials": int(args.n_trials),
        "seed": int(args.seed),
        "online_config": {
            "update_mode": args.update_mode,
            "update_every": args.update_every,
            "mem_cap": args.mem_cap,
            "hill_lr": args.hill_lr,
            "n_init_regions": args.n_init_regions,
            "batch_size": args.batch_size,
        },
        "include_static": args.include_static,
        "methods": list(build_methods().keys()) + [f"Online({args.update_mode})"],
    })

    best = ranking[0]["method"] if ranking else None
    if best:
        print(f"\nBest by mean micro TPR: {best}")
    print(f"[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
