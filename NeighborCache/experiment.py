#!/usr/bin/env python3
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

# ---------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # .../Multi_Dim_Threshold
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NC_ROOT = ROOT / "NeighborCache"
OUT_BASE = NC_ROOT / "outputs" / "region_local_threshold"

from np_bench.methods import (
    AndBoxHCMethod,
    AndBoxWgtMethod,
    CosineMethod,
    LDAMethod,
    LogisticRegressionMethod,
    NaiveBayesMethod,
    TinyMLPMethod,
    VectorWeightedMethod,
)

try:
    from np_bench.methods import RobustAndBoxMethod
except ImportError:
    try:
        from NeighborCache.andbox import RobustAndBoxMethod
    except ImportError:
        print("Warning: RobustAndBoxMethod not found. Skipping.")
        RobustAndBoxMethod = None

from np_bench.utils import make_run_dir, save_csv_rows, save_json

try:
    from np_bench.methods.xgboost import XGBoostLightMethod, HAS_XGB
except Exception:
    HAS_XGB = False


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def resolve_npz_path(data_path: str) -> Tuple[Path, Optional[Path]]:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    if p.suffix == ".npz":
        return p, None

    if p.suffix == ".json" and p.name.endswith("_stats.json"):
        npz_path = p.with_name(p.name.replace("_stats.json", ".npz"))
        if not npz_path.exists():
            raise FileNotFoundError(f"Stats file provided but paired NPZ is missing: {npz_path}")
        return npz_path, p

    raise ValueError("Use either the .npz file or *_stats.json file path for --data.")


def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(npz_path)) as ds:
        return {k: ds[k] for k in ds.files}


def resolve_features(ds: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Select feature matrix X.
    Priority: high-dim arrays like emb/vectors/features/embedding.
    Fallback: cosine_to_anchor (1D).
    """
    candidates = ["emb", "vectors", "features", "embedding"]
    for key in candidates:
        if key in ds:
            X = ds[key].astype(np.float32, copy=False)
            if X.ndim == 2 and X.shape[1] > 1:
                print(f"[INFO] Using high-dimensional vectors from key: '{key}' (dim={X.shape[1]})")
                return X

    if "cosine_to_anchor" in ds:
        print("[WARNING] High-dimensional vectors NOT found in NPZ.")
        print("          Using 1D 'cosine_to_anchor' as features.")
        return ds["cosine_to_anchor"].astype(np.float32, copy=False).reshape(-1, 1)

    raise ValueError(f"No suitable feature array found in NPZ keys: {sorted(ds.keys())}")


def score_quantiles(scores: np.ndarray) -> Dict[str, float]:
    ps = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if s.size == 0:
        return {f"p{p:02d}": float("nan") for p in ps}
    return {f"p{p:02d}": float(np.percentile(s, p)) for p in ps}


# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------

def build_methods() -> Dict[str, Any]:
    methods: Dict[str, Any] = {
        "Cosine": CosineMethod(),
        "Vec (Wgt)": VectorWeightedMethod(),
        "Naive Bayes": NaiveBayesMethod(),
        "Log Reg": LogisticRegressionMethod(),
        "LDA": LDAMethod(),
    }
    if HAS_XGB:
        methods["XGBoost"] = XGBoostLightMethod()
    methods["Tiny MLP"] = TinyMLPMethod()
    methods["AndBox-HC"] = AndBoxHCMethod()
    methods["AndBox-Wgt"] = AndBoxWgtMethod()
    if RobustAndBoxMethod is not None:
        methods["RobustAndBox"] = RobustAndBoxMethod()
    return methods


def needs_weights(method: Any) -> bool:
    return bool(getattr(method, "needs_weights", False))


def needs_seed(method: Any) -> bool:
    return bool(getattr(method, "needs_seed", False))


def try_fit_method(
    method: Any,
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    weights: Optional[np.ndarray],
    seed: int,
    alpha: float,
) -> None:
    """
    Best-effort fitting across differing method signatures.
    """
    if not hasattr(method, "fit"):
        return

    try_orders: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
        ((), {"weights": weights, "seed": seed, "alpha": alpha}),
        ((), {"weights": weights, "alpha": alpha}),
        ((), {"seed": seed, "alpha": alpha}),
        ((), {"alpha": alpha}),
        ((), {"weights": weights, "seed": seed}),
        ((), {"weights": weights}),
        ((), {"seed": seed}),
        ((), {}),
    ]

    last_exc: Optional[Exception] = None
    for args, kwargs in try_orders:
        try:
            method.fit(X0, X1, *args, **kwargs)
            return
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            raise exc

    raise TypeError(f"Could not fit method={type(method).__name__}: {last_exc}")


# ---------------------------------------------------------------------
# Splits per region (FLEXIBLE caps + hard mins)
# ---------------------------------------------------------------------

@dataclass
class RegionSplit:
    rid: int
    H0_train: np.ndarray
    H1_train: np.ndarray
    H0_eval: np.ndarray
    H1_eval: np.ndarray


def _take_split_flexible(
    idx: np.ndarray,
    rng: np.random.Generator,
    n_train_cap: int,
    n_eval_cap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample WITHOUT replacement, up to caps.
    Priority order: train -> eval.
    """
    idx = np.asarray(idx, dtype=np.int64).copy()
    rng.shuffle(idx)

    n = idx.size
    n_train_eff = min(n_train_cap, n) if n_train_cap > 0 else 0
    tr = idx[:n_train_eff]

    rem = idx[n_train_eff:]
    n_eval_eff = min(n_eval_cap, rem.size)
    ev = rem[:n_eval_eff]

    return tr, ev


def split_indices_per_region(
    region_id: np.ndarray,
    y: np.ndarray,
    *,
    n_train_cap: int,
    n_eval_cap: int,
    seed: int,
    # hard mins
    min_h0_eval: int,
    min_h1_eval: int,
) -> Tuple[List[RegionSplit], Dict[str, int]]:
    """
    Split each region into TRAIN + EVAL.
    Regions selection is based on EVAL mins only.
    """
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

        # Must be able to satisfy eval mins (otherwise this region is useless for scoring).
        if idx0.size < min_h0_eval or idx1.size < min_h1_eval:
            stats["skipped_region_insufficient_eval_mins"] += 1
            continue

        h0_tr, h0_ev = _take_split_flexible(idx0, rng, n_train_cap, n_eval_cap)
        h1_tr, h1_ev = _take_split_flexible(idx1, rng, n_train_cap, n_eval_cap)

        # enforce eval mins after actual sampling
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
                H0_eval=h0_ev.astype(np.int64),
                H1_eval=h1_ev.astype(np.int64),
            )
        )
        stats["used_regions"] += 1

    return splits, dict(stats)


# ---------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------

def apply_threshold(scores: np.ndarray, tau: float, tie_mode: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if tie_mode == "gt":
        return (s > tau).astype(np.int32)
    return (s >= tau).astype(np.int32)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "VectorQ-like region threshold experiment (local/global tau).\n"
            "Use --region_key anchor_qid for per-anchor behavior."
        )
    )
    ap.add_argument("--data", type=str, default="NeighborCache/data/bge_full_emb_sem_k100.npz")
    ap.add_argument(
        "--region_key",
        type=str,
        default="bucket_id",
        help="Key in NPZ used as region id (e.g., bucket_id, sem_bucket, anchor_qid).",
    )

    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tie_mode", type=str, default="ge", choices=["ge", "gt"])
    ap.add_argument(
        "--tau_mode",
        type=str,
        default="local",
        choices=["local", "global"],
        help="Threshold scope: one global tau or per-region local tau.",
    )

    # caps per region per class
    ap.add_argument("--n_train", type=int, default=20, help="Cap on train samples per region per class.")
    ap.add_argument("--n_eval", type=int, default=20, help="Cap on eval samples per region per class.")

    # hard mins (eval mins always enforced)
    ap.add_argument("--min_h0_eval", type=int, default=20, help="Hard minimum H0 eval per region.")
    ap.add_argument("--min_h1_eval", type=int, default=20, help="Hard minimum H1 eval per region.")

    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()

    npz_path, stats_path = resolve_npz_path(args.data)
    ds = load_npz(npz_path)

    required = {args.region_key, "label"}
    miss = sorted(required - set(ds.keys()))
    if miss:
        raise ValueError(f"{npz_path} missing required arrays: {miss}; have={sorted(ds.keys())}")

    region_id = ds[args.region_key].astype(np.int64, copy=False)
    y = ds["label"].astype(np.int32, copy=False)
    X = resolve_features(ds)

    run_dir = make_run_dir(base_dir=str(OUT_BASE), run_name=args.run_name)

    print("\n=== VectorQ-like Region Threshold Benchmark ===")
    print(f"region_key={args.region_key}")
    print(f"dataset_npz={npz_path}")
    print(f"rows={X.shape[0]} dim={X.shape[1]}")
    print(f"alpha={args.alpha} tie_mode={args.tie_mode} trials={args.n_trials} base_seed={args.seed}")
    print(f"tau_mode={args.tau_mode}")
    print(f"caps per region per class: train={args.n_train} eval={args.n_eval}")
    print(f"mins: min_h0_eval={args.min_h0_eval} min_h1_eval={args.min_h1_eval}")
    print(f"run_dir={run_dir}")

    method_names = list(build_methods().keys())

    trial_summary_rows: List[Dict[str, Any]] = []
    per_region_rows: List[Dict[str, Any]] = []
    failures: Dict[str, List[str]] = defaultdict(list)
    worst_regions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for trial in range(args.n_trials):
        seed = args.seed + trial

        splits, split_stats = split_indices_per_region(
            region_id=region_id,
            y=y,
            n_train_cap=args.n_train,
            n_eval_cap=args.n_eval,
            seed=seed,
            min_h0_eval=args.min_h0_eval,
            min_h1_eval=args.min_h1_eval,
        )
        if len(splits) == 0:
            raise RuntimeError(
                "No regions satisfied eval mins. Lower eval mins, increase caps, or change regioning."
            )

        # -------------------------
        # Build pooled train set
        # -------------------------
        h0_train_idx_list = [s.H0_train for s in splits if s.H0_train.size > 0]
        h1_train_idx_list = [s.H1_train for s in splits if s.H1_train.size > 0]

        if len(h0_train_idx_list) == 0 or len(h1_train_idx_list) == 0:
            raise RuntimeError("No data to fit methods (train empty). Increase n_train or check dataset labels.")

        h0_train_idx = np.concatenate(h0_train_idx_list)
        h1_train_idx = np.concatenate(h1_train_idx_list)
        H0_train = X[h0_train_idx]
        H1_train = X[h1_train_idx]

        # Weights heuristic (for weighted methods)
        v0 = np.var(H0_train, axis=0)
        v1 = np.var(H1_train, axis=0)
        weights = (v1 / (v0 + 1e-12)).astype(np.float32, copy=False)

        print(f"\n[trial={trial} seed={seed}] used_regions={len(splits)} split_stats={split_stats}")

        # Fit methods globally
        methods = build_methods()
        for name, method in methods.items():
            try:
                w = weights if needs_weights(method) else None
                s_for_method = seed if needs_seed(method) else seed
                try_fit_method(method, H0_train, H1_train, weights=w, seed=s_for_method, alpha=args.alpha)
            except Exception as exc:
                failures[name].append(f"trial={trial}: fit failed: {exc}")

        # Evaluate each method
        for name in method_names:
            if name not in methods:
                continue
            method = methods[name]

            micro_tp = micro_fp = micro_tn = micro_fn = 0
            macro_tprs: List[float] = []
            macro_fprs: List[float] = []

            per_region_tmp: List[Tuple[int, float, float, float]] = []  # (rid, tpr, fpr, tau)
            tau_values: List[float] = []

            t_method_start = time.perf_counter()
            ok_regions = 0

            tau_global: Optional[float] = None
            if args.tau_mode == "global":
                # Compute one global tau from pooled H0_train scores.
                try:
                    sc0_train = np.asarray(method.score(H0_train), dtype=np.float32).reshape(-1)
                    tau_global = float(np.quantile(sc0_train, 1.0 - args.alpha))
                except Exception as exc:
                    failures[name].append(f"trial={trial}: global threshold computation failed: {exc}")
                    continue

            for s in splits:
                rid = s.rid
                H0_tr_r = X[s.H0_train]
                H1_tr_r = X[s.H1_train]
                H0_ev = X[s.H0_eval]
                H1_ev = X[s.H1_eval]

                try:
                    if args.tau_mode == "local":
                        # Per-region tau from this region's H0 train split.
                        sc0_tr_r = np.asarray(method.score(H0_tr_r), dtype=np.float32).reshape(-1)
                        if sc0_tr_r.size == 0:
                            failures[name].append(f"trial={trial} region={rid}: empty H0_train for local tau.")
                            continue
                        tau_r = float(np.quantile(sc0_tr_r, 1.0 - args.alpha))
                    else:
                        tau_r = float(tau_global)

                    sc0_ev = np.asarray(method.score(H0_ev), dtype=np.float32).reshape(-1)
                    sc1_ev = np.asarray(method.score(H1_ev), dtype=np.float32).reshape(-1)

                    p0 = apply_threshold(sc0_ev, tau_r, args.tie_mode)
                    p1 = apply_threshold(sc1_ev, tau_r, args.tie_mode)

                    fpr_r = float(np.mean(p0 == 1))
                    tpr_r = float(np.mean(p1 == 1))

                    micro_fp += int(np.sum(p0 == 1))
                    micro_tn += int(np.sum(p0 == 0))
                    micro_tp += int(np.sum(p1 == 1))
                    micro_fn += int(np.sum(p1 == 0))

                    macro_fprs.append(fpr_r)
                    macro_tprs.append(tpr_r)

                    tau_values.append(float(tau_r))
                    per_region_tmp.append((rid, tpr_r, fpr_r, float(tau_r)))
                    ok_regions += 1

                    per_region_rows.append(
                        {
                            "trial": trial,
                            "seed": seed,
                            "method": name,
                            "region_key": args.region_key,
                            "region_id": int(rid),
                            "tau_mode": args.tau_mode,
                            "tau": float(tau_r),
                            "n0_train": int(H0_tr_r.shape[0]),
                            "n1_train": int(H1_tr_r.shape[0]),
                            "n0_eval": int(H0_ev.shape[0]),
                            "n1_eval": int(H1_ev.shape[0]),
                            "tpr_r": float(tpr_r),
                            "fpr_r": float(fpr_r),
                            "sep_med_eval": float(np.median(sc1_ev) - np.median(sc0_ev)),
                            "H0_eval_q": json.dumps(score_quantiles(sc0_ev)),
                            "H1_eval_q": json.dumps(score_quantiles(sc1_ev)),
                        }
                    )

                except Exception as exc:
                    failures[name].append(f"trial={trial} region={rid}: score failed: {exc}")
                    continue

            t_method_ms = (time.perf_counter() - t_method_start) * 1000.0
            if ok_regions == 0:
                failures[name].append(f"trial={trial}: no regions evaluated.")
                continue

            micro_tpr = float(micro_tp / max(1, (micro_tp + micro_fn)))
            micro_fpr = float(micro_fp / max(1, (micro_fp + micro_tn)))
            macro_tpr = float(np.mean(macro_tprs)) if macro_tprs else float("nan")
            macro_fpr = float(np.mean(macro_fprs)) if macro_fprs else float("nan")
            tau_mean = float(np.mean(tau_values)) if tau_values else float("nan")
            tau_summary = float(tau_global) if args.tau_mode == "global" and tau_global is not None else float("nan")

            trial_summary_rows.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "method": name,
                    "region_key": args.region_key,
                    "tau_mode": args.tau_mode,
                    "tau": tau_summary,
                    "tau_mean": tau_mean,
                    "micro_tpr": micro_tpr,
                    "micro_fpr": micro_fpr,
                    "macro_tpr": macro_tpr,
                    "macro_fpr": macro_fpr,
                    "ok_regions": int(ok_regions),
                    "time_ms": float(t_method_ms),
                }
            )

            # worst regions by TPR
            per_region_tmp.sort(key=lambda t: t[1])
            for rid, tpr_r, fpr_r, tau_r in per_region_tmp[:25]:
                worst_regions[name].append(
                    {
                        "trial": trial,
                        "region_id": int(rid),
                        "tpr_r": float(tpr_r),
                        "fpr_r": float(fpr_r),
                        "tau": float(tau_r),
                    }
                )

            tau_txt = f"{tau_summary:.4f}" if args.tau_mode == "global" else f"local(mean={tau_mean:.4f})"
            print(
                f"  {name:<14} micro(TPR={micro_tpr:.4f},FPR={micro_fpr:.4f}) "
                f"macro(TPR={macro_tpr:.4f},FPR={macro_fpr:.4f}) regions={ok_regions} "
                f"tau={tau_txt} time={t_method_ms:.1f}ms"
            )

    # Aggregate ranking
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

    print("\n=== Ranking by mean micro TPR ===")
    for i, r in enumerate(ranking, start=1):
        print(
            f"{i:2d}. {r['method']:<14} "
            f"micro_TPR={r['mean_micro_tpr']:.4f}±{r['std_micro_tpr']:.4f} "
            f"micro_FPR={r['mean_micro_fpr']:.4f}±{r['std_micro_fpr']:.4f} | "
            f"macro_TPR={r['mean_macro_tpr']:.4f}"
        )

    # Save artifacts
    save_csv_rows(
        run_dir / "trial_summary.csv",
        trial_summary_rows,
        fieldnames=[
            "trial", "seed", "method", "region_key", "tau_mode", "tau", "tau_mean",
            "micro_tpr", "micro_fpr", "macro_tpr", "macro_fpr",
            "ok_regions", "time_ms",
        ],
    )

    save_csv_rows(
        run_dir / "per_region_metrics.csv",
        per_region_rows,
        fieldnames=[
            "trial", "seed", "method", "region_key", "region_id",
            "tau_mode", "tau", "n0_train", "n1_train", "n0_eval", "n1_eval",
            "tpr_r", "fpr_r", "sep_med_eval",
            "H0_eval_q", "H1_eval_q",
        ],
    )

    save_json(run_dir / "ranking.json", {"ranking": ranking})
    save_json(run_dir / "worst_regions_by_tpr.json", worst_regions)
    save_json(
        run_dir / "notes.json",
        {
            "experiment": "vectorq_like_region_threshold",
            "tau_mode": args.tau_mode,
            "region_key": args.region_key,
            "data_arg": args.data,
            "dataset_npz": str(npz_path),
            "stats_json": str(stats_path) if stats_path else None,
            "alpha": float(args.alpha),
            "tie_mode": args.tie_mode,
            "n_trials": int(args.n_trials),
            "seed": int(args.seed),
            "caps_per_region_per_class": {
                "n_train": int(args.n_train),
                "n_eval": int(args.n_eval),
            },
            "mins": {
                "min_h0_eval": int(args.min_h0_eval),
                "min_h1_eval": int(args.min_h1_eval),
            },
            "methods": list(build_methods().keys()),
            "xgboost_available": bool(HAS_XGB),
            "failures": failures,
        },
    )

    best = ranking[0]["method"] if ranking else None
    if best:
        print(f"\nBest by mean micro TPR: {best}")
    print(f"[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
