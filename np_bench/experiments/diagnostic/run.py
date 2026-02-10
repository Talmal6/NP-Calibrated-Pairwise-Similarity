#!/usr/bin/env python3
"""
Diagnostic script for analyzing NP-calibrated pairwise similarity methods.

Runs a single method on a fixed train/calib/eval split, extracts score
distributions for every group, records the NP threshold, and saves rich
diagnostics (JSON + CSV) for downstream analysis.

Usage:
    python -m np_bench.experiments.diagnostic.run \
        --pkl np_bench/data/quora_question_pairs_with_embeddings.pkl \
        --d 1024 --n 2000 --alpha 0.05 --method "Cosine"
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from np_bench.data import load_quora_hadamard
from np_bench.methods import (
    CosineMethod,
    VectorWeightedMethod,
    NaiveBayesMethod,
    LogisticRegressionMethod,
    LDAMethod,
    TinyMLPMethod,
    AndBoxHCMethod,
    AndBoxWgtMethod,
)
from np_bench.methods.base import _np_eval_from_calib
from np_bench.utils import get_fisher_scores, save_json, save_csv_rows
from np_bench.utils.split import split_by_class_triplet

try:
    from np_bench.methods.xgboost import XGBoostLightMethod, HAS_XGB
except Exception:
    HAS_XGB = False


# ── helpers ──────────────────────────────────────────────────────────────────

METHOD_REGISTRY: Dict[str, type] = {
    "Cosine": CosineMethod,
    "Vec (Wgt)": VectorWeightedMethod,
    "Naive Bayes": NaiveBayesMethod,
    "Log Reg": LogisticRegressionMethod,
    "LDA": LDAMethod,
    "Tiny MLP": TinyMLPMethod,
    "AndBox-HC": AndBoxHCMethod,
    "AndBox-Wgt": AndBoxWgtMethod,
}

WEIGHT_METHODS = {"Vec (Wgt)", "AndBox-HC", "AndBox-Wgt"}


def _register_xgb():
    if HAS_XGB:
        METHOD_REGISTRY["XGBoost"] = XGBoostLightMethod


def _score_stats(scores: np.ndarray) -> Dict[str, float]:
    """Summary statistics for a 1-D score array."""
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "n": int(scores.shape[0]),
    }


def _safe_filename(method_name: str) -> str:
    """Sanitise a method display name into a filesystem-safe string."""
    return method_name.replace(" ", "_").replace("(", "").replace(")", "")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    _register_xgb()

    ap = argparse.ArgumentParser(
        description="NP diagnostic: deep analysis of a single method"
    )
    ap.add_argument("--pkl", required=True, help="Path to pickle dataset")
    ap.add_argument("--split_key", type=str, default="train")
    ap.add_argument("--d", type=int, default=1024,
                    help="Number of features to select via Fisher scores")
    ap.add_argument("--n", type=int, default=2000,
                    help="Total examples per class (split via fractions)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="NP FPR constraint")
    ap.add_argument("--method", type=str, required=True,
                    help=f"Method name. Choices: {list(METHOD_REGISTRY.keys())}")

    ap.add_argument("--train_frac", type=float, default=0.2)
    ap.add_argument("--calib_frac", type=float, default=0.3)
    ap.add_argument("--eval_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tie_mode", type=str, default="ge",
                    choices=["ge", "gt"])
    args = ap.parse_args()

    # ── validate ─────────────────────────────────────────────────────────
    frac_sum = args.train_frac + args.calib_frac + args.eval_frac
    if not np.isclose(frac_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"train_frac + calib_frac + eval_frac must equal 1.0, got {frac_sum:.6f}"
        )

    if args.method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{args.method}'. "
            f"Available: {list(METHOD_REGISTRY.keys())}"
        )

    # ── resolve split sizes ──────────────────────────────────────────────
    n_train = int(np.floor(args.n * args.train_frac))
    n_calib = int(np.floor(args.n * args.calib_frac))
    n_eval  = args.n - n_train - n_calib
    if n_eval <= 0:
        raise ValueError(f"n={args.n} too small for the given fractions.")

    # ── output directory ─────────────────────────────────────────────────
    safe_name = _safe_filename(args.method)
    out_dir = Path("outputs") / "diagnostic_run" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  NP Diagnostic – method={args.method}")
    print(f"  d={args.d}, n={args.n}  "
          f"(train={n_train}, calib={n_calib}, eval={n_eval})")
    print(f"  alpha={args.alpha}, seed={args.seed}, tie_mode={args.tie_mode}")
    print(f"  output → {out_dir}")
    print(f"{'=' * 60}\n")

    # ── load data ────────────────────────────────────────────────────────
    X_full, y_full = load_quora_hadamard(args.pkl, split_key=args.split_key)

    # ── Fisher feature selection ─────────────────────────────────────────
    fisher = get_fisher_scores(X_full, y_full)
    sorted_idx = np.argsort(fisher)
    if args.d > X_full.shape[1]:
        raise ValueError(f"d={args.d} > feature_dim={X_full.shape[1]}")
    top_k = sorted_idx[-args.d:]
    w_d = fisher[top_k]

    # ── split ────────────────────────────────────────────────────────────
    sp = split_by_class_triplet(
        X_full, y_full,
        n_train=n_train, n_calib=n_calib, n_eval=n_eval,
        seed=args.seed,
    )

    H0_train = sp.H0_train[:, top_k]
    H1_train = sp.H1_train[:, top_k]
    H0_calib = sp.H0_calib[:, top_k]
    H0_eval  = sp.H0_eval[:, top_k]
    H1_eval  = sp.H1_eval[:, top_k]

    # ── instantiate & fit ────────────────────────────────────────────────
    method = METHOD_REGISTRY[args.method]()
    weights = w_d if args.method in WEIGHT_METHODS else None

    # AndBox methods need alpha at fit time (used internally for box fitting)
    ANDBOX_METHODS = {"AndBox-HC", "AndBox-Wgt"}
    if args.method in ANDBOX_METHODS:
        method.fit(H0_train, H1_train, weights=weights, seed=args.seed,
                   alpha=args.alpha)
    else:
        method.fit(H0_train, H1_train, weights=weights, seed=args.seed)

    # ── score every group ────────────────────────────────────────────────
    scores: Dict[str, np.ndarray] = OrderedDict()
    scores["H0_train"] = method.score(H0_train)
    scores["H1_train"] = method.score(H1_train)
    scores["H0_calib"] = method.score(H0_calib)
    scores["H0_eval"]  = method.score(H0_eval)
    scores["H1_eval"]  = method.score(H1_eval)

    # ── NP calibration: extract threshold ────────────────────────────────
    tpr_eval, fpr_eval, threshold = _np_eval_from_calib(
        scores["H0_calib"], scores["H0_eval"], scores["H1_eval"],
        args.alpha, tie_mode=args.tie_mode,
    )

    # ── train metrics using the same threshold ───────────────────────────
    if args.tie_mode == "gt":
        tpr_train = float(np.mean(scores["H1_train"] > threshold))
        fpr_train = float(np.mean(scores["H0_train"] > threshold))
    else:
        tpr_train = float(np.mean(scores["H1_train"] >= threshold))
        fpr_train = float(np.mean(scores["H0_train"] >= threshold))

    # ── per-group statistics ─────────────────────────────────────────────
    group_stats: Dict[str, Dict[str, Any]] = {}
    for gname, sarr in scores.items():
        stats = _score_stats(sarr)
        stats["dist_to_threshold"] = float(threshold - stats["mean"])
        group_stats[gname] = stats

    # ── assemble full diagnostics dict ───────────────────────────────────
    diagnostics: Dict[str, Any] = {
        "method": args.method,
        "d": args.d,
        "n": args.n,
        "n_train": n_train,
        "n_calib": n_calib,
        "n_eval": n_eval,
        "alpha": args.alpha,
        "seed": args.seed,
        "tie_mode": args.tie_mode,
        "threshold": threshold,
        "eval_tpr": tpr_eval,
        "eval_fpr": fpr_eval,
        "train_tpr": tpr_train,
        "train_fpr": fpr_train,
        "group_stats": group_stats,
    }

    # ── console summary ──────────────────────────────────────────────────
    print(f"Threshold = {threshold:.6f}")
    print(f"  Train  TPR={tpr_train:.4f}  FPR={fpr_train:.4f}")
    print(f"  Eval   TPR={tpr_eval:.4f}   FPR={fpr_eval:.4f}")
    print()
    for gname, gs in group_stats.items():
        print(f"  {gname:12s}  mean={gs['mean']:+.5f}  std={gs['std']:.5f}  "
              f"[{gs['min']:+.5f}, {gs['max']:+.5f}]  "
              f"dist_to_thresh={gs['dist_to_threshold']:+.5f}")
    print()

    # ── save JSON diagnostics ────────────────────────────────────────────
    save_json(out_dir / "diagnostics.json", diagnostics)

    # ── save raw scores as .npy for later plotting ───────────────────────
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for gname, sarr in scores.items():
        np.save(scores_dir / f"{gname}.npy", sarr)
        print(f"[NPY] Saved: {scores_dir / gname}.npy")

    # ── save group stats as CSV ──────────────────────────────────────────
    csv_fields = ["group", "mean", "std", "min", "max", "median", "n",
                  "dist_to_threshold"]
    csv_rows: List[Dict[str, Any]] = []
    for gname, gs in group_stats.items():
        row = {"group": gname, **gs}
        csv_rows.append(row)
    save_csv_rows(out_dir / "group_stats.csv", csv_rows, csv_fields)

    # ── save summary metrics CSV ─────────────────────────────────────────
    summary_fields = ["method", "d", "n", "alpha", "seed", "threshold",
                      "train_tpr", "train_fpr", "eval_tpr", "eval_fpr"]
    summary_row = {k: diagnostics[k] for k in summary_fields}
    save_csv_rows(out_dir / "summary.csv", [summary_row], summary_fields)

    print(f"\nDiagnostic run complete → {out_dir}")


if __name__ == "__main__":
    main()
