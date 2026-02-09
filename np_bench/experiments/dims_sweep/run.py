#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from np_bench.data import load_quora_hadamard
from np_bench.utils import get_fisher_scores, make_run_dir, save_json, save_csv_rows
from np_bench.utils.split import split_by_class_triplet

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

try:
    from np_bench.methods.xgboost import XGBoostLightMethod, HAS_XGB
except Exception:
    HAS_XGB = False


def plot_lines(dims: List[int], series: Dict[str, List[List[float]]], ylabel: str, title: str, out: Path):
    plt.figure(figsize=(14, 7))
    for name, vals_by_dim in series.items():
        means = [float(np.mean(v)) if len(v) else 0.0 for v in vals_by_dim]
        stds  = [float(np.std(v))  if len(v) else 0.0 for v in vals_by_dim]
        plt.errorbar(dims, means, yerr=stds, capsize=4, marker="o", linewidth=2, label=name)
    plt.xscale("log", base=2)
    plt.xticks(dims, labels=[str(d) for d in dims])
    plt.xlabel("d")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Graph] Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--split_key", type=str, default="train")

    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_trials", type=int, default=3)

    ap.add_argument("--n_list", type=str, default="500,1000,2000",
                    help="Comma-separated total samples per class to sweep over")
    ap.add_argument("--train_frac", type=float, default=0.2,
                    help="Fraction of n for training (0-1)")
    ap.add_argument("--calib_frac", type=float, default=0.2,
                    help="Fraction of n for calibration (0-1)")
    ap.add_argument("--eval_frac", type=float, default=0.6,
                    help="Fraction of n for evaluation (0-1)")

    ap.add_argument("--dims", type=str, default="8,16,32,64,128,256,512,1024")
    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()

    # Validate fractions
    frac_sum = args.train_frac + args.calib_frac + args.eval_frac
    if not np.isclose(frac_sum, 1.0, atol=1e-6):
        raise ValueError(f"train_frac + calib_frac + eval_frac must equal 1.0, got {frac_sum:.6f}")
    if not (0.0 < args.train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0,1), got {args.train_frac}")
    if not (0.0 < args.calib_frac < 1.0):
        raise ValueError(f"calib_frac must be in (0,1), got {args.calib_frac}")
    if not (0.0 < args.eval_frac < 1.0):
        raise ValueError(f"eval_frac must be in (0,1), got {args.eval_frac}")

    dims_list = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    run_dir = make_run_dir(base_dir=str(Path("outputs") / "dims_sweep"), run_name=args.run_name)

    X_full, y_full = load_quora_hadamard(args.pkl, split_key=args.split_key)
    fisher = get_fisher_scores(X_full, y_full)
    sorted_idx = np.argsort(fisher)

    method_names = ["Cosine", "Vec (Wgt)", "Naive Bayes", "Log Reg", "LDA"]
    if HAS_XGB:
        method_names.append("XGBoost")
    method_names.extend(["Tiny MLP", "AndBox-HC", "AndBox-Wgt"])

    csv_rows: List[Dict[str, Any]] = []

    def resolve_sizes(n_total: int):
        n_train = int(np.floor(n_total * args.train_frac))
        n_calib = int(np.floor(n_total * args.calib_frac))
        n_eval = n_total - n_train - n_calib
        if n_eval <= 0:
            raise ValueError(f"n_total={n_total} too small for the given fractions.")
        return n_train, n_calib, n_eval

    print(f"\n=== dims_sweep (train/calib/eval) @ FPR≈{args.alpha} ===")
    print(f"n_list={n_list}, trials={args.n_trials}, dims={dims_list}")
    print(f"fractions: train={args.train_frac}, calib={args.calib_frac}, eval={args.eval_frac}")
    print(f"run_dir={run_dir}")
    for n_tag in n_list:
        n_train, n_calib, n_eval = resolve_sizes(n_tag)
        n_tag_label = str(n_tag)
        seed_base = 1000 * n_tag
        res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for trial in range(args.n_trials):
            seed = 42 + seed_base + trial
            sp = split_by_class_triplet(
                X_full, y_full,
                n_train=n_train, n_calib=n_calib, n_eval=n_eval,
                seed=seed
            )

            for d in dims_list:
                if d > X_full.shape[1]:
                    raise ValueError(f"d={d} > feature_dim={X_full.shape[1]}")

                top_k = sorted_idx[-d:]
                w_d = fisher[top_k]

                H0_tr = sp.H0_train[:, top_k]
                H1_tr = sp.H1_train[:, top_k]
                H0_ca = sp.H0_calib[:, top_k]
                H1_ev = sp.H1_eval[:, top_k]
                H0_ev = sp.H0_eval[:, top_k]

                # Create method instances
                methods = {
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

                for name, method in methods.items():
                    # Pass weights for methods that need it
                    method_weights = w_d if name in ["Vec (Wgt)", "AndBox-HC", "AndBox-Wgt"] else None
                    result = method.run(
                        H0_tr, H1_tr,
                        H0_ca, H0_ev, H1_ev,
                        args.alpha,
                        weights=method_weights,
                        seed=seed
                    )

                    res[name]["tpr"][d].append(result.tpr)
                    res[name]["fpr"][d].append(result.fpr)
                    res[name]["train_tpr"][d].append(result.train_tpr)
                    res[name]["train_fpr"][d].append(result.train_fpr)
                    res[name]["time_ms"][d].append(result.time_ms)

                    csv_rows.append({
                        "experiment": "dims_sweep",
                        "trial": trial,
                        "seed": seed,
                        "d": d,
                        "n_tag": n_tag_label,
                        "n_train": n_train,
                        "n_calib": n_calib,
                        "n_eval": n_eval,
                        "alpha": args.alpha,
                        "method": name,
                        "tpr": result.tpr,
                        "fpr": result.fpr,
                        "train_tpr": result.train_tpr,
                        "train_fpr": result.train_fpr,
                        "time_ms": result.time_ms,
                    })

        # plots
        tpr_series = {m: [res[m]["tpr"][d] for d in dims_list] for m in method_names if m in res}
        fpr_series = {m: [res[m]["fpr"][d] for d in dims_list] for m in method_names if m in res}
        train_tpr_series = {m: [res[m]["train_tpr"][d] for d in dims_list] for m in method_names if m in res}
        train_fpr_series = {m: [res[m]["train_fpr"][d] for d in dims_list] for m in method_names if m in res}
        tm_series  = {m: [res[m]["time_ms"][d] for d in dims_list] for m in method_names if m in res}

        suffix = f"_n{n_tag_label}"
        title_suffix = f" (n={n_tag_label})"

        plot_lines(dims_list, tpr_series, ylabel=f"TPR @ FPR≈{args.alpha}",
                   title=f"TPR vs d (NP-calibrated on calib(H0)){title_suffix}",
                   out=run_dir / f"benchmark_tpr_final{suffix}.png")

        plot_lines(dims_list, tm_series, ylabel="Inference time (ms) (single call on eval(H1))",
                   title=f"Inference time vs d{title_suffix}",
                   out=run_dir / f"benchmark_time_final{suffix}.png")

        # (אופציונלי) גרף FPR מול alpha
        plot_lines(dims_list, fpr_series, ylabel="FPR on eval",
                   title=f"FPR vs d (target α={args.alpha}){title_suffix}",
                   out=run_dir / f"benchmark_fpr_final{suffix}.png")

        plot_lines(dims_list, train_tpr_series, ylabel=f"Train TPR @ FPR≈{args.alpha}",
                   title=f"Train TPR vs d (NP-calibrated on calib(H0)){title_suffix}",
                   out=run_dir / f"benchmark_train_tpr{suffix}.png")

        plot_lines(dims_list, train_fpr_series, ylabel="Train FPR",
                   title=f"Train FPR vs d (target α={args.alpha}){title_suffix}",
                   out=run_dir / f"benchmark_train_fpr{suffix}.png")

    # save results
    save_csv_rows(
        run_dir / "results.csv",
        csv_rows,
        fieldnames=[
            "experiment", "trial", "seed", "d", "n_tag",
            "n_train", "n_calib", "n_eval", "alpha",
            "method", "tpr", "fpr", "train_tpr", "train_fpr", "time_ms",
        ]
    )
    save_json(run_dir / "summary.json", {
        "experiment": "dims_sweep",
        "pkl": args.pkl,
        "split_key": args.split_key,
        "alpha": args.alpha,
        "n_trials": args.n_trials,
        "n_list": n_list,
        "train_frac": args.train_frac,
        "calib_frac": args.calib_frac,
        "eval_frac": args.eval_frac,
        "dims": dims_list,
        "methods": method_names,
        "note": "NP threshold calibrated on calib(H0) only; eval metrics on eval set.",
    })

    print(f"\n[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
