#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from np_bench.data import load_quora_hadamard
from np_bench.utils import get_fisher_scores, make_run_dir, save_json, save_csv_rows, plot_grouped_bars
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--split_key", type=str, default="train")

    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_trials", type=int, default=3)

    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--n_list", type=str, default="100,200,500,1000,2000")

    # for each n in n_list we do: train/calib/eval sizes per class
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=1000)

    ap.add_argument("--sweep", type=str, choices=["none", "train", "calib", "eval", "total"], default="none")
    ap.add_argument("--train_frac", type=float, default=0.2)
    ap.add_argument("--calib_frac", type=float, default=0.2)

    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    run_dir = make_run_dir(base_dir=str(Path("outputs") / "n_sweep"), run_name=args.run_name)

    X_full, y_full = load_quora_hadamard(args.pkl, split_key=args.split_key)
    fisher = get_fisher_scores(X_full, y_full)
    sorted_idx = np.argsort(fisher)

    if args.d > X_full.shape[1]:
        raise ValueError(f"d={args.d} > feature_dim={X_full.shape[1]}")
    top_k = sorted_idx[-args.d:]
    w_d = fisher[top_k]

    method_names = ["Cosine", "Vec (Wgt)", "Naive Bayes", "Log Reg", "LDA"]
    if HAS_XGB:
        method_names.append("XGBoost")
    method_names.extend(["Tiny MLP", "AndBox-HC", "AndBox-Wgt"])

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    csv_rows: List[Dict[str, Any]] = []

    if args.sweep == "none":
        sweep_values = [None]
        sweep_n_list = []
    else:
        sweep_values = n_list
        sweep_n_list = n_list

    def resolve_sizes(n_tag):
        n_train = args.n_train
        n_calib = args.n_calib
        n_eval = args.n_eval

        if args.sweep == "train":
            n_train = int(n_tag)
        elif args.sweep == "calib":
            n_calib = int(n_tag)
        elif args.sweep == "eval":
            n_eval = int(n_tag)
        elif args.sweep == "total":
            total = int(n_tag)
            if not (0.0 < args.train_frac < 1.0 and 0.0 <= args.calib_frac < 1.0):
                raise ValueError("train_frac must be in (0,1) and calib_frac in [0,1).")
            if args.train_frac + args.calib_frac >= 1.0:
                raise ValueError("train_frac + calib_frac must be < 1.")
            n_train = int(np.floor(total * args.train_frac))
            n_calib = int(np.floor(total * args.calib_frac))
            n_eval = int(total - n_train - n_calib)
            if n_eval <= 0:
                raise ValueError("Total too small for derived split sizes.")
        return n_train, n_calib, n_eval

    print(f"\n=== n_sweep (train/calib/eval) fixed d={args.d} @ FPR≈{args.alpha} ===")
    if args.sweep == "none":
        print(f"sweep=none, trials={args.n_trials}")
    else:
        print(f"sweep={args.sweep}, n_list={sweep_n_list}, trials={args.n_trials}")
    print(f"fixed: n_train={args.n_train}, n_calib={args.n_calib}, n_eval={args.n_eval}")
    if args.sweep == "total":
        print(f"fractions: train_frac={args.train_frac}, calib_frac={args.calib_frac}")
    print(f"run_dir={run_dir}")

    for n_tag in sweep_values:
        n_train, n_calib, n_eval = resolve_sizes(n_tag)

        for trial in range(args.n_trials):
            seed = 42 + 1000 * (n_tag if n_tag is not None else 0) + trial

            sp = split_by_class_triplet(
                X_full, y_full,
                n_train=n_train, n_calib=n_calib, n_eval=n_eval,
                seed=seed
            )

            # additionally, enforce enough samples by also subsampling within each split to size=n if wanted
            # Here n_list represents *overall difficulty*; simplest: use it to set n_eval etc. externally.
            # If you want n to control total size, pass --n_train/--n_calib/--n_eval derived from n.

            H0_tr = sp.H0_train[:, top_k]
            H1_tr = sp.H1_train[:, top_k]
            H0_ca = sp.H0_calib[:, top_k]
            H0_ev = sp.H0_eval[:, top_k]
            H1_ev = sp.H1_eval[:, top_k]

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

                results[name]["tpr"][n_tag].append(result.tpr)
                results[name]["fpr"][n_tag].append(result.fpr)
                results[name]["train_tpr"][n_tag].append(result.train_tpr)
                results[name]["train_fpr"][n_tag].append(result.train_fpr)
                results[name]["time_ms"][n_tag].append(result.time_ms)

                csv_rows.append({
                    "experiment": "n_sweep",
                    "trial": trial,
                    "seed": seed,
                    "d": args.d,
                    "n_tag": (n_tag if n_tag is not None else "fixed"),
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

        print(f"\n[n_tag={n_tag if n_tag is not None else 'fixed'}]")
        for name in method_names:
            if name not in results:
                continue
            tpr_mu = float(np.mean(results[name]["tpr"][n_tag]))
            fpr_mu = float(np.mean(results[name]["fpr"][n_tag]))
            tr_tpr_mu = float(np.mean(results[name]["train_tpr"][n_tag]))
            tr_fpr_mu = float(np.mean(results[name]["train_fpr"][n_tag]))
            tm_mu  = float(np.mean(results[name]["time_ms"][n_tag]))
            print(f"  {name:<14}  TPR={tpr_mu:.4f}  FPR={fpr_mu:.4f}  train_TPR={tr_tpr_mu:.4f}  train_FPR={tr_fpr_mu:.4f}  time={tm_mu:.3f} ms")

    # plots (grouped bars)
    x_labels = [str(n) if n is not None else "fixed" for n in sweep_values]
    tpr_series = {}
    train_tpr_series = {}
    train_fpr_series = {}
    time_series = {}

    for name in method_names:
        if name not in results:
            continue
        tpr_series[name] = (
            [float(np.mean(results[name]["tpr"][n])) for n in sweep_values],
            [float(np.std(results[name]["tpr"][n])) for n in sweep_values],
        )
        train_tpr_series[name] = (
            [float(np.mean(results[name]["train_tpr"][n])) for n in sweep_values],
            [float(np.std(results[name]["train_tpr"][n])) for n in sweep_values],
        )
        train_fpr_series[name] = (
            [float(np.mean(results[name]["train_fpr"][n])) for n in sweep_values],
            [float(np.std(results[name]["train_fpr"][n])) for n in sweep_values],
        )
        time_series[name] = (
            [float(np.mean(results[name]["time_ms"][n])) for n in sweep_values],
            [float(np.std(results[name]["time_ms"][n])) for n in sweep_values],
        )

    plot_grouped_bars(
        x_labels=x_labels,
        series_dict=tpr_series,
        ylabel=f"TPR @ FPR≈{args.alpha}",
        title=f"TPR vs n_tag (sweep={args.sweep}, fixed d={args.d})",
        output_path=str(run_dir / f"tpr_vs_n_d{args.d}.png"),
    )
    plot_grouped_bars(
        x_labels=x_labels,
        series_dict=time_series,
        ylabel="Inference time (ms)",
        title=f"Inference time vs n_tag (sweep={args.sweep}, fixed d={args.d})",
        output_path=str(run_dir / f"time_vs_n_d{args.d}.png"),
    )
    plot_grouped_bars(
        x_labels=x_labels,
        series_dict=train_tpr_series,
        ylabel=f"Train TPR @ FPR≈{args.alpha}",
        title=f"Train TPR vs n_tag (sweep={args.sweep}, fixed d={args.d})",
        output_path=str(run_dir / f"train_tpr_vs_n_d{args.d}.png"),
    )
    plot_grouped_bars(
        x_labels=x_labels,
        series_dict=train_fpr_series,
        ylabel="Train FPR",
        title=f"Train FPR vs n_tag (sweep={args.sweep}, fixed d={args.d})",
        output_path=str(run_dir / f"train_fpr_vs_n_d{args.d}.png"),
    )

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
        "experiment": "n_sweep",
        "pkl": args.pkl,
        "split_key": args.split_key,
        "alpha": args.alpha,
        "d": args.d,
        "n_trials": args.n_trials,
        "n_list": sweep_n_list,
        "sweep": args.sweep,
        "n_train": args.n_train,
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "train_frac": args.train_frac,
        "calib_frac": args.calib_frac,
        "methods": method_names,
        "note": "NP threshold calibrated on calib(H0) only; eval metrics on eval set.",
    })

    print(f"\n[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
