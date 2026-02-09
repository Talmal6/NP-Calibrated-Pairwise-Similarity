#!/usr/bin/env python3
"""
Experiment: sweep n_samples per class at fixed d.

Outputs (under outputs/n_sweep/run_*/):
- results.csv
- summary.json
- tpr_vs_n_d{d}.png
- time_vs_n_d{d}.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from np_bench.data import load_quora_hadamard, subsample_by_class
from np_bench.utils import (
    get_fisher_scores,
    plot_grouped_bars,
    make_run_dir,
    save_json,
    save_csv_rows,
)
from np_bench.methods import get_default_methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--n_list", type=str, default="100,200,500,1000,2000")
    parser.add_argument("--d", type=int, default=1024)
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    if not n_list:
        raise ValueError("Empty --n_list")

    run_dir = make_run_dir(base_dir=str(Path("outputs") / "n_sweep"), run_name=args.run_name)

    X_full, y_full = load_quora_hadamard(args.pkl, split_key=args.split_key)
    fisher = get_fisher_scores(X_full, y_full)
    sorted_idx = np.argsort(fisher)

    d = args.d
    if d > X_full.shape[1]:
        raise ValueError(f"d={d} > feature_dim={X_full.shape[1]}")

    top_k = sorted_idx[-d:]
    w_d = fisher[top_k]

    methods = get_default_methods()

    print(f"\n=== n_sweep @ FPR≈{args.alpha} fixed d={d} ===")
    print(f"trials={args.n_trials}")
    print("n_list:", n_list)
    print("methods:", [m.name for m in methods])

    # results[name]["tpr"][n] = list over trials
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    csv_rows: List[Dict[str, Any]] = []

    for n in n_list:
        for trial in range(args.n_trials):
            seed = 42 + 1000 * n + trial
            H0_full, H1_full = subsample_by_class(X_full, y_full, n, seed)

            H0 = H0_full[:, top_k]
            H1 = H1_full[:, top_k]

            for m in methods:
                out = m.run(
                    H0=H0,
                    H1=H1,
                    alpha=args.alpha,
                    weights=(w_d if m.needs_weights else None),
                    seed=(seed if m.needs_seed else None),
                )
                results[m.name]["tpr"][n].append(out.tpr)
                results[m.name]["fpr"][n].append(out.fpr)
                results[m.name]["time_ms"][n].append(out.time_ms)

                csv_rows.append({
                    "experiment": "n_sweep",
                    "trial": trial,
                    "seed": seed,
                    "d": d,
                    "n_samples_per_class": n,
                    "alpha": args.alpha,
                    "method": m.name,
                    "tpr": out.tpr,
                    "fpr": out.fpr,
                    "time_ms": out.time_ms,
                })

        print(f"\n[n={n}]")
        for m in methods:
            tpr_mu = float(np.mean(results[m.name]["tpr"][n]))
            tpr_sd = float(np.std(results[m.name]["tpr"][n]))
            tm_mu  = float(np.mean(results[m.name]["time_ms"][n]))
            tm_sd  = float(np.std(results[m.name]["time_ms"][n]))
            fpr_mu = float(np.mean(results[m.name]["fpr"][n]))
            print(f"  {m.name:<14} TPR={tpr_mu:.4f}±{tpr_sd:.4f}  FPR={fpr_mu:.4f}  time={tm_mu:.3f}±{tm_sd:.3f} ms")

    # Build plot series: per method, list aligned with n_list
    x_labels = [str(n) for n in n_list]

    tpr_series = {}
    time_series = {}

    for m in methods:
        name = m.name
        tpr_series[name] = (
            [float(np.mean(results[name]["tpr"][n])) for n in n_list],
            [float(np.std(results[name]["tpr"][n])) for n in n_list],
        )
        time_series[name] = (
            [float(np.mean(results[name]["time_ms"][n])) for n in n_list],
            [float(np.std(results[name]["time_ms"][n])) for n in n_list],
        )

    plot_grouped_bars(
        x_labels=x_labels,
        series_dict=tpr_series,
        ylabel=f"TPR @ FPR≈{args.alpha}",
        title=f"TPR vs n_samples (per class), d={d}",
        output_path=str(run_dir / f"tpr_vs_n_d{d}.png"),
    )

    plot_grouped_bars(
        x_labels=x_labels,
        series_dict=time_series,
        ylabel="Inference time (ms)",
        title=f"Inference time vs n_samples (per class), d={d}",
        output_path=str(run_dir / f"time_vs_n_d{d}.png"),
    )

    # Save CSV + summary.json
    fieldnames = [
        "experiment", "trial", "seed", "d", "n_samples_per_class", "alpha",
        "method", "tpr", "fpr", "time_ms"
    ]
    save_csv_rows(run_dir / "results.csv", csv_rows, fieldnames=fieldnames)

    summary = {
        "experiment": "n_sweep",
        "pkl": args.pkl,
        "split_key": args.split_key,
        "alpha": args.alpha,
        "d": d,
        "n_trials": args.n_trials,
        "n_list": n_list,
        "methods": [m.name for m in methods],
    }
    save_json(run_dir / "summary.json", summary)

    print(f"\n[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
