#!/usr/bin/env python3
"""
NP-style Benchmark (FPR ~= alpha via H0-quantile thresholding)

Experiment: sweep dimensions (d) at fixed n_samples per class.

Outputs (under outputs/dims_sweep/run_*/):
- results.csv
- summary.json
- benchmark_tpr_final.png
- benchmark_time_final.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from np_bench.methods import get_default_methods
from np_bench.data import load_quora_hadamard, subsample_by_class
from np_bench.utils import (
    make_run_dir, save_json, save_csv_rows,
    get_fisher_scores
)


def plot_lines_with_errorbars(
    dims: List[int],
    series: Dict[str, Dict[str, List[List[float]]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    """
    series[name][metric_key] is list over dims, each dim has list of trial values.
    """
    plt.figure(figsize=(14, 7))

    for name, data in series.items():
        vals_by_dim = data[metric_key]
        means = [float(np.mean(v)) if len(v) else 0.0 for v in vals_by_dim]
        stds  = [float(np.std(v)) if len(v) else 0.0 for v in vals_by_dim]
        plt.errorbar(dims, means, yerr=stds, capsize=4, marker="o", linewidth=2, label=name)

    plt.xscale("log", base=2)
    plt.xticks(dims, labels=[str(d) for d in dims])
    plt.xlabel("Number of Dimensions (d)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Graph] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n_samples", type=int, default=1000, help="per class")
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--dims", type=str, default="8,16,32,64,128,256,512,1024")
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument("--run_name", type=str, default=None, help="optional custom run dir name")
    args = parser.parse_args()

    dims_list = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    if not dims_list:
        raise ValueError("Empty --dims")

    # outputs/dims_sweep/run_xxx
    run_dir = make_run_dir(base_dir=str(Path("outputs") / "dims_sweep"), run_name=args.run_name)

    X_full, y_full = load_quora_hadamard(args.pkl, split_key=args.split_key)
    fisher = get_fisher_scores(X_full, y_full)
    sorted_idx = np.argsort(fisher)

    methods = get_default_methods()

    print(f"\n=== dims_sweep @ FPR≈{args.alpha} ===")
    print(f"n_samples per class={args.n_samples}, trials={args.n_trials}")
    print("dims:", dims_list)
    print("methods:", [m.name for m in methods])

    # res[name]["tpr"][d] = list over trials
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # For CSV rows
    csv_rows: List[Dict[str, Any]] = []

    for trial in range(args.n_trials):
        seed = 42 + trial
        H0_full, H1_full = subsample_by_class(X_full, y_full, args.n_samples, seed)

        for d in dims_list:
            if d > X_full.shape[1]:
                raise ValueError(f"d={d} > feature_dim={X_full.shape[1]}")

            top_k = sorted_idx[-d:]
            H0 = H0_full[:, top_k]
            H1 = H1_full[:, top_k]
            w_d = fisher[top_k]

            for m in methods:
                out = m.run(
                    H0=H0,
                    H1=H1,
                    alpha=args.alpha,
                    weights=(w_d if m.needs_weights else None),
                    seed=(seed if m.needs_seed else None),
                )
                res[m.name]["tpr"][d].append(out.tpr)
                res[m.name]["fpr"][d].append(out.fpr)
                res[m.name]["time_ms"][d].append(out.time_ms)

                csv_rows.append({
                    "experiment": "dims_sweep",
                    "trial": trial,
                    "seed": seed,
                    "d": d,
                    "n_samples_per_class": args.n_samples,
                    "alpha": args.alpha,
                    "method": m.name,
                    "tpr": out.tpr,
                    "fpr": out.fpr,
                    "time_ms": out.time_ms,
                })

    # Build plotting series (list aligned with dims_list)
    plot_series = {}
    for m in methods:
        plot_series[m.name] = {
            "tpr": [res[m.name]["tpr"][d] for d in dims_list],
            "time_ms": [res[m.name]["time_ms"][d] for d in dims_list],
        }

    plot_lines_with_errorbars(
        dims=dims_list,
        series=plot_series,
        metric_key="tpr",
        ylabel=f"TPR @ FPR≈{args.alpha}",
        title=f"TPR vs d (n_samples/class={args.n_samples})",
        output_path=run_dir / "benchmark_tpr_final.png",
    )

    plot_lines_with_errorbars(
        dims=dims_list,
        series=plot_series,
        metric_key="time_ms",
        ylabel="Inference Time (ms)",
        title=f"Inference Time vs d (n_samples/class={args.n_samples})",
        output_path=run_dir / "benchmark_time_final.png",
    )

    # Save CSV + summary.json
    fieldnames = [
        "experiment", "trial", "seed", "d", "n_samples_per_class", "alpha",
        "method", "tpr", "fpr", "time_ms"
    ]
    save_csv_rows(run_dir / "results.csv", csv_rows, fieldnames=fieldnames)

    summary = {
        "experiment": "dims_sweep",
        "pkl": args.pkl,
        "split_key": args.split_key,
        "alpha": args.alpha,
        "n_samples_per_class": args.n_samples,
        "n_trials": args.n_trials,
        "dims": dims_list,
        "methods": [m.name for m in methods],
    }
    save_json(run_dir / "summary.json", summary)

    print(f"\n[Done] outputs at: {run_dir}")


if __name__ == "__main__":
    main()
