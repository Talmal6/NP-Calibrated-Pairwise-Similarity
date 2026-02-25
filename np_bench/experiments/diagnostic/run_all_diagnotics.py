#!/usr/bin/env python3
"""Run diagnostic script for all available methods across multiple seeds."""

import argparse
import subprocess
from pathlib import Path
import pandas as pd

METHODS = [
    "Cosine",
    "Vec (Wgt)",
    "Naive Bayes",
    "Log Reg",
    "LDA",
    "Tiny MLP",
    "AndBox-HC",
    "AndBox-Wgt",
    "XGBoost",
]

def method_is_available(name):
    # exclude XGBoost if not available
    if name == "XGBoost":
        try:
            import xgboost
            return True
        except ImportError:
            return False
    return True

def run_diagnostics(method, args):
    print(f"\n==== Running: {method} (seed={args.current_seed}) ====")
    cmd = [
        "python", "-m", "np_bench.experiments.diagnostic.run",
        "--pkl", args.pkl,
        "--d", str(args.d),
        "--n", str(args.n),
        "--alpha", str(args.alpha),
        "--train_frac", str(args.train_frac),
        "--eval_frac", str(args.eval_frac),
        "--seed", str(args.current_seed),
        "--method", method,
    ]
    subprocess.run(cmd, check=True)

def collect_summaries():
    base = Path("outputs/diagnostic_run")
    rows = []
    # Recursively find all summary.csv files
    for summary_file in base.rglob("summary.csv"):
        try:
            df = pd.read_csv(summary_file)
            rows.append(df)
        except Exception as e:
            print(f"⚠️ Failed to read {summary_file}: {e}")
    
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        out_path = base / "all_methods_summary.csv"
        all_df.to_csv(out_path, index=False)
        print(f"\n✅ All summaries saved to: {out_path} ({len(rows)} files)")
    else:
        print("⚠️ No summaries found.")

def main():
    ap = argparse.ArgumentParser(
        description="Run diagnostic script for all methods across multiple seeds"
    )
    ap.add_argument("--pkl", required=True, help="Path to pickle dataset")
    ap.add_argument("--d", type=int, default=1024,
                    help="Number of features to select")
    ap.add_argument("--n", type=int, default=2000,
                    help="Total examples per class")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="NP FPR constraint")
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--eval_frac", type=float, default=0.5)
    ap.add_argument("--seeds", type=str, default="42",
                    help="Comma-separated list of seeds to run")
    args = ap.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    print(f"\n{'=' * 60}")
    print(f"Running diagnostics for {len(METHODS)} methods × {len(seeds)} seeds")
    print(f"n={args.n}, d={args.d}, alpha={args.alpha}")
    print(f"fractions: train={args.train_frac}, eval={args.eval_frac}")
    print(f"seeds={seeds}")
    print(f"{'=' * 60}")

    for seed in seeds:
        args.current_seed = seed
        for method in METHODS:
            if method_is_available(method):
                try:
                    run_diagnostics(method, args)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed for {method} with seed={seed}: {e}")
                    continue
    
    collect_summaries()

if __name__ == "__main__":
    main()
