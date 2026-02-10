# save as: np_bench/experiments/run_all_diagnostics.py

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
    "XGBoost",  # יהיה מותנה בהמשך
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

def run_diagnostics(method):
    print(f"\n==== Running: {method} ====")
    cmd = [
        "python", "-m", "np_bench.experiments.diagnostic.run",
        "--pkl", "np_bench/data/quora_question_pairs_with_embeddings.pkl",
        "--d", "1024",
        "--n", "2000",
        "--alpha", "0.05",
        "--method", method,
    ]
    subprocess.run(cmd, check=True)

def collect_summaries():
    base = Path("outputs/diagnostic_run")
    rows = []
    for method_dir in base.iterdir():
        summary_file = method_dir / "summary.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            rows.append(df)
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        out_path = base / "all_methods_summary.csv"
        all_df.to_csv(out_path, index=False)
        print(f"\n✅ All summaries saved to: {out_path}")
    else:
        print("⚠️ No summaries found.")

def main():
    for method in METHODS:
        if method_is_available(method):
            run_diagnostics(method)
    collect_summaries()

if __name__ == "__main__":
    main()
