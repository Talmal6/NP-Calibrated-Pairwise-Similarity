"""Pretty-print helpers for trial tables and rankings."""
from __future__ import annotations

from typing import Any, Dict, List


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:  # nan
        return "nan"
    return f"{x:.{nd}f}"


def print_trial_table(rows: List[Dict[str, Any]], *, alpha: float) -> None:
    """
    Pretty print per-trial results with PASS/FAIL tag.
    Expects rows with keys:
      method, train_samples_needed, micro_tpr, micro_fpr, macro_tpr, macro_fpr, ok_regions, tau_mean, time_ms
    """
    headers = ["Method", "TrainN", "TPR", "FPR", "TrTPR", "TrFPR", "MacroTPR", "MacroFPR", "Regions", "TauMean", "ms", "OK?"]
    table: List[List[str]] = []
    for r in rows:
        fpr = float(r["micro_fpr"])
        ok = "PASS" if fpr <= alpha + 1e-12 else "FAIL"
        table.append([
            str(r["method"]),
            str(int(r.get("train_samples_needed", r.get("train_total_samples", 0)))),
            _fmt(float(r["micro_tpr"]), 4),
            _fmt(fpr, 4),
            _fmt(float(r.get("train_tpr", float("nan"))), 4),
            _fmt(float(r.get("train_fpr", float("nan"))), 4),
            _fmt(float(r["macro_tpr"]), 4),
            _fmt(float(r["macro_fpr"]), 4),
            str(int(r["ok_regions"])),
            _fmt(float(r["tau_mean"]), 4),
            _fmt(float(r["time_ms"]), 1),
            ok,
        ])

    # column widths
    widths = [len(h) for h in headers]
    for row in table:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(str(cell)))

    def fmt_row(row: List[str]) -> str:
        return " | ".join(str(row[j]).ljust(widths[j]) for j in range(len(headers)))

    def line(sep: str = "-") -> str:
        return sep * (sum(widths) + 3 * (len(widths) - 1))

    print("\n" + fmt_row(headers))
    print(line("-"))
    for row in table:
        print(fmt_row(row))


def print_ranking(
    ranking: List[Dict[str, Any]],
    *,
    alpha: float,
    tpr_tolerance: float = 0.01,
) -> List[Dict[str, Any]]:
    """Print safety-first and cache-efficiency rankings."""
    alpha = float(alpha)

    primary = sorted(ranking, key=lambda r: int(r.get("primary_safety_rank", 10**9)))
    safety_valid = [
        r for r in primary
        if float(r.get("valid_rate", 0.0)) >= 1.0 - 1e-12
        and float(r.get("mean_eval_fpr", r.get("mean_micro_fpr", float("inf")))) <= alpha + 1e-12
        and float(r.get("max_eval_fpr", float("inf"))) <= alpha + 1e-12
    ]

    def train_count_text(r: Dict[str, Any]) -> str:
        if "mean_train_n" in r:
            mean_n = float(r["mean_train_n"])
            if mean_n != mean_n:
                return ""
            min_raw = r.get("min_train_n", None)
            max_raw = r.get("max_train_n", None)
            min_n = int(min_raw) if min_raw is not None else int(round(mean_n))
            max_n = int(max_raw) if max_raw is not None else int(round(mean_n))
        elif "mean_train_samples_needed" in r:
            mean_n = float(r["mean_train_samples_needed"])
            if mean_n != mean_n:
                return ""
            min_raw = r.get("min_train_samples_needed", None)
            max_raw = r.get("max_train_samples_needed", None)
            min_n = int(min_raw) if min_raw is not None else int(round(mean_n))
            max_n = int(max_raw) if max_raw is not None else int(round(mean_n))
        else:
            return ""
        if min_n == max_n:
            return f"TrainN={min_n} "
        return f"TrainN={mean_n:.1f}[{min_n}-{max_n}] "

    print("\n=== Primary Safety Ranking ===")
    for i, r in enumerate(primary, start=1):
        valid_rate = float(r.get("valid_rate", float("nan")))
        mean_fpr = float(r.get("mean_eval_fpr", r.get("mean_micro_fpr", float("nan"))))
        max_fpr = float(r.get("max_eval_fpr", float("nan")))
        tag = (
            "PASS"
            if valid_rate >= 1.0 - 1e-12 and mean_fpr <= alpha + 1e-12 and max_fpr <= alpha + 1e-12
            else "FAIL"
        )
        print(
            f"{i:2d}. {r['method']:<18} "
            f"{train_count_text(r)}"
            f"TPR={float(r.get('mean_eval_tpr', r['mean_micro_tpr'])):.4f}+/-{r['std_micro_tpr']:.4f} "
            f"FPR={mean_fpr:.4f}+/-{r['std_micro_fpr']:.4f} "
            f"MaxFPR={max_fpr:.4f} Valid={valid_rate:.2f} [{tag}]"
        )

    print("\n=== Cache-Efficiency Ranking (safe and near-best TPR) ===")
    if not safety_valid:
        print("  (none satisfy valid_rate=1 and FPR <= alpha)")
    else:
        best_tpr = max(float(r.get("mean_eval_tpr", r.get("mean_micro_tpr", float("-inf")))) for r in safety_valid)
        cache_rows = [
            r for r in safety_valid
            if float(r.get("mean_eval_tpr", r.get("mean_micro_tpr", float("-inf")))) >= best_tpr - float(tpr_tolerance)
        ]
        def cache_sort_train_n(row: Dict[str, Any]) -> float:
            out = float(row.get("mean_train_n", row.get("mean_train_samples_needed", float("inf"))))
            return out if out == out else float("inf")

        cache_rows.sort(
            key=lambda r: (
                cache_sort_train_n(r),
                -float(r.get("mean_eval_tpr", r.get("mean_micro_tpr", float("-inf")))),
                float(r.get("mean_eval_fpr", r.get("mean_micro_fpr", float("inf")))),
            )
        )
        print(f"  best_safe_tpr={best_tpr:.4f} tolerance={float(tpr_tolerance):.4f}")
        for i, r in enumerate(cache_rows, start=1):
            print(
                f"{i:2d}. {r['method']:<18} "
                f"{train_count_text(r)}"
                f"TPR={float(r.get('mean_eval_tpr', r['mean_micro_tpr'])):.4f}+/-{r['std_micro_tpr']:.4f} "
                f"FPR={float(r.get('mean_eval_fpr', r['mean_micro_fpr'])):.4f}+/-{r['std_micro_fpr']:.4f} "
                f"MaxFPR={float(r.get('max_eval_fpr', float('nan'))):.4f}"
            )

    return safety_valid
