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
      method, micro_tpr, micro_fpr, macro_tpr, macro_fpr, ok_regions, tau_mean, time_ms
    """
    headers = ["Method", "TPR", "FPR", "TrTPR", "TrFPR", "MacroTPR", "MacroFPR", "Regions", "TauMean", "ms", "OK?"]
    table: List[List[str]] = []
    for r in rows:
        fpr = float(r["micro_fpr"])
        ok = "PASS" if fpr <= alpha + 1e-12 else "FAIL"
        table.append([
            str(r["method"]),
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


def print_ranking(ranking: List[Dict[str, Any]], *, alpha: float) -> None:
    """Print unconstrained and constrained rankings."""
    constrained = [r for r in ranking if r["mean_micro_fpr"] <= float(alpha) + 1e-12]
    constrained.sort(key=lambda r: r["mean_micro_tpr"], reverse=True)

    print("\n=== Ranking by mean micro TPR (UNCONSTRAINED) ===")
    for i, r in enumerate(ranking, start=1):
        tag = "PASS" if r["mean_micro_fpr"] <= float(alpha) + 1e-12 else "FAIL"
        print(
            f"{i:2d}. {r['method']:<18} "
            f"TPR={r['mean_micro_tpr']:.4f}\u00b1{r['std_micro_tpr']:.4f} "
            f"FPR={r['mean_micro_fpr']:.4f}\u00b1{r['std_micro_fpr']:.4f} [{tag}]"
        )

    print("\n=== Ranking by mean micro TPR (CONSTRAINED: FPR <= alpha) ===")
    if not constrained:
        print("  (none satisfy the constraint)")
    else:
        for i, r in enumerate(constrained, start=1):
            print(
                f"{i:2d}. {r['method']:<18} "
                f"TPR={r['mean_micro_tpr']:.4f}\u00b1{r['std_micro_tpr']:.4f} "
                f"FPR={r['mean_micro_fpr']:.4f}\u00b1{r['std_micro_fpr']:.4f}"
            )

    return constrained
