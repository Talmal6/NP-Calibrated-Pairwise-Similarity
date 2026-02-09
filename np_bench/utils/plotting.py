from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_grouped_bars(
    x_labels: List[str],
    series_dict: Dict[str, Tuple[List[float], List[float]]],
    ylabel: str,
    title: str,
    output_path: str,
):
    """
    Grouped bar chart.

    Args:
      x_labels: list of group labels (e.g., n_samples values as strings)
      series_dict: {method_name: (means_list, stds_list)} aligned with x_labels
    """
    methods = list(series_dict.keys())
    n_groups = len(x_labels)
    n_methods = len(methods)

    if n_groups == 0:
        raise ValueError("x_labels is empty")
    if n_methods == 0:
        raise ValueError("series_dict is empty")

    x = np.arange(n_groups)
    width = 0.8 / max(1, n_methods)

    plt.figure(figsize=(16, 6))
    for i, m in enumerate(methods):
        means, stds = series_dict[m]
        if len(means) != n_groups or len(stds) != n_groups:
            raise ValueError(f"Series '{m}' length mismatch with x_labels")

        offset = (i - (n_methods - 1) / 2) * width
        plt.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=m)

    plt.xticks(x, x_labels)
    plt.xlabel("n_samples per class")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"[Graph] Saved: {output_path}")
