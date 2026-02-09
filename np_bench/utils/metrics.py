from __future__ import annotations
from typing import Tuple
import numpy as np


def get_metrics_at_fpr(
    scores0: np.ndarray,
    scores1: np.ndarray,
    alpha: float,
    tie_mode: str = "ge",  # "ge" or "gt"
) -> Tuple[float, float]:
    """
    Given scores for H0 and H1, set threshold as (1-alpha) quantile of H0,
    then compute (TPR, FPR) under tie policy.

    tie_mode:
      - "ge": positive if score >= thresh
      - "gt": positive if score > thresh
    """
    if scores0.size == 0 or scores1.size == 0:
        return 0.0, 0.0

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    thresh = float(np.quantile(scores0, 1.0 - alpha))

    if tie_mode not in ("ge", "gt"):
        raise ValueError(f"tie_mode must be 'ge' or 'gt', got {tie_mode}")

    if tie_mode == "gt":
        tpr = float(np.mean(scores1 > thresh))
        fpr = float(np.mean(scores0 > thresh))
    else:
        tpr = float(np.mean(scores1 >= thresh))
        fpr = float(np.mean(scores0 >= thresh))

    return tpr, fpr
