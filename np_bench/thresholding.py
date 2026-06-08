from __future__ import annotations

import numpy as np


def _beta_ppf(q: float, a: float, b: float) -> float:
    try:
        from scipy.stats import beta as scipy_beta  # type: ignore

        return float(scipy_beta.ppf(q, a, b))
    except Exception:
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return 1.0
        return float(q)


def _normal_ppf(q: float) -> float:
    try:
        from scipy.stats import norm  # type: ignore

        return float(norm.ppf(q))
    except Exception:
        if q >= 0.995:
            return 2.5758
        if q >= 0.99:
            return 2.3263
        if q >= 0.975:
            return 1.9600
        if q >= 0.95:
            return 1.6449
        return 1.2816


def _fpr_ucb(k: int, n: int, *, method: str, delta: float) -> float:
    if n <= 0:
        return 1.0
    k = int(max(0, min(k, n)))
    delta = float(np.clip(delta, 1e-12, 0.5))

    if method == "clopper_pearson":
        if k >= n:
            return 1.0
        return _beta_ppf(1.0 - delta, k + 1.0, n - k)

    if method == "beta_ucb":
        return _beta_ppf(1.0 - delta, k + 1.0, n - k + 1.0)

    if method == "wilson":
        phat = k / n
        z = _normal_ppf(1.0 - delta)
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (phat + z2 / (2.0 * n)) / denom
        radius = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
        return float(min(1.0, max(0.0, center + radius)))

    return float(k / n)


def select_np_threshold(
    scores_h0: np.ndarray,
    *,
    alpha: float,
    tie_mode: str = "ge",
    guardrail: str = "none",
    guardrail_delta: float = 0.01,
) -> float:
    """Tie-aware empirical NP threshold selector used by offline and online paths."""
    if tie_mode not in {"ge", "gt"}:
        raise ValueError("tie_mode must be 'ge' or 'gt'.")
    if guardrail not in {"none", "clopper_pearson", "wilson", "beta_ucb"}:
        raise ValueError(
            "guardrail must be one of 'none', 'clopper_pearson', 'wilson', or 'beta_ucb'."
        )

    s = np.asarray(scores_h0, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("empty H0 calibration scores")

    alpha_f = float(alpha)
    uniq, counts = np.unique(s, return_counts=True)
    n = int(s.size)
    cumsum = np.cumsum(counts)

    for i, tau in enumerate(uniq):
        if tie_mode == "gt":
            k = int(n - cumsum[i])
        else:
            k = int(n - (cumsum[i - 1] if i > 0 else 0))

        if guardrail == "none":
            if (k / max(1, n)) <= alpha_f:
                return float(tau)
            continue

        ucb = _fpr_ucb(k, n, method=guardrail, delta=guardrail_delta)
        if ucb <= alpha_f:
            return float(tau)

    return float("inf")
