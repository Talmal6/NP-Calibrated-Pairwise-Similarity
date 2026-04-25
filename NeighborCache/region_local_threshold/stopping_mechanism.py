"""Online stopping utilities for NP threshold stabilization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class StopConfig:
    stop_check_every: int
    stop_window: int
    stop_patience: int
    stop_eps_tpr: float
    stop_eps_fpr: float
    stop_eps_tau: float
    stop_fpr_margin: float
    alpha: float


@dataclass(frozen=True)
class StopHistoryEntry:
    checkpoint: int
    tpr_monitor: float
    fpr_monitor: float
    tau: float
    slope_tpr: float
    slope_fpr: float
    slope_tau: float
    condition_passed: bool
    stop_streak: int
    should_stop: bool


def least_squares_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    n = y.size
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


class OnlineStopper:
    def __init__(self, config: StopConfig) -> None:
        self.cfg = config
        self.history: List[StopHistoryEntry] = []
        self._streak = 0

    def update(self, checkpoint: int, tpr: float, fpr: float, tau: float) -> StopHistoryEntry:
        window = max(1, int(self.cfg.stop_window))
        recent = self.history[-(window - 1):] if window > 1 else []

        tpr_series = np.array([e.tpr_monitor for e in recent] + [tpr], dtype=np.float64)
        fpr_series = np.array([e.fpr_monitor for e in recent] + [fpr], dtype=np.float64)
        tau_series = np.array([e.tau for e in recent] + [tau], dtype=np.float64)

        slope_tpr = least_squares_slope(tpr_series)
        slope_fpr = least_squares_slope(fpr_series)
        slope_tau = least_squares_slope(tau_series)
        mean_fpr = float(np.mean(fpr_series))

        condition_passed = (
            mean_fpr <= float(self.cfg.alpha) + float(self.cfg.stop_fpr_margin)
            and np.isfinite(slope_tpr)
            and np.isfinite(slope_fpr)
            and np.isfinite(slope_tau)
            and abs(float(slope_tpr)) < float(self.cfg.stop_eps_tpr)
            and abs(float(slope_fpr)) < float(self.cfg.stop_eps_fpr)
            and abs(float(slope_tau)) < float(self.cfg.stop_eps_tau)
        )

        if condition_passed:
            self._streak += 1
        else:
            self._streak = 0

        should_stop = self._streak >= int(self.cfg.stop_patience)

        entry = StopHistoryEntry(
            checkpoint=int(checkpoint),
            tpr_monitor=float(tpr),
            fpr_monitor=float(fpr),
            tau=float(tau),
            slope_tpr=float(slope_tpr),
            slope_fpr=float(slope_fpr),
            slope_tau=float(slope_tau),
            condition_passed=bool(condition_passed),
            stop_streak=int(self._streak),
            should_stop=bool(should_stop),
        )
        self.history.append(entry)
        return entry
