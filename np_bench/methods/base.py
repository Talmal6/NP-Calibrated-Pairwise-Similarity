from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class MethodResult:
    tpr: float
    fpr: float
    time_ms: float
    train_tpr: float = 0.0
    train_fpr: float = 0.0


def _np_eval_from_calib(
    s0_cal: np.ndarray,
    s0_eval: np.ndarray,
    s1_eval: np.ndarray,
    alpha: float,
    tie_mode: str = "ge",
) -> tuple[float, float, float]:
    thresh = float(np.quantile(s0_cal, 1.0 - alpha))
    if tie_mode == "gt":
        tpr = float(np.mean(s1_eval > thresh))
        fpr = float(np.mean(s0_eval > thresh))
    else:
        tpr = float(np.mean(s1_eval >= thresh))
        fpr = float(np.mean(s0_eval >= thresh))
    return tpr, fpr, thresh


class BaseMethod(ABC):
    """
    NP-correct base method:
      - fit on TRAIN
      - threshold on CALIB(H0)
      - eval on EVAL
    """
    name: str = "BaseMethod"
    needs_weights: bool = False
    needs_seed: bool = False

    @abstractmethod
    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "BaseMethod":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns higher-is-more-positive scores, shape (n,).
        """
        ...

    def run(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        H0_calib: np.ndarray,
        H0_eval: np.ndarray,
        H1_eval: np.ndarray,
        alpha: float,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        tie_mode: str = "ge",
    ) -> MethodResult:
        # Fit
        self.fit(H0_train, H1_train, weights=weights, seed=seed)

        # Calib + eval scores
        s0_cal = self.score(H0_calib)
        s0_ev  = self.score(H0_eval)
        s1_ev  = self.score(H1_eval)

        # NP metrics (eval)
        tpr, fpr, thresh = _np_eval_from_calib(s0_cal, s0_ev, s1_ev, alpha, tie_mode=tie_mode)

        # Train metrics (reuse same threshold from calib)
        s0_tr = self.score(H0_train)
        s1_tr = self.score(H1_train)
        if tie_mode == "gt":
            train_tpr = float(np.mean(s1_tr > thresh))
            train_fpr = float(np.mean(s0_tr > thresh))
        else:
            train_tpr = float(np.mean(s1_tr >= thresh))
            train_fpr = float(np.mean(s0_tr >= thresh))

        # Inference timing on H1_eval
        import time
        t0 = time.perf_counter()
        _ = self.score(H1_eval)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return MethodResult(tpr=tpr, fpr=fpr, time_ms=float(dt_ms),
                            train_tpr=train_tpr, train_fpr=train_fpr)
