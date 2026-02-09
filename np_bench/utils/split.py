from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class SplitTriplet:
    H0_train: np.ndarray
    H1_train: np.ndarray
    H0_calib: np.ndarray
    H1_calib: np.ndarray
    H0_eval: np.ndarray
    H1_eval: np.ndarray


def split_by_class_triplet(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_calib: int,
    n_eval: int,
    seed: int,
) -> SplitTriplet:
    """
    Balanced split by class into (train, calib, eval) WITHOUT replacement.

    Returns:
      H0_train, H1_train, H0_calib, H1_calib, H0_eval, H1_eval
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows != y size: {X.shape[0]} != {y.shape[0]}")
    if min(n_train, n_calib, n_eval) <= 0:
        raise ValueError("n_train/n_calib/n_eval must be positive")

    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        return SplitTriplet(X[idx0], X[idx1], X[idx0][:0], X[idx1][:0], X[idx0][:0], X[idx1][:0])

    need = n_train + n_calib + n_eval
    n0 = min(idx0.size, need)
    n1 = min(idx1.size, need)
    if n0 < need or n1 < need:
        raise ValueError(
            f"Not enough samples per class. Need {need} each, have: H0={idx0.size}, H1={idx1.size}"
        )

    sel0 = rng.choice(idx0, size=need, replace=False)
    sel1 = rng.choice(idx1, size=need, replace=False)

    def chunk(sel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = sel[:n_train]
        b = sel[n_train:n_train + n_calib]
        c = sel[n_train + n_calib:]
        return a, b, c

    tr0, ca0, ev0 = chunk(sel0)
    tr1, ca1, ev1 = chunk(sel1)

    return SplitTriplet(
        H0_train=X[tr0], H1_train=X[tr1],
        H0_calib=X[ca0], H1_calib=X[ca1],
        H0_eval=X[ev0],  H1_eval=X[ev1],
    )
