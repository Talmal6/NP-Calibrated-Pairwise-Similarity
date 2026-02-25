"""Method building, fitting helpers, and train-requirement checks."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from np_bench.methods import get_default_methods


def build_methods() -> Dict[str, Any]:
    methods: Dict[str, Any] = {}
    for method in get_default_methods():
        name = str(getattr(method, "name", type(method).__name__))
        methods[name] = method
    return methods


def needs_weights(method: Any) -> bool:
    return bool(getattr(method, "needs_weights", False))


def needs_seed(method: Any) -> bool:
    return bool(getattr(method, "needs_seed", False))


def try_fit_method(
    method: Any,
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    weights: Optional[np.ndarray],
    seed: int,
    alpha: float,
) -> None:
    if not hasattr(method, "fit"):
        return

    try_orders: List[Dict[str, Any]] = [
        {"weights": weights, "seed": seed, "alpha": alpha},
        {"weights": weights, "alpha": alpha},
        {"seed": seed, "alpha": alpha},
        {"alpha": alpha},
        {"weights": weights, "seed": seed},
        {"weights": weights},
        {"seed": seed},
        {},
    ]

    last_exc: Optional[Exception] = None
    for kwargs in try_orders:
        try:
            method.fit(X0, X1, **kwargs)
            return
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception:
            raise

    raise TypeError(f"Could not fit method={type(method).__name__}: {last_exc}")


_TRAIN_REQUIRED_NAMES = {
    "Naive Bayes",
    "Log Reg",
    "LDA",
    "Tiny MLP",
    "XGBoost",
    "WeightedEnsemble",
}


def method_train_required(method: Any) -> bool:
    """Check whether a method (or its wrapped base) needs a dedicated train split."""
    name = str(getattr(method, "name", ""))
    if name in _TRAIN_REQUIRED_NAMES:
        return True
    # ProjectedMethod wraps a base method — check that too
    base = getattr(method, "base", None)
    if base is not None:
        base_name = str(getattr(base, "name", ""))
        if base_name in _TRAIN_REQUIRED_NAMES:
            return True
    return False
