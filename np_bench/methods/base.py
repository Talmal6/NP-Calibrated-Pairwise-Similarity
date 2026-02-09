from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np

@dataclass(frozen=True)
class MethodResult:
    tpr: float
    fpr: float
    time_ms: float

class Method(Protocol):
    name: str
    needs_weights: bool
    needs_seed: bool

    def run(
        self,
        H0: np.ndarray,
        H1: np.ndarray,
        alpha: float,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> MethodResult:
        ...
