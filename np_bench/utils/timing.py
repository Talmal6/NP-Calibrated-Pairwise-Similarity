from __future__ import annotations
import time
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")


def time_ms(fn: Callable[[], T], reps: int = 1, warmup: int = 0) -> Tuple[T, float]:
    """
    Time a callable and return (last_output, mean_ms_per_call).

    Notes:
      - warmup calls are executed but not timed.
      - reps times are averaged.
    """
    for _ in range(max(0, warmup)):
        _ = fn()

    t0 = time.perf_counter()
    out = None
    reps = max(1, int(reps))

    for _ in range(reps):
        out = fn()

    dt = (time.perf_counter() - t0) * 1000.0 / reps
    return out, float(dt)
