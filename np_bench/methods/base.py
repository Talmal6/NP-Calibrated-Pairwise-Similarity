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


class BaseMethod(ABC):
    """
    Base method:
      - fit on TRAIN
      - threshold from quantile(score(H0_train), 1-alpha)
      - eval on TEST
    """
    name: str = "BaseMethod"
    needs_weights: bool = False
    needs_seed: bool = False
    saved_vectors: np.ndarray | None = None
    


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

        # Threshold from H0_train
        s0_tr = self.score(H0_train)
        thresh = float(np.quantile(s0_tr, 1.0 - alpha))

        # Eval scores
        s0_ev  = self.score(H0_eval)
        s1_ev  = self.score(H1_eval)

        # Eval metrics
        if tie_mode == "gt":
            tpr = float(np.mean(s1_ev > thresh))
            fpr = float(np.mean(s0_ev > thresh))
        else:
            tpr = float(np.mean(s1_ev >= thresh))
            fpr = float(np.mean(s0_ev >= thresh))

        # Train metrics
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


class OnlineBaseMethod(BaseMethod):
    """
    Online method with training memory.
    Threshold computed from quantile(score(mem_H0), 1-alpha).
    """
    name: str = "OnlineBaseMethod"

    def __init__(
        self,
        *,
        mem_cap_H0: int = 2000,
        mem_cap_H1: int = 2000,
        update_mode: str = "refit",
        update_every: int = 1,
        hill_lr: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        if update_mode not in {"refit", "hill_climb"}:
            raise ValueError("update_mode must be 'refit' or 'hill_climb'")
        self.mem_cap_H0 = int(mem_cap_H0)
        self.mem_cap_H1 = int(mem_cap_H1)
        self.update_mode = update_mode
        self.update_every = int(update_every)
        self.hill_lr = float(hill_lr)
        self.rng = np.random.default_rng(seed)

        self.mem_H0: np.ndarray | None = None
        self.mem_H1: np.ndarray | None = None
        self.total_seen_H0 = 0
        self.total_seen_H1 = 0
        self.tau_np: float = float("inf")
        self.alpha: Optional[float] = None
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self._update_count = 0

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "OnlineBaseMethod":
        del weights
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.initialize(H0_train, H1_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            return np.zeros(X.shape[0], dtype=float)
        return X @ self.w + self.b

    def initialize(
        self,
        H0_init: np.ndarray,
        H1_init: np.ndarray,
    ) -> None:
        self.mem_H0 = np.array(H0_init, copy=True)
        self.mem_H1 = np.array(H1_init, copy=True)
        self.total_seen_H0 = int(H0_init.shape[0])
        self.total_seen_H1 = int(H1_init.shape[0])
        self.refit()

    def update(self, batch_X: np.ndarray, batch_y: np.ndarray) -> None:
        _ = self.score(batch_X)

        y = batch_y.astype(int)
        H0_mask = y == 0
        H1_mask = y == 1

        H0_batch = batch_X[H0_mask]
        H1_batch = batch_X[H1_mask]

        if H0_batch.size:
            self._reservoir_add(H0_batch, is_h0=True)

        if H1_batch.size:
            self._reservoir_add(H1_batch, is_h0=False)

        self._update_count += 1
        if self._update_count % self.update_every == 0:
            if self.update_mode == "refit":
                self.refit()
            else:
                self.hill_climb_step()

    def refit(self) -> None:
        if self.mem_H0 is None or self.mem_H1 is None:
            return
        if self.mem_H0.size == 0 or self.mem_H1.size == 0:
            self.w = np.zeros(self.mem_H0.shape[1], dtype=float)
            self.b = 0.0
            return
        mu0 = np.mean(self.mem_H0, axis=0)
        mu1 = np.mean(self.mem_H1, axis=0)
        self.w = mu1 - mu0
        self.b = -0.5 * float(np.dot(mu1 + mu0, self.w))

    def hill_climb_step(self) -> None:
        if self.mem_H0 is None or self.mem_H1 is None:
            return
        if self.mem_H0.size == 0 or self.mem_H1.size == 0:
            return
        mu0 = np.mean(self.mem_H0, axis=0)
        mu1 = np.mean(self.mem_H1, axis=0)
        target_w = mu1 - mu0
        target_b = -0.5 * float(np.dot(mu1 + mu0, target_w))
        if self.w is None:
            self.w = target_w
            self.b = target_b
        else:
            self.w = (1.0 - self.hill_lr) * self.w + self.hill_lr * target_w
            self.b = (1.0 - self.hill_lr) * self.b + self.hill_lr * target_b

    def recalibrate_threshold(self, alpha: float) -> float:
        """Recompute tau from training memory: quantile(score(mem_H0), 1-alpha)."""
        self.alpha = float(alpha)
        if self.mem_H0 is None or self.mem_H0.size == 0:
            self.tau_np = float("inf")
            return self.tau_np
        s0_tr = self.score(self.mem_H0)
        self.tau_np = float(np.quantile(s0_tr, 1.0 - self.alpha))
        return self.tau_np

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores >= self.tau_np).astype(int)

    def _reservoir_add(self, X: np.ndarray, *, is_h0: bool) -> None:
        if is_h0:
            cap = self.mem_cap_H0
            buf = self.mem_H0
            seen_attr = "total_seen_H0"
        else:
            cap = self.mem_cap_H1
            buf = self.mem_H1
            seen_attr = "total_seen_H1"

        if buf is None:
            buf = np.empty((0, X.shape[1]), dtype=X.dtype)

        seen = getattr(self, seen_attr)
        for row in X:
            seen += 1
            if buf.shape[0] < cap:
                buf = np.vstack([buf, row[None, :]])
                continue
            j = self.rng.integers(0, seen)
            if j < cap:
                buf[j, :] = row

        setattr(self, seen_attr, seen)
        if is_h0:
            self.mem_H0 = buf
        else:
            self.mem_H1 = buf


def _example_online_loop() -> None:
    rng = np.random.default_rng(7)
    n_init = 200
    n_stream = 500
    d = 5
    alpha = 0.1

    H0_init = rng.normal(0.0, 1.0, size=(n_init, d))
    H1_init = rng.normal(0.7, 1.0, size=(n_init, d))

    model = OnlineBaseMethod(
        mem_cap_H0=500,
        mem_cap_H1=500,
        update_mode="refit",
        update_every=5,
        seed=7,
    )
    model.initialize(H0_init, H1_init)
    model.recalibrate_threshold(alpha)

    for _ in range(0, n_stream, 50):
        X0 = rng.normal(0.0, 1.0, size=(25, d))
        X1 = rng.normal(0.7, 1.0, size=(25, d))
        X_batch = np.vstack([X0, X1])
        y_batch = np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0])])

        model.update(X_batch, y_batch)
        model.recalibrate_threshold(alpha)

    X_eval0 = rng.normal(0.0, 1.0, size=(200, d))
    X_eval1 = rng.normal(0.7, 1.0, size=(200, d))
    pred0 = model.predict(X_eval0)
    pred1 = model.predict(X_eval1)
    fpr = float(np.mean(pred0 == 1))
    tpr = float(np.mean(pred1 == 1))
    print("eval fpr=%.3f tpr=%.3f tau=%.3f" % (fpr, tpr, model.tau_np))


if __name__ == "__main__":
    _example_online_loop()
