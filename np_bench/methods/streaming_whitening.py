from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from np_bench.thresholding import select_np_threshold

from .whitened_cosine import WhitenedCosineMethod, _l2_normalize
from .whitening_core import (
    RankMode,
    WhiteningType,
    _retained_eigenpairs,
    _sorted_eigh,
)


@dataclass(frozen=True)
class WhiteningSnapshot:
    version: int
    W: np.ndarray
    mean: np.ndarray
    selected_rank: int
    whitening_type: WhiteningType
    samples_seen: int
    effective_n: float
    diagnostics: dict[str, float]


class EWMACovarianceAccumulator:
    def __init__(
        self,
        dim: int | None = None,
        *,
        forgetting_half_life_samples: int | None = 10000,
    ) -> None:
        if forgetting_half_life_samples is not None and forgetting_half_life_samples <= 0:
            raise ValueError("forgetting_half_life_samples must be positive or None.")
        self.dim = None if dim is None else int(dim)
        self.forgetting_half_life_samples = forgetting_half_life_samples
        self.samples_seen = 0
        self.sum_weights = 0.0
        self.sum_weights_sq = 0.0
        self.mean_: np.ndarray | None = None
        self.M2_: np.ndarray | None = None

    def update_batch(self, X: np.ndarray) -> None:
        Xf = np.asarray(X, dtype=np.float64)
        if Xf.ndim != 2:
            raise ValueError(f"X must have shape (n, d). Got {Xf.shape}.")
        if Xf.shape[0] == 0:
            return
        if self.dim is None:
            self.dim = int(Xf.shape[1])
        elif Xf.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: got {Xf.shape[1]}, expected {self.dim}.")

        m = int(Xf.shape[0])
        mu_b = Xf.mean(axis=0)
        Xc = Xf - mu_b[None, :]
        M2_b = Xc.T @ Xc
        w_b = float(m)

        if self.sum_weights <= 0.0 or self.mean_ is None or self.M2_ is None:
            self.mean_ = mu_b
            self.M2_ = M2_b
            self.sum_weights = w_b
            self.sum_weights_sq = w_b
            self.samples_seen += m
            return

        if self.forgetting_half_life_samples is None:
            decay = 1.0
        else:
            decay = 2.0 ** (-m / float(self.forgetting_half_life_samples))

        w_o = decay * self.sum_weights
        M2_o = decay * self.M2_
        delta = mu_b - self.mean_
        w_n = w_o + w_b
        if w_n <= 0.0:
            self.reset()
            self.update_batch(Xf)
            return

        self.mean_ = self.mean_ + delta * (w_b / w_n)
        self.M2_ = M2_o + M2_b + np.outer(delta, delta) * (w_o * w_b / w_n)
        self.sum_weights = w_n
        self.sum_weights_sq = decay * decay * self.sum_weights_sq + w_b
        self.samples_seen += m

    def covariance(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self.dim is None or self.mean_ is None or self.M2_ is None or self.sum_weights <= 0.0:
            raise RuntimeError("Cannot compute covariance before update_batch(...).")

        denom = self.sum_weights - (self.sum_weights_sq / self.sum_weights)
        if denom <= 0.0:
            if self.forgetting_half_life_samples is None:
                denom = max(self.sum_weights - 1.0, 1.0)
            else:
                denom = max(self.sum_weights, 1.0)
        cov = self.M2_ / denom
        cov = 0.5 * (cov + cov.T)
        return cov, self.mean_.copy(), self.effective_n

    @property
    def effective_n(self) -> float:
        if self.sum_weights_sq <= 0.0:
            return 0.0
        return float((self.sum_weights * self.sum_weights) / self.sum_weights_sq)

    def reset(self) -> None:
        self.samples_seen = 0
        self.sum_weights = 0.0
        self.sum_weights_sq = 0.0
        self.mean_ = None
        self.M2_ = None


class PageHinkleyDriftDetector:
    def __init__(
        self,
        *,
        delta: float = 0.01,
        threshold: float = 25.0,
        min_samples: int = 512,
        cooldown_samples: int = 10000,
    ) -> None:
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.cooldown_samples = int(cooldown_samples)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.min_cumulative = 0.0
        self.last_alarm_sample = -10**18
        self.alarm_count = 0

    def update(self, value: float, *, sample_index: int) -> bool:
        if not np.isfinite(value):
            return False
        self.n += 1
        self.mean += (float(value) - self.mean) / self.n
        self.cumulative += float(value) - self.mean - self.delta
        self.min_cumulative = min(self.min_cumulative, self.cumulative)

        if self.n < self.min_samples:
            return False
        if sample_index - self.last_alarm_sample < self.cooldown_samples:
            return False
        if (self.cumulative - self.min_cumulative) <= self.threshold:
            return False

        self.last_alarm_sample = int(sample_index)
        self.alarm_count += 1
        self.cumulative = 0.0
        self.min_cumulative = 0.0
        return True

    def diagnostics(self) -> dict[str, float]:
        return {
            "page_hinkley_n": float(self.n),
            "page_hinkley_mean": float(self.mean),
            "page_hinkley_cumulative": float(self.cumulative),
            "page_hinkley_min_cumulative": float(self.min_cumulative),
            "page_hinkley_alarm_count": float(self.alarm_count),
        }


class RollingH0CalibrationBuffer:
    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self._data: np.ndarray | None = None

    def add_batch(self, X: np.ndarray) -> None:
        Xf = np.asarray(X)
        if Xf.ndim != 2:
            raise ValueError(f"X must have shape (n, d). Got {Xf.shape}.")
        if Xf.shape[0] == 0:
            return
        if self._data is None:
            self._data = np.array(Xf[-self.capacity :], copy=True)
            return
        if Xf.shape[1] != self._data.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: got {Xf.shape[1]}, expected {self._data.shape[1]}.")
        merged = np.concatenate([self._data, Xf], axis=0)
        self._data = np.array(merged[-self.capacity :], copy=True)

    def values(self) -> np.ndarray:
        if self._data is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.array(self._data, copy=True)

    def __len__(self) -> int:
        return 0 if self._data is None else int(self._data.shape[0])


def _apply_shrinkage(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1].")
    out = np.asarray(cov, dtype=np.float64)
    if shrinkage > 0.0:
        diag = np.diag(np.diag(out))
        out = (1.0 - shrinkage) * out + shrinkage * diag
    return 0.5 * (out + out.T)


def _eigen_scale(vals: np.ndarray, gamma: float) -> np.ndarray:
    if vals.size == 0:
        return vals.astype(np.float64, copy=True)
    if gamma == 0.5:
        return 1.0 / np.sqrt(vals)
    return np.power(vals, -float(gamma))


def _zca_from_eigenpairs_gamma(vals: np.ndarray, vecs: np.ndarray, d: int, gamma: float) -> np.ndarray:
    if vals.size == 0:
        return np.zeros((d, d), dtype=np.float64)
    scale = _eigen_scale(vals, gamma)
    return (vecs * scale[None, :]) @ vecs.T


def _pca_from_eigenpairs_gamma(vals: np.ndarray, vecs: np.ndarray, d: int, gamma: float) -> np.ndarray:
    if vals.size == 0:
        return np.zeros((0, d), dtype=np.float64)
    scale = _eigen_scale(vals, gamma)
    return scale[:, None] * vecs.T


def _compute_whitening_from_covariance(
    cov: np.ndarray,
    mean: np.ndarray,
    *,
    n: int,
    whitening_type: WhiteningType,
    abs_eps: float,
    rel_eps: float,
    max_rank: Optional[int],
    rank_mode: RankMode,
    explained_variance: float,
    shrinkage: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    if whitening_type not in {"zca", "pca", "zca_cor", "pca_cor"}:
        raise ValueError(f"Unknown whitening_type: {whitening_type!r}")
    if not 0.0 < gamma <= 0.5:
        raise ValueError("gamma must satisfy 0 < gamma <= 0.5.")

    cov_f = _apply_shrinkage(cov, shrinkage)
    d = int(cov_f.shape[0])

    if whitening_type in {"zca", "pca"}:
        eigvals_desc, eigvecs_desc = _sorted_eigh(cov_f)
        vals, vecs, rank = _retained_eigenpairs(
            eigvals_desc,
            eigvecs_desc,
            n=n,
            d=d,
            max_rank=max_rank,
            rank_mode=rank_mode,
            explained_variance=explained_variance,
            abs_eps=abs_eps,
            rel_eps=rel_eps,
        )
        if whitening_type == "zca":
            W = _zca_from_eigenpairs_gamma(vals, vecs, d, gamma)
        else:
            W = _pca_from_eigenpairs_gamma(vals, vecs, d, gamma)
    else:
        diag = np.maximum(np.diag(cov_f), 0.0)
        max_diag = max(float(np.max(diag)), 0.0) if diag.size else 0.0
        diag_threshold = max(abs_eps, rel_eps * max_diag)
        inv_std = np.zeros(d, dtype=np.float64)
        valid_std = diag > diag_threshold
        inv_std[valid_std] = 1.0 / np.sqrt(diag[valid_std])

        corr = (inv_std[:, None] * cov_f) * inv_std[None, :]
        corr = 0.5 * (corr + corr.T)
        eigvals_desc, eigvecs_desc = _sorted_eigh(corr)
        vals, vecs, rank = _retained_eigenpairs(
            eigvals_desc,
            eigvecs_desc,
            n=n,
            d=d,
            max_rank=max_rank,
            rank_mode=rank_mode,
            explained_variance=explained_variance,
            abs_eps=abs_eps,
            rel_eps=rel_eps,
        )
        if whitening_type == "zca_cor":
            W = _zca_from_eigenpairs_gamma(vals, vecs, d, gamma) @ np.diag(inv_std)
        else:
            W = _pca_from_eigenpairs_gamma(vals, vecs, d, gamma) * inv_std[None, :]

    max_eigval = max(float(eigvals_desc[0]), 0.0) if eigvals_desc.size else 0.0
    min_retained = float(np.min(vals)) if vals.size else 0.0
    condition_number = float(max_eigval / max(min_retained, abs_eps)) if max_eigval > 0.0 else 0.0
    valid_vals = eigvals_desc[eigvals_desc > max(abs_eps, rel_eps * max_eigval)]
    total_valid = float(np.sum(valid_vals)) if valid_vals.size else 0.0
    retained_var = float(np.sum(vals)) if vals.size else 0.0
    explained_retained = retained_var / total_valid if total_valid > 0.0 else 0.0
    diagnostics = {
        "selected_rank": float(rank),
        "max_eigenvalue": float(max_eigval),
        "min_retained_eigenvalue": float(min_retained),
        "condition_number": float(condition_number),
        "explained_variance_retained": float(explained_retained),
    }
    return W, np.asarray(mean, dtype=np.float64).reshape(-1), rank, diagnostics


class StreamingWhitening:
    def __init__(
        self,
        *,
        whitening_type: WhiteningType = "zca",
        abs_eps: float = 1e-6,
        rel_eps: float = 1e-6,
        max_rank: int | None = 128,
        rank_mode: RankMode = "explained_variance",
        explained_variance: float = 0.99,
        shrinkage: float = 0.0,
        gamma: float = 0.5,
        forgetting_half_life_samples: int | None = 10000,
        refresh_every_samples: int = 2500,
        min_refresh_samples: int = 2048,
        drift_detector: str = "page_hinkley",
    ) -> None:
        if whitening_type not in {"zca", "pca", "zca_cor", "pca_cor"}:
            raise ValueError(f"Unknown whitening_type: {whitening_type!r}")
        if not 0.0 < gamma <= 0.5:
            raise ValueError("gamma must satisfy 0 < gamma <= 0.5.")
        if refresh_every_samples <= 0:
            raise ValueError("refresh_every_samples must be positive.")
        if min_refresh_samples <= 0:
            raise ValueError("min_refresh_samples must be positive.")
        if drift_detector not in {"page_hinkley", "none"}:
            raise ValueError("drift_detector must be 'page_hinkley' or 'none'.")

        self.whitening_type = whitening_type
        self.abs_eps = float(abs_eps)
        self.rel_eps = float(rel_eps)
        self.max_rank = max_rank
        self.rank_mode = rank_mode
        self.explained_variance = float(explained_variance)
        self.shrinkage = float(shrinkage)
        self.gamma = float(gamma)
        self.refresh_every_samples = int(refresh_every_samples)
        self.min_refresh_samples = int(min_refresh_samples)
        self.accumulator = EWMACovarianceAccumulator(
            forgetting_half_life_samples=forgetting_half_life_samples,
        )
        self.detector = PageHinkleyDriftDetector() if drift_detector == "page_hinkley" else None
        self.snapshot: WhiteningSnapshot | None = None
        self.version = 0
        self.last_refresh_samples = 0
        self.last_refresh_trigger = "none"
        self._last_batch_log_norm_ratio = float("nan")

    def initialize(self, X: np.ndarray) -> int:
        self.accumulator.reset()
        if self.detector is not None:
            self.detector.reset()
        self.snapshot = None
        self.version = 0
        self.last_refresh_samples = 0
        self.accumulator.update_batch(X)
        return self.refresh_transform(force=True)

    def update_batch(self, X: np.ndarray) -> bool:
        before = self.accumulator.samples_seen
        self.accumulator.update_batch(X)
        if self.snapshot is None:
            return False

        Xf = np.asarray(X, dtype=np.float64)
        drift_alarm = False
        if Xf.ndim == 2 and Xf.shape[0] > 0:
            Z = self.transform(Xf, center=True)
            norm2 = np.sum(Z * Z, axis=1)
            denom = max(float(self.snapshot.selected_rank), self.abs_eps)
            log_ratio = float(np.mean(np.log((norm2 + self.abs_eps) / denom)))
            self._last_batch_log_norm_ratio = log_ratio
            if self.detector is not None:
                drift_alarm = self.detector.update(
                    log_ratio,
                    sample_index=self.accumulator.samples_seen,
                )

        due = (self.accumulator.samples_seen - self.last_refresh_samples) >= self.refresh_every_samples
        if not drift_alarm and not due:
            return False
        if self.accumulator.effective_n < self.min_refresh_samples:
            return False

        self.last_refresh_trigger = "drift" if drift_alarm else "schedule"
        old_version = self.version
        self.refresh_transform(force=True)
        return self.version != old_version and self.accumulator.samples_seen > before

    def refresh_transform(self, *, force: bool = False) -> int:
        if self.snapshot is not None and not force:
            due = (self.accumulator.samples_seen - self.last_refresh_samples) >= self.refresh_every_samples
            if not due:
                return self.snapshot.version
        if self.accumulator.effective_n < self.min_refresh_samples and self.snapshot is not None:
            return self.snapshot.version

        cov, mean, effective_n = self.accumulator.covariance()
        W, mean, rank, diagnostics = _compute_whitening_from_covariance(
            cov,
            mean,
            n=max(1, int(round(effective_n))),
            whitening_type=self.whitening_type,
            abs_eps=self.abs_eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank,
            rank_mode=self.rank_mode,
            explained_variance=self.explained_variance,
            shrinkage=self.shrinkage,
            gamma=self.gamma,
        )
        diagnostics.update(
            {
                "effective_n": float(effective_n),
                "samples_seen": float(self.accumulator.samples_seen),
                "last_batch_log_norm_ratio": float(self._last_batch_log_norm_ratio),
            }
        )
        if self.detector is not None:
            diagnostics.update(self.detector.diagnostics())

        self.version += 1
        self.last_refresh_samples = self.accumulator.samples_seen
        self.snapshot = WhiteningSnapshot(
            version=self.version,
            W=W,
            mean=mean,
            selected_rank=rank,
            whitening_type=self.whitening_type,
            samples_seen=self.accumulator.samples_seen,
            effective_n=float(effective_n),
            diagnostics=diagnostics,
        )
        return self.version

    def transform(self, X: np.ndarray, *, center: bool = False) -> np.ndarray:
        if self.snapshot is None:
            raise RuntimeError("Missing whitening transform. Call initialize(...) first.")
        Xf = np.asarray(X, dtype=np.float64)
        if Xf.ndim != 2:
            raise ValueError(f"X must have shape (n, d). Got {Xf.shape}.")
        if center:
            Xf = Xf - self.snapshot.mean[None, :]
        return Xf @ self.snapshot.W.T

    def diagnostics(self) -> dict[str, float]:
        out = {
            "version": float(self.version),
            "samples_seen": float(self.accumulator.samples_seen),
            "effective_n": float(self.accumulator.effective_n),
            "last_refresh_samples": float(self.last_refresh_samples),
            "last_batch_log_norm_ratio": float(self._last_batch_log_norm_ratio),
        }
        if self.snapshot is not None:
            out.update(self.snapshot.diagnostics)
        return out


class StreamingWhitenedCosineMethod(WhitenedCosineMethod):
    def __init__(
        self,
        name: str = "StreamingWhitenedCosine",
        *,
        eps: float = 1e-6,
        rel_eps: float = 1e-6,
        max_rank: Optional[int] = 128,
        rank_mode: RankMode = "explained_variance",
        explained_variance: float = 0.99,
        norm_eps: float = 1e-12,
        whitening_type: WhiteningType = "zca",
        shrinkage: float = 0.0,
        gamma: float = 0.5,
        forgetting_half_life_samples: int | None = 10000,
        refresh_every_samples: int = 2500,
        min_refresh_samples: int = 2048,
        drift_detector: str = "page_hinkley",
        h0_buffer_capacity: int = 4096,
        min_h0_for_threshold: int = 512,
        abs_eps: Optional[float] = None,
        pca_whiten_abs_eps: Optional[float] = None,
        pca_whiten_rel_eps: Optional[float] = None,
        pca_whiten_max_rank: Optional[int] = None,
        pca_whiten_rank_mode: Optional[RankMode] = None,
        pca_whiten_explained_variance: Optional[float] = None,
        pca_whiten_norm_eps: Optional[float] = None,
    ) -> None:
        super().__init__(
            name=name,
            eps=eps,
            rel_eps=rel_eps,
            max_rank=max_rank,
            rank_mode=rank_mode,
            explained_variance=explained_variance,
            norm_eps=norm_eps,
            whitening_type=whitening_type,
            shrinkage=shrinkage,
            abs_eps=abs_eps,
            pca_whiten_abs_eps=pca_whiten_abs_eps,
            pca_whiten_rel_eps=pca_whiten_rel_eps,
            pca_whiten_max_rank=pca_whiten_max_rank,
            pca_whiten_rank_mode=pca_whiten_rank_mode,
            pca_whiten_explained_variance=pca_whiten_explained_variance,
            pca_whiten_norm_eps=pca_whiten_norm_eps,
        )
        self.streaming_whitening = StreamingWhitening(
            whitening_type=self.whitening_type,
            abs_eps=self.eps,
            rel_eps=self.rel_eps,
            max_rank=self.max_rank,
            rank_mode=self.rank_mode,
            explained_variance=self.explained_variance,
            shrinkage=self.shrinkage,
            gamma=gamma,
            forgetting_half_life_samples=forgetting_half_life_samples,
            refresh_every_samples=refresh_every_samples,
            min_refresh_samples=min_refresh_samples,
            drift_detector=drift_detector,
        )
        self.h0_calib_buffer = RollingH0CalibrationBuffer(capacity=h0_buffer_capacity)
        self.min_h0_for_threshold = int(min_h0_for_threshold)
        self.transform_version_ = 0
        self._pending_snapshot: WhiteningSnapshot | None = None
        self._last_activation_ok = False
        self._last_activation_reason = "not_initialized"

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: Optional[float] = None,
        tie_mode: str = "ge",
        guardrail: str = "none",
        guardrail_delta: float = 0.01,
    ) -> "StreamingWhitenedCosineMethod":
        super().fit(H0_train, H1_train, weights=weights, seed=seed)
        all_data = np.concatenate([H0_train, H1_train], axis=0)
        self.streaming_whitening.initialize(all_data)
        snapshot = self.streaming_whitening.snapshot
        if snapshot is None:
            raise RuntimeError("Streaming whitening failed to initialize.")
        self.W = snapshot.W
        self.mean_ = snapshot.mean
        self.selected_rank_ = snapshot.selected_rank
        self.transform_version_ = snapshot.version
        self._refit_whitened()
        self.h0_calib_buffer.add_batch(H0_train)
        self._last_activation_ok = True
        self._last_activation_reason = "fit"
        if alpha is not None:
            self.refresh_transform_and_threshold(
                alpha=float(alpha),
                tie_mode=tie_mode,
                guardrail=guardrail,
                guardrail_delta=guardrail_delta,
            )
        return self

    def _linear_form_for_W(self, W: np.ndarray) -> tuple[np.ndarray, float]:
        if self.mem_H0 is None or self.mem_H1 is None:
            raise RuntimeError("Missing stored training data. Call fit(...) first.")
        if self.whitening_type == "zca":
            WH0 = self.mem_H0 @ W
            WH1 = self.mem_H1 @ W
        else:
            WH0 = self.mem_H0 @ W.T
            WH1 = self.mem_H1 @ W.T
        mu0 = WH0.mean(axis=0)
        mu1 = WH1.mean(axis=0)
        d_w = mu1 - mu0
        if self.whitening_type == "zca":
            w = W @ d_w
        else:
            w = W.T @ d_w
        b = -0.5 * float((mu0 + mu1) @ d_w)
        return np.asarray(w, dtype=np.float64), float(b)

    def partial_fit(
        self,
        batch: np.ndarray,
        *,
        H0_calib_batch: np.ndarray | None = None,
    ) -> dict[str, object]:
        if H0_calib_batch is not None:
            self.h0_calib_buffer.add_batch(H0_calib_batch)
        refreshed = self.streaming_whitening.update_batch(batch)
        if refreshed and self.alpha is not None:
            self.refresh_transform_and_threshold(alpha=self.alpha)
        return {
            "refreshed": bool(refreshed),
            "active_version": int(self.transform_version_),
            "pending_version": int(self.streaming_whitening.version),
            "activation_ok": bool(self._last_activation_ok),
            "activation_reason": self._last_activation_reason,
        }

    def refresh_transform_and_threshold(
        self,
        *,
        alpha: float,
        tie_mode: str = "ge",
        guardrail: str = "none",
        guardrail_delta: float = 0.01,
    ) -> int:
        self.alpha = float(alpha)
        old_version = int(self.transform_version_)
        snapshot = self.streaming_whitening.snapshot
        if snapshot is None or snapshot.version == self.transform_version_:
            candidate_version = self.streaming_whitening.refresh_transform(force=True)
            snapshot = self.streaming_whitening.snapshot
        else:
            candidate_version = snapshot.version
        if snapshot is None:
            self._last_activation_ok = False
            self._last_activation_reason = "missing_snapshot"
            return old_version
        self._pending_snapshot = snapshot

        H0 = self.h0_calib_buffer.values()
        if H0.shape[0] < self.min_h0_for_threshold:
            if not np.isfinite(self.tau_np):
                self.tau_np = float("inf")
            self._last_activation_ok = False
            self._last_activation_reason = "insufficient_h0"
            return old_version

        w, b = self._linear_form_for_W(snapshot.W)
        s0 = H0 @ w + b
        if not np.all(np.isfinite(s0)):
            self._last_activation_ok = False
            self._last_activation_reason = "nonfinite_h0_scores"
            return old_version

        tau = select_np_threshold(
            s0,
            alpha=self.alpha,
            tie_mode=tie_mode,
            guardrail=guardrail,
            guardrail_delta=guardrail_delta,
        )
        self.W = snapshot.W
        self.mean_ = snapshot.mean
        self.selected_rank_ = snapshot.selected_rank
        self.w = w
        self.b = b
        self.tau_np = float(tau)
        self.transform_version_ = int(candidate_version)
        self._last_activation_ok = True
        self._last_activation_reason = "activated"
        return self.transform_version_

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Missing whitening matrix W. Call fit(...) first.")
        WA = _l2_normalize(np.asarray(A, dtype=np.float64) @ self.W.T, eps=self.norm_eps)
        WB = _l2_normalize(np.asarray(B, dtype=np.float64) @ self.W.T, eps=self.norm_eps)
        return np.sum(WA * WB, axis=1)

    def diagnostics(self) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = self.streaming_whitening.diagnostics()
        out.update(
            {
                "active_version": float(self.transform_version_),
                "tau_np": float(self.tau_np),
                "h0_buffer_size": float(len(self.h0_calib_buffer)),
                "activation_ok": bool(self._last_activation_ok),
                "activation_reason": self._last_activation_reason,
            }
        )
        return out
