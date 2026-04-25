from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .base import BaseMethod
from .weighted_ensemble import EnsembleConfig, WeightedEnsembleMethod


@dataclass
class RegionalEnsembleConfig:
    alpha: float = 0.05
    ridge: float = 1e-3
    standardize: bool = False
    nonneg_simplex: bool = True
    eps: float = 1e-12
    meta_frac: float = 0.30
    tpr_tie_tol: float = 1e-4
    tie_break_entropy: bool = True
    k_shrink: float = 200.0
    min_region_h0: int = 30
    min_region_h1: int = 30

    def to_ensemble_config(self) -> EnsembleConfig:
        return EnsembleConfig(
            alpha=float(self.alpha),
            ridge=float(self.ridge),
            standardize=bool(self.standardize),
            nonneg_simplex=bool(self.nonneg_simplex),
            eps=float(self.eps),
            meta_frac=float(self.meta_frac),
            tpr_tie_tol=float(self.tpr_tie_tol),
            tie_break_entropy=bool(self.tie_break_entropy),
        )


class RegionalWeightedEnsembleMethod(WeightedEnsembleMethod):
    """Standalone shrunk regional ensemble over fixed judges.

    Judges are fit once on TRAIN (same as WeightedEnsembleMethod). Meta weights and
    tau are fit globally on pooled CALIB, then region-local parameters are fit on
    each region's CALIB subset and shrunk toward global parameters.
    """

    name: str = "RegionalWeightedEnsemble"
    needs_seed: bool = True
    needs_weights: bool = False
    input_space: str = "mixed"

    uses_internal_thresholds: bool = True
    requires_region_ids: bool = True

    def __init__(
        self,
        judges: Sequence[BaseMethod],
        *,
        config: Optional[RegionalEnsembleConfig] = None,
    ) -> None:
        self.r_cfg = config or RegionalEnsembleConfig()
        super().__init__(judges=judges, config=self.r_cfg.to_ensemble_config())

        self.global_meta_w: Optional[np.ndarray] = None
        self.global_tau: float = float("inf")

        self.meta_w_by_region: Dict[int, np.ndarray] = {}
        self.tau_by_region: Dict[int, float] = {}
        self.lambda_by_region: Dict[int, float] = {}
        self.local_fit_ok_by_region: Dict[int, bool] = {}

    def _tau_for_regions(self, region_ids: np.ndarray) -> np.ndarray:
        rid = np.asarray(region_ids, dtype=np.int64).reshape(-1)
        out = np.full(rid.shape[0], float(self.global_tau), dtype=np.float64)
        for i, r in enumerate(rid.tolist()):
            out[i] = float(self.tau_by_region.get(int(r), self.global_tau))
        return out

    def get_region_tau(self, region_id: int) -> float:
        return float(self.tau_by_region.get(int(region_id), self.global_tau))

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        alpha: Optional[float] = None,
        H0_calib: Optional[np.ndarray] = None,
        H1_calib: Optional[np.ndarray] = None,
        H0_train_alt: Optional[np.ndarray] = None,
        H1_train_alt: Optional[np.ndarray] = None,
        H0_calib_alt: Optional[np.ndarray] = None,
        H1_calib_alt: Optional[np.ndarray] = None,
        H0_calib_region_ids: Optional[np.ndarray] = None,
        H1_calib_region_ids: Optional[np.ndarray] = None,
        judge_input: Optional[Dict[str, str]] = None,
        tie_mode: str = "ge",
        guardrail: str = "none",
        guardrail_delta: float = 0.01,
        fit_context: str = "unknown",
    ) -> "RegionalWeightedEnsembleMethod":
        super().fit(
            H0_train,
            H1_train,
            weights=weights,
            seed=seed,
            alpha=alpha,
            H0_calib=H0_calib,
            H1_calib=H1_calib,
            H0_train_alt=H0_train_alt,
            H1_train_alt=H1_train_alt,
            H0_calib_alt=H0_calib_alt,
            H1_calib_alt=H1_calib_alt,
            judge_input=judge_input,
            tie_mode=tie_mode,
            guardrail=guardrail,
            guardrail_delta=guardrail_delta,
            fit_context=fit_context,
        )

        self.global_meta_w = None if self.meta_w is None else np.asarray(self.meta_w, dtype=np.float64).copy()
        self.global_tau = float(self.tau)
        self.meta_w_by_region = {}
        self.tau_by_region = {}
        self.lambda_by_region = {}
        self.local_fit_ok_by_region = {}

        if (
            H0_calib is None
            or H1_calib is None
            or H0_calib_region_ids is None
            or H1_calib_region_ids is None
            or self.global_meta_w is None
            or self.global_meta_w.size == 0
        ):
            print("[RegionalWeightedEnsemble] Region calib ids unavailable; using global-only fallback")
            return self

        rid0 = np.asarray(H0_calib_region_ids, dtype=np.int64).reshape(-1)
        rid1 = np.asarray(H1_calib_region_ids, dtype=np.int64).reshape(-1)
        if rid0.shape[0] != H0_calib.shape[0] or rid1.shape[0] != H1_calib.shape[0]:
            print("[RegionalWeightedEnsemble] Region-id length mismatch; using global-only fallback")
            return self

        Z0 = self._judge_matrix(H0_calib, H0_calib_alt)
        Z1 = self._judge_matrix(H1_calib, H1_calib_alt)
        Z0 = self._standardize_apply(Z0)
        Z1 = self._standardize_apply(Z1)

        all_regions = sorted(set(rid0.tolist()) | set(rid1.tolist()))
        print(
            "[RegionalWeightedEnsemble] "
            f"global_tau={self.global_tau:.4f} "
            f"regions={len(all_regions)} "
            f"k_shrink={float(self.r_cfg.k_shrink):.2f}"
        )
        top_g = int(np.argmax(self.global_meta_w))
        top_g_name = getattr(self.judges[top_g], "name", type(self.judges[top_g]).__name__)
        print(
            "[RegionalWeightedEnsemble] "
            f"global_top_judge={top_g_name} weight={float(self.global_meta_w[top_g]):.4f}"
        )

        for r in all_regions:
            m0 = rid0 == int(r)
            m1 = rid1 == int(r)
            n0 = int(np.sum(m0))
            n1 = int(np.sum(m1))

            fit_ok = n0 >= int(self.r_cfg.min_region_h0) and n1 >= int(self.r_cfg.min_region_h1)
            self.local_fit_ok_by_region[int(r)] = bool(fit_ok)

            if not fit_ok:
                self.lambda_by_region[int(r)] = 0.0
                self.meta_w_by_region[int(r)] = self.global_meta_w.copy()
                self.tau_by_region[int(r)] = float(self.global_tau)
                print(
                    "  [Region] "
                    f"rid={int(r)} n0={n0} n1={n1} "
                    "local_fit=no fallback=global lambda=0.0000"
                )
                continue

            Z0_r = Z0[m0]
            Z1_r = Z1[m1]
            w_local = self._fit_meta_weights_neyman_pearson(
                Z0_r,
                Z1_r,
                float(alpha if alpha is not None else self.r_cfg.alpha),
                tie_mode,
                guardrail,
                guardrail_delta,
            )

            s0_local = Z0_r @ w_local
            tau_local = self._select_tau(
                s0_local,
                alpha=float(alpha if alpha is not None else self.r_cfg.alpha),
                tie_mode=tie_mode,
                guardrail=guardrail,
                guardrail_delta=guardrail_delta,
            )
            if not np.isfinite(tau_local):
                tau_local = float(np.quantile(s0_local, 1.0 - float(alpha if alpha is not None else self.r_cfg.alpha)))

            n_r = float(n0 + n1)
            k = float(max(self.r_cfg.k_shrink, 1e-9))
            lam = float(n_r / (n_r + k))

            w_final = lam * np.asarray(w_local, dtype=np.float64) + (1.0 - lam) * self.global_meta_w
            if self.cfg.nonneg_simplex:
                w_final = np.maximum(w_final, 0.0)
                w_final = self._proj_simplex(w_final)

            tau_final = float(lam * float(tau_local) + (1.0 - lam) * float(self.global_tau))

            self.lambda_by_region[int(r)] = float(lam)
            self.meta_w_by_region[int(r)] = np.asarray(w_final, dtype=np.float64)
            self.tau_by_region[int(r)] = float(tau_final)

            top_idx = int(np.argmax(w_final))
            top_name = getattr(self.judges[top_idx], "name", type(self.judges[top_idx]).__name__)
            print(
                "  [Region] "
                f"rid={int(r)} n0={n0} n1={n1} "
                "local_fit=yes "
                f"lambda={lam:.4f} "
                f"local_tau={float(tau_local):.4f} "
                f"final_tau={float(tau_final):.4f} "
                f"top_judge={top_name} top_weight={float(w_final[top_idx]):.4f}"
            )

        return self

    def score(
        self,
        X: np.ndarray,
        X_alt: Optional[np.ndarray] = None,
        region_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.global_meta_w is None:
            if self.meta_w is None:
                return np.zeros(X.shape[0], dtype=np.float64)
            self.global_meta_w = np.asarray(self.meta_w, dtype=np.float64).copy()

        Z = self._judge_matrix(X, X_alt)
        Z = self._standardize_apply(Z)

        if region_ids is None:
            return Z @ self.global_meta_w

        rid = np.asarray(region_ids, dtype=np.int64).reshape(-1)
        if rid.shape[0] != Z.shape[0]:
            raise ValueError(
                f"RegionalWeightedEnsemble.score region_ids length mismatch: {rid.shape[0]} vs {Z.shape[0]}"
            )

        W = np.zeros((Z.shape[0], Z.shape[1]), dtype=np.float64)
        for i, r in enumerate(rid.tolist()):
            W[i] = self.meta_w_by_region.get(int(r), self.global_meta_w)
        return np.sum(Z * W, axis=1)

    def predict(
        self,
        X: np.ndarray,
        *,
        X_alt: Optional[np.ndarray] = None,
        region_ids: Optional[np.ndarray] = None,
        tie_mode: str = "ge",
    ) -> np.ndarray:
        s = np.asarray(self.score(X, X_alt=X_alt, region_ids=region_ids), dtype=np.float64).reshape(-1)
        if region_ids is None:
            tau_vec = np.full(s.shape[0], float(self.global_tau), dtype=np.float64)
        else:
            tau_vec = self._tau_for_regions(region_ids)

        if tie_mode == "gt":
            return (s > tau_vec).astype(np.int32)
        return (s >= tau_vec).astype(np.int32)
