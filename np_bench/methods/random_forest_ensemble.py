from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import BaseMethod
from .weighted_ensemble import EnsembleConfig, WeightedEnsembleMethod


@dataclass
class RandomForestEnsembleConfig:
    """Random-forest meta-model settings for judge-score ensembling."""

    n_estimators: int = 256
    max_depth: Optional[int] = 4
    min_samples_leaf: int = 5
    max_features: str | int | float | None = "sqrt"
    class_weight: str | Dict[int, float] | None = None
    n_jobs: int = 1


class RandomForestEnsembleMethod(WeightedEnsembleMethod):
    """
    Split-calibrated ensemble with a random-forest meta-model.

    This uses the same base judges, judge-score matrix, split separation,
    score normalization, and final NP threshold selection as WeightedEnsemble.
    Instead of a linear score-level combination, it fits a random forest on
    the META judge-score features and uses P(H1) as the final score.
    """

    name: str = "RandomForestEnsemble"
    needs_seed: bool = True
    needs_weights: bool = False
    input_space: str = "mixed"

    def __init__(
        self,
        judges: Sequence[BaseMethod],
        *,
        config: Optional[EnsembleConfig] = None,
        rf_config: Optional[RandomForestEnsembleConfig] = None,
    ) -> None:
        super().__init__(judges=judges, config=config)
        self.rf_cfg = rf_config or RandomForestEnsembleConfig()
        self.clf = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _fit_meta_random_forest(
        self,
        Z0_meta: np.ndarray,
        Z1_meta: np.ndarray,
        *,
        seed: int,
    ) -> None:
        if Z0_meta.ndim != 2 or Z1_meta.ndim != 2:
            raise ValueError("Z0_meta and Z1_meta must be 2D matrices")
        if Z0_meta.shape[1] != Z1_meta.shape[1]:
            raise ValueError("Z0_meta and Z1_meta must have the same number of columns")
        if Z0_meta.shape[0] == 0 or Z1_meta.shape[0] == 0:
            raise ValueError("RandomForestEnsemble requires non-empty META H0 and H1 splits")

        from sklearn.ensemble import RandomForestClassifier

        X_meta = np.vstack([Z0_meta, Z1_meta]).astype(np.float64, copy=False)
        y_meta = np.concatenate(
            [
                np.zeros(Z0_meta.shape[0], dtype=np.int32),
                np.ones(Z1_meta.shape[0], dtype=np.int32),
            ]
        )

        self.clf = RandomForestClassifier(
            n_estimators=int(self.rf_cfg.n_estimators),
            max_depth=self.rf_cfg.max_depth,
            min_samples_leaf=int(self.rf_cfg.min_samples_leaf),
            max_features=self.rf_cfg.max_features,
            class_weight=self.rf_cfg.class_weight,
            n_jobs=int(self.rf_cfg.n_jobs),
            random_state=int(seed),
        )
        self.clf.fit(X_meta, y_meta)

        importances = getattr(self.clf, "feature_importances_", None)
        self.feature_importances_ = (
            None
            if importances is None
            else np.asarray(importances, dtype=np.float64).reshape(-1)
        )

    def _score_normalized_matrix(self, Z: np.ndarray) -> np.ndarray:
        if self.clf is None:
            return np.zeros(Z.shape[0], dtype=np.float64)

        classes = np.asarray(getattr(self.clf, "classes_", []))
        if classes.size == 0:
            return np.zeros(Z.shape[0], dtype=np.float64)

        proba = np.asarray(self.clf.predict_proba(Z), dtype=np.float64)
        pos = np.flatnonzero(classes == 1)
        if pos.size == 0:
            return np.zeros(Z.shape[0], dtype=np.float64)
        return proba[:, int(pos[0])]

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
        judge_input: Optional[Dict[str, str]] = None,
        tie_mode: str = "ge",
        guardrail: str = "none",
        guardrail_delta: float = 0.01,
        fit_context: str = "unknown",
    ) -> "RandomForestEnsembleMethod":
        if alpha is None:
            alpha = self.cfg.alpha
        alpha = float(alpha)

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0,1)")

        self.judge_input = dict(judge_input or {})
        self.last_fit_context = str(fit_context)
        self.clf = None
        self.feature_importances_ = None
        self.meta_w = None

        alt_judges = [
            name for name, route in self.judge_input.items()
            if route == "alt"
        ]
        if alt_judges:
            print(
                "  [RandomForestEnsemble] Judges routed to 'alt' input: "
                + ", ".join(alt_judges)
            )

        used_external_calib = H0_calib is not None and H1_calib is not None
        self.last_used_external_calib = bool(used_external_calib)
        base_seed = seed if seed is not None else self.cfg.random_seed

        if used_external_calib:
            H0_judge = H0_train
            H1_judge = H1_train
            H0_judge_alt = H0_train_alt
            H1_judge_alt = H1_train_alt

            H0_meta_idx, H0_risk_idx = self._split_two_indices(
                H0_calib.shape[0],
                seed=base_seed,
                risk_frac=self.cfg.external_risk_calib_frac,
            )
            H1_meta_idx, H1_risk_idx = self._split_two_indices(
                H1_calib.shape[0],
                seed=base_seed + 1,
                risk_frac=self.cfg.external_risk_calib_frac,
            )

            H0_meta = H0_calib[H0_meta_idx]
            H1_meta = H1_calib[H1_meta_idx]
            H0_risk = H0_calib[H0_risk_idx]
            H1_risk = H1_calib[H1_risk_idx]

            H0_meta_alt = self._take_optional(H0_calib_alt, H0_meta_idx)
            H1_meta_alt = self._take_optional(H1_calib_alt, H1_meta_idx)
            H0_risk_alt = self._take_optional(H0_calib_alt, H0_risk_idx)
            H1_risk_alt = self._take_optional(H1_calib_alt, H1_risk_idx)
        else:
            H0_j_idx, H0_m_idx, H0_r_idx = self._split_three_indices(
                H0_train.shape[0],
                seed=base_seed,
                meta_frac=self.cfg.meta_frac,
                risk_frac=self.cfg.risk_calib_frac,
            )
            H1_j_idx, H1_m_idx, H1_r_idx = self._split_three_indices(
                H1_train.shape[0],
                seed=base_seed + 1,
                meta_frac=self.cfg.meta_frac,
                risk_frac=self.cfg.risk_calib_frac,
            )

            H0_judge = H0_train[H0_j_idx]
            H1_judge = H1_train[H1_j_idx]
            H0_meta = H0_train[H0_m_idx]
            H1_meta = H1_train[H1_m_idx]
            H0_risk = H0_train[H0_r_idx]
            H1_risk = H1_train[H1_r_idx]

            H0_judge_alt = self._take_optional(H0_train_alt, H0_j_idx)
            H1_judge_alt = self._take_optional(H1_train_alt, H1_j_idx)
            H0_meta_alt = self._take_optional(H0_train_alt, H0_m_idx)
            H1_meta_alt = self._take_optional(H1_train_alt, H1_m_idx)
            H0_risk_alt = self._take_optional(H0_train_alt, H0_r_idx)
            H1_risk_alt = self._take_optional(H1_train_alt, H1_r_idx)

        if fit_context in {"local", "matched_global_on_local"}:
            print(
                "  [RandomForestEnsemble][DEBUG] "
                f"fit_context={fit_context} "
                f"external_calib_used={bool(used_external_calib)} "
                f"H0_train={int(H0_train.shape[0])} "
                f"H1_train={int(H1_train.shape[0])} "
                f"H0_judge={int(H0_judge.shape[0])} "
                f"H1_judge={int(H1_judge.shape[0])} "
                f"H0_meta={int(H0_meta.shape[0])} "
                f"H1_meta={int(H1_meta.shape[0])} "
                f"H0_risk={int(H0_risk.shape[0])} "
                f"H1_risk={int(H1_risk.shape[0])}"
            )

        if H0_meta.shape[0] == 0 or H1_meta.shape[0] == 0:
            raise ValueError(
                "RandomForestEnsembleMethod requires non-empty META H0 and H1 splits"
            )
        if H0_risk.shape[0] == 0:
            raise ValueError(
                "RandomForestEnsembleMethod requires non-empty RISK_CALIB H0 split"
            )

        for judge in self.judges:
            H0_use = self._select_X(judge, H0_judge, H0_judge_alt, self.judge_input)
            H1_use = self._select_X(judge, H1_judge, H1_judge_alt, self.judge_input)

            kwargs = {}
            if weights is not None and getattr(judge, "needs_weights", False):
                kwargs["weights"] = weights
            if seed is not None and getattr(judge, "needs_seed", False):
                kwargs["seed"] = seed

            try:
                judge.fit(H0_use, H1_use, **kwargs)
            except TypeError:
                try:
                    judge.fit(H0_use, H1_use)
                except Exception as exc:
                    print(
                        f"  [RandomForestEnsemble][Warning] "
                        f"Judge {self._judge_name(judge)} failed during fit: {exc}"
                    )
            except Exception as exc:
                print(
                    f"  [RandomForestEnsemble][Warning] "
                    f"Judge {self._judge_name(judge)} failed during fit: {exc}"
                )

        Z0_meta_raw = self._judge_matrix(H0_meta, H0_meta_alt)
        Z1_meta_raw = self._judge_matrix(H1_meta, H1_meta_alt)

        self._normalization_fit(Z0_meta_raw, Z1_meta_raw)
        Z0_meta = self._normalization_apply(Z0_meta_raw)
        Z1_meta = self._normalization_apply(Z1_meta_raw)

        self._fit_meta_random_forest(Z0_meta, Z1_meta, seed=int(base_seed))

        Z0_risk = self._normalization_apply(
            self._judge_matrix(H0_risk, H0_risk_alt)
        )
        s0_risk = self._score_normalized_matrix(Z0_risk)

        self.tau = self._select_tau(
            s0_risk,
            alpha=alpha,
            tie_mode=tie_mode,
            guardrail=guardrail,
            guardrail_delta=guardrail_delta,
        )

        s0_meta = self._score_normalized_matrix(Z0_meta)
        s1_meta = self._score_normalized_matrix(Z1_meta)
        tau_meta = self._select_tau(
            s0_meta,
            alpha=alpha,
            tie_mode=tie_mode,
            guardrail=guardrail,
            guardrail_delta=guardrail_delta,
        )

        if np.isfinite(tau_meta):
            if tie_mode == "gt":
                self.last_meta_tpr = float(np.mean(s1_meta > tau_meta))
                self.last_meta_fpr = float(np.mean(s0_meta > tau_meta))
            else:
                self.last_meta_tpr = float(np.mean(s1_meta >= tau_meta))
                self.last_meta_fpr = float(np.mean(s0_meta >= tau_meta))
        else:
            self.last_meta_tpr = float("nan")
            self.last_meta_fpr = float("nan")

        Z1_risk = self._normalization_apply(
            self._judge_matrix(H1_risk, H1_risk_alt)
        ) if H1_risk.shape[0] > 0 else np.empty((0, len(self.judges)))
        s1_risk = self._score_normalized_matrix(Z1_risk) if Z1_risk.size else np.empty(0)

        if np.isfinite(self.tau):
            if tie_mode == "gt":
                self.last_risk_calib_fpr = float(np.mean(s0_risk > self.tau))
                self.last_risk_calib_tpr = (
                    float(np.mean(s1_risk > self.tau))
                    if s1_risk.size
                    else float("nan")
                )
            else:
                self.last_risk_calib_fpr = float(np.mean(s0_risk >= self.tau))
                self.last_risk_calib_tpr = (
                    float(np.mean(s1_risk >= self.tau))
                    if s1_risk.size
                    else float("nan")
                )
        else:
            self.last_risk_calib_fpr = float("nan")
            self.last_risk_calib_tpr = float("nan")

        calib_msg = (
            "external calib split into META/RISK_CALIB"
            if used_external_calib
            else "internal TRAIN split into JUDGE/META/RISK_CALIB"
        )

        print(
            f"[RandomForestEnsemble] Fitted split-calibrated RF ensemble "
            f"(alpha={alpha:.4f}, {calib_msg}, "
            f"normalization={self._normalization_mode()}, "
            f"n_estimators={int(self.rf_cfg.n_estimators)}, "
            f"max_depth={self.rf_cfg.max_depth}):"
        )
        print(
            f"  tau={self.tau:.6f}, "
            f"META: TPR={self.last_meta_tpr:.4f}, FPR={self.last_meta_fpr:.4f}, "
            f"RISK_CALIB: TPR={self.last_risk_calib_tpr:.4f}, "
            f"FPR={self.last_risk_calib_fpr:.4f}"
        )

        if self.feature_importances_ is not None:
            for judge, imp in zip(self.judges, self.feature_importances_):
                print(f"  {self._judge_name(judge):24s}: importance={float(imp):.6f}")

        return self

    def score(
        self,
        X: np.ndarray,
        X_alt: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.clf is None:
            return np.zeros(X.shape[0], dtype=np.float64)

        Z = self._judge_matrix(X, X_alt)
        Z = self._normalization_apply(Z)
        return self._score_normalized_matrix(Z)
