# NeighborCache/np_bench/methods/weighted_ensemble.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import BaseMethod, MethodResult


@dataclass
class EnsembleConfig:
    """
    Configuration for NP-calibrated weighted score ensembling.

    The important design choice is split separation:

    - Base judges are fitted on TRAIN.
    - Ensemble weights are selected on META.
    - The final deployment threshold tau is selected on a disjoint RISK_CALIB H0 split.

    This avoids using the same calibration examples both to choose meta-weights
    and to select the final Neyman--Pearson threshold.
    """

    alpha: float = 0.05

    # Kept for compatibility/future smooth objectives. The current default
    # meta-objective is NP-TPR candidate search, so ridge is not used directly.
    ridge: float = 1e-3

    # Legacy option. Prefer score_normalization="h0_cdf" instead.
    standardize: bool = False

    # One of:
    #   "none"   : use raw judge scores
    #   "zscore" : z-score normalize judge scores using META scores
    #   "h0_cdf" : convert each judge score to its empirical H0 CDF percentile
    #
    # h0_cdf is recommended because it puts all judges on a common
    # Neyman--Pearson-style scale: "how extreme is this score relative to H0?"
    score_normalization: str = "h0_cdf"

    # Enforce w_j >= 0 and sum_j w_j = 1.
    nonneg_simplex: bool = True

    eps: float = 1e-12

    # If external calibration split is not provided, split the training data into:
    #   judge-train / meta / risk-calib.
    meta_frac: float = 0.30
    risk_calib_frac: float = 0.30

    # If external calibration split is provided, split that external calibration
    # split into:
    #   meta / risk-calib
    # where risk_calib_frac controls how much calibration H0 is reserved only
    # for final tau selection.
    external_risk_calib_frac: float = 0.50

    # Candidate-search parameters.
    n_random_weights: int = 128
    random_seed: int = 42
    use_slsqp_refine: bool = True

    # If two feasible weight vectors have nearly identical META TPR,
    # prefer the one with larger entropy, i.e. less collapse into one judge.
    tpr_tie_tol: float = 1e-4
    tie_break_entropy: bool = True

    # Smoothing used by H0-CDF score normalization.
    # With smoothing=0.5:
    #   transformed_score = (rank + 0.5) / (n_h0 + 1.0)
    h0_cdf_smoothing: float = 0.5


class WeightedEnsembleMethod(BaseMethod):
    """
    Split-calibrated stacking ensemble.

    This method combines several base scorers, called "judges", into a single
    score-level ensemble.

    Design:

    1. Fit base judges on TRAIN only.
    2. Compute judge scores on a META split.
    3. Learn nonnegative simplex weights on META.
    4. Freeze weights.
    5. Select final NP threshold tau using a disjoint RISK_CALIB H0 split.
    6. Evaluate on EVAL.

    This is safer than learning weights and threshold on the same calibration
    split, because the final risk threshold is selected from H0 examples that
    were not used to choose the ensemble weights.

    Supports per-judge input routing:

        judge_input = {
            "Cosine": "alt",
            "TinyMLP": "main",
        }

    so different judges may operate on different feature matrices.
    """

    name: str = "WeightedEnsemble"
    needs_seed: bool = True
    needs_weights: bool = False
    input_space: str = "mixed"

    def __init__(
        self,
        judges: Sequence[BaseMethod],
        *,
        config: Optional[EnsembleConfig] = None,
    ) -> None:
        if len(judges) == 0:
            raise ValueError("WeightedEnsembleMethod: judges list is empty")

        self.judges: List[BaseMethod] = list(judges)
        self.cfg = config or EnsembleConfig()
        self._validate_config()

        self.meta_w: Optional[np.ndarray] = None
        self.tau: float = float("inf")

        # z-score normalization state
        self.z_mu: Optional[np.ndarray] = None
        self.z_std: Optional[np.ndarray] = None

        # H0-CDF normalization state: one sorted H0-score vector per judge
        self.h0_cdf_values: Optional[List[np.ndarray]] = None

        # Per-judge feature routing.
        self.judge_input: Dict[str, str] = {}

        # Diagnostics from the last fit.
        self.last_meta_tpr: float = float("nan")
        self.last_meta_fpr: float = float("nan")
        self.last_risk_calib_tpr: float = float("nan")
        self.last_risk_calib_fpr: float = float("nan")
        self.last_used_external_calib: bool = False
        self.last_fit_context: str = "unknown"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if not 0.0 < float(self.cfg.alpha) < 1.0:
            raise ValueError("alpha must be in (0,1)")
        if float(self.cfg.ridge) < 0.0:
            raise ValueError("ridge must be non-negative")
        if float(self.cfg.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if not 0.0 <= float(self.cfg.meta_frac) < 1.0:
            raise ValueError("meta_frac must be in [0,1)")
        if not 0.0 <= float(self.cfg.risk_calib_frac) < 1.0:
            raise ValueError("risk_calib_frac must be in [0,1)")
        if not 0.0 < float(self.cfg.external_risk_calib_frac) < 1.0:
            raise ValueError("external_risk_calib_frac must be in (0,1)")
        if int(self.cfg.n_random_weights) < 0:
            raise ValueError("n_random_weights must be non-negative")
        if float(self.cfg.tpr_tie_tol) < 0.0:
            raise ValueError("tpr_tie_tol must be non-negative")
        if float(self.cfg.h0_cdf_smoothing) < 0.0:
            raise ValueError("h0_cdf_smoothing must be non-negative")

        mode = self._normalization_mode()
        if mode not in {"none", "zscore", "h0_cdf"}:
            raise ValueError(
                "score_normalization must be one of: "
                "'none', 'zscore', 'h0_cdf'"
            )

    def _normalization_mode(self) -> str:
        # Backward compatibility: old config used standardize=True.
        if self.cfg.standardize and self.cfg.score_normalization == "none":
            return "zscore"
        return str(self.cfg.score_normalization).lower()

    # ------------------------------------------------------------------
    # Input routing and score matrices
    # ------------------------------------------------------------------

    @staticmethod
    def _judge_name(judge: BaseMethod) -> str:
        return str(getattr(judge, "name", type(judge).__name__))

    @staticmethod
    def _select_X(
        judge: BaseMethod,
        X_main: np.ndarray,
        X_alt: Optional[np.ndarray],
        judge_input: Dict[str, str],
    ) -> np.ndarray:
        """Select which input matrix this judge should use."""
        judge_name = WeightedEnsembleMethod._judge_name(judge)
        input_type = judge_input.get(judge_name, "main")

        if input_type == "alt":
            if X_alt is None:
                raise ValueError(
                    f"Judge '{judge_name}' requires 'alt' input, "
                    "but X_alt is None"
                )
            return X_alt

        if input_type != "main":
            raise ValueError(
                f"Invalid input routing for judge '{judge_name}': {input_type!r}. "
                "Expected 'main' or 'alt'."
            )

        return X_main

    def _judge_matrix(
        self,
        X_main: np.ndarray,
        X_alt: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a matrix of base-judge scores.

        Returns:
            Z of shape (N, M), where M is the number of judges.
        """
        if X_main.ndim != 2:
            raise ValueError("X_main must be a 2D array")

        cols: List[np.ndarray] = []

        for judge in self.judges:
            X_use = self._select_X(judge, X_main, X_alt, self.judge_input)
            scores = np.asarray(judge.score(X_use), dtype=np.float64).reshape(-1)

            if scores.shape[0] != X_main.shape[0]:
                raise ValueError(
                    f"Judge {self._judge_name(judge)!r} returned "
                    f"{scores.shape[0]} scores for {X_main.shape[0]} rows"
                )

            cols.append(scores)

        return np.stack(cols, axis=1)

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_two_indices(
        n: int,
        *,
        seed: Optional[int],
        risk_frac: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split indices into META and RISK.

        Used when an external calibration split exists.

        Returns:
            meta_idx, risk_idx
        """
        if n <= 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        idx = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed if seed is not None else 42)
        rng.shuffle(idx)

        if n == 1:
            # Degenerate fallback. Caller may reuse this carefully.
            return idx, idx.copy()

        n_risk = max(1, int(round(float(risk_frac) * n)))
        n_risk = min(n_risk, n - 1)

        risk_idx = idx[:n_risk]
        meta_idx = idx[n_risk:]

        return meta_idx, risk_idx

    @staticmethod
    def _split_three_indices(
        n: int,
        *,
        seed: Optional[int],
        meta_frac: float,
        risk_frac: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split indices into JUDGE_TRAIN, META, RISK.

        Used when an external calibration split does not exist.

        Returns:
            judge_idx, meta_idx, risk_idx
        """
        if n <= 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty

        idx = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed if seed is not None else 42)
        rng.shuffle(idx)

        if n < 3:
            # Not enough data to separate roles.
            # Keep all data for judge training and leave meta/risk empty.
            empty = np.empty(0, dtype=np.int64)
            return idx, empty, empty

        n_meta = max(1, int(round(float(meta_frac) * n)))
        n_risk = max(1, int(round(float(risk_frac) * n)))

        # Leave at least one row for judge training.
        if n_meta + n_risk > n - 1:
            overflow = n_meta + n_risk - (n - 1)
            shrink_risk = min(overflow, max(0, n_risk - 1))
            n_risk -= shrink_risk
            overflow -= shrink_risk
            shrink_meta = min(overflow, max(0, n_meta - 1))
            n_meta -= shrink_meta

        n_meta = max(1, n_meta)
        n_risk = max(1, n_risk)

        meta_idx = idx[:n_meta]
        risk_idx = idx[n_meta:n_meta + n_risk]
        judge_idx = idx[n_meta + n_risk:]

        if judge_idx.size == 0:
            # Last-resort fallback.
            judge_idx = risk_idx
            risk_idx = meta_idx

        return judge_idx, meta_idx, risk_idx

    @staticmethod
    def _take_optional(
        X: Optional[np.ndarray],
        idx: np.ndarray,
    ) -> Optional[np.ndarray]:
        if X is None:
            return None
        return X[idx]

    # ------------------------------------------------------------------
    # Score normalization
    # ------------------------------------------------------------------

    def _normalization_fit(self, Z0_meta: np.ndarray, Z1_meta: np.ndarray) -> None:
        """
        Fit score normalization using META data only.

        For h0_cdf mode, only H0 META scores are used.
        """
        mode = self._normalization_mode()

        self.z_mu = None
        self.z_std = None
        self.h0_cdf_values = None

        if mode == "none":
            return

        if mode == "zscore":
            Z = np.vstack([Z0_meta, Z1_meta])
            mu = Z.mean(axis=0)
            std = Z.std(axis=0)
            std = np.maximum(std, self.cfg.eps)
            self.z_mu = mu
            self.z_std = std
            return

        if mode == "h0_cdf":
            values: List[np.ndarray] = []
            for j in range(Z0_meta.shape[1]):
                col = np.asarray(Z0_meta[:, j], dtype=np.float64).reshape(-1)
                if col.size == 0:
                    values.append(np.array([0.0], dtype=np.float64))
                else:
                    values.append(np.sort(col))
            self.h0_cdf_values = values
            return

        raise ValueError(f"Unsupported normalization mode: {mode!r}")

    def _normalization_apply(self, Z: np.ndarray) -> np.ndarray:
        """Apply the fitted score normalization."""
        mode = self._normalization_mode()

        if mode == "none":
            return Z

        if mode == "zscore":
            if self.z_mu is None or self.z_std is None:
                return Z
            return (Z - self.z_mu) / self.z_std

        if mode == "h0_cdf":
            if self.h0_cdf_values is None:
                return Z

            Z = np.asarray(Z, dtype=np.float64)
            out = np.empty_like(Z, dtype=np.float64)
            smoothing = float(self.cfg.h0_cdf_smoothing)

            for j, sorted_h0 in enumerate(self.h0_cdf_values):
                sorted_h0 = np.asarray(sorted_h0, dtype=np.float64).reshape(-1)
                n = int(sorted_h0.size)

                if n == 0:
                    out[:, j] = 0.5
                    continue

                # rank = number of H0 scores <= current score.
                # Higher output means more extreme relative to H0.
                rank = np.searchsorted(sorted_h0, Z[:, j], side="right").astype(np.float64)

                if smoothing > 0.0:
                    out[:, j] = (rank + smoothing) / (float(n) + 2.0 * smoothing)
                else:
                    out[:, j] = rank / max(1.0, float(n))

            return out

        raise ValueError(f"Unsupported normalization mode: {mode!r}")

    # ------------------------------------------------------------------
    # Simplex / entropy
    # ------------------------------------------------------------------

    @staticmethod
    def _proj_simplex(v: np.ndarray) -> np.ndarray:
        """Project onto the probability simplex {w >= 0, sum(w) = 1}."""
        v = np.asarray(v, dtype=np.float64).reshape(-1)

        if v.size == 0:
            return v

        if v.size == 1:
            return np.array([1.0], dtype=np.float64)

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, v.size + 1) > (cssv - 1.0))[0]

        if rho.size == 0:
            return np.ones_like(v, dtype=np.float64) / float(v.size)

        rho_i = int(rho[-1])
        theta = (cssv[rho_i] - 1.0) / float(rho_i + 1)
        w = np.maximum(v - theta, 0.0)

        s = float(w.sum())
        if s <= 0.0:
            return np.ones_like(v, dtype=np.float64) / float(v.size)

        return w / s

    @staticmethod
    def _weight_entropy(w: np.ndarray, eps: float = 1e-12) -> float:
        ww = np.asarray(w, dtype=np.float64).reshape(-1)
        ww = np.maximum(ww, 0.0)

        s = float(ww.sum())
        if s <= eps:
            return 0.0

        ww = ww / s
        return float(-np.sum(ww * np.log(np.maximum(ww, eps))))

    # ------------------------------------------------------------------
    # Neyman--Pearson tau selection
    # ------------------------------------------------------------------

    @staticmethod
    def _beta_ppf(q: float, a: float, b: float) -> float:
        try:
            from scipy.stats import beta as scipy_beta

            return float(scipy_beta.ppf(q, a, b))
        except Exception:
            if q <= 0.0:
                return 0.0
            if q >= 1.0:
                return 1.0
            return float(q)

    @staticmethod
    def _normal_ppf(q: float) -> float:
        try:
            from scipy.stats import norm

            return float(norm.ppf(q))
        except Exception:
            if q >= 0.995:
                return 2.5758
            if q >= 0.99:
                return 2.3263
            if q >= 0.975:
                return 1.9600
            if q >= 0.95:
                return 1.6449
            return 1.2816

    @staticmethod
    def _fpr_ucb(
        k: int,
        n: int,
        *,
        method: str,
        delta: float,
    ) -> float:
        """
        Upper confidence bound on FPR after observing k accepts among n H0 samples.
        """
        if n <= 0:
            return 1.0

        k = int(max(0, min(k, n)))
        delta = float(np.clip(delta, 1e-12, 0.5))
        method = str(method)

        if method == "clopper_pearson":
            if k >= n:
                return 1.0
            return WeightedEnsembleMethod._beta_ppf(
                1.0 - delta,
                k + 1.0,
                n - k,
            )

        if method == "beta_ucb":
            return WeightedEnsembleMethod._beta_ppf(
                1.0 - delta,
                k + 1.0,
                n - k + 1.0,
            )

        if method == "wilson":
            phat = k / n
            z = WeightedEnsembleMethod._normal_ppf(1.0 - delta)
            z2 = z * z
            denom = 1.0 + z2 / n
            center = (phat + z2 / (2.0 * n)) / denom
            radius = (z / denom) * np.sqrt(
                (phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))
            )
            return float(min(1.0, max(0.0, center + radius)))

        return float(k / n)

    @staticmethod
    def _select_tau(
        scores: np.ndarray,
        *,
        alpha: float,
        tie_mode: str,
        guardrail: str,
        guardrail_delta: float,
    ) -> float:
        """
        Select a threshold from H0 scores.

        The selected tau is the lowest threshold whose empirical or guarded H0
        accept rate is <= alpha.

        For tie_mode="ge":
            accept iff score >= tau.

        For tie_mode="gt":
            accept iff score > tau.
        """
        s = np.asarray(scores, dtype=np.float64).reshape(-1)

        if s.size == 0:
            return float("inf")

        alpha = float(alpha)
        tie_mode = str(tie_mode)
        guardrail = str(guardrail)

        if tie_mode not in {"ge", "gt"}:
            raise ValueError("tie_mode must be 'ge' or 'gt'")

        if guardrail not in {"none", "clopper_pearson", "beta_ucb", "wilson"}:
            raise ValueError(
                "guardrail must be one of: "
                "'none', 'clopper_pearson', 'beta_ucb', 'wilson'"
            )

        uniq, counts = np.unique(s, return_counts=True)
        n = int(s.size)
        cumsum = np.cumsum(counts)

        for i, tau in enumerate(uniq):
            if tie_mode == "gt":
                # accepted: scores strictly greater than tau
                k = int(n - cumsum[i])
            else:
                # accepted: scores greater than or equal to tau
                k = int(n - (cumsum[i - 1] if i > 0 else 0))

            if guardrail == "none":
                if (k / max(1, n)) <= alpha:
                    return float(tau)
            else:
                ucb = WeightedEnsembleMethod._fpr_ucb(
                    k,
                    n,
                    method=guardrail,
                    delta=guardrail_delta,
                )
                if ucb <= alpha:
                    return float(tau)

        return float("inf")

    # ------------------------------------------------------------------
    # Meta weight optimization
    # ------------------------------------------------------------------

    def _fit_meta_weights_neyman_pearson(
        self,
        Z0_meta: np.ndarray,
        Z1_meta: np.ndarray,
        alpha: float,
        tie_mode: str,
        guardrail: str,
        guardrail_delta: float,
    ) -> np.ndarray:
        """
        Select ensemble weights on META.

        For each candidate weight vector w:
            s0 = Z0_meta @ w
            s1 = Z1_meta @ w
            tau_meta = NP threshold selected from s0
            TPR_meta = fraction of s1 accepted by tau_meta

        Choose feasible w with highest META TPR.

        Important:
            The tau selected here is used only to evaluate candidate weights.
            The final deployment tau is selected later from a disjoint
            RISK_CALIB H0 split.
        """
        if Z0_meta.ndim != 2 or Z1_meta.ndim != 2:
            raise ValueError("Z0_meta and Z1_meta must be 2D matrices")
        if Z0_meta.shape[1] != Z1_meta.shape[1]:
            raise ValueError("Z0_meta and Z1_meta must have the same number of columns")

        m = int(Z0_meta.shape[1])

        if m == 0:
            raise ValueError("No judges available for ensemble")

        if m == 1:
            return np.array([1.0], dtype=np.float64)

        def eval_candidate(w: np.ndarray) -> Tuple[bool, float, float]:
            w = np.asarray(w, dtype=np.float64).reshape(-1)

            if self.cfg.nonneg_simplex:
                w = self._proj_simplex(w)

            s0 = Z0_meta @ w
            s1 = Z1_meta @ w

            tau = self._select_tau(
                s0,
                alpha=alpha,
                tie_mode=tie_mode,
                guardrail=guardrail,
                guardrail_delta=guardrail_delta,
            )

            if not np.isfinite(tau):
                return False, 0.0, tau

            if tie_mode == "gt":
                tpr = float(np.mean(s1 > tau)) if s1.size else 0.0
            else:
                tpr = float(np.mean(s1 >= tau)) if s1.size else 0.0

            return True, tpr, tau

        candidates: List[np.ndarray] = []

        # 1. Each judge alone.
        for j in range(m):
            w = np.zeros(m, dtype=np.float64)
            w[j] = 1.0
            candidates.append(w)

        # 2. Uniform weights.
        candidates.append(np.ones(m, dtype=np.float64) / float(m))

        # 3. Random simplex samples.
        rng = np.random.default_rng(int(self.cfg.random_seed))
        for _ in range(int(self.cfg.n_random_weights)):
            candidates.append(rng.dirichlet(np.ones(m, dtype=np.float64)))

        best_w: Optional[np.ndarray] = None
        best_tpr = -1.0
        best_entropy = -1.0
        infeasible_count = 0

        for w_cand in candidates:
            is_feasible, tpr, _ = eval_candidate(w_cand)

            if not is_feasible:
                infeasible_count += 1
                continue

            entropy = self._weight_entropy(w_cand, eps=self.cfg.eps)

            if tpr > best_tpr + self.cfg.tpr_tie_tol:
                best_w = w_cand.copy()
                best_tpr = tpr
                best_entropy = entropy
            elif abs(tpr - best_tpr) <= self.cfg.tpr_tie_tol:
                if self.cfg.tie_break_entropy and entropy > best_entropy:
                    best_w = w_cand.copy()
                    best_tpr = tpr
                    best_entropy = entropy

        # Optional local refinement. The objective is not truly smooth because
        # tau changes discretely, so this is only a refinement heuristic.
        if best_w is not None and self.cfg.use_slsqp_refine:
            try:
                from scipy.optimize import minimize

                def objective(w_raw: np.ndarray) -> float:
                    w = np.asarray(w_raw, dtype=np.float64)
                    if self.cfg.nonneg_simplex:
                        w = self._proj_simplex(w)
                    is_feasible, tpr, _ = eval_candidate(w)
                    if not is_feasible:
                        return 1e6
                    return -float(tpr)

                result = minimize(
                    objective,
                    best_w,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * m,
                    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                    options={"maxiter": 50, "ftol": 1e-6},
                )

                if result.success:
                    w_opt = np.asarray(result.x, dtype=np.float64)
                    w_opt = self._proj_simplex(w_opt)

                    is_feasible, tpr_opt, _ = eval_candidate(w_opt)
                    if is_feasible:
                        entropy_opt = self._weight_entropy(w_opt, eps=self.cfg.eps)

                        if tpr_opt > best_tpr + self.cfg.tpr_tie_tol:
                            best_w = w_opt
                            best_tpr = tpr_opt
                            best_entropy = entropy_opt
                        elif abs(tpr_opt - best_tpr) <= self.cfg.tpr_tie_tol:
                            if self.cfg.tie_break_entropy and entropy_opt > best_entropy:
                                best_w = w_opt
                                best_tpr = tpr_opt
                                best_entropy = entropy_opt

            except Exception:
                # Candidate search is the stable fallback.
                pass

        if best_w is None:
            print(
                "[WeightedEnsemble][Warning] No feasible META weights found "
                f"(tie_mode={tie_mode}, guardrail={guardrail}, "
                f"delta={guardrail_delta}). "
                f"{infeasible_count}/{len(candidates)} candidates were infeasible. "
                "Falling back to uniform weights."
            )
            best_w = np.ones(m, dtype=np.float64) / float(m)

        if self.cfg.nonneg_simplex:
            best_w = self._proj_simplex(best_w)

        return best_w

    # ------------------------------------------------------------------
    # BaseMethod API
    # ------------------------------------------------------------------

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
    ) -> "WeightedEnsembleMethod":
        """
        Fit the split-calibrated weighted ensemble.

        If external calibration is provided:
            - judges fit on H*_train
            - H*_calib is split into META and RISK_CALIB

        If external calibration is not provided:
            - H*_train is split into JUDGE_TRAIN, META, and RISK_CALIB
        """
        if alpha is None:
            alpha = self.cfg.alpha
        alpha = float(alpha)

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0,1)")

        self.judge_input = dict(judge_input or {})
        self.last_fit_context = str(fit_context)

        alt_judges = [
            name for name, route in self.judge_input.items()
            if route == "alt"
        ]
        if alt_judges:
            print(
                "  [WeightedEnsemble] Judges routed to 'alt' input: "
                + ", ".join(alt_judges)
            )

        used_external_calib = H0_calib is not None and H1_calib is not None
        self.last_used_external_calib = bool(used_external_calib)

        base_seed = seed if seed is not None else self.cfg.random_seed

        if used_external_calib:
            # Judges use the full train split.
            H0_judge = H0_train
            H1_judge = H1_train
            H0_judge_alt = H0_train_alt
            H1_judge_alt = H1_train_alt

            # External calibration is split into META and RISK_CALIB.
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
            # No external calibration. Split train into three disjoint roles.
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
                "  [WeightedEnsemble][DEBUG] "
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
                "WeightedEnsembleMethod requires non-empty META H0 and H1 splits"
            )

        if H0_risk.shape[0] == 0:
            raise ValueError(
                "WeightedEnsembleMethod requires non-empty RISK_CALIB H0 split"
            )

        # 1. Fit base judges on judge-training split only.
        for judge in self.judges:
            H0_use = self._select_X(judge, H0_judge, H0_judge_alt, self.judge_input)
            H1_use = self._select_X(judge, H1_judge, H1_judge_alt, self.judge_input)

            kwargs = {}

            # Kept for compatibility. If weights are used heavily, this should
            # be split consistently with the selected rows.
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
                        f"  [WeightedEnsemble][Warning] "
                        f"Judge {self._judge_name(judge)} failed during fit: {exc}"
                    )
            except Exception as exc:
                print(
                    f"  [WeightedEnsemble][Warning] "
                    f"Judge {self._judge_name(judge)} failed during fit: {exc}"
                )

        # 2. Build META score matrices.
        Z0_meta_raw = self._judge_matrix(H0_meta, H0_meta_alt)
        Z1_meta_raw = self._judge_matrix(H1_meta, H1_meta_alt)

        # 3. Fit score normalization on META only, then apply it.
        self._normalization_fit(Z0_meta_raw, Z1_meta_raw)
        Z0_meta = self._normalization_apply(Z0_meta_raw)
        Z1_meta = self._normalization_apply(Z1_meta_raw)

        # 4. Learn weights on META only.
        self.meta_w = self._fit_meta_weights_neyman_pearson(
            Z0_meta,
            Z1_meta,
            alpha=alpha,
            tie_mode=tie_mode,
            guardrail=guardrail,
            guardrail_delta=guardrail_delta,
        )

        # 5. Select final tau on disjoint RISK_CALIB H0 only.
        Z0_risk = self._normalization_apply(
            self._judge_matrix(H0_risk, H0_risk_alt)
        )
        s0_risk = Z0_risk @ self.meta_w

        self.tau = self._select_tau(
            s0_risk,
            alpha=alpha,
            tie_mode=tie_mode,
            guardrail=guardrail,
            guardrail_delta=guardrail_delta,
        )

        # Diagnostics on META.
        s0_meta = Z0_meta @ self.meta_w
        s1_meta = Z1_meta @ self.meta_w

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

        # Diagnostics on RISK_CALIB.
        # H0 risk is authoritative for FPR. H1 risk is diagnostic only.
        Z1_risk = self._normalization_apply(
            self._judge_matrix(H1_risk, H1_risk_alt)
        ) if H1_risk.shape[0] > 0 else np.empty((0, len(self.judges)))

        s1_risk = Z1_risk @ self.meta_w if Z1_risk.size else np.empty(0)

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
            f"[WeightedEnsemble] Fitted split-calibrated ensemble "
            f"(alpha={alpha:.4f}, {calib_msg}, "
            f"normalization={self._normalization_mode()}):"
        )
        print(
            f"  tau={self.tau:.6f}, "
            f"META: TPR={self.last_meta_tpr:.4f}, FPR={self.last_meta_fpr:.4f}, "
            f"RISK_CALIB: TPR={self.last_risk_calib_tpr:.4f}, "
            f"FPR={self.last_risk_calib_fpr:.4f}"
        )

        for judge, weight_value in zip(self.judges, self.meta_w):
            print(f"  {self._judge_name(judge):24s}: {float(weight_value):+.6f}")

        if fit_context in {"local", "matched_global_on_local"}:
            top_idx = int(np.argmax(self.meta_w))
            top_name = self._judge_name(self.judges[top_idx])
            top_weight = float(self.meta_w[top_idx])
            collapsed = bool(top_weight >= 0.999)
            print(
                "  [WeightedEnsemble][DEBUG] "
                f"collapse={collapsed} "
                f"top_judge={top_name} "
                f"top_weight={top_weight:.6f}"
            )

        return self

    def score(
        self,
        X: np.ndarray,
        X_alt: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score samples using the fitted ensemble.

        Returns zero scores if the ensemble has not been fitted yet.
        """
        if self.meta_w is None:
            return np.zeros(X.shape[0], dtype=np.float64)

        Z = self._judge_matrix(X, X_alt)
        Z = self._normalization_apply(Z)

        return Z @ self.meta_w

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
        H0_calib: Optional[np.ndarray] = None,
        H1_calib: Optional[np.ndarray] = None,
        H0_train_alt: Optional[np.ndarray] = None,
        H1_train_alt: Optional[np.ndarray] = None,
        H0_eval_alt: Optional[np.ndarray] = None,
        H1_eval_alt: Optional[np.ndarray] = None,
        H0_calib_alt: Optional[np.ndarray] = None,
        H1_calib_alt: Optional[np.ndarray] = None,
        judge_input: Optional[Dict[str, str]] = None,
        guardrail: str = "none",
        guardrail_delta: float = 0.01,
        fit_context: str = "unknown",
    ) -> MethodResult:
        """
        Fit the ensemble and evaluate on held-out H0/H1 evaluation sets.
        """
        self.fit(
            H0_train,
            H1_train,
            weights=weights,
            seed=seed,
            alpha=float(alpha),
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

        threshold = float(self.tau)

        s0_eval = self.score(H0_eval, H0_eval_alt)
        s1_eval = self.score(H1_eval, H1_eval_alt)

        if np.isfinite(threshold):
            if tie_mode == "gt":
                tpr = float(np.mean(s1_eval > threshold))
                fpr = float(np.mean(s0_eval > threshold))
            else:
                tpr = float(np.mean(s1_eval >= threshold))
                fpr = float(np.mean(s0_eval >= threshold))
        else:
            tpr = 0.0
            fpr = 0.0

        import time

        t0 = time.perf_counter()
        _ = self.score(H1_eval, H1_eval_alt)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # In MethodResult, train_tpr/train_fpr are used as diagnostics.
        # Here they report RISK_CALIB diagnostics, not training-set metrics.
        return MethodResult(
            tpr=tpr,
            fpr=fpr,
            time_ms=float(dt_ms),
            train_tpr=float(self.last_risk_calib_tpr),
            train_fpr=float(self.last_risk_calib_fpr),
        )