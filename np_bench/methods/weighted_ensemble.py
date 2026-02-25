# NeighborCache/np_bench/methods/weighted_ensemble.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict

import numpy as np

from .base import BaseMethod, MethodResult


@dataclass
class EnsembleConfig:
    alpha: float = 0.05
    ridge: float = 1e-3
    standardize: bool = False  # Disabled: breaks NP optimization on standardized scores
    nonneg_simplex: bool = True
    eps: float = 1e-12
    meta_frac: float = 0.30  # fraction for internal calib split when external not provided


class WeightedEnsembleMethod(BaseMethod):
    """
    Stacking ensemble that uses EXTERNAL calib split (from pipeline).
    
    Supports per-judge input routing (main vs alt feature matrices) so that
    different judges can work on different representations (e.g., Cosine on Hadamard vectors).

    - Fit judges on TRAIN only.
    - Fit meta weights on CALIB only using NP-feasible optimization.
    - Calibrate tau on CALIB H0 using _select_tau (respects tie_mode + guardrail).
    """
    name: str = "WeightedEnsemble"
    needs_seed: bool = True
    needs_weights: bool = False

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

        self.meta_w: Optional[np.ndarray] = None  # (M,)
        self.z_mu: Optional[np.ndarray] = None    # (M,)
        self.z_std: Optional[np.ndarray] = None   # (M,)
        self.tau: float = float("inf")
        
        # Store judge routing for score()
        self.judge_input: Dict[str, str] = {}

    # -----------------------------
    # helpers
    # -----------------------------
    @staticmethod
    def _select_X(judge: BaseMethod, X_main: np.ndarray, X_alt: Optional[np.ndarray], 
                  judge_input: Dict[str, str]) -> np.ndarray:
        """Select which input matrix to use for this judge."""
        judge_name = getattr(judge, "name", type(judge).__name__)
        input_type = judge_input.get(judge_name, "main")
        
        if input_type == "alt":
            if X_alt is None:
                raise ValueError(f"Judge '{judge_name}' requires 'alt' input but X_alt is None")
            return X_alt
        return X_main
    def _split_train(self, H0: np.ndarray, H1: np.ndarray, seed: Optional[int]) -> tuple:
        """Split train into judge-fit and meta-fit parts"""
        rng = np.random.default_rng(seed if seed is not None else 42)
        
        def split(X: np.ndarray) -> tuple:
            n = X.shape[0]
            if n < 2:
                return X, X[:0]
            idx = np.arange(n)
            rng.shuffle(idx)
            n_meta = max(1, int(round(self.cfg.meta_frac * n)))
            return X[idx[n_meta:]], X[idx[:n_meta]]
        
        H0_j, H0_m = split(H0)
        H1_j, H1_m = split(H1)
        return H0_j, H1_j, H0_m, H1_m

    def _judge_matrix(self, X_main: np.ndarray, X_alt: Optional[np.ndarray] = None) -> np.ndarray:
        """Build judge score matrix, routing each judge to its selected input."""
        cols = []
        for m in self.judges:
            X_use = self._select_X(m, X_main, X_alt, self.judge_input)
            s = np.asarray(m.score(X_use), dtype=np.float64).reshape(-1)
            cols.append(s)
        return np.stack(cols, axis=1)  # (N, M)

    def _standardize_fit(self, Z: np.ndarray) -> None:
        mu = Z.mean(axis=0)
        std = Z.std(axis=0)
        std = np.maximum(std, self.cfg.eps)
        self.z_mu = mu
        self.z_std = std

    def _standardize_apply(self, Z: np.ndarray) -> np.ndarray:
        if not self.cfg.standardize or self.z_mu is None or self.z_std is None:
            return Z
        return (Z - self.z_mu) / self.z_std

    @staticmethod
    def _proj_simplex(v: np.ndarray) -> np.ndarray:
        # Projection onto {w>=0, sum w = 1}
        v = np.asarray(v, dtype=np.float64)
        if v.size == 1:
            return np.array([1.0], dtype=np.float64)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, v.size + 1) > (cssv - 1.0))[0]
        if rho.size == 0:
            return np.ones_like(v) / v.size
        rho = int(rho[-1])
        theta = (cssv[rho] - 1.0) / float(rho + 1)
        w = np.maximum(v - theta, 0.0)
        s = w.sum()
        if s > 0:
            w /= s
        return w
    
    # -----------------------------
    # NP tau selection (from evaluation.py)
    # -----------------------------
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
    def _fpr_ucb(k: int, n: int, *, method: str, delta: float) -> float:
        if n <= 0:
            return 1.0
        k = int(max(0, min(k, n)))
        delta = float(np.clip(delta, 1e-12, 0.5))

        if method == "clopper_pearson":
            if k >= n:
                return 1.0
            return WeightedEnsembleMethod._beta_ppf(1.0 - delta, k + 1.0, n - k)

        if method == "beta_ucb":
            return WeightedEnsembleMethod._beta_ppf(1.0 - delta, k + 1.0, n - k + 1.0)

        if method == "wilson":
            phat = k / n
            z = WeightedEnsembleMethod._normal_ppf(1.0 - delta)
            z2 = z * z
            denom = 1.0 + z2 / n
            center = (phat + z2 / (2.0 * n)) / denom
            radius = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
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
        """NP tau selection with tie handling and guardrails."""
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if s.size == 0:
            return float("inf")

        if guardrail == "none":
            return float(np.quantile(s, 1.0 - alpha))

        uniq, counts = np.unique(s, return_counts=True)
        n = int(s.size)
        cumsum = np.cumsum(counts)

        for i, tau in enumerate(uniq):
            if tie_mode == "gt":
                k = int(n - cumsum[i])
            else:
                k = int(n - (cumsum[i - 1] if i > 0 else 0))
            ucb = WeightedEnsembleMethod._fpr_ucb(k, n, method=guardrail, delta=guardrail_delta)
            if ucb <= alpha:
                return float(tau)

        return float("inf")

    def _fit_meta_weights_neyman_pearson(
        self, 
        Z0: np.ndarray, 
        Z1: np.ndarray, 
        alpha: float,
        tie_mode: str,
        guardrail: str,
        guardrail_delta: float
    ) -> np.ndarray:
        """
        Find weights that maximize TPR subject to NP-feasible FPR ≤ alpha.
        
        For any weight vector w:
        - Ensemble scores: s0 = Z0 @ w, s1 = Z1 @ w  
        - NP threshold: tau = _select_tau(s0, alpha, tie_mode, guardrail, guardrail_delta)
        - Feasible if tau is finite
        - TPR = mean(s1 >= tau) or mean(s1 > tau) depending on tie_mode
        
        We find: argmax_w TPR(w) subject to w >= 0, sum(w) = 1, and w is feasible
        """
        M = Z0.shape[1]
        
        def eval_feasibility_and_tpr(w: np.ndarray) -> tuple[bool, float, float]:
            """Return (is_feasible, tpr, tau)."""
            s0 = Z0 @ w
            s1 = Z1 @ w
            tau = self._select_tau(
                s0, alpha=alpha, tie_mode=tie_mode,
                guardrail=guardrail, guardrail_delta=guardrail_delta
            )
            
            if not np.isfinite(tau):
                return False, 0.0, tau
            
            # Compute TPR using same tie_mode as tau selection
            if tie_mode == "gt":
                tpr = float(np.mean(s1 > tau))
            else:  # "ge"
                tpr = float(np.mean(s1 >= tau))
            
            return True, tpr, tau
        
        # Build candidate set
        candidates = []
        
        # 1. Each individual judge
        for j in range(M):
            w = np.zeros(M)
            w[j] = 1.0
            candidates.append(w)
        
        # 2. Uniform weights
        candidates.append(np.ones(M) / M)
        
        # 3. Random Dirichlet samples
        rng = np.random.default_rng(42)
        for _ in range(64):
            candidates.append(rng.dirichlet(np.ones(M)))
        
        # Evaluate all candidates and keep best feasible
        best_w = None
        best_tpr = -1.0
        best_tau = float("inf")
        infeasible_count = 0
        
        for w_cand in candidates:
            is_feas, tpr, tau = eval_feasibility_and_tpr(w_cand)
            if not is_feas:
                infeasible_count += 1
                continue
            
            if tpr > best_tpr:
                best_tpr = tpr
                best_w = w_cand.copy()
                best_tau = tau
        
        # Optional: refine with scipy if available
        if best_w is not None:
            try:
                from scipy.optimize import minimize
                
                def objective(w):
                    """Negative TPR with heavy penalty for infeasibility."""
                    is_feas, tpr, _ = eval_feasibility_and_tpr(w)
                    if not is_feas:
                        return 1e6  # Large penalty
                    return -tpr
                
                # Refine from best candidate
                result = minimize(
                    objective,
                    best_w,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * M,
                    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    options={'maxiter': 50, 'ftol': 1e-6}
                )
                
                if result.success:
                    w_opt = result.x
                    w_opt = np.maximum(w_opt, 0.0)
                    w_opt = w_opt / (np.sum(w_opt) + 1e-12)
                    
                    is_feas, tpr_opt, tau_opt = eval_feasibility_and_tpr(w_opt)
                    if is_feas and tpr_opt > best_tpr:
                        best_tpr = tpr_opt
                        best_w = w_opt
                        best_tau = tau_opt
            except Exception:
                pass  # Fall back to candidates
        
        # Fallback if no feasible solution
        if best_w is None:
            print(f"  [Warning] No feasible weights found (tie_mode={tie_mode}, guardrail={guardrail}, delta={guardrail_delta})")
            print(f"            {infeasible_count}/{len(candidates)} candidates infeasible")
            print(f"            Falling back to best single judge by minimum constraint violation")
            
            # Find single judge with smallest violation
            best_violation = float("inf")
            for j in range(M):
                w = np.zeros(M)
                w[j] = 1.0
                is_feas, tpr, tau = eval_feasibility_and_tpr(w)
                # Violation = how much FPR exceeds alpha (or inf if no finite tau)
                if np.isfinite(tau):
                    s0 = Z0 @ w
                    if tie_mode == "gt":
                        fpr = float(np.mean(s0 > tau))
                    else:
                        fpr = float(np.mean(s0 >= tau))
                    violation = max(0.0, fpr - alpha)
                else:
                    violation = float("inf")
                
                if violation < best_violation:
                    best_violation = violation
                    best_w = w.copy()
                    best_tau = tau
            
            if best_w is None:
                best_w = np.ones(M) / M
        
        # Apply simplex constraint if configured
        if self.cfg.nonneg_simplex and best_w is not None:
            best_w = np.maximum(best_w, 0.0)
            best_w = self._proj_simplex(best_w)
        
        return best_w if best_w is not None else np.ones(M) / M

    # -----------------------------
    # BaseMethod API
    # -----------------------------
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
    ) -> "WeightedEnsembleMethod":
        
        if alpha is None:
            alpha = self.cfg.alpha
        alpha = float(alpha)
        
        # Store judge routing
        self.judge_input = judge_input if judge_input is not None else {}
        
        # Log which judges are routed to alt
        alt_judges = [j for j in self.judge_input.keys() if self.judge_input[j] == "alt"]
        if alt_judges:
            print(f"  [Ensemble] Judges routed to 'alt' input: {', '.join(alt_judges)}")

        # If external calib not provided, do internal split
        used_external_calib = (H0_calib is not None and H1_calib is not None)
        if not used_external_calib:
            H0_j, H1_j, H0_calib, H1_calib = self._split_train(H0_train, H1_train, seed)
            # If alt matrices provided, split them too
            if H0_train_alt is not None and H1_train_alt is not None:
                H0_j_alt, H1_j_alt, H0_calib_alt, H1_calib_alt = self._split_train(H0_train_alt, H1_train_alt, seed)
            else:
                H0_j_alt, H1_j_alt = None, None
        else:
            H0_j, H1_j = H0_train, H1_train
            H0_j_alt, H1_j_alt = H0_train_alt, H1_train_alt

        # 1) Fit judges on judge-train part (using their selected input)
        for m in self.judges:
            # Select input for this judge
            H0_j_use = self._select_X(m, H0_j, H0_j_alt, self.judge_input)
            H1_j_use = self._select_X(m, H1_j, H1_j_alt, self.judge_input)
            
            kwargs = {}
            if weights is not None and getattr(m, "needs_weights", False):
                kwargs["weights"] = weights
            if seed is not None and getattr(m, "needs_seed", False):
                kwargs["seed"] = seed
            try:
                m.fit(H0_j_use, H1_j_use, **kwargs)
            except TypeError:
                try:
                    m.fit(H0_j_use, H1_j_use)
                except Exception as e:
                    print(f"  [Warning] Judge {getattr(m, 'name', type(m).__name__)} failed: {e}")

        # 2) Build meta features on CALIB only (using per-judge routing)
        Z0 = self._judge_matrix(H0_calib, H0_calib_alt)
        Z1 = self._judge_matrix(H1_calib, H1_calib_alt)

        if self.cfg.standardize:
            Zc = np.vstack([Z0, Z1])
            self._standardize_fit(Zc)
            Z0 = self._standardize_apply(Z0)
            Z1 = self._standardize_apply(Z1)

        # 3) Fit meta weights on CALIB only via NP-feasible optimization
        self.meta_w = self._fit_meta_weights_neyman_pearson(
            Z0, Z1, alpha, tie_mode, guardrail, guardrail_delta
        )

        # 4) Calibrate tau on CALIB-H0 using _select_tau
        s0_cal = Z0 @ self.meta_w
        s1_cal = Z1 @ self.meta_w
        self.tau = self._select_tau(
            s0_cal, alpha=alpha, tie_mode=tie_mode,
            guardrail=guardrail, guardrail_delta=guardrail_delta
        )
        
        # Compute achieved TPR/FPR on calib for logging
        if np.isfinite(self.tau):
            if tie_mode == "gt":
                calib_tpr = float(np.mean(s1_cal > self.tau))
                calib_fpr = float(np.mean(s0_cal > self.tau))
            else:  # "ge"
                calib_tpr = float(np.mean(s1_cal >= self.tau))
                calib_fpr = float(np.mean(s0_cal >= self.tau))
        else:
            calib_tpr = calib_fpr = float("nan")

        # log
        calib_msg = "external calib" if used_external_calib else f"internal split (meta_frac={self.cfg.meta_frac:.2f})"
        print(f"[WeightedEnsemble] Fitted weights (alpha={alpha:.4f}, {calib_msg}):")
        print(f"  tau={self.tau:.4f}, Calib: TPR={calib_tpr:.4f}, FPR={calib_fpr:.4f}")
        for j, wj in zip(self.judges, self.meta_w):
            name = getattr(j, "name", type(j).__name__)
            print(f"  {name:20s}: {wj:+.4f}")

        return self

    def score(self, X: np.ndarray, X_alt: Optional[np.ndarray] = None) -> np.ndarray:
        """Score samples using ensemble. Supports per-judge routing to X_alt if provided."""
        if self.meta_w is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        Z = self._judge_matrix(X, X_alt)
        Z = self._standardize_apply(Z)
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
        # External calib split from pipeline
        H0_calib: Optional[np.ndarray] = None,
        H1_calib: Optional[np.ndarray] = None,
        # Alternate feature matrices for per-judge routing
        H0_train_alt: Optional[np.ndarray] = None,
        H1_train_alt: Optional[np.ndarray] = None,
        H0_eval_alt: Optional[np.ndarray] = None,
        H1_eval_alt: Optional[np.ndarray] = None,
        H0_calib_alt: Optional[np.ndarray] = None,
        H1_calib_alt: Optional[np.ndarray] = None,
        judge_input: Optional[Dict[str, str]] = None,
        # NP tau selection parameters
        guardrail: str = "none",
        guardrail_delta: float = 0.01,
    ) -> MethodResult:
        # Fit (judges=train, meta+tau=calib)
        self.fit(
            H0_train, H1_train,
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
        )

        thresh = float(self.tau)

        s0_ev = self.score(H0_eval, H0_eval_alt)
        s1_ev = self.score(H1_eval, H1_eval_alt)

        if tie_mode == "gt":
            tpr = float(np.mean(s1_ev > thresh))
            fpr = float(np.mean(s0_ev > thresh))
        else:
            tpr = float(np.mean(s1_ev >= thresh))
            fpr = float(np.mean(s0_ev >= thresh))

        # honest "train metrics" on calib (where tau is calibrated)
        if H0_calib is None or H1_calib is None:
            train_tpr = 0.0
            train_fpr = 0.0
        else:
            s0_cal = self.score(H0_calib, H0_calib_alt)
            s1_cal = self.score(H1_calib, H1_calib_alt)
            if tie_mode == "gt":
                train_tpr = float(np.mean(s1_cal > thresh))
                train_fpr = float(np.mean(s0_cal > thresh))
            else:
                train_tpr = float(np.mean(s1_cal >= thresh))
                train_fpr = float(np.mean(s0_cal >= thresh))

        import time
        t0 = time.perf_counter()
        _ = self.score(H1_eval, H1_eval_alt)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return MethodResult(
            tpr=tpr, fpr=fpr, time_ms=float(dt_ms),
            train_tpr=train_tpr, train_fpr=train_fpr
        )