"""
Stabilized Whitened Cosine method.

Whitening pipeline with PCA truncation, covariance shrinkage, and
per-region fallback to plain Cosine when estimation is unreliable.

Score interface:
    score(X)          — linear Fisher direction in stabilized whitened space
                        (X is a Hadamard product matrix, same as other methods)
    score_pairs(A, B) — full whiten → L2-normalize → cosine on separate vectors
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import OnlineBaseMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def _ledoit_wolf_covariance(Z: np.ndarray) -> Optional[np.ndarray]:
    """Return LedoitWolf covariance if sklearn is available, else None."""
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore
        return LedoitWolf().fit(Z).covariance_
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------

class StabilizedWhitenedCosineMethod(OnlineBaseMethod):
    """Whitened cosine with PCA truncation + shrinkage + per-region fallback.

    Parameters
    ----------
    k : int
        Number of PCA components to retain (default 64).
    shrinkage : float
        Manual shrinkage coefficient  ``C = (1-s)*cov + s*I`` (default 0.1).
        Ignored when sklearn LedoitWolf is available (automatic shrinkage).
    eps : float
        Floor for eigenvalues and numerical guards (default 1e-6).
    min_samples : int
        Minimum total samples (H0 + H1) to attempt whitening (default 200).
    fallback : bool
        If True, fall back to plain Cosine when whitening is unstable.
    """

    # Flag inspected by the evaluation loop for per-region fitting.
    supports_local_fit: bool = True

    def __init__(
        self,
        name: str = "StabilizedWhitenedCosine",
        k: int = 64,
        shrinkage: float = 0.1,
        eps: float = 1e-6,
        min_samples: int = 200,
        fallback: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.k = k
        self.shrinkage = shrinkage
        self.eps = eps
        self.min_samples = min_samples
        self.fallback = fallback
        self.verbose = verbose

        # Internal state set by _fit_whitening / _set_cosine_fallback
        self._V_k: Optional[np.ndarray] = None       # (d, k')
        self._W_inv_sqrt: Optional[np.ndarray] = None # (k', k')
        self._mean: Optional[np.ndarray] = None       # (d,)
        self._is_fallback: bool = False
        self._dim: Optional[int] = None

        # Saved global-fit state — restored when fit_region falls back
        self._global_w: Optional[np.ndarray] = None
        self._global_b: float = 0.0
        self._global_V_k: Optional[np.ndarray] = None
        self._global_W_inv_sqrt: Optional[np.ndarray] = None
        self._global_mean: Optional[np.ndarray] = None
        self._global_is_fallback: bool = True

    # ------------------------------------------------------------------
    # Public fit API
    # ------------------------------------------------------------------

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> "StabilizedWhitenedCosineMethod":
        """Global fit (used in both GLOBAL and LOCAL modes)."""
        del weights, alpha  # unused
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        all_data = np.concatenate([H0_train, H1_train], axis=0)
        self._dim = all_data.shape[1]
        self.mem_H0 = H0_train.copy()
        self.mem_H1 = H1_train.copy()

        if not self._fit_whitening(all_data, H0_train, H1_train, label="global"):
            self._set_cosine_fallback()

        # Snapshot the global fit so fit_region fallback can restore it
        self._save_global_state()
        return self

    def fit_region(
        self,
        H0_region: np.ndarray,
        H1_region: np.ndarray,
    ) -> bool:
        """Per-region fit with fallback.

        Returns True if per-region whitening succeeded.
        When fallback triggers, restores the *global* whitening fit
        (from the last ``fit()`` call) rather than degrading to flat
        cosine — a much stronger baseline than ``w = ones``.
        Only falls back to flat cosine if no global fit exists.
        """
        n_total = H0_region.shape[0] + H1_region.shape[0]
        d = H0_region.shape[1]
        self._dim = d

        min_needed = max(self.min_samples, self.k + 1, 5)

        # Guard: not enough samples for per-region whitening
        if n_total < min_needed:
            if self.verbose:
                print(f"  [SWC] fit_region: n_total={n_total} < {min_needed} "
                      f"→ restoring global fit")
            self._restore_global_state()
            return False

        all_data = np.concatenate([H0_region, H1_region], axis=0)
        if self._fit_whitening(all_data, H0_region, H1_region, label="region"):
            return True

        if self.verbose:
            print("  [SWC] fit_region: whitening failed → restoring global fit")
        self._restore_global_state()
        return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray) -> np.ndarray:
        """Linear score on Hadamard-product vectors (inherited w/b)."""
        if self.w is None:
            return np.zeros(X.shape[0], dtype=np.float32)
        s = X @ self.w + self.b
        return np.asarray(s, dtype=np.float32)

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Whiten → L2-normalise → cosine on separate embedding pairs."""
        if self._is_fallback or self._V_k is None or self._W_inv_sqrt is None:
            return np.sum(_l2_normalize(A) * _l2_normalize(B), axis=1)

        T = self._V_k @ self._W_inv_sqrt  # (d, k')
        WA = _l2_normalize((A - self._mean) @ T)
        WB = _l2_normalize((B - self._mean) @ T)
        return np.sum(WA * WB, axis=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save_global_state(self) -> None:
        """Snapshot current model as the global fit."""
        self._global_w = self.w.copy() if self.w is not None else None
        self._global_b = self.b
        self._global_V_k = self._V_k.copy() if self._V_k is not None else None
        self._global_W_inv_sqrt = self._W_inv_sqrt.copy() if self._W_inv_sqrt is not None else None
        self._global_mean = self._mean.copy() if self._mean is not None else None
        self._global_is_fallback = self._is_fallback

    def _restore_global_state(self) -> None:
        """Restore the global-fit snapshot (stronger than flat cosine)."""
        if self._global_w is not None:
            self.w = self._global_w.copy()
            self.b = self._global_b
            self._V_k = self._global_V_k.copy() if self._global_V_k is not None else None
            self._W_inv_sqrt = self._global_W_inv_sqrt.copy() if self._global_W_inv_sqrt is not None else None
            self._mean = self._global_mean.copy() if self._global_mean is not None else None
            self._is_fallback = self._global_is_fallback
        else:
            # No global fit available — true cosine fallback
            self._set_cosine_fallback()

    def _fit_whitening(
        self,
        all_data: np.ndarray,
        H0: np.ndarray,
        H1: np.ndarray,
        label: str = "",
    ) -> bool:
        """Core whitening pipeline.  Returns True on success."""
        n, d = all_data.shape

        # (a) Center
        self._mean = all_data.mean(axis=0)
        X_c = all_data - self._mean

        # (b) PCA truncation via SVD
        k_eff = min(self.k, n - 1, d)
        if k_eff < 1:
            if self.verbose:
                print(f"  [SWC {label}] k_eff={k_eff} < 1, cannot whiten")
            return False

        try:
            # Economy SVD — only first k_eff components needed
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            V_k = Vt[:k_eff].T  # (d, k_eff)
        except np.linalg.LinAlgError:
            if self.verbose:
                print(f"  [SWC {label}] SVD failed")
            return False

        self._V_k = V_k

        # Project into PCA space
        Z = X_c @ V_k  # (n, k_eff)

        # (c) Covariance in PCA space with shrinkage
        lw_cov = _ledoit_wolf_covariance(Z)
        shrinkage_used = "LedoitWolf" if lw_cov is not None else f"manual={self.shrinkage}"
        if lw_cov is not None:
            C = lw_cov
        else:
            sample_cov = (Z.T @ Z) / max(1, n - 1)
            C = (1.0 - self.shrinkage) * sample_cov + self.shrinkage * np.eye(k_eff)

        # (d) Inverse sqrt via eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            if self.verbose:
                print(f"  [SWC {label}] eigh failed")
            return False

        # Clip eigenvalues
        eigvals = np.maximum(eigvals, self.eps)

        # Condition-number check
        cond = float(eigvals.max() / eigvals.min())
        if cond > 1e10:
            if self.verbose:
                print(f"  [SWC {label}] cond={cond:.1e} too high, whitening unstable")
            if self.fallback:
                return False
            # Without fallback, proceed anyway (eigenvalues already clipped)

        if self.verbose:
            print(f"  [SWC {label}] n={n} k_eff={k_eff} shrinkage={shrinkage_used} "
                  f"eigval=[{eigvals.min():.4e}, {eigvals.max():.4e}] cond={cond:.2e} "
                  f"→ whitening OK")

        inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
        W_inv_sqrt = (eigvecs * inv_sqrt_eigvals[None, :]) @ eigvecs.T  # (k_eff, k_eff)
        self._W_inv_sqrt = W_inv_sqrt

        # (e) Fisher direction in whitened space
        H0_c = H0 - self._mean
        H1_c = H1 - self._mean

        Z0 = (H0_c @ V_k) @ W_inv_sqrt
        Z1 = (H1_c @ V_k) @ W_inv_sqrt

        mu0 = Z0.mean(axis=0) if Z0.shape[0] > 0 else np.zeros(k_eff)
        mu1 = Z1.mean(axis=0) if Z1.shape[0] > 0 else np.zeros(k_eff)

        d_w = mu1 - mu0  # Fisher direction in whitened space

        # Back-project to original space for linear score(X):
        #   score(X) = (X - μ) @ V_k @ W_inv_sqrt @ d_w + bias
        #            = X @ w + b
        T_dw = V_k @ (W_inv_sqrt @ d_w)  # (d,)

        self.w = T_dw.astype(np.float64, copy=False)
        self.b = float(-self._mean @ T_dw - 0.5 * (mu0 + mu1) @ d_w)
        self._is_fallback = False
        return True

    def _set_cosine_fallback(self) -> None:
        """Set scoring to plain cosine (sum of Hadamard product)."""
        d = self._dim if self._dim is not None else 1
        self.w = np.ones(d, dtype=np.float64)
        self.b = 0.0
        self._V_k = None
        self._W_inv_sqrt = None
        self._is_fallback = True


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Minimal smoke test: run on random data and verify fallback logic."""
    rng = np.random.default_rng(0)
    d = 128

    # --- Test 1: normal fit with enough data ---------------------------
    H0 = rng.standard_normal((500, d)).astype(np.float32)
    H1 = rng.standard_normal((500, d)).astype(np.float32) + 0.3
    m = StabilizedWhitenedCosineMethod(k=32, shrinkage=0.1, min_samples=100)
    m.fit(H0, H1)
    scores = m.score(np.concatenate([H0, H1]))
    assert scores.shape == (1000,), f"score shape: {scores.shape}"
    assert not m.is_fallback, "should NOT have fallen back with 1000 samples"
    print("[PASS] Test 1 — normal fit, whitening active")

    # --- Test 2: fit_region fallback → restores global whitening ------
    H0_tiny = rng.standard_normal((10, d)).astype(np.float32)
    H1_tiny = rng.standard_normal((10, d)).astype(np.float32)
    m2 = StabilizedWhitenedCosineMethod(k=32, shrinkage=0.1, min_samples=200)
    m2.fit(H0, H1)  # global fit first (succeeds)
    w_global = m2.w.copy()
    assert not m2.is_fallback, "global fit should succeed"
    ok = m2.fit_region(H0_tiny, H1_tiny)
    assert not ok, "fit_region should return False for tiny region"
    # KEY FIX: after fallback, w should be the GLOBAL whitened Fisher,
    # not flat-cosine ones vector
    assert not m2.is_fallback, "should restore global (non-fallback) state"
    assert np.allclose(m2.w, w_global), "w should match the global fit"
    s = m2.score(H0_tiny)
    assert s.shape == (10,), f"score shape: {s.shape}"
    print("[PASS] Test 2 — fit_region fallback restores global whitening")

    # --- Test 3: score_pairs consistency ------------------------------
    A = rng.standard_normal((50, d)).astype(np.float32)
    B = rng.standard_normal((50, d)).astype(np.float32)
    m3 = StabilizedWhitenedCosineMethod(k=16, shrinkage=0.1, min_samples=50)
    m3.fit(H0, H1)
    sp = m3.score_pairs(A, B)
    assert sp.shape == (50,), f"score_pairs shape: {sp.shape}"
    assert np.all(np.isfinite(sp)), "score_pairs produced non-finite values"
    print("[PASS] Test 3 — score_pairs produces finite output")

    # --- Test 4: fit_region succeeds with enough data -----------------
    m4 = StabilizedWhitenedCosineMethod(k=16, shrinkage=0.1, min_samples=100)
    m4.fit(H0, H1)
    ok = m4.fit_region(H0[:200], H1[:200])
    assert ok, "fit_region should succeed with 400 samples"
    assert not m4.is_fallback, "should NOT be in fallback"
    print("[PASS] Test 4 — fit_region succeeds with enough data")

    # --- Test 5: fallback=False still works ---------------------------
    m5 = StabilizedWhitenedCosineMethod(k=32, shrinkage=0.1, min_samples=200, fallback=False)
    m5.fit(H0, H1)
    s5 = m5.score(H0)
    assert np.all(np.isfinite(s5)), "scores should be finite"
    print("[PASS] Test 5 — fallback=False with enough data")

    # --- Test 6: verbose diagnostics ----------------------------------
    m6 = StabilizedWhitenedCosineMethod(k=16, shrinkage=0.1, min_samples=200, verbose=True)
    print("--- expected verbose output below ---")
    m6.fit(H0, H1)
    m6.fit_region(H0_tiny, H1_tiny)
    print("--- end verbose output ---")
    print("[PASS] Test 6 — verbose diagnostics print without error")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
