import numpy as np
from typing import Optional
from .base import OnlineBaseMethod

def _inv_sqrt_cov(X: np.ndarray, eps: float = 1e-6,
                  max_rank: Optional[int] = None) -> np.ndarray:
    """
    Compute truncated inverse-square-root of the covariance of X.

    Returns W of shape (d, d) such that  X @ W  is approximately whitened
    in the top-k PCA directions, where k = min(n-1, d, max_rank).
    Directions beyond k are zeroed out rather than amplified.
    """
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, n)
    # eigendecompose (ascending order)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Determine rank: keep at most min(n-1, d, max_rank) components
    # whose eigenvalue exceeds eps
    k = min(n - 1, d)
    if max_rank is not None:
        k = min(k, max_rank)
    k = max(k, 1)

    # Take the top-k eigenvalues (largest, at end due to ascending order)
    top_vals = eigvals[-k:]
    top_vecs = eigvecs[:, -k:]  # (d, k)

    # Only keep components with meaningful variance
    mask = top_vals > eps
    if not mask.any():
        # Degenerate: return identity-scaled
        return np.eye(d, dtype=X.dtype) / np.sqrt(eps)

    top_vals = top_vals[mask]
    top_vecs = top_vecs[:, mask]  # (d, k')

    # W = V_k @ diag(1/sqrt(lambda_k)) @ V_k^T
    inv_sqrt = 1.0 / np.sqrt(top_vals)
    W = (top_vecs * inv_sqrt[None, :]) @ top_vecs.T  # (d, d)
    return W

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

class WhitenedCosineMethod(OnlineBaseMethod):
    def __init__(self, name="WhitenedCosine", eps=1e-6, max_rank: int = 128):
        super().__init__()
        self.name = name
        self.eps = eps
        self.max_rank = max_rank
        self.W = None  # (d,d)

    def fit(self, H0_train: np.ndarray, H1_train: np.ndarray, *,
            weights=None, seed=None) -> "WhitenedCosineMethod":
        # Learn whitening from pooled data
        all_data = np.concatenate([H0_train, H1_train], axis=0)
        self.W = _inv_sqrt_cov(all_data, eps=self.eps, max_rank=self.max_rank)
        # Store for potential refit via online update
        self.mem_H0 = H0_train.copy()
        self.mem_H1 = H1_train.copy()
        self._refit_whitened()
        return self

    def _refit_whitened(self) -> None:
        # Fisher direction in whitened space  (W is symmetric)
        WH0 = self.mem_H0 @ self.W
        WH1 = self.mem_H1 @ self.W
        mu0 = WH0.mean(axis=0)
        mu1 = WH1.mean(axis=0)
        d_w = mu1 - mu0
        # score(x) = (x @ W) @ d_w + b = x @ (W @ d_w) + b
        self.w = self.W @ d_w
        self.b = -0.5 * float((mu0 + mu1) @ d_w)

    def refit(self) -> None:
        if self.W is not None and self.mem_H0 is not None and self.mem_H1 is not None:
            self._refit_whitened()
        else:
            super().refit()

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # A,B: (m,d)
        WA = _l2_normalize(A @ self.W.T)
        WB = _l2_normalize(B @ self.W.T)
        return np.sum(WA * WB, axis=1)