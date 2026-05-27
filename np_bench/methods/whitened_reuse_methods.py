from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .base import OnlineBaseMethod


RankMode = Literal["fixed", "explained_variance", "threshold"]


BEST_WHITENED_LINEAR_PARAMS = {
    "pca_whiten_rank_mode": "fixed",
    "pca_whiten_max_rank": 128,
    "pca_whiten_explained_variance": 0.911621945041671,
    "pca_whiten_abs_eps": 0.0006149097843486478,
    "pca_whiten_rel_eps": 4.243156000968392e-06,
    "pca_whiten_norm_eps": 7.224083939349313e-08,
}


@dataclass(frozen=True)
class MLPTrainConfig:
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    batch_size: int = 256
    residual_scale: float = 0.25
    grad_clip_norm: Optional[float] = 5.0
    device: str = "cpu"


@dataclass(frozen=True)
class TailAwareConfig(MLPTrainConfig):
    # Fraction of H0 examples in each batch treated as hard-tail negatives.
    tail_fraction: float = 0.25
    # Pairwise softplus margin between positives and top-scoring negatives.
    tail_margin: float = 1.0
    # Weight on the low-FPR tail-ranking objective.
    tail_pair_weight: float = 0.50
    # Extra penalty that pushes top-scoring H0 logits below zero.
    tail_negative_weight: float = 0.10


def _validate_2d(name: str, X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d). Got {X.shape}.")


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _select_rank(
    eigvals_desc: np.ndarray,
    *,
    n: int,
    d: int,
    max_rank: Optional[int],
    rank_mode: RankMode,
    explained_variance: float,
    abs_eps: float,
    rel_eps: float,
) -> int:
    """Select the number of PCA directions to retain."""
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be positive or None.")
    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    rank_cap = min(max(n - 1, 1), d)
    if max_rank is not None:
        rank_cap = min(rank_cap, max_rank)

    if eigvals_desc.size == 0:
        return 1

    max_eigval = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eigval)
    valid = eigvals_desc > threshold

    if not np.any(valid):
        return 1

    valid_vals = eigvals_desc[valid]
    valid_count = len(valid_vals)

    if rank_mode == "fixed":
        k = rank_cap
    elif rank_mode == "threshold":
        k = valid_count
    elif rank_mode == "explained_variance":
        total = float(np.sum(valid_vals))
        if total <= 0.0:
            return 1
        ratios = np.cumsum(valid_vals) / total
        k = int(np.searchsorted(ratios, explained_variance) + 1)
    else:
        raise ValueError(f"Unknown rank_mode: {rank_mode!r}")

    return max(1, min(k, valid_count, rank_cap))


def _inv_sqrt_cov(
    X: np.ndarray,
    *,
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-6,
    max_rank: Optional[int] = 128,
    rank_mode: RankMode = "explained_variance",
    explained_variance: float = 0.99,
) -> np.ndarray:
    """
    Compute a truncated PCA inverse-square-root covariance transform.

    Given X with shape (n, d), returns W with shape (d, d). Directions outside
    the retained PCA subspace are zeroed out rather than amplified.
    """
    _validate_2d("X", X)
    n, d = X.shape
    if n == 0:
        raise ValueError("X must contain at least one row.")

    X_float = np.asarray(X, dtype=np.float64)
    X_centered = X_float - X_float.mean(axis=0, keepdims=True)

    denom = max(n - 1, 1)
    cov = (X_centered.T @ X_centered) / denom
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_desc = np.maximum(eigvals[::-1], 0.0)
    eigvecs_desc = eigvecs[:, ::-1]

    k = _select_rank(
        eigvals_desc,
        n=n,
        d=d,
        max_rank=max_rank,
        rank_mode=rank_mode,
        explained_variance=explained_variance,
        abs_eps=abs_eps,
        rel_eps=rel_eps,
    )

    vals = eigvals_desc[:k]
    vecs = eigvecs_desc[:, :k]

    max_eigval = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eigval)
    mask = vals > threshold

    if not np.any(mask):
        return np.zeros((d, d), dtype=np.float64)

    vals = vals[mask]
    vecs = vecs[:, mask]
    inv_sqrt = 1.0 / np.sqrt(vals)
    return (vecs * inv_sqrt[None, :]) @ vecs.T


class WhitenedLinearMethod(OnlineBaseMethod):
    """
    Version 1: supervised whitened linear scorer.

    Actual score path:
        x = emb(query) * emb(anchor)      # Hadamard pair feature
        z = x @ W                         # truncated pooled-covariance whitening
        score = z @ (mu1 - mu0) + b

    This is LDA/nearest-centroid-like, but with a truncated global covariance
    inverse and explicit FPR calibration expected outside this scorer.
    """

    def __init__(
        self,
        name: str = "WhitenedLinear",
        *,
        pca_whiten_rank_mode: RankMode = "fixed",
        pca_whiten_max_rank: Optional[int] = 128,
        pca_whiten_explained_variance: float = 0.911621945041671,
        pca_whiten_abs_eps: float = 0.0006149097843486478,
        pca_whiten_rel_eps: float = 4.243156000968392e-06,
        pca_whiten_norm_eps: float = 7.224083939349313e-08,
        eps: Optional[float] = None,
        rel_eps: Optional[float] = None,
        max_rank: Optional[int] = None,
        rank_mode: Optional[RankMode] = None,
        explained_variance: Optional[float] = None,
        norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.name = name

        if eps is not None:
            pca_whiten_abs_eps = eps
        if rel_eps is not None:
            pca_whiten_rel_eps = rel_eps
        if max_rank is not None:
            pca_whiten_max_rank = max_rank
        if rank_mode is not None:
            pca_whiten_rank_mode = rank_mode
        if explained_variance is not None:
            pca_whiten_explained_variance = explained_variance
        if norm_eps is not None:
            pca_whiten_norm_eps = norm_eps

        self.pca_whiten_rank_mode = pca_whiten_rank_mode
        self.pca_whiten_max_rank = pca_whiten_max_rank
        self.pca_whiten_explained_variance = float(pca_whiten_explained_variance)
        self.pca_whiten_abs_eps = float(pca_whiten_abs_eps)
        self.pca_whiten_rel_eps = float(pca_whiten_rel_eps)
        self.pca_whiten_norm_eps = float(pca_whiten_norm_eps)

        self.W: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.mu0_: Optional[np.ndarray] = None
        self.mu1_: Optional[np.ndarray] = None
        self.mem_H0: Optional[np.ndarray] = None
        self.mem_H1: Optional[np.ndarray] = None

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
    ) -> "WhitenedLinearMethod":
        del weights, seed
        _validate_2d("H0_train", H0_train)
        _validate_2d("H1_train", H1_train)

        if H0_train.shape[1] != H1_train.shape[1]:
            raise ValueError(
                "H0_train and H1_train must have the same feature dimension. "
                f"Got {H0_train.shape[1]} and {H1_train.shape[1]}."
            )
        if H0_train.shape[0] == 0 or H1_train.shape[0] == 0:
            raise ValueError("H0_train and H1_train must both contain at least one row.")

        H0 = np.asarray(H0_train, dtype=np.float64)
        H1 = np.asarray(H1_train, dtype=np.float64)
        pooled = np.concatenate([H0, H1], axis=0)

        self.W = _inv_sqrt_cov(
            pooled,
            abs_eps=self.pca_whiten_abs_eps,
            rel_eps=self.pca_whiten_rel_eps,
            max_rank=self.pca_whiten_max_rank,
            rank_mode=self.pca_whiten_rank_mode,
            explained_variance=self.pca_whiten_explained_variance,
        )

        self.mem_H0 = H0.copy()
        self.mem_H1 = H1.copy()
        self._refit_whitened()
        return self

    def _check_is_fitted(self) -> None:
        if self.W is None or self.w is None:
            raise RuntimeError("Method is not fitted. Call fit(...) first.")

    def _refit_whitened(self) -> None:
        if self.W is None:
            raise RuntimeError("Missing whitening matrix W. Call fit(...) first.")
        if self.mem_H0 is None or self.mem_H1 is None:
            raise RuntimeError("Missing stored training data. Call fit(...) first.")

        Z0 = self.mem_H0 @ self.W
        Z1 = self.mem_H1 @ self.W

        mu0 = Z0.mean(axis=0)
        mu1 = Z1.mean(axis=0)
        d_z = mu1 - mu0

        self.mu0_ = mu0
        self.mu1_ = mu1
        self.w = self.W @ d_z
        self.b = -0.5 * float((mu0 + mu1) @ d_z)

    def refit(self) -> None:
        if self.W is not None and self.mem_H0 is not None and self.mem_H1 is not None:
            self._refit_whitened()
        else:
            super().refit()

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        _validate_2d("X", X)
        assert self.W is not None
        if X.shape[1] != self.W.shape[0]:
            raise ValueError(
                "X feature dimension does not match fitted W. "
                f"Got {X.shape[1]}, expected {self.W.shape[0]}."
            )
        return np.asarray(X, dtype=np.float64) @ self.W

    def base_score(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        _validate_2d("X", X)
        assert self.w is not None
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                "X feature dimension does not match fitted scorer. "
                f"Got {X.shape[1]}, expected {self.w.shape[0]}."
            )
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.base_score(X)

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        _validate_2d("A", A)
        _validate_2d("B", B)
        if A.shape != B.shape:
            raise ValueError(f"A and B must have the same shape. Got {A.shape} and {B.shape}.")
        return self.score(np.asarray(A, dtype=np.float64) * np.asarray(B, dtype=np.float64))

    def score_hadamard_features(self, H: np.ndarray) -> np.ndarray:
        return self.score(H)

    def score_pairs_whitened_cosine(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Diagnostic only: true cosine(A @ W, B @ W). Not the main score."""
        self._check_is_fitted()
        _validate_2d("A", A)
        _validate_2d("B", B)
        if A.shape != B.shape:
            raise ValueError(f"A and B must have the same shape. Got {A.shape} and {B.shape}.")
        assert self.W is not None
        WA = _l2_normalize(np.asarray(A, dtype=np.float64) @ self.W, eps=self.pca_whiten_norm_eps)
        WB = _l2_normalize(np.asarray(B, dtype=np.float64) @ self.W, eps=self.pca_whiten_norm_eps)
        return np.sum(WA * WB, axis=1)


class _TorchResidualMLP:
    """Lazy wrapper so importing this file does not hard-require torch."""

    @staticmethod
    def require_torch():
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError(
                "WhitenedResidualMLPMethod and TailAwareResidualMLPMethod require PyTorch. "
                "Install torch or use WhitenedLinearMethod."
            ) from exc
        return torch, nn, F

    @staticmethod
    def make_model(input_dim: int, hidden_dim: int, depth: int, dropout: float):
        torch, nn, _ = _TorchResidualMLP.require_torch()
        if depth <= 0:
            raise ValueError("depth must be positive.")

        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers)

        # Start the final residual head near zero so the method begins close to
        # the stable whitened-linear scorer instead of immediately overwriting it.
        final = model[-1]
        if hasattr(final, "weight"):
            nn.init.zeros_(final.weight)
        if hasattr(final, "bias"):
            nn.init.zeros_(final.bias)
        return model


class WhitenedResidualMLPMethod(WhitenedLinearMethod):
    """
    Version 2: whitened linear base + small nonlinear residual MLP.

    score(x) = base_linear(x) + residual_scale * MLP(x @ W)

    The whitening/base scorer is fitted first and then frozen. The residual MLP
    is trained to correct nonlinear hard-neighbor mistakes without discarding the
    stable LDA-like base scorer.
    """

    def __init__(
        self,
        name: str = "WhitenedResidualMLP",
        *,
        mlp_config: Optional[MLPTrainConfig] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.mlp_config = mlp_config or MLPTrainConfig()
        self.residual_model = None
        self._torch_device = self.mlp_config.device

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
    ) -> "WhitenedResidualMLPMethod":
        super().fit(H0_train, H1_train, weights=weights, seed=seed)
        self._fit_residual_mlp(
            H0_train=np.asarray(H0_train, dtype=np.float64),
            H1_train=np.asarray(H1_train, dtype=np.float64),
            seed=seed,
            tail_aware=False,
        )
        return self

    def _prepare_train_tensors(self, H0_train: np.ndarray, H1_train: np.ndarray):
        torch, _, _ = _TorchResidualMLP.require_torch()
        Z0 = self.transform(H0_train).astype(np.float32)
        Z1 = self.transform(H1_train).astype(np.float32)
        base0 = self.base_score(H0_train).astype(np.float32)
        base1 = self.base_score(H1_train).astype(np.float32)

        device = torch.device(self.mlp_config.device)
        return (
            torch.from_numpy(Z0).to(device),
            torch.from_numpy(Z1).to(device),
            torch.from_numpy(base0).to(device),
            torch.from_numpy(base1).to(device),
        )

    def _sample_balanced_batch(self, n0: int, n1: int, batch_size: int, rng: np.random.Generator):
        half = max(1, batch_size // 2)
        i0 = rng.integers(0, n0, size=half)
        i1 = rng.integers(0, n1, size=batch_size - half)
        return i0, i1

    def _fit_residual_mlp(
        self,
        *,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        seed=None,
        tail_aware: bool,
    ) -> None:
        torch, nn, F = _TorchResidualMLP.require_torch()
        cfg = self.mlp_config
        if seed is not None:
            torch.manual_seed(int(seed))
        rng = np.random.default_rng(seed)

        Z0, Z1, base0, base1 = self._prepare_train_tensors(H0_train, H1_train)
        input_dim = Z0.shape[1]
        self.residual_model = _TorchResidualMLP.make_model(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
        ).to(torch.device(cfg.device))

        opt = torch.optim.AdamW(
            self.residual_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        bce = nn.BCEWithLogitsLoss()

        n0, n1 = Z0.shape[0], Z1.shape[0]
        steps_per_epoch = max(1, int(np.ceil((n0 + n1) / max(cfg.batch_size, 1))))
        scale = float(cfg.residual_scale)

        self.residual_model.train()
        for _ in range(cfg.epochs):
            for _ in range(steps_per_epoch):
                i0, i1 = self._sample_balanced_batch(n0, n1, cfg.batch_size, rng)
                idx0 = torch.as_tensor(i0, device=Z0.device, dtype=torch.long)
                idx1 = torch.as_tensor(i1, device=Z1.device, dtype=torch.long)

                z = torch.cat([Z0[idx0], Z1[idx1]], dim=0)
                base = torch.cat([base0[idx0], base1[idx1]], dim=0)
                y = torch.cat([
                    torch.zeros(len(i0), device=Z0.device),
                    torch.ones(len(i1), device=Z1.device),
                ])

                residual = self.residual_model(z).squeeze(-1)
                logits = base + scale * residual
                loss = bce(logits, y)

                if tail_aware:
                    loss = self._add_tail_loss(loss, logits[: len(i0)], logits[len(i0) :])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.residual_model.parameters(), cfg.grad_clip_norm)
                opt.step()

        self.residual_model.eval()

    def _add_tail_loss(self, loss, neg_logits, pos_logits):
        return loss

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        base = self.base_score(X)
        if self.residual_model is None:
            return base

        torch, _, _ = _TorchResidualMLP.require_torch()
        cfg = self.mlp_config
        Z = self.transform(X).astype(np.float32)
        device = torch.device(cfg.device)

        outs = []
        self.residual_model.eval()
        with torch.no_grad():
            for start in range(0, Z.shape[0], max(cfg.batch_size, 1)):
                z_batch = torch.from_numpy(Z[start : start + cfg.batch_size]).to(device)
                residual = self.residual_model(z_batch).squeeze(-1)
                outs.append(residual.detach().cpu().numpy())

        residual_np = np.concatenate(outs, axis=0) if outs else np.empty((0,), dtype=np.float32)
        return base + float(cfg.residual_scale) * residual_np.astype(np.float64)


class TailAwareResidualMLPMethod(WhitenedResidualMLPMethod):
    """
    Version 3: residual MLP trained with a low-FPR tail-aware loss.

    Adds a ranking penalty between positives and the highest-scoring H0 examples
    in each batch. This targets the part of the negative score distribution that
    will determine the calibrated FPR threshold.
    """

    def __init__(
        self,
        name: str = "TailAwareResidualMLP",
        *,
        mlp_config: Optional[TailAwareConfig] = None,
        **kwargs,
    ):
        super().__init__(name=name, mlp_config=mlp_config or TailAwareConfig(), **kwargs)
        if not isinstance(self.mlp_config, TailAwareConfig):
            # Allows callers to pass a structurally compatible config, but keeps
            # the tail fields available.
            self.mlp_config = TailAwareConfig(**self.mlp_config.__dict__)

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
    ) -> "TailAwareResidualMLPMethod":
        WhitenedLinearMethod.fit(self, H0_train, H1_train, weights=weights, seed=seed)
        self._fit_residual_mlp(
            H0_train=np.asarray(H0_train, dtype=np.float64),
            H1_train=np.asarray(H1_train, dtype=np.float64),
            seed=seed,
            tail_aware=True,
        )
        return self

    def _add_tail_loss(self, loss, neg_logits, pos_logits):
        _, _, F = _TorchResidualMLP.require_torch()
        cfg: TailAwareConfig = self.mlp_config  # type: ignore[assignment]

        if neg_logits.numel() == 0 or pos_logits.numel() == 0:
            return loss

        k = max(1, int(np.ceil(float(cfg.tail_fraction) * int(neg_logits.numel()))))
        k = min(k, int(neg_logits.numel()))
        hard_neg = neg_logits.topk(k=k, largest=True).values

        # Penalize hard negatives that outrank positives by margin.
        pair_loss = F.softplus(
            hard_neg[:, None] + float(cfg.tail_margin) - pos_logits[None, :]
        ).mean()

        # Penalize hard negatives that are confidently on the positive side.
        neg_tail_loss = F.softplus(hard_neg).mean()

        return (
            loss
            + float(cfg.tail_pair_weight) * pair_loss
            + float(cfg.tail_negative_weight) * neg_tail_loss
        )


__all__ = [
    "RankMode",
    "BEST_WHITENED_LINEAR_PARAMS",
    "MLPTrainConfig",
    "TailAwareConfig",
    "WhitenedLinearMethod",
    "WhitenedResidualMLPMethod",
    "TailAwareResidualMLPMethod",
]
