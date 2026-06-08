from __future__ import annotations

from typing import Optional

import numpy as np

from NeighborCache.region_local_threshold.methods import build_methods
from np_bench.methods.whitened_cosine import WhitenedCosineMethod, _inv_sqrt_cov


def _legacy_select_rank(
    eigvals_desc: np.ndarray,
    *,
    n: int,
    d: int,
    max_rank: Optional[int],
    rank_mode: str,
    explained_variance: float,
    abs_eps: float,
    rel_eps: float,
) -> int:
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
    valid_count = int(valid_vals.size)

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


def _legacy_inv_sqrt_cov(
    X: np.ndarray,
    *,
    abs_eps: float = 1e-6,
    rel_eps: float = 1e-6,
    max_rank: Optional[int] = 128,
    rank_mode: str = "explained_variance",
    explained_variance: float = 0.99,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d). Got {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one row.")

    n, d = X.shape
    Xf = np.asarray(X, dtype=np.float64)
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, n - 1)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_desc = np.maximum(eigvals[::-1], 0.0)
    eigvecs_desc = eigvecs[:, ::-1]

    k = _legacy_select_rank(
        eigvals_desc,
        n=n,
        d=d,
        max_rank=max_rank,
        rank_mode=rank_mode,
        explained_variance=explained_variance,
        abs_eps=abs_eps,
        rel_eps=rel_eps,
    )

    top_vals = eigvals_desc[:k]
    top_vecs = eigvecs_desc[:, :k]

    max_eigval = max(float(eigvals_desc[0]), 0.0)
    threshold = max(abs_eps, rel_eps * max_eigval)
    mask = top_vals > threshold

    if not np.any(mask):
        return np.zeros((d, d), dtype=np.float64)

    top_vals = top_vals[mask]
    top_vecs = top_vecs[:, mask]
    inv_sqrt = 1.0 / np.sqrt(top_vals)
    return (top_vecs * inv_sqrt[None, :]) @ top_vecs.T


def _fit_variant(
    whitening_type: str,
    H0: np.ndarray,
    H1: np.ndarray,
    **kwargs,
) -> WhitenedCosineMethod:
    method = WhitenedCosineMethod(
        name=f"test:{whitening_type}",
        whitening_type=whitening_type,  # type: ignore[arg-type]
        **kwargs,
    )
    return method.fit(H0, H1)


def test_backward_compat_zca_matches_legacy() -> None:
    rng = np.random.default_rng(10)
    H0 = rng.normal(loc=-0.1, scale=1.3, size=(90, 32)).astype(np.float32)
    H1 = rng.normal(loc=0.2, scale=0.9, size=(95, 32)).astype(np.float32)
    all_data = np.concatenate([H0, H1], axis=0)

    legacy_W = _legacy_inv_sqrt_cov(all_data)
    wrapper_W = _inv_sqrt_cov(all_data)
    method = _fit_variant("zca", H0, H1, shrinkage=0.0)

    np.testing.assert_allclose(wrapper_W, legacy_W, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(method.W, legacy_W, rtol=0.0, atol=1e-10)

    legacy = WhitenedCosineMethod(name="legacy")
    legacy.W = legacy_W
    legacy.mem_H0 = H0.copy()
    legacy.mem_H1 = H1.copy()
    legacy._refit_whitened()

    X_eval = rng.normal(size=(24, 32)).astype(np.float32)
    A = rng.normal(size=(24, 32)).astype(np.float32)
    B = rng.normal(size=(24, 32)).astype(np.float32)
    np.testing.assert_allclose(method.score(X_eval), legacy.score(X_eval), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(method.score_pairs(A, B), legacy.score_pairs(A, B), rtol=0.0, atol=1e-10)


def test_all_four_methods_are_registered_distinct() -> None:
    methods = build_methods()
    expected = {
        "PCAWhitenedCosine": "zca",
        "PCAWhitenedCosine_PCA": "pca",
        "PCAWhitenedCosine_ZCAcor": "zca_cor",
        "PCAWhitenedCosine_PCAcor": "pca_cor",
    }

    for name, whitening_type in expected.items():
        assert name in methods
        assert isinstance(methods[name], WhitenedCosineMethod)
        assert methods[name].whitening_type == whitening_type

    assert len({id(methods[name]) for name in expected}) == len(expected)


def test_pca_output_shape_is_k_dim() -> None:
    rng = np.random.default_rng(20)
    H0 = rng.normal(size=(120, 64)).astype(np.float32)
    H1 = rng.normal(loc=0.2, size=(125, 64)).astype(np.float32)
    method = _fit_variant("pca", H0, H1, max_rank=16, rank_mode="fixed")

    assert method.W.shape[1] == 64
    assert method.W.shape[0] == method.selected_rank_
    assert 1 <= method.W.shape[0] <= 16

    scores = method.score_pairs(
        rng.normal(size=(17, 64)).astype(np.float32),
        rng.normal(size=(17, 64)).astype(np.float32),
    )
    assert scores.shape == (17,)


def test_zca_cor_uses_correlation() -> None:
    rng = np.random.default_rng(30)
    scales = np.array([10.0, 1.0, 0.5, 3.0, 0.2, 5.0], dtype=np.float64)
    X = (rng.normal(size=(800, scales.size)) * scales + np.array([4.0, -2.0, 0.5, 1.5, 0.0, -3.0])).astype(
        np.float32
    )
    H0, H1 = X[:400], X[400:]

    zca = _fit_variant("zca", H0, H1, max_rank=None, rank_mode="threshold", explained_variance=1.0)
    zca_cor = _fit_variant("zca_cor", H0, H1, max_rank=None, rank_mode="threshold", explained_variance=1.0)

    assert not np.allclose(zca.W, zca_cor.W)

    centered = X.astype(np.float64) - zca_cor.mean_[None, :]
    Z = centered @ zca_cor.W.T
    cov_z = (Z.T @ Z) / max(1, Z.shape[0] - 1)
    np.testing.assert_allclose(np.diag(cov_z), np.ones(scales.size), rtol=1e-5, atol=1e-5)


def test_score_pairs_finite_and_bounded() -> None:
    rng = np.random.default_rng(40)
    H0 = rng.normal(size=(180, 32)).astype(np.float32)
    H1 = rng.normal(loc=0.1, size=(180, 32)).astype(np.float32)
    A = rng.normal(size=(60, 32)).astype(np.float32)
    B = rng.normal(size=(60, 32)).astype(np.float32)

    for whitening_type in ("zca", "pca", "zca_cor", "pca_cor"):
        method = _fit_variant(whitening_type, H0, H1)
        scores = method.score_pairs(A, B)
        assert scores.shape == (60,)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= -1.0 - 1e-10)
        assert np.all(scores <= 1.0 + 1e-10)


def test_rank_truncation_applies_to_all_variants() -> None:
    rng = np.random.default_rng(50)
    H0 = rng.normal(size=(180, 64)).astype(np.float32)
    H1 = rng.normal(loc=0.15, size=(180, 64)).astype(np.float32)

    for whitening_type in ("zca", "zca_cor"):
        method = _fit_variant(whitening_type, H0, H1, max_rank=4, rank_mode="fixed")
        effective_rank = int(np.sum(np.linalg.svd(method.W, compute_uv=False) > 1e-8))
        assert effective_rank <= 4
        assert method.selected_rank_ <= 4

    for whitening_type in ("pca", "pca_cor"):
        method = _fit_variant(whitening_type, H0, H1, max_rank=4, rank_mode="fixed")
        assert method.W.shape[0] <= 4
        assert method.selected_rank_ <= 4


def test_rank_diagnostics_across_variants(capsys) -> None:
    rng = np.random.default_rng(60)
    scales = np.linspace(0.2, 5.0, 32)
    X0 = (rng.normal(size=(240, 32)) * scales).astype(np.float32)
    X1 = (rng.normal(loc=0.1, size=(240, 32)) * scales).astype(np.float32)

    ranks = {}
    for whitening_type in ("zca", "pca", "zca_cor", "pca_cor"):
        method = _fit_variant(whitening_type, X0, X1, max_rank=24, rank_mode="explained_variance")
        ranks[whitening_type] = int(method.selected_rank_)

    print(f"selected whitening ranks: {ranks}")
    captured = capsys.readouterr()
    assert "selected whitening ranks" in captured.out
    assert all(1 <= rank <= 24 for rank in ranks.values())


def test_rotation_invariant_pairs_have_expected_equal_scores() -> None:
    rng = np.random.default_rng(70)
    H0 = rng.normal(size=(220, 48)).astype(np.float32)
    H1 = rng.normal(loc=0.2, size=(220, 48)).astype(np.float32)
    A = rng.normal(size=(35, 48)).astype(np.float32)
    B = rng.normal(size=(35, 48)).astype(np.float32)

    zca = _fit_variant("zca", H0, H1, max_rank=12, rank_mode="fixed")
    pca = _fit_variant("pca", H0, H1, max_rank=12, rank_mode="fixed")
    zca_cor = _fit_variant("zca_cor", H0, H1, max_rank=12, rank_mode="fixed")
    pca_cor = _fit_variant("pca_cor", H0, H1, max_rank=12, rank_mode="fixed")

    np.testing.assert_allclose(zca.score_pairs(A, B), pca.score_pairs(A, B), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(zca_cor.score_pairs(A, B), pca_cor.score_pairs(A, B), rtol=1e-10, atol=1e-10)
