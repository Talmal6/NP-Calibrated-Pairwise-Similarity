from __future__ import annotations

import numpy as np

from NeighborCache.region_local_threshold.methods import build_methods
from NeighborCache.region_local_threshold.evaluation import _select_tau
from experiments.online_policies import _select_np_tau
from np_bench.methods.streaming_whitening import (
    EWMACovarianceAccumulator,
    PageHinkleyDriftDetector,
    StreamingWhitening,
    StreamingWhitenedCosineMethod,
)
from np_bench.methods.whitened_cosine import WhitenedCosineMethod
from np_bench.thresholding import select_np_threshold


def _weighted_cov_reference(X: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xf = np.asarray(X, dtype=np.float64)
    wf = np.asarray(weights, dtype=np.float64)
    mean = np.average(Xf, axis=0, weights=wf)
    Xc = Xf - mean[None, :]
    denom = float(np.sum(wf) - np.sum(wf * wf) / np.sum(wf))
    cov = (Xc.T * wf) @ Xc / denom
    return 0.5 * (cov + cov.T), mean


def test_accumulator_no_forgetting_matches_numpy_cov() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 7)).astype(np.float32)

    acc = EWMACovarianceAccumulator(forgetting_half_life_samples=None)
    acc.update_batch(X[:11])
    acc.update_batch(X[11:])
    cov, mean, effective_n = acc.covariance()

    np.testing.assert_allclose(mean, np.mean(X.astype(np.float64), axis=0), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cov, np.cov(X.astype(np.float64), rowvar=False), rtol=1e-12, atol=1e-12)
    assert effective_n == float(X.shape[0])


def test_accumulator_ewma_matches_explicit_batch_weight_reference() -> None:
    rng = np.random.default_rng(456)
    X0 = rng.normal(size=(6, 5))
    X1 = rng.normal(loc=0.4, size=(4, 5))
    half_life = 8
    decay = 2.0 ** (-X1.shape[0] / half_life)

    acc = EWMACovarianceAccumulator(forgetting_half_life_samples=half_life)
    acc.update_batch(X0)
    acc.update_batch(X1)
    cov, mean, effective_n = acc.covariance()

    X = np.vstack([X0, X1])
    weights = np.concatenate([np.full(X0.shape[0], decay), np.ones(X1.shape[0])])
    cov_ref, mean_ref = _weighted_cov_reference(X, weights)

    np.testing.assert_allclose(mean, mean_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-12, atol=1e-12)
    expected_effective_n = float(np.sum(weights) ** 2 / np.sum(weights * weights))
    np.testing.assert_allclose(effective_n, expected_effective_n, rtol=1e-12, atol=1e-12)


def test_streaming_fit_matches_offline_for_all_variants_without_partial_fit() -> None:
    rng = np.random.default_rng(789)
    H0 = rng.normal(loc=-0.2, scale=1.1, size=(90, 24)).astype(np.float32)
    H1 = rng.normal(loc=0.25, scale=0.9, size=(95, 24)).astype(np.float32)
    X_eval = rng.normal(size=(32, 24)).astype(np.float32)
    A = rng.normal(size=(32, 24)).astype(np.float32)
    B = rng.normal(size=(32, 24)).astype(np.float32)

    for whitening_type in ("zca", "pca", "zca_cor", "pca_cor"):
        offline = WhitenedCosineMethod(
            name=f"offline:{whitening_type}",
            whitening_type=whitening_type,  # type: ignore[arg-type]
            max_rank=12,
            rank_mode="fixed",
        ).fit(H0, H1)
        streaming = StreamingWhitenedCosineMethod(
            name=f"streaming:{whitening_type}",
            whitening_type=whitening_type,  # type: ignore[arg-type]
            max_rank=12,
            rank_mode="fixed",
            forgetting_half_life_samples=None,
            min_refresh_samples=1,
            refresh_every_samples=10_000,
            drift_detector="none",
            min_h0_for_threshold=8,
        ).fit(H0, H1)

        np.testing.assert_allclose(streaming.W, offline.W, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(streaming.score(X_eval), offline.score(X_eval), atol=1e-6, rtol=1e-8)
        np.testing.assert_allclose(streaming.score_pairs(A, B), offline.score_pairs(A, B), atol=1e-6, rtol=1e-8)


def test_streaming_methods_are_opt_in_not_default_registered() -> None:
    default_methods = build_methods()
    streaming_methods = build_methods(include_streaming=True)

    assert "StreamingPCAWhitenedCosine" not in default_methods
    assert "StreamingPCAWhitenedCosine" in streaming_methods
    assert "StreamingPCAWhitenedCosine_PCAcor" in streaming_methods


def test_region_cli_flag_adds_streaming_methods() -> None:
    from NeighborCache.region_local_threshold.cli import _build_configured_methods, parse_args

    args = parse_args(
        [
            "--data",
            "dummy.npz",
            "--region_key",
            "global_cluster",
            "--include_streaming_whitening",
        ]
    )
    methods = _build_configured_methods(args, X_cos=None, X_text=None, quiet=True)

    assert "StreamingPCAWhitenedCosine" in methods
    assert "StreamingPCAWhitenedCosine_ZCAcor" in methods


def test_streaming_whitening_tempered_gamma_and_degenerate_data_are_finite() -> None:
    X = np.ones((20, 6), dtype=np.float32)
    sw = StreamingWhitening(
        whitening_type="zca_cor",
        gamma=0.25,
        max_rank=None,
        rank_mode="threshold",
        min_refresh_samples=1,
        drift_detector="none",
    )
    sw.initialize(X)
    Z = sw.transform(X)
    assert Z.shape == (20, 6)
    assert np.all(np.isfinite(Z))
    assert np.all(np.isfinite(sw.snapshot.W))  # type: ignore[union-attr]


def test_page_hinkley_triggers_on_synthetic_shift() -> None:
    detector = PageHinkleyDriftDetector(delta=0.0, threshold=2.0, min_samples=5, cooldown_samples=0)
    alarms = []
    for i, value in enumerate([0.0] * 8 + [5.0] * 4, start=1):
        alarms.append(detector.update(value, sample_index=i))
    assert any(alarms)


def test_shared_threshold_wrappers_are_consistent() -> None:
    scores = np.array([0.1, 0.2, 0.2, 0.5, 0.7, 0.9])
    direct = select_np_threshold(scores, alpha=0.25, tie_mode="ge", guardrail="none")
    eval_tau = _select_tau(scores, alpha=0.25, tie_mode="ge", guardrail="none", guardrail_delta=0.01)
    online_tau = _select_np_tau(scores, alpha=0.25, tie_mode="ge")
    assert direct == eval_tau == online_tau


def test_threshold_activation_is_atomic_when_h0_is_insufficient() -> None:
    rng = np.random.default_rng(999)
    H0 = rng.normal(size=(24, 10)).astype(np.float32)
    H1 = rng.normal(loc=0.2, size=(24, 10)).astype(np.float32)
    batch = rng.normal(loc=2.0, size=(40, 10)).astype(np.float32)

    method = StreamingWhitenedCosineMethod(
        forgetting_half_life_samples=None,
        min_refresh_samples=1,
        refresh_every_samples=10,
        drift_detector="none",
        min_h0_for_threshold=1000,
    ).fit(H0, H1)
    old_W = method.W.copy()
    old_version = method.transform_version_

    returned_version = method.refresh_transform_and_threshold(alpha=0.05)

    assert returned_version == old_version
    np.testing.assert_allclose(method.W, old_W, rtol=0.0, atol=0.0)
    assert method.diagnostics()["activation_reason"] == "insufficient_h0"

    method_ok = StreamingWhitenedCosineMethod(
        forgetting_half_life_samples=None,
        min_refresh_samples=1,
        refresh_every_samples=10,
        drift_detector="none",
        min_h0_for_threshold=8,
    ).fit(H0, H1)
    old_version_ok = method_ok.transform_version_
    method_ok.partial_fit(batch, H0_calib_batch=H0)
    new_version = method_ok.refresh_transform_and_threshold(alpha=0.05)

    assert new_version > old_version_ok
    assert np.isfinite(method_ok.tau_np)
    assert method_ok.diagnostics()["activation_reason"] == "activated"


def test_fit_alpha_seeds_threshold_for_auto_partial_fit_refresh() -> None:
    rng = np.random.default_rng(1001)
    H0 = rng.normal(size=(24, 10)).astype(np.float32)
    H1 = rng.normal(loc=0.2, size=(24, 10)).astype(np.float32)
    batch = rng.normal(loc=1.5, size=(40, 10)).astype(np.float32)

    method = StreamingWhitenedCosineMethod(
        forgetting_half_life_samples=None,
        min_refresh_samples=1,
        refresh_every_samples=10,
        drift_detector="none",
        min_h0_for_threshold=8,
    ).fit(H0, H1, alpha=0.05)
    old_version = method.transform_version_

    assert method.alpha == 0.05
    assert np.isfinite(method.tau_np)

    result = method.partial_fit(batch, H0_calib_batch=H0)

    assert result["refreshed"] is True
    assert method.transform_version_ > old_version
    assert np.isfinite(method.tau_np)
    assert method.diagnostics()["activation_reason"] == "activated"
