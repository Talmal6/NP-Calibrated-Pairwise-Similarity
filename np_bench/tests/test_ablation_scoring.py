from __future__ import annotations

import unittest

import numpy as np

from NeighborCache.region_local_threshold.methods import build_methods


class AblationScoringTest(unittest.TestCase):
    def test_methods_are_registered_as_distinct_objects(self) -> None:
        methods = build_methods()

        pca_cos = methods["PCAWhitenedCosine"]
        pca_hadamard = methods["ablation:pca_whitened_hadamard_linear"]

        self.assertIsNot(pca_cos, pca_hadamard)
        self.assertEqual(getattr(pca_hadamard, "resolved_ablation", None), "pca_whitened_hadamard_linear")
        self.assertEqual(getattr(pca_cos, "name", None), "PCAWhitenedCosine")
        self.assertEqual(getattr(pca_hadamard, "name", None), "ablation:pca_whitened_hadamard_linear")
        self.assertNotEqual(getattr(pca_cos, "score_mode", None), getattr(pca_hadamard, "score_mode", None))

    def test_feature_space_scores_diverge(self) -> None:
        rng = np.random.default_rng(123)
        n_train, n_eval, d = 240, 64, 48

        H0_train = rng.normal(loc=-0.15, scale=1.0, size=(n_train, d)).astype(np.float32)
        H1_train = rng.normal(loc=0.20, scale=1.0, size=(n_train, d)).astype(np.float32)
        X_eval = rng.normal(loc=0.05, scale=1.0, size=(n_eval, d)).astype(np.float32)

        methods = build_methods()
        pca_cos = methods["PCAWhitenedCosine"]
        pca_hadamard = methods["ablation:pca_whitened_hadamard_linear"]

        pca_cos.fit(H0_train, H1_train)
        pca_hadamard.fit(H0_train, H1_train)

        scores_cos = np.asarray(pca_cos.score(X_eval), dtype=np.float64).reshape(-1)
        scores_hadamard = np.asarray(pca_hadamard.score(X_eval), dtype=np.float64).reshape(-1)

        self.assertEqual(scores_cos.shape, scores_hadamard.shape)
        self.assertFalse(np.allclose(scores_cos, scores_hadamard))
        self.assertGreater(float(np.max(np.abs(scores_cos - scores_hadamard))), 1e-4)

    def test_pair_score_and_feature_score_paths_are_explicit(self) -> None:
        rng = np.random.default_rng(321)
        n_train, n_eval, d = 160, 32, 24

        A_train = rng.normal(loc=-0.1, scale=1.0, size=(n_train, d)).astype(np.float32)
        B_train = rng.normal(loc=0.25, scale=1.0, size=(n_train, d)).astype(np.float32)
        X0_train = A_train * B_train
        X1_train = (
            rng.normal(loc=0.2, scale=1.1, size=(n_train, d))
            * rng.normal(loc=0.35, scale=0.9, size=(n_train, d))
        ).astype(np.float32)
        X_eval = rng.normal(loc=0.05, scale=1.0, size=(n_eval, d)).astype(np.float32)

        methods = build_methods()
        pca_cos = methods["PCAWhitenedCosine"]
        pca_hadamard = methods["ablation:pca_whitened_hadamard_linear"]

        pca_cos.fit(X0_train, X1_train)
        pca_hadamard.fit(X0_train, X1_train)

        feature_scores_cos = np.asarray(pca_cos.score(X_eval), dtype=np.float64).reshape(-1)
        feature_scores_hadamard = np.asarray(pca_hadamard.score(X_eval), dtype=np.float64).reshape(-1)

        self.assertFalse(np.allclose(feature_scores_cos, feature_scores_hadamard))
        self.assertEqual(feature_scores_cos.shape, feature_scores_hadamard.shape)


if __name__ == "__main__":
    unittest.main()
