from __future__ import annotations

import unittest

import numpy as np

from NeighborCache.region_local_threshold.methods import build_methods
from np_bench.methods.cosine import CosineMethod
from np_bench.methods.lda import LDAMethod
from np_bench.methods.random_forest_ensemble import (
    RandomForestEnsembleConfig,
    RandomForestEnsembleMethod,
)
from np_bench.methods.weighted_ensemble import EnsembleConfig


class RandomForestEnsembleTest(unittest.TestCase):
    def test_registered_in_default_methods(self) -> None:
        methods = build_methods()
        self.assertIn("RandomForestEnsemble", methods)
        self.assertIn("WeightedEnsembleNoPCAWhitenedCosine", methods)
        self.assertIn("RandomForestEnsembleNoPCAWhitenedCosine", methods)

        for name in [
            "WeightedEnsembleNoPCAWhitenedCosine",
            "RandomForestEnsembleNoPCAWhitenedCosine",
        ]:
            judge_names = [
                str(getattr(judge, "name", type(judge).__name__))
                for judge in getattr(methods[name], "judges", [])
            ]
            self.assertNotIn("WhitenedCosine", judge_names)
            self.assertIn("LDA", judge_names)

    def test_fit_score_with_external_calibration(self) -> None:
        rng = np.random.default_rng(7)
        d = 10
        n_train = 80
        n_calib = 70
        n_eval = 24

        H0_train = rng.normal(loc=-0.25, scale=1.0, size=(n_train, d)).astype(np.float32)
        H1_train = rng.normal(loc=0.25, scale=1.0, size=(n_train, d)).astype(np.float32)
        H0_calib = rng.normal(loc=-0.20, scale=1.0, size=(n_calib, d)).astype(np.float32)
        H1_calib = rng.normal(loc=0.20, scale=1.0, size=(n_calib, d)).astype(np.float32)
        X_eval = rng.normal(loc=0.0, scale=1.0, size=(n_eval, d)).astype(np.float32)

        method = RandomForestEnsembleMethod(
            judges=[CosineMethod(), LDAMethod()],
            config=EnsembleConfig(alpha=0.05, n_random_weights=4),
            rf_config=RandomForestEnsembleConfig(
                n_estimators=16,
                max_depth=3,
                min_samples_leaf=2,
            ),
        )

        method.fit(
            H0_train,
            H1_train,
            seed=123,
            alpha=0.05,
            H0_calib=H0_calib,
            H1_calib=H1_calib,
        )
        scores = np.asarray(method.score(X_eval), dtype=np.float64).reshape(-1)

        self.assertEqual(scores.shape, (n_eval,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertIsNotNone(method.clf)
        self.assertIsNotNone(method.feature_importances_)


if __name__ == "__main__":
    unittest.main()
