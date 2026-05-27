from __future__ import annotations

import unittest

import numpy as np

from np_bench.methods.fisher_hadamard_methods import (
    HadamardCosineMethod,
    FisherWhitenedHadamardPooledMethod,
    FisherWhitenedHadamardWithinMethod,
)


class FisherHadamardMethodsTest(unittest.TestCase):
    def test_hadamard_cosine_equivalence(self) -> None:
        rng = np.random.default_rng(1)
        n, d = 64, 48

        A = rng.normal(size=(n, d))
        B = rng.normal(size=(n, d))
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)

        H = A * B
        m = HadamardCosineMethod().fit(H, H)

        s_h = m.score(H)
        s_cos = np.sum(A * B, axis=1)

        np.testing.assert_allclose(s_h, s_cos, rtol=1e-10, atol=1e-10)

    def test_shape_validation(self) -> None:
        rng = np.random.default_rng(2)
        H0 = rng.normal(size=(30, 20))
        H1 = rng.normal(size=(30, 20))

        m = FisherWhitenedHadamardPooledMethod().fit(H0, H1)

        A = rng.normal(size=(10, 20))
        B_bad = rng.normal(size=(11, 20))
        with self.assertRaises(ValueError):
            _ = m.score_pairs(A, B_bad)

        X_bad = rng.normal(size=(10, 19))
        with self.assertRaises(ValueError):
            _ = m.score(X_bad)

    def test_no_mixed_space_bug(self) -> None:
        rng = np.random.default_rng(3)
        n_train, n_eval, d = 100, 40, 32

        H0 = rng.normal(loc=-0.05, scale=0.8, size=(n_train, d))
        H1 = rng.normal(loc=0.20, scale=0.8, size=(n_train, d))

        m = FisherWhitenedHadamardWithinMethod().fit(H0, H1)

        A = rng.normal(size=(n_eval, d))
        B = rng.normal(size=(n_eval, d))

        expected = m.score(A * B)
        got = m.score_pairs(A, B)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)

    def test_pooled_and_within_fit_finite_scores(self) -> None:
        rng = np.random.default_rng(4)
        n, d = 160, 40

        H0 = rng.normal(loc=-0.1, scale=1.0, size=(n, d))
        H1 = rng.normal(loc=0.25, scale=1.0, size=(n, d))
        X = rng.normal(size=(50, d))

        methods = [
            FisherWhitenedHadamardPooledMethod(max_rank=None),
            FisherWhitenedHadamardWithinMethod(max_rank=None),
        ]

        for method in methods:
            method.fit(H0, H1)
            scores = method.score(X)
            self.assertTrue(np.all(np.isfinite(scores)))

            if not method.fallback_used:
                self.assertGreaterEqual(method.selected_rank_, 1)
                self.assertLessEqual(method.selected_rank_, d)


if __name__ == "__main__":
    unittest.main()
