from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sklearn")

from NeighborCache.region_local_threshold.evaluation import evaluate_methods_global
from NeighborCache.region_local_threshold.splits import GlobalSplit
from np_bench.methods.faiss_backed import PairContext, build_faiss_variant


class _LinearHadamardMethod:
    input_space = "embedding"

    def __init__(self, weights: np.ndarray, bias: float = 0.0) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)

    def score(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) @ self.weights + self.bias).astype(np.float32)

    def linear_form(self) -> tuple[str, np.ndarray, float]:
        return ("hadamard_linear", self.weights, self.bias)


def _synthetic_pair_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(36, 6)).astype(np.float32)
    a_unique = rng.normal(size=(9, 6)).astype(np.float32)
    anchor_ids = np.repeat(np.arange(9), 4)
    a = a_unique[anchor_ids]
    X = q * a
    return q, a, X, anchor_ids


def test_faiss_hadamard_linear_scores_match_with_negative_weights() -> None:
    q, a, X, anchor_ids = _synthetic_pair_data()
    weights = np.asarray([1.25, -0.75, 0.5, -1.5, 0.2, 2.0], dtype=np.float64)
    source = _LinearHadamardMethod(weights, bias=-0.3)
    pair_context = PairContext(
        query=q,
        anchor=a,
        anchor_ids=anchor_ids,
        features_are_hadamard=True,
        source="synthetic",
    )

    scorer, result = build_faiss_variant(
        source_name="synthetic_linear",
        source_method=source,
        pair_context=pair_context,
        X_main=X,
        X_cos=None,
    )

    assert result.eligible, result.reason
    assert scorer is not None
    idx = np.arange(X.shape[0], dtype=np.int64)
    assert np.allclose(scorer.score_with_indices(idx), source.score(X), atol=1e-4, rtol=1e-4)
    assert scorer.diagnostic_max_abs_diff <= 1e-4


def test_faiss_variant_matches_source_eval_rows_exactly() -> None:
    q, a, X, anchor_ids = _synthetic_pair_data()
    weights = np.asarray([0.7, -1.1, 0.4, 1.3, -0.6, 0.9], dtype=np.float64)
    source = _LinearHadamardMethod(weights, bias=0.05)
    pair_context = PairContext(
        query=q,
        anchor=a,
        anchor_ids=anchor_ids,
        features_are_hadamard=True,
        source="synthetic",
    )
    scorer, result = build_faiss_variant(
        source_name="linear",
        source_method=source,
        pair_context=pair_context,
        X_main=X,
        X_cos=None,
    )
    assert result.eligible, result.reason
    assert scorer is not None

    gs = GlobalSplit(
        H0_train=np.arange(0, 4, dtype=np.int64),
        H1_train=np.arange(18, 22, dtype=np.int64),
        H0_calib=np.arange(4, 12, dtype=np.int64),
        H1_calib=np.arange(22, 30, dtype=np.int64),
        H0_eval=np.arange(12, 18, dtype=np.int64),
        H1_eval=np.arange(30, 36, dtype=np.int64),
    )
    rows = evaluate_methods_global(
        {"linear": source, "linear [faiss]": scorer},
        ["linear", "linear [faiss]"],
        gs,
        X_main=X,
        X_cos=None,
        X_text=None,
        alpha=0.25,
        tie_mode="ge",
        tau_guardrail="none",
        tau_guardrail_delta=0.01,
        trial=0,
        seed=0,
        region_key="synthetic",
        region_id=np.zeros(X.shape[0], dtype=np.int64),
        failures=defaultdict(list),
    )
    by_method: dict[str, dict[str, Any]] = {str(r["method"]): r for r in rows}

    assert by_method["linear [faiss]"]["micro_tpr"] == by_method["linear"]["micro_tpr"]
    assert by_method["linear [faiss]"]["micro_fpr"] == by_method["linear"]["micro_fpr"]
    assert by_method["linear [faiss]"]["train_tpr"] == by_method["linear"]["train_tpr"]
    assert by_method["linear [faiss]"]["train_fpr"] == by_method["linear"]["train_fpr"]
    assert scorer.diagnostic_max_abs_diff <= 1e-4


def test_real_vcache_npz_slice_if_available() -> None:
    candidates = [
        Path("NeighborCache/data/vcache_lmarena_gte_pairs.npz"),
        Path("NeighborCache/data/vcache_classification_gte_pairs.npz"),
        Path("NeighborCache/data/vcache_searchqueries_gte_pairs.npz"),
        Path("NeighborCache/data/vcache_combo_gte_pairs.npz"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("no local vCache pair NPZ available")

    with np.load(path, allow_pickle=True) as data:
        if "query_emb" not in data or "anchor_emb" not in data or "emb" not in data:
            pytest.skip(f"{path} lacks query_emb/anchor_emb/emb")
        q = np.asarray(data["query_emb"][:128], dtype=np.float32)
        a = np.asarray(data["anchor_emb"][:128], dtype=np.float32)
        X = np.asarray(data["emb"][:128], dtype=np.float32)
        anchor_ids = np.asarray(data["anchor_qid"][:128]) if "anchor_qid" in data else None

    if q.shape != a.shape or X.shape != q.shape:
        pytest.skip("real fixture has incompatible pair shapes")
    if not np.allclose(X, q * a, atol=1e-4, rtol=1e-4):
        pytest.skip("real fixture emb is not exact query_emb*anchor_emb")

    weights = np.linspace(-1.0, 1.0, X.shape[1], dtype=np.float64)
    source = _LinearHadamardMethod(weights, bias=0.1)
    scorer, result = build_faiss_variant(
        source_name="real_linear",
        source_method=source,
        pair_context=PairContext(
            query=q,
            anchor=a,
            anchor_ids=anchor_ids,
            features_are_hadamard=True,
            source=str(path),
        ),
        X_main=X,
        X_cos=None,
    )
    assert result.eligible, result.reason
    assert scorer is not None
    idx = np.arange(X.shape[0], dtype=np.int64)
    assert np.allclose(scorer.score_with_indices(idx), source.score(X), atol=1e-4, rtol=1e-4)
