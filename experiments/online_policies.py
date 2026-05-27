from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset_stream import StreamExample
from .equivalence import EquivalenceJudge
from .vcache_metrics import add_cumulative_fields


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    return (arr / max(float(np.linalg.norm(arr)), eps)).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


@dataclass
class CacheEntry:
    cache_id: int
    example: StreamExample
    norm_embedding: np.ndarray
    created_at: int
    last_accessed: int


class ExactVectorCache:
    def __init__(self, capacity: int, eviction_policy: str = "mru") -> None:
        self.capacity = int(capacity)
        self.eviction_policy = eviction_policy
        self.entries: List[CacheEntry] = []
        self._matrix: Optional[np.ndarray] = None
        self._next_id = 0
        self._clock = 0

    def add(self, example: StreamExample) -> Optional[int]:
        if self.capacity <= 0 or example.embedding is None:
            return None
        while len(self.entries) >= self.capacity:
            self._evict_one()
        self._clock += 1
        cache_id = self._next_id
        self._next_id += 1
        norm_embedding = _normalize(example.embedding)
        if self._matrix is None:
            self._matrix = np.empty((self.capacity, int(norm_embedding.size)), dtype=np.float32)
        elif self._matrix.shape[1] != int(norm_embedding.size):
            raise ValueError(f"Embedding dimension mismatch in cache: {self._matrix.shape[1]} vs {norm_embedding.size}")
        self.entries.append(
            CacheEntry(
                cache_id=cache_id,
                example=example,
                norm_embedding=norm_embedding,
                created_at=self._clock,
                last_accessed=self._clock,
            )
        )
        self._matrix[len(self.entries) - 1] = norm_embedding
        return cache_id

    def nearest(self, embedding: Optional[np.ndarray]) -> Tuple[Optional[CacheEntry], Optional[float]]:
        if embedding is None or not self.entries:
            return None, None
        q = _normalize(embedding)
        if self._matrix is None:
            return None, None
        mat = self._matrix[: len(self.entries)]
        sims = mat @ q
        idx = int(np.argmax(sims))
        entry = self.entries[idx]
        self._clock += 1
        entry.last_accessed = self._clock
        return entry, float(sims[idx])

    def _evict_one(self) -> None:
        if not self.entries:
            return
        policy = self.eviction_policy.lower()
        if policy == "fifo":
            idx = int(np.argmin([e.created_at for e in self.entries]))
        elif policy == "lru":
            idx = int(np.argmin([e.last_accessed for e in self.entries]))
        elif policy == "mru":
            idx = int(np.argmax([e.last_accessed for e in self.entries]))
        else:
            idx = int(np.argmin([e.created_at for e in self.entries]))
        self.entries.pop(idx)
        if self._matrix is not None and idx < len(self.entries):
            self._matrix[idx : len(self.entries)] = self._matrix[idx + 1 : len(self.entries) + 1]


def _base_record(
    *,
    dataset: str,
    method: str,
    seed: int,
    param: Any,
    request_index: int,
    example: StreamExample,
    nearest: Optional[CacheEntry],
    decision: str,
    returned_response: str,
    correctness: Optional[bool],
    would_correct: Optional[bool],
    similarity_score: Optional[float],
    method_score: Optional[float],
    latency: float,
    llm_calls: int,
    online_judge_calls: int,
    evaluation_judge_calls: int,
    cache_size: int,
    embedding_model: str,
    judge_name: str,
    stream_hash: str,
) -> Dict[str, Any]:
    is_hit = decision in {"hit", "exploit"}
    tp = int(is_hit and correctness is True)
    fp = int(is_hit and correctness is False)
    tn = int((not is_hit) and would_correct is False)
    fn = int((not is_hit) and would_correct is True)
    nearest_id = nearest.example.id if nearest is not None else ""
    nearest_prompt = nearest.example.prompt if nearest is not None else ""
    nearest_label = "H1" if would_correct is True else "H0" if would_correct is False else ""
    return {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "delta_or_threshold": param,
        "delta_or_policy_param": param,
        "request_index": int(request_index),
        "stream_index": int(request_index),
        "prompt_id": example.id,
        "prompt": example.prompt,
        "query_id": example.id,
        "query_text": example.prompt,
        "nearest_prompt_id": nearest_id,
        "nearest_prompt": nearest_prompt,
        "nearest_candidate_id": nearest_id,
        "nearest_candidate_text_id": nearest_id,
        "nearest_candidate_response_id": "",
        "nearest_candidate_cosine": similarity_score,
        "nearest_candidate_label_H1_or_H0": nearest_label,
        "decision": decision,
        "decision_type": decision,
        "cache_hit_or_miss": "hit" if is_hit else "miss",
        "returned_response": returned_response,
        "gold_response": example.gold_response,
        "correctness": correctness,
        "nearest_would_be_correct": would_correct,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "similarity_score": similarity_score,
        "method_score": method_score,
        "latency": float(latency),
        "llm_calls": int(llm_calls),
        "online_judge_calls": int(online_judge_calls),
        "evaluation_judge_calls": int(evaluation_judge_calls),
        "online_judge_called": bool(online_judge_calls),
        "eval_judge_called": bool(evaluation_judge_calls),
        "judge_calls": int(online_judge_calls) + int(evaluation_judge_calls),
        "total_estimated_cost": "",
        "cache_size": int(cache_size),
        "embedding_model": embedding_model,
        "judge": judge_name,
        "stream_hash": stream_hash,
    }


def _evaluate_decision(
    judge: EquivalenceJudge,
    example: StreamExample,
    nearest: Optional[CacheEntry],
    *,
    is_hit: bool,
    returned_response: str,
) -> Tuple[Optional[bool], Optional[bool], int]:
    if nearest is None:
        return None, None, 0
    before = judge.calls
    if is_hit:
        correctness = judge.equivalent_examples(
            example,
            nearest.example,
            returned_response=returned_response,
        )
        would_correct = correctness
    else:
        correctness = None
        would_correct = judge.equivalent_examples(example, nearest.example)
    return correctness, would_correct, judge.calls - before


def run_cosine_policy(
    stream: Sequence[StreamExample],
    *,
    threshold: float,
    dataset: str,
    seed: int,
    cache_size: int,
    eviction_policy: str,
    judge: EquivalenceJudge,
    embedding_model: str,
    stream_hash: str,
) -> List[Dict[str, Any]]:
    cache = ExactVectorCache(capacity=cache_size, eviction_policy=eviction_policy)
    records: List[Dict[str, Any]] = []
    for i, example in enumerate(stream):
        t0 = time.perf_counter()
        nearest, sim = cache.nearest(example.embedding)
        is_hit = nearest is not None and sim is not None and sim >= float(threshold)
        if is_hit:
            returned = nearest.example.gold_response
            llm_calls = 0
        else:
            returned = example.gold_response
            llm_calls = 1
            cache.add(example)
        latency = time.perf_counter() - t0
        correctness, would_correct, eval_calls = _evaluate_decision(
            judge, example, nearest, is_hit=is_hit, returned_response=returned
        )
        records.append(
            _base_record(
                dataset=dataset,
                method="cosine",
                seed=seed,
                param=float(threshold),
                request_index=i,
                example=example,
                nearest=nearest,
                decision="hit" if is_hit else "miss",
                returned_response=returned,
                correctness=correctness,
                would_correct=would_correct,
                similarity_score=sim,
                method_score=sim,
                latency=latency,
                llm_calls=llm_calls,
                online_judge_calls=0,
                evaluation_judge_calls=eval_calls,
                cache_size=cache_size,
                embedding_model=embedding_model,
                judge_name=judge.name,
                stream_hash=stream_hash,
            )
        )
    add_cumulative_fields(records)
    return records


def make_pair_feature(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    if av.shape != bv.shape:
        raise ValueError(f"Embedding dimension mismatch: {av.shape} vs {bv.shape}")
    if mode == "hadamard":
        return av * bv
    if mode == "absdiff":
        return np.abs(av - bv)
    if mode == "concat":
        return np.concatenate([av, bv, np.abs(av - bv), av * bv], axis=0)
    if mode == "cosine":
        return np.asarray([cosine_similarity(av, bv)], dtype=np.float32)
    raise ValueError("--ours_pair_feature must be hadamard, absdiff, concat, or cosine")


def _select_np_tau(scores_h0: np.ndarray, alpha: float, tie_mode: str = "ge") -> float:
    s = np.asarray(scores_h0, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("empty H0 calibration scores")
    uniq, counts = np.unique(s, return_counts=True)
    n = int(s.size)
    cumsum = np.cumsum(counts)
    for i, tau in enumerate(uniq):
        if tie_mode == "gt":
            k = int(n - cumsum[i])
        else:
            k = int(n - (cumsum[i - 1] if i > 0 else 0))
        if k / max(1, n) <= float(alpha):
            return float(tau)
    return float("inf")


def _get_method_by_name(name: str) -> Any:
    from np_bench.methods import get_default_methods

    methods = get_default_methods(include_ablations=True, include_optional=True, include_ensemble=True)
    by_name = {str(getattr(m, "name", type(m).__name__)): m for m in methods}
    if name not in by_name:
        raise ValueError(f"Unknown ours method {name!r}. Available: {sorted(by_name)}")
    return by_name[name]


def _fit_method(method: Any, h0_train: np.ndarray, h1_train: np.ndarray, *, seed: int, alpha: float) -> None:
    from NeighborCache.region_local_threshold.methods import try_fit_method

    try_fit_method(method, h0_train, h1_train, weights=None, seed=seed, alpha=alpha)


def load_pairwise_train_calib(
    path: str,
    *,
    feature_key: str,
    label_key: str,
    alt_feature_key: Optional[str] = None,
    n_train: int,
    n_calib: int,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Pairwise data not found: {npz_path}")
    with np.load(str(npz_path), allow_pickle=True) as ds:
        if feature_key not in ds or label_key not in ds:
            raise ValueError(
                f"Pairwise NPZ must contain {feature_key!r} and {label_key!r}. Keys={list(ds.files)}"
            )
        X = np.asarray(ds[feature_key], dtype=np.float32)
        y = np.asarray(ds[label_key]).reshape(-1)
        X_alt = None
        if alt_feature_key:
            if alt_feature_key not in ds:
                raise ValueError(
                    f"Pairwise NPZ missing alt feature key {alt_feature_key!r}. Keys={list(ds.files)}"
                )
            X_alt = np.asarray(ds[alt_feature_key], dtype=np.float32).reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Pairwise feature matrix must be 2D, got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Pairwise X/y row mismatch: {X.shape[0]} vs {y.shape[0]}")
    if X_alt is not None and X_alt.shape[0] != y.shape[0]:
        raise ValueError(f"Pairwise X_alt/y row mismatch: {X_alt.shape[0]} vs {y.shape[0]}")

    rng = np.random.default_rng(int(seed))
    h0_idx = np.where(y == 0)[0]
    h1_idx = np.where(y == 1)[0]
    rng.shuffle(h0_idx)
    rng.shuffle(h1_idx)
    need0 = int(n_train) + int(n_calib)
    need1 = int(n_train)
    if h0_idx.size < need0 or h1_idx.size < need1:
        raise ValueError(
            "Not enough pairwise examples for ours calibration: "
            f"H0={h0_idx.size} need={need0}, H1={h1_idx.size} need={need1}"
        )
    h0_train = X[h0_idx[: int(n_train)]]
    h0_calib = X[h0_idx[int(n_train) : need0]]
    h1_train = X[h1_idx[: int(n_train)]]
    h1_calib = X[h1_idx[int(n_train) : min(h1_idx.size, int(n_train) + int(n_calib))]]
    if X_alt is None:
        return h0_train, h1_train, h0_calib, h1_calib, None, None, None, None
    h0_train_alt = X_alt[h0_idx[: int(n_train)]]
    h0_calib_alt = X_alt[h0_idx[int(n_train) : need0]]
    h1_train_alt = X_alt[h1_idx[: int(n_train)]]
    h1_calib_alt = X_alt[h1_idx[int(n_train) : min(h1_idx.size, int(n_train) + int(n_calib))]]
    return h0_train, h1_train, h0_calib, h1_calib, h0_train_alt, h1_train_alt, h0_calib_alt, h1_calib_alt


@dataclass
class OursModel:
    method: Any
    tau: float
    alpha: float
    pair_feature: str
    uses_alt_score: bool = False
    orientation: str = "higher"
    threshold_source: str = "calibrated"


def score_accepts(score: float, tau: float, orientation: str = "higher") -> bool:
    if orientation == "higher":
        return float(score) >= float(tau)
    if orientation == "lower":
        return float(score) <= float(tau)
    raise ValueError(f"Unknown score orientation {orientation!r}")


def train_ours_model(
    *,
    pairwise_data: str,
    method_name: str,
    feature_key: str,
    label_key: str,
    alt_feature_key: Optional[str],
    n_train: int,
    n_calib: int,
    seed: int,
    alpha: float,
    pair_feature: str,
) -> OursModel:
    method = copy.deepcopy(_get_method_by_name(method_name))
    input_space = str(getattr(method, "input_space", "embedding"))
    if input_space not in {"embedding", "mixed"}:
        raise ValueError(
            f"Method {method_name!r} uses input_space={getattr(method, 'input_space', None)!r}; "
            "this online adapter currently supports embedding or mixed pair-feature methods only."
        )
    (
        h0_train,
        h1_train,
        h0_calib,
        h1_calib,
        h0_train_alt,
        h1_train_alt,
        h0_calib_alt,
        h1_calib_alt,
    ) = load_pairwise_train_calib(
        pairwise_data,
        feature_key=feature_key,
        label_key=label_key,
        alt_feature_key=alt_feature_key if input_space == "mixed" else None,
        n_train=n_train,
        n_calib=n_calib,
        seed=seed,
    )

    uses_alt_score = False
    if input_space == "mixed":
        from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod

        if h0_train_alt is None or h1_train_alt is None or h0_calib_alt is None or h1_calib_alt is None:
            raise ValueError(
                f"Method {method_name!r} requires --ours_pairwise_alt_feature_key for scalar judge routing"
            )
        new_judges = []
        for judge in list(getattr(method, "judges", [])):
            if str(getattr(judge, "name", "")) == "Cosine":
                new_judges.append(PrecomputedCosineMethod())
            else:
                new_judges.append(judge)
        method.judges = new_judges
        judge_input = {}
        for judge in getattr(method, "judges", []):
            if str(getattr(judge, "input_space", "embedding")) == "scalar_score":
                judge_input[str(getattr(judge, "name", type(judge).__name__))] = "alt"
        method.fit(
            h0_train,
            h1_train,
            seed=seed,
            alpha=alpha,
            H0_calib=h0_calib,
            H1_calib=h1_calib,
            H0_train_alt=h0_train_alt,
            H1_train_alt=h1_train_alt,
            H0_calib_alt=h0_calib_alt,
            H1_calib_alt=h1_calib_alt,
            judge_input=judge_input,
            fit_context="vcache_comparison",
        )
        tau = float(getattr(method, "tau", _select_np_tau(method.score(h0_calib, h0_calib_alt), alpha=alpha)))
        uses_alt_score = True
    else:
        _fit_method(method, h0_train, h1_train, seed=seed, alpha=alpha)
        tau = _select_np_tau(method.score(h0_calib), alpha=alpha)
    return OursModel(
        method=method,
        tau=tau,
        alpha=float(alpha),
        pair_feature=pair_feature,
        uses_alt_score=uses_alt_score,
        orientation="higher",
        threshold_source="static_pairwise_npz_calibrated",
    )


def run_ours_policy(
    stream: Sequence[StreamExample],
    *,
    model: OursModel,
    method_label: str = "ours",
    dataset: str,
    seed: int,
    cache_size: int,
    eviction_policy: str,
    judge: EquivalenceJudge,
    embedding_model: str,
    stream_hash: str,
) -> List[Dict[str, Any]]:
    cache = ExactVectorCache(capacity=cache_size, eviction_policy=eviction_policy)
    records: List[Dict[str, Any]] = []
    for i, example in enumerate(stream):
        t0 = time.perf_counter()
        nearest, sim = cache.nearest(example.embedding)
        score = None
        is_hit = False
        if nearest is not None and example.embedding is not None and nearest.example.embedding is not None:
            feat = make_pair_feature(example.embedding, nearest.example.embedding, model.pair_feature).reshape(1, -1)
            if model.uses_alt_score:
                alt = np.asarray([[sim]], dtype=np.float32)
                score_arr = model.method.score(feat, X_alt=alt)
            else:
                score_arr = model.method.score(feat)
            score = float(np.asarray(score_arr).reshape(-1)[0])
            is_hit = score_accepts(score, float(model.tau), model.orientation)
        if is_hit:
            returned = nearest.example.gold_response
            llm_calls = 0
        else:
            returned = example.gold_response
            llm_calls = 1
            cache.add(example)
        latency = time.perf_counter() - t0
        correctness, would_correct, eval_calls = _evaluate_decision(
            judge, example, nearest, is_hit=is_hit, returned_response=returned
        )
        records.append(
            _base_record(
                dataset=dataset,
                method=method_label,
                seed=seed,
                param=float(model.alpha),
                request_index=i,
                example=example,
                nearest=nearest,
                decision="hit" if is_hit else "miss",
                returned_response=returned,
                correctness=correctness,
                would_correct=would_correct,
                similarity_score=sim,
                method_score=score,
                latency=latency,
                llm_calls=llm_calls,
                online_judge_calls=0,
                evaluation_judge_calls=eval_calls,
                cache_size=cache_size,
                embedding_model=embedding_model,
                judge_name=judge.name,
                stream_hash=stream_hash,
            )
        )
    add_cumulative_fields(records)
    return records
