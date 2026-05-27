from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .online_policies import OursModel, _fit_method, _get_method_by_name


@dataclass
class OnlinePairs:
    X: np.ndarray
    y: np.ndarray
    cosine: np.ndarray
    query_id: np.ndarray
    candidate_id: np.ndarray
    query_emb: Optional[np.ndarray] = None
    candidate_emb: Optional[np.ndarray] = None


def load_pairs(path: Path) -> OnlinePairs:
    if not path.exists():
        raise FileNotFoundError(f"Missing online-mined pairs file: {path}")
    with np.load(path, allow_pickle=True) as ds:
        if "X_hadamard" in ds.files:
            X = np.asarray(ds["X_hadamard"], dtype=np.float32)
        elif "query_emb" in ds.files and "candidate_emb" in ds.files:
            X = np.asarray(ds["query_emb"], dtype=np.float32) * np.asarray(ds["candidate_emb"], dtype=np.float32)
        else:
            raise ValueError(f"{path} must contain X_hadamard or query_emb/candidate_emb. Keys={ds.files}")
        y = np.asarray(ds["label"]).reshape(-1).astype(np.int32)
        cosine = np.asarray(ds["cosine_similarity"], dtype=np.float32).reshape(-1)
        query_id = np.asarray(ds["query_id"]).reshape(-1)
        candidate_id = np.asarray(ds["candidate_id"]).reshape(-1)
        query_emb = np.asarray(ds["query_emb"], dtype=np.float32) if "query_emb" in ds.files else None
        candidate_emb = np.asarray(ds["candidate_emb"], dtype=np.float32) if "candidate_emb" in ds.files else None
    if X.ndim != 2:
        raise ValueError(f"{path} feature matrix must be 2D, got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{path} X/label row mismatch: {X.shape[0]} vs {y.shape[0]}")
    bad = set(np.unique(y).tolist()) - {0, 1}
    if bad:
        raise ValueError(f"{path} contains unexpected labels {sorted(bad)}")
    return OnlinePairs(X=X, y=y, cosine=cosine, query_id=query_id, candidate_id=candidate_id, query_emb=query_emb, candidate_emb=candidate_emb)


def load_pairs_dir(pairs_dir: Path) -> Tuple[OnlinePairs, OnlinePairs, OnlinePairs, Dict[str, Any]]:
    pairs_dir = Path(pairs_dir)
    with (pairs_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return (
        load_pairs(pairs_dir / "online_train_pairs.npz"),
        load_pairs(pairs_dir / "online_calib_pairs.npz"),
        load_pairs(pairs_dir / "online_eval_pairs.npz"),
        config,
    )


def load_eval_stream_from_pairs_dir(pairs_dir: Path, *, limit: Optional[int] = None):
    from .compare_vcache_real_datasets import SPECS, _embedding_choice, load_real_stream

    pairs_dir = Path(pairs_dir)
    with (pairs_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    dataset = str(config["dataset"])
    args = config.get("args", {})
    embedding_arg = str(args.get("embedding_model", "GTE"))
    embedding_key, embedding_model_name = _embedding_choice(embedding_arg)
    spec = SPECS[dataset]
    stream, info = load_real_stream(
        spec,
        embedding_key=embedding_key,
        response_key=str(config.get("response_key", spec.default_response_key)),
        seed=int(args.get("seed", 42)),
        limit=args.get("max_examples"),
    )
    split_info = config.get("split_info", {})
    start = int(split_info.get("eval_start", 0))
    n_eval = int(split_info.get("n_eval", max(0, len(stream) - start)))
    eval_stream = list(stream[start : start + n_eval])
    if limit is not None:
        eval_stream = eval_stream[: int(limit)]
    return eval_stream, info, config, embedding_model_name


def h0_h1(pairs: OnlinePairs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h0 = pairs.y == 0
    h1 = pairs.y == 1
    return pairs.X[h0], pairs.X[h1], pairs.cosine[h0].reshape(-1, 1), pairs.cosine[h1].reshape(-1, 1)


def _prepare_method(method_name: str) -> Tuple[Any, bool, Dict[str, str]]:
    method = copy.deepcopy(_get_method_by_name(method_name))
    input_space = str(getattr(method, "input_space", "embedding"))
    if input_space not in {"embedding", "mixed"}:
        raise ValueError(f"Method {method_name!r} uses unsupported input_space={input_space!r}")
    judge_input: Dict[str, str] = {}
    uses_alt = False
    if input_space == "mixed":
        from np_bench.methods.precomputed_cosine import PrecomputedCosineMethod

        new_judges = []
        for judge in list(getattr(method, "judges", [])):
            if str(getattr(judge, "name", "")) == "Cosine":
                new_judges.append(PrecomputedCosineMethod())
            else:
                new_judges.append(judge)
        method.judges = new_judges
        for judge in getattr(method, "judges", []):
            if str(getattr(judge, "input_space", "embedding")) == "scalar_score":
                judge_input[str(getattr(judge, "name", type(judge).__name__))] = "alt"
        uses_alt = True
    return method, uses_alt, judge_input


def score_method(method: Any, X: np.ndarray, X_alt: Optional[np.ndarray], uses_alt: bool) -> np.ndarray:
    if uses_alt:
        if X_alt is None:
            raise ValueError("Method requires scalar alt scores but X_alt is None")
        return np.asarray(method.score(X, X_alt=X_alt), dtype=np.float64).reshape(-1)
    return np.asarray(method.score(X), dtype=np.float64).reshape(-1)


def infer_orientation(scores_h0: np.ndarray, scores_h1: np.ndarray) -> Tuple[str, str]:
    mean0 = float(np.mean(scores_h0))
    mean1 = float(np.mean(scores_h1))
    pooled = np.concatenate([np.asarray(scores_h0).reshape(-1), np.asarray(scores_h1).reshape(-1)])
    spread = float(np.std(pooled))
    diff = mean1 - mean0
    if abs(diff) <= max(1e-8, 0.02 * spread):
        return "ambiguous", f"score distributions have weak separation: mean_H1={mean1}, mean_H0={mean0}, std={spread}"
    return ("higher", "") if diff > 0.0 else ("lower", "")


def select_tau(scores_h0: np.ndarray, alpha: float, orientation: str) -> float:
    s = np.asarray(scores_h0, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return float("inf") if orientation == "higher" else float("-inf")
    uniq, counts = np.unique(s, return_counts=True)
    n = int(s.size)
    if orientation == "higher":
        cumsum = np.cumsum(counts)
        for i, tau in enumerate(uniq):
            accepted = n - (int(cumsum[i - 1]) if i > 0 else 0)
            if accepted / max(1, n) <= float(alpha):
                return float(tau)
        return float("inf")
    if orientation == "lower":
        cumsum = np.cumsum(counts)
        best = float("-inf")
        for tau, accepted in zip(uniq, cumsum):
            if int(accepted) / max(1, n) <= float(alpha):
                best = float(tau)
        return best
    raise ValueError(f"Unknown orientation {orientation!r}")


def accept_scores(scores: np.ndarray, tau: float, orientation: str) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if orientation == "higher":
        return s >= float(tau)
    if orientation == "lower":
        return s <= float(tau)
    raise ValueError(f"Unknown orientation {orientation!r}")


def train_online_mined_model(
    *,
    pairs_dir: Path,
    method_name: str,
    alpha: float,
    seed: int,
    pair_feature: str = "hadamard",
) -> Tuple[OursModel, Dict[str, Any]]:
    train, calib, eval_pairs, config = load_pairs_dir(Path(pairs_dir))
    h0_train, h1_train, h0_train_alt, h1_train_alt = h0_h1(train)
    h0_calib, h1_calib, h0_calib_alt, h1_calib_alt = h0_h1(calib)
    h0_eval, h1_eval, h0_eval_alt, h1_eval_alt = h0_h1(eval_pairs)
    if h0_train.size == 0 or h1_train.size == 0:
        raise ValueError("Online-mined train pairs must contain both H0 and H1")
    if h0_calib.size == 0 or h1_calib.size == 0:
        raise ValueError("Online-mined calibration pairs must contain both H0 and H1")

    method, uses_alt, judge_input = _prepare_method(method_name)
    if uses_alt:
        method.fit(
            h0_train,
            h1_train,
            seed=seed,
            alpha=float(alpha),
            H0_train_alt=h0_train_alt,
            H1_train_alt=h1_train_alt,
            judge_input=judge_input,
            fit_context="online_mined_train_only",
        )
    else:
        _fit_method(method, h0_train, h1_train, seed=seed, alpha=float(alpha))

    s0_calib = score_method(method, h0_calib, h0_calib_alt, uses_alt)
    s1_calib = score_method(method, h1_calib, h1_calib_alt, uses_alt)
    orientation, orientation_warning = infer_orientation(s0_calib, s1_calib)
    if orientation == "ambiguous":
        orientation = "higher"
    tau = select_tau(s0_calib, float(alpha), orientation)

    s0_eval = score_method(method, h0_eval, h0_eval_alt, uses_alt)
    s1_eval = score_method(method, h1_eval, h1_eval_alt, uses_alt)
    diag = {
        "orientation": orientation,
        "orientation_warning": orientation_warning,
        "tau": float(tau),
        "calib_accept_rate_H0": float(np.mean(accept_scores(s0_calib, tau, orientation))),
        "calib_accept_rate_H1": float(np.mean(accept_scores(s1_calib, tau, orientation))),
        "eval_accept_rate_H0": float(np.mean(accept_scores(s0_eval, tau, orientation))),
        "eval_accept_rate_H1": float(np.mean(accept_scores(s1_eval, tau, orientation))),
        "n_train_pairs": int(train.y.size),
        "n_calib_pairs": int(calib.y.size),
        "n_eval_pairs": int(eval_pairs.y.size),
        "calibration_labels_used": int(train.y.size + calib.y.size),
        "calibration_split_size": int(calib.y.size),
        "calibration_equivalence_mode": str(config.get("equivalence_mode", config.get("args", {}).get("equivalence_mode", ""))),
        "pairs_config": config,
    }
    model = OursModel(
        method=method,
        tau=float(tau),
        alpha=float(alpha),
        pair_feature=pair_feature,
        uses_alt_score=uses_alt,
        orientation=orientation,
        threshold_source="online_mined_calibrated",
    )
    return model, diag
