from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-neighborcache")

from .dataset_stream import DatasetInfo, StreamExample, _to_embedding, load_dataset_stream
from .equivalence import EquivalenceJudge
from .online_policies import ExactVectorCache, run_cosine_policy, run_ours_policy, train_ours_model
from .online_mined_ours import load_eval_stream_from_pairs_dir, train_online_mined_model
from .vcache_adapter import VCacheAdapterConfig, VCacheUnavailableError, run_vcache_policy
from .vcache_metrics import (
    RAW_DECISION_FIELDS,
    RAW_DECISION_MINIMAL_FIELDS,
    SUMMARY_FIELDS,
    generate_plots,
    summarize_all,
    summarize_run,
    write_csv,
    write_json,
)


DEFAULT_DELTAS = [0.01, 0.02, 0.03, 0.05, 0.08]
DEFAULT_THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
DEFAULT_BUDGETS = [0.01, 0.02, 0.03, 0.05, 0.08]
DEFAULT_HARD_THRESHOLDS = [0.80, 0.85, 0.90]


EMBEDDING_COLUMNS = {
    "GTE": ("emb_gte", "GteLargeENv1_5"),
    "GteLargeENv1_5": ("emb_gte", "GteLargeENv1_5"),
    "thenlper/gte-large": ("emb_gte", "GteLargeENv1_5"),
    "E5_LARGE_V2": ("emb_e5_large_v2", "E5_Large_v2"),
    "E5_Large_v2": ("emb_e5_large_v2", "E5_Large_v2"),
    "intfloat/e5-large-v2": ("emb_e5_large_v2", "E5_Large_v2"),
    "text-embedding-3-small": ("emb_text-embedding-3-small", "text-embedding-3-small"),
}


LEARNED_METHODS = {
    "ours_whitened_hadamard": "WhitenedCosine",
    "ours_weighted_ensemble": "WeightedEnsemble",
    "ours_xgboost": "XGBoost",
    "ours_lda": "LDA",
    "ours_tiny_mlp": "Tiny MLP",
}


@dataclass
class RealDatasetSpec:
    canonical_name: str
    display_name: str
    local_path: str
    hf_id: str
    prompt_key: str
    id_key: str
    cluster_key: str
    default_response_key: str
    default_pairwise_path: Optional[str]
    pairwise_mode: str


SPECS = {
    "SemCacheLMArena": RealDatasetSpec(
        canonical_name="SemCacheLMArena",
        display_name="SemCacheLMArena",
        local_path="NeighborCache/data/SemBenchmarkLmArena_local",
        hf_id="vCache/SemBenchmarkLmArena",
        prompt_key="prompt",
        id_key="id",
        cluster_key="ID_Set",
        default_response_key="response_gpt-4o-mini",
        default_pairwise_path="NeighborCache/data/arena_emb_gte.npz",
        pairwise_mode="anchor_qid",
    ),
    "SemCacheSearchQueries": RealDatasetSpec(
        canonical_name="SemCacheSearchQueries",
        display_name="SemCacheSearchQueries",
        local_path="NeighborCache/data/SemBenchmarkSearchQueries_train.npz",
        hf_id="vCache/SemBenchmarkSearchQueries",
        prompt_key="prompt",
        id_key="id",
        cluster_key="id_set",
        default_response_key="response_llama_3_8b",
        default_pairwise_path=None,
        pairwise_mode="from_stream_annotations",
    ),
}


DATASET_STATS_FIELDS = [
    "dataset",
    "source",
    "n_examples",
    "prompt_key",
    "response_key",
    "id_key",
    "cluster_key",
    "embedding_key",
    "embedding_model",
    "warnings",
]


HARD_STATS_FIELDS = [
    "dataset",
    "subset_name",
    "threshold",
    "n_examples",
    "n_with_nearest",
    "n_reusable",
    "n_non_reusable",
    "positive_ratio",
    "cosine_min",
    "cosine_p50",
    "cosine_p95",
    "cosine_max",
    "warning",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare vCache on official real semantic-cache datasets.")
    ap.add_argument("--datasets", nargs="+", default=["SemCacheLMArena", "SemCacheSearchQueries"], choices=sorted(SPECS))
    ap.add_argument("--output_dir", default="results/vcache_real_datasets_comparison")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["vcache", "cosine", "ours_whitened_hadamard", "ours_weighted_ensemble"],
        choices=["vcache", "cosine", *sorted(LEARNED_METHODS)],
    )
    ap.add_argument("--delta_values", nargs="+", type=float, default=DEFAULT_DELTAS)
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--target_budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    ap.add_argument("--hard_neighbor_thresholds", nargs="+", type=float, default=DEFAULT_HARD_THRESHOLDS)
    ap.add_argument("--embedding_model", default="GTE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_size", type=int, default=4096)
    ap.add_argument("--eviction_policy", choices=["mru", "lru", "fifo"], default="mru")
    ap.add_argument("--limit", type=int, default=None, help="Debug/smoke limit. Leave unset for full streams.")
    ap.add_argument("--max_examples", type=int, default=None, help="Alias for --limit for smoke runs.")
    ap.add_argument("--vcache_repo_path", default="/tmp/vcache-official")
    ap.add_argument("--vcache_policy", choices=["local", "global"], default="local")
    ap.add_argument("--vcache_async_updates", action="store_true")
    ap.add_argument("--ours_n_train", type=int, default=1200)
    ap.add_argument("--ours_n_calib", type=int, default=1200)
    ap.add_argument("--ours_calibration_source", choices=["pairwise_npz", "online_mined"], default="pairwise_npz")
    ap.add_argument("--online_pairs_dir", default=None)
    ap.add_argument(
        "--raw_log_mode",
        choices=["full", "minimal", "none"],
        default="full",
        help=(
            "full writes raw_decisions.csv with prompt/response text; minimal writes "
            "raw_decisions_minimal.csv with ids/scores/decisions only; none skips raw decision CSV."
        ),
    )
    ap.add_argument("--pairwise_sample_per_class", type=int, default=8000)
    ap.add_argument(
        "--allow_pairwise_from_stream_annotations",
        action="store_true",
        help=(
            "Allow constructing learned-method pairwise train/calibration rows from official stream id_set labels "
            "when no separate pairwise file exists. This is recorded as a fairness warning."
        ),
    )
    return ap.parse_args(argv)


def _embedding_choice(name: str) -> Tuple[str, str]:
    if name not in EMBEDDING_COLUMNS:
        raise SystemExit(
            f"Unsupported --embedding_model {name!r}. Available official precomputed choices: "
            f"{sorted(EMBEDDING_COLUMNS)}. Do not pass BGE unless the official stream has BGE embeddings."
        )
    return EMBEDDING_COLUMNS[name]


def stream_hash(stream: Sequence[StreamExample]) -> str:
    h = hashlib.sha256()
    for ex in stream:
        h.update(str(ex.id).encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(ex.prompt.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def _load_hf_saved_stream(
    path: Path,
    *,
    spec: RealDatasetSpec,
    embedding_key: str,
    response_key: str,
    seed: int,
    limit: Optional[int],
) -> Tuple[List[StreamExample], DatasetInfo]:
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise ImportError("Loading saved Hugging Face datasets requires the `datasets` package") from exc

    ds = load_from_disk(str(path))
    keys = list(ds.column_names)
    missing = [k for k in [spec.prompt_key, spec.id_key, spec.cluster_key, embedding_key, response_key] if k not in keys]
    if missing:
        raise ValueError(f"{spec.canonical_name} missing columns {missing}. Existing columns={keys}")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(ds))
    if limit is not None:
        order = order[: int(limit)]

    examples: List[StreamExample] = []
    for out_i, row_i in enumerate(order):
        row = ds[int(row_i)]
        emb = _to_embedding(row.get(embedding_key), key=embedding_key)
        examples.append(
            StreamExample(
                id=str(row.get(spec.id_key, out_i)),
                prompt=str(row.get(spec.prompt_key, "")),
                gold_response=str(row.get(response_key, "")),
                cluster=str(row.get(spec.cluster_key)),
                metadata={k: row.get(k) for k in [spec.id_key, spec.cluster_key, "dataset_name"] if k in row},
                embedding=emb,
            )
        )

    info = DatasetInfo(
        path=str(path),
        name=spec.canonical_name,
        n_examples=len(examples),
        prompt_key=spec.prompt_key,
        response_key=response_key,
        id_key=spec.id_key,
        cluster_key=spec.cluster_key,
        embedding_key=embedding_key,
        warnings=[],
    )
    return examples, info


def load_real_stream(
    spec: RealDatasetSpec,
    *,
    embedding_key: str,
    response_key: str,
    seed: int,
    limit: Optional[int],
) -> Tuple[List[StreamExample], DatasetInfo]:
    path = Path(spec.local_path)
    if path.exists() and path.is_dir():
        return _load_hf_saved_stream(
            path,
            spec=spec,
            embedding_key=embedding_key,
            response_key=response_key,
            seed=seed,
            limit=limit,
        )
    if path.exists():
        return load_dataset_stream(
            str(path),
            prompt_key=spec.prompt_key,
            response_key=response_key,
            id_key=spec.id_key,
            cluster_key=spec.cluster_key,
            embedding_key=embedding_key,
            seed=seed,
            shuffle=True,
            limit=limit,
        )
    raise FileNotFoundError(
        f"Official dataset {spec.canonical_name} not found locally at {path}. "
        f"Expected Hugging Face dataset: {spec.hf_id}. Download it before running, or place it at {path}."
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr if norm <= 1e-12 else (arr / norm).astype(np.float32, copy=False)


def compute_hard_subset_info(
    stream: Sequence[StreamExample],
    *,
    cache_size: int,
    eviction_policy: str,
) -> List[Dict[str, Any]]:
    cache = ExactVectorCache(capacity=cache_size, eviction_policy=eviction_policy)
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(stream):
        nearest, sim = cache.nearest(ex.embedding)
        reusable = None
        if nearest is not None:
            reusable = str(ex.cluster) == str(nearest.example.cluster)
        rows.append(
            {
                "index": i,
                "nearest_cosine": sim,
                "nearest_reusable": reusable,
                "nearest_prompt_id": nearest.example.id if nearest is not None else "",
            }
        )
        cache.add(ex)
    return rows


def _hard_stats(dataset: str, subset_name: str, threshold: Optional[float], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sims = np.asarray([r["nearest_cosine"] for r in rows if r.get("nearest_cosine") is not None], dtype=float)
    labels = [r.get("nearest_reusable") for r in rows if r.get("nearest_reusable") is not None]
    pos = sum(1 for x in labels if x is True)
    neg = sum(1 for x in labels if x is False)
    ratio = None if pos + neg == 0 else pos / float(pos + neg)
    warning = ""
    if len(rows) < 100:
        warning = "hard subset too small"
    if ratio is not None and (ratio <= 0.05 or ratio >= 0.95):
        warning = (warning + "; " if warning else "") + "hard subset has weak H0/H1 mixture"
    return {
        "dataset": dataset,
        "subset_name": subset_name,
        "threshold": "" if threshold is None else float(threshold),
        "n_examples": len(rows),
        "n_with_nearest": int(sims.size),
        "n_reusable": pos,
        "n_non_reusable": neg,
        "positive_ratio": ratio,
        "cosine_min": float(np.min(sims)) if sims.size else None,
        "cosine_p50": float(np.quantile(sims, 0.50)) if sims.size else None,
        "cosine_p95": float(np.quantile(sims, 0.95)) if sims.size else None,
        "cosine_max": float(np.max(sims)) if sims.size else None,
        "warning": warning,
    }


def build_subsets(
    dataset: str,
    stream: Sequence[StreamExample],
    hard_rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
) -> Tuple[Dict[str, List[StreamExample]], List[Dict[str, Any]], Dict[str, List[float]]]:
    subsets: Dict[str, List[StreamExample]] = {f"{dataset}_full": list(stream)}
    stats = [_hard_stats(dataset, f"{dataset}_full", None, hard_rows)]
    hist: Dict[str, List[float]] = {
        f"{dataset}_full": [float(r["nearest_cosine"]) for r in hard_rows if r.get("nearest_cosine") is not None]
    }
    for threshold in thresholds:
        name = f"{dataset}_hard_cos_ge_{threshold:.2f}"
        selected_idx = [
            int(r["index"])
            for r in hard_rows
            if r.get("nearest_cosine") is not None and float(r["nearest_cosine"]) >= float(threshold)
        ]
        selected_rows = [hard_rows[i] for i in selected_idx]
        subsets[name] = [stream[i] for i in selected_idx]
        stats.append(_hard_stats(dataset, name, threshold, selected_rows))
        hist[name] = [float(r["nearest_cosine"]) for r in selected_rows if r.get("nearest_cosine") is not None]
    return subsets, stats, hist


def _sample_balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out: List[np.ndarray] = []
    for label in [0, 1]:
        idx = np.flatnonzero(labels.reshape(-1) == label)
        if idx.size < per_class:
            raise ValueError(f"Not enough pairwise label={label} rows: {idx.size} < {per_class}")
        rng.shuffle(idx)
        out.append(idx[:per_class])
    all_idx = np.concatenate(out)
    rng.shuffle(all_idx)
    return all_idx.astype(np.int64, copy=False)


def prepare_pairwise_hadamard(
    spec: RealDatasetSpec,
    stream: Sequence[StreamExample],
    *,
    output_dir: Path,
    seed: int,
    per_class: int,
    embedding_model_name: str,
    allow_from_stream_annotations: bool,
) -> Tuple[Path, List[str]]:
    warnings: List[str] = []
    out_dir = output_dir / "pairwise_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.canonical_name}_{embedding_model_name}_hadamard_s{seed}_k{per_class}.npz"
    if out_path.exists():
        return out_path, warnings

    id_to_embedding = {str(ex.id): np.asarray(ex.embedding, dtype=np.float32).reshape(-1) for ex in stream}

    feats: List[np.ndarray] = []
    labels: List[int] = []
    cosines: List[float] = []

    if spec.default_pairwise_path and Path(spec.default_pairwise_path).exists():
        with np.load(spec.default_pairwise_path, allow_pickle=True) as ds:
            needed = ["emb", "label", "qid", "anchor_qid", "cosine_to_anchor"]
            missing = [k for k in needed if k not in ds.files]
            if missing:
                raise ValueError(f"Pairwise file {spec.default_pairwise_path} missing keys {missing}")
            y = np.asarray(ds["label"]).reshape(-1)
            idx = _sample_balanced_indices(y, per_class=per_class, seed=seed)
            q_emb = np.asarray(ds["emb"], dtype=np.float32)
            qids = np.asarray(ds["qid"]).reshape(-1)
            anchors = np.asarray(ds["anchor_qid"]).reshape(-1)
            source_id_to_embedding = {
                str(int(qid)): q_emb[pos]
                for pos, qid in enumerate(qids)
            }
            for i in idx:
                anchor_id = str(int(anchors[int(i)]))
                anchor = id_to_embedding.get(anchor_id)
                if anchor is None:
                    anchor = source_id_to_embedding.get(anchor_id)
                if anchor is None:
                    continue
                q = q_emb[int(i)]
                feats.append((q * anchor).astype(np.float32, copy=False))
                labels.append(int(y[int(i)]))
                cosines.append(float(ds["cosine_to_anchor"][int(i)]))
    else:
        if not allow_from_stream_annotations:
            raise SystemExit(
                f"No separate Hadamard pairwise train/calibration file found for {spec.canonical_name}. "
                "Pass --allow_pairwise_from_stream_annotations to build a labeled pairwise cache from official id_set "
                "annotations, or provide a matching pairwise file in the dataset spec."
            )
        warnings.append(
            f"{spec.canonical_name}: pairwise training generated from official stream id_set annotations; "
            "use a separate pairwise file for strict no-future experimental claims"
        )
        rng = np.random.default_rng(int(seed))
        by_cluster: Dict[str, List[StreamExample]] = {}
        for ex in stream:
            by_cluster.setdefault(str(ex.cluster), []).append(ex)
        usable = [xs for xs in by_cluster.values() if len(xs) >= 2]
        if not usable:
            raise ValueError(f"{spec.canonical_name} has no id_set groups with at least two examples")
        clusters = list(by_cluster.keys())
        for _ in range(per_class):
            group = usable[int(rng.integers(0, len(usable)))]
            a, b = rng.choice(len(group), size=2, replace=False)
            q = group[int(a)]
            c = group[int(b)]
            qv = np.asarray(q.embedding, dtype=np.float32)
            cv = np.asarray(c.embedding, dtype=np.float32)
            feats.append((qv * cv).astype(np.float32, copy=False))
            labels.append(1)
            cosines.append(float(np.dot(_normalize(qv), _normalize(cv))))
        for _ in range(per_class):
            q = stream[int(rng.integers(0, len(stream)))]
            other_keys = [k for k in clusters if k != str(q.cluster)]
            other_group = by_cluster[other_keys[int(rng.integers(0, len(other_keys)))]]
            c = other_group[int(rng.integers(0, len(other_group)))]
            qv = np.asarray(q.embedding, dtype=np.float32)
            cv = np.asarray(c.embedding, dtype=np.float32)
            feats.append((qv * cv).astype(np.float32, copy=False))
            labels.append(0)
            cosines.append(float(np.dot(_normalize(qv), _normalize(cv))))

    if not feats:
        raise ValueError(f"Could not construct any pairwise Hadamard rows for {spec.canonical_name}")
    X = np.vstack(feats).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int32)
    cos = np.asarray(cosines, dtype=np.float32)
    np.savez_compressed(
        out_path,
        emb=X,
        label=y,
        cosine_to_anchor=cos,
        model=np.asarray([embedding_model_name], dtype=object),
    )
    return out_path, warnings


def _tag_records(records: List[Dict[str, Any]], *, dataset: str, subset_name: str) -> None:
    for row in records:
        row["dataset"] = dataset
        row["subset_name"] = subset_name


def run_subset_methods(
    subset: Sequence[StreamExample],
    *,
    dataset: str,
    subset_name: str,
    methods: Sequence[str],
    pairwise_path: Path,
    args: argparse.Namespace,
    info: DatasetInfo,
    embedding_model_name: str,
    stream_digest: str,
) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    if "vcache" in methods:
        for delta in args.delta_values:
            judge = EquivalenceJudge(mode="cluster")
            try:
                rows = run_vcache_policy(
                    subset,
                    config=VCacheAdapterConfig(
                        delta=float(delta),
                        repo_path=args.vcache_repo_path,
                        policy=args.vcache_policy,
                        sync_updates=not args.vcache_async_updates,
                        cache_size=args.cache_size,
                        eviction_policy=args.eviction_policy,
                    ),
                    dataset=dataset,
                    seed=args.seed,
                    judge=judge,
                    embedding_model=embedding_model_name,
                    stream_hash=stream_digest,
                )
            except VCacheUnavailableError as exc:
                raise SystemExit(f"vCache baseline cannot run: {exc}") from exc
            _tag_records(rows, dataset=dataset, subset_name=subset_name)
            raw.extend(rows)

    for method in methods:
        if method in LEARNED_METHODS:
            method_name = LEARNED_METHODS[method]
            for alpha in args.target_budgets:
                if args.ours_calibration_source == "online_mined":
                    if not args.online_pairs_dir:
                        raise SystemExit("--ours_calibration_source online_mined requires --online_pairs_dir")
                    model, _diag = train_online_mined_model(
                        pairs_dir=Path(args.online_pairs_dir),
                        method_name=method_name,
                        alpha=float(alpha),
                        seed=args.seed,
                        pair_feature="hadamard",
                    )
                    diag = _diag
                else:
                    model = train_ours_model(
                        pairwise_data=str(pairwise_path),
                        method_name=method_name,
                        feature_key="emb",
                        label_key="label",
                        alt_feature_key="cosine_to_anchor",
                        n_train=args.ours_n_train,
                        n_calib=args.ours_n_calib,
                        seed=args.seed,
                        alpha=float(alpha),
                        pair_feature="hadamard",
                    )
                    diag = {
                        "n_train_pairs": int(args.ours_n_train) * 2,
                        "n_calib_pairs": int(args.ours_n_calib) * 2,
                        "n_eval_pairs": None,
                        "calibration_labels_used": (int(args.ours_n_train) + int(args.ours_n_calib)) * 2,
                        "calibration_split_size": int(args.ours_n_calib) * 2,
                        "calibration_equivalence_mode": "pairwise_npz_labels",
                    }
                judge = EquivalenceJudge(mode="cluster")
                rows = run_ours_policy(
                    subset,
                    model=model,
                    method_label=method,
                    dataset=dataset,
                    seed=args.seed,
                    cache_size=args.cache_size,
                    eviction_policy=args.eviction_policy,
                    judge=judge,
                    embedding_model=embedding_model_name,
                    stream_hash=stream_digest,
                )
                _annotate_learned_rows(rows, model, diag, args.ours_calibration_source)
                _tag_records(rows, dataset=dataset, subset_name=subset_name)
                raw.extend(rows)

    if "cosine" in methods:
        for threshold in args.thresholds:
            judge = EquivalenceJudge(mode="cluster")
            rows = run_cosine_policy(
                subset,
                threshold=float(threshold),
                dataset=dataset,
                seed=args.seed,
                cache_size=args.cache_size,
                eviction_policy=args.eviction_policy,
                judge=judge,
                embedding_model=embedding_model_name,
                stream_hash=stream_digest,
            )
            _tag_records(rows, dataset=dataset, subset_name=subset_name)
            raw.extend(rows)
    return raw


def iter_subset_method_runs(
    subset: Sequence[StreamExample],
    *,
    dataset: str,
    subset_name: str,
    methods: Sequence[str],
    pairwise_path: Path,
    args: argparse.Namespace,
    info: DatasetInfo,
    embedding_model_name: str,
    stream_digest: str,
) -> Sequence[Tuple[str, float, List[Dict[str, Any]]]]:
    del info
    if "vcache" in methods:
        for delta in args.delta_values:
            judge = EquivalenceJudge(mode="cluster")
            try:
                rows = run_vcache_policy(
                    subset,
                    config=VCacheAdapterConfig(
                        delta=float(delta),
                        repo_path=args.vcache_repo_path,
                        policy=args.vcache_policy,
                        sync_updates=not args.vcache_async_updates,
                        cache_size=args.cache_size,
                        eviction_policy=args.eviction_policy,
                    ),
                    dataset=dataset,
                    seed=args.seed,
                    judge=judge,
                    embedding_model=embedding_model_name,
                    stream_hash=stream_digest,
                )
            except VCacheUnavailableError as exc:
                raise SystemExit(f"vCache baseline cannot run: {exc}") from exc
            _tag_records(rows, dataset=dataset, subset_name=subset_name)
            yield "vcache", float(delta), rows

    for method in methods:
        if method in LEARNED_METHODS:
            method_name = LEARNED_METHODS[method]
            for alpha in args.target_budgets:
                if args.ours_calibration_source == "online_mined":
                    if not args.online_pairs_dir:
                        raise SystemExit("--ours_calibration_source online_mined requires --online_pairs_dir")
                    model, _diag = train_online_mined_model(
                        pairs_dir=Path(args.online_pairs_dir),
                        method_name=method_name,
                        alpha=float(alpha),
                        seed=args.seed,
                        pair_feature="hadamard",
                    )
                    diag = _diag
                else:
                    model = train_ours_model(
                        pairwise_data=str(pairwise_path),
                        method_name=method_name,
                        feature_key="emb",
                        label_key="label",
                        alt_feature_key="cosine_to_anchor",
                        n_train=args.ours_n_train,
                        n_calib=args.ours_n_calib,
                        seed=args.seed,
                        alpha=float(alpha),
                        pair_feature="hadamard",
                    )
                    diag = {
                        "n_train_pairs": int(args.ours_n_train) * 2,
                        "n_calib_pairs": int(args.ours_n_calib) * 2,
                        "n_eval_pairs": None,
                        "calibration_labels_used": (int(args.ours_n_train) + int(args.ours_n_calib)) * 2,
                        "calibration_split_size": int(args.ours_n_calib) * 2,
                        "calibration_equivalence_mode": "pairwise_npz_labels",
                    }
                judge = EquivalenceJudge(mode="cluster")
                rows = run_ours_policy(
                    subset,
                    model=model,
                    method_label=method,
                    dataset=dataset,
                    seed=args.seed,
                    cache_size=args.cache_size,
                    eviction_policy=args.eviction_policy,
                    judge=judge,
                    embedding_model=embedding_model_name,
                    stream_hash=stream_digest,
                )
                _annotate_learned_rows(rows, model, diag, args.ours_calibration_source)
                _tag_records(rows, dataset=dataset, subset_name=subset_name)
                yield method, float(alpha), rows

    if "cosine" in methods:
        for threshold in args.thresholds:
            judge = EquivalenceJudge(mode="cluster")
            rows = run_cosine_policy(
                subset,
                threshold=float(threshold),
                dataset=dataset,
                seed=args.seed,
                cache_size=args.cache_size,
                eviction_policy=args.eviction_policy,
                judge=judge,
                embedding_model=embedding_model_name,
                stream_hash=stream_digest,
            )
            _tag_records(rows, dataset=dataset, subset_name=subset_name)
            yield "cosine", float(threshold), rows


def append_raw_decisions(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _raw_log_path_and_fields(output_dir: Path, mode: str) -> Tuple[Optional[Path], Sequence[str]]:
    if mode == "none":
        return None, []
    if mode == "minimal":
        return output_dir / "raw_decisions_minimal.csv", RAW_DECISION_MINIMAL_FIELDS
    return output_dir / "raw_decisions.csv", RAW_DECISION_FIELDS


def _write_examples_lookup(path: Path, stream: Sequence[StreamExample], dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ex in stream:
            f.write(
                json.dumps(
                    {
                        "dataset": dataset,
                        "id": ex.id,
                        "prompt": ex.prompt,
                        "gold_response": ex.gold_response,
                        "cluster": ex.cluster,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _annotate_learned_rows(rows: Sequence[Dict[str, Any]], model: Any, diag: Dict[str, Any], source: str) -> None:
    for row in rows:
        row["threshold_source"] = getattr(model, "threshold_source", "")
        row["score_orientation"] = getattr(model, "orientation", "")
        row["calibration_source"] = source
        row["calibration_train_pairs"] = diag.get("n_train_pairs")
        row["calibration_calib_pairs"] = diag.get("n_calib_pairs")
        row["calibration_eval_pairs"] = diag.get("n_eval_pairs")
        row["calibration_labels_used"] = diag.get("calibration_labels_used")
        row["calibration_split_size"] = diag.get("calibration_split_size")
        row["calibration_equivalence_mode"] = diag.get("calibration_equivalence_mode")


def _sample_for_plots(rows: Sequence[Dict[str, Any]], max_points: int = 5000) -> List[Dict[str, Any]]:
    if len(rows) <= max_points:
        return list(rows)
    step = max(1, len(rows) // max_points)
    out = [row for i, row in enumerate(rows) if i % step == 0]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def _plot_hist(values: Sequence[float], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.5))
    if values:
        plt.hist(values, bins=40, color="#4c78a8", alpha=0.85)
    plt.xlabel("nearest_neighbor_cosine")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_config(output_dir: Path, args: argparse.Namespace, warnings: Sequence[str]) -> None:
    write_json(
        output_dir / "config.json",
        {
            "args": vars(args),
            "warnings": list(warnings),
            "protocol_notes": [
                "Banking77 is not used as a main benchmark.",
                "vCache is evaluated as an online cache policy, not as a pairwise classifier.",
                "Primary comparison is hit rate at matched empirical stream-level error FP/n.",
                "Learned methods score retrieved candidates with Hadamard pair features.",
            ],
        },
    )


def _write_checkpoint(
    output_dir: Path,
    *,
    dataset_stats: Sequence[Dict[str, Any]],
    hard_stats: Sequence[Dict[str, Any]],
    summary: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    warnings: Sequence[str],
) -> None:
    write_csv(output_dir / "dataset_stats.csv", dataset_stats, DATASET_STATS_FIELDS)
    write_csv(output_dir / "hard_subset_stats.csv", hard_stats, HARD_STATS_FIELDS)
    write_csv(output_dir / "summary_metrics.csv", summary, SUMMARY_FIELDS)
    write_json(output_dir / "summary_metrics.json", list(summary))
    _write_config(output_dir, args, warnings)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.max_examples is not None:
        args.limit = int(args.max_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_key, embedding_model_name = _embedding_choice(args.embedding_model)
    learned_requested = any(m in LEARNED_METHODS for m in args.methods)
    if learned_requested and args.ours_calibration_source == "online_mined" and not args.online_pairs_dir:
        raise SystemExit("--ours_calibration_source online_mined requires --online_pairs_dir")
    if learned_requested and args.ours_calibration_source == "pairwise_npz" and not args.allow_pairwise_from_stream_annotations:
        missing_pairwise = [
            name
            for name in args.datasets
            if not (
                SPECS[name].default_pairwise_path
                and Path(str(SPECS[name].default_pairwise_path)).exists()
            )
        ]
        if missing_pairwise:
            raise SystemExit(
                "Missing separate Hadamard pairwise train/calibration data for: "
                f"{', '.join(missing_pairwise)}. "
                "Provide matching pairwise files in the dataset spec, remove learned methods for those datasets, "
                "or pass --allow_pairwise_from_stream_annotations to explicitly build pairwise rows from official id_set annotations."
            )
    raw_path, raw_fields = _raw_log_path_and_fields(output_dir, args.raw_log_mode)
    if raw_path is not None:
        with raw_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_fields), extrasaction="ignore")
            writer.writeheader()
    if args.raw_log_mode == "minimal":
        lookup_path = output_dir / "examples_lookup.jsonl"
        lookup_path.write_text("", encoding="utf-8")

    all_summary: List[Dict[str, Any]] = []
    dataset_stats: List[Dict[str, Any]] = []
    hard_stats: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for dataset_name in args.datasets:
        print(f"[compare_vcache_real_datasets] loading {dataset_name}", flush=True)
        spec = SPECS[dataset_name]
        response_key = spec.default_response_key
        if args.ours_calibration_source == "online_mined":
            stream, info, pairs_config, pairs_embedding_model = load_eval_stream_from_pairs_dir(
                Path(args.online_pairs_dir),
                limit=args.limit,
            )
            if str(pairs_config.get("dataset")) != dataset_name:
                raise SystemExit(
                    f"--online_pairs_dir dataset {pairs_config.get('dataset')!r} does not match requested {dataset_name!r}"
                )
            if str(pairs_embedding_model) != embedding_model_name:
                raise SystemExit(
                    f"--online_pairs_dir embedding model {pairs_embedding_model!r} does not match requested {embedding_model_name!r}"
                )
        else:
            stream, info = load_real_stream(
                spec,
                embedding_key=embedding_key,
                response_key=response_key,
                seed=args.seed,
                limit=args.limit,
            )
        if not stream:
            raise ValueError(f"{dataset_name} stream is empty")
        if any(ex.embedding is None for ex in stream):
            raise ValueError(f"{dataset_name} has missing embeddings for key {embedding_key}")
        if any(ex.cluster is None for ex in stream):
            raise ValueError(f"{dataset_name} has missing id_set/equivalence labels")
        if args.raw_log_mode == "minimal":
            _write_examples_lookup(output_dir / "examples_lookup.jsonl", stream, dataset_name)

        dataset_stats.append(
            {
                "dataset": dataset_name,
                "source": info.path,
                "n_examples": len(stream),
                "prompt_key": info.prompt_key,
                "response_key": info.response_key,
                "id_key": info.id_key,
                "cluster_key": info.cluster_key,
                "embedding_key": info.embedding_key,
                "embedding_model": embedding_model_name,
                "warnings": "; ".join(info.warnings),
            }
        )

        learned_requested = any(m in LEARNED_METHODS for m in args.methods)
        pairwise_path: Optional[Path] = None
        if learned_requested and args.ours_calibration_source == "pairwise_npz":
            needed = int(args.ours_n_train) + int(args.ours_n_calib)
            per_class = max(int(args.pairwise_sample_per_class), needed)
            print(
                f"[compare_vcache_real_datasets] preparing pairwise data for {dataset_name} "
                f"(per_class={per_class})",
                flush=True,
            )
            pairwise_path, pair_warnings = prepare_pairwise_hadamard(
                spec,
                stream,
                output_dir=output_dir,
                seed=args.seed,
                per_class=per_class,
                embedding_model_name=embedding_model_name,
                allow_from_stream_annotations=bool(args.allow_pairwise_from_stream_annotations),
            )
            warnings.extend(pair_warnings)

        print(f"[compare_vcache_real_datasets] computing hard-neighbor subsets for {dataset_name}", flush=True)
        hard_rows = compute_hard_subset_info(
            stream,
            cache_size=args.cache_size,
            eviction_policy=args.eviction_policy,
        )
        subsets, subset_stats, hist_values = build_subsets(
            dataset_name,
            stream,
            hard_rows,
            args.hard_neighbor_thresholds,
        )
        hard_stats.extend(subset_stats)
        for stat in subset_stats:
            if stat.get("warning"):
                warnings.append(f"{stat['subset_name']}: {stat['warning']}")

        for subset_name, subset_stream in subsets.items():
            if not subset_stream:
                warnings.append(f"{subset_name}: empty subset, skipping method runs")
                continue
            digest = stream_hash(subset_stream)
            print(
                f"[compare_vcache_real_datasets] running {subset_name} "
                f"n={len(subset_stream)} runs={len(args.delta_values) if 'vcache' in args.methods else 0}"
                f"+{len(args.target_budgets) * sum(1 for m in args.methods if m in LEARNED_METHODS)}"
                f"+{len(args.thresholds) if 'cosine' in args.methods else 0}",
                flush=True,
            )
            subset_summary: List[Dict[str, Any]] = []
            subset_plot_raw: List[Dict[str, Any]] = []
            for method, param, rows in iter_subset_method_runs(
                subset_stream,
                dataset=dataset_name,
                subset_name=subset_name,
                methods=args.methods,
                pairwise_path=pairwise_path if pairwise_path is not None else Path(""),
                args=args,
                info=info,
                embedding_model_name=embedding_model_name,
                stream_digest=digest,
            ):
                if len(rows) != len(subset_stream):
                    warnings.append(
                        f"different number of evaluated examples for "
                        f"{dataset_name}/{subset_name}/{method}/{param}: {len(rows)} != {len(subset_stream)}"
                    )
                if any(r.get("decision") not in {"hit", "miss", "exploit", "explore"} for r in rows):
                    warnings.append(f"missing hit/miss logs for {dataset_name}/{subset_name}/{method}/{param}")
                if any(str(r.get("stream_hash")) != digest for r in rows):
                    warnings.append(f"different stream order across methods for {dataset_name}/{subset_name}/{method}/{param}")
                row_summary = summarize_run(rows)
                all_summary.append(row_summary)
                subset_summary.append(row_summary)
                subset_plot_raw.extend(_sample_for_plots(rows))
                if raw_path is not None:
                    append_raw_decisions(raw_path, rows, raw_fields)
                _write_checkpoint(
                    output_dir,
                    dataset_stats=dataset_stats,
                    hard_stats=hard_stats,
                    summary=all_summary,
                    args=args,
                    warnings=warnings,
                )
                print(
                    "[compare_vcache_real_datasets] finished "
                    f"{subset_name} {method}={param} "
                    f"error={row_summary.get('error_rate_stream')} hit={row_summary.get('hit_rate')}",
                    flush=True,
                )

            generate_plots(subset_summary, subset_plot_raw, output_dir / "plots" / subset_name)
            _plot_hist(
                hist_values.get(subset_name, []),
                output_dir / "plots" / subset_name / "nearest_neighbor_cosine_histogram.png",
                f"{subset_name} nearest-neighbor cosine",
            )

    if not all_summary:
        raise ValueError("No raw decision records produced")

    _write_checkpoint(
        output_dir,
        dataset_stats=dataset_stats,
        hard_stats=hard_stats,
        summary=all_summary,
        args=args,
        warnings=warnings,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
