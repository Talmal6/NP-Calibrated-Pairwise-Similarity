from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-neighborcache")

from .dataset_stream import DatasetInfo, StreamExample, load_dataset_stream
from .equivalence import EquivalenceJudge
from .online_policies import run_cosine_policy, run_ours_policy, train_ours_model
from .vcache_adapter import VCacheAdapterConfig, VCacheUnavailableError, run_vcache_policy
from .vcache_metrics import (
    RAW_DECISION_FIELDS,
    SUMMARY_FIELDS,
    generate_plots,
    summarize_all,
    write_csv,
    write_json,
)


DEFAULT_DELTAS = [0.01, 0.02, 0.03, 0.05, 0.08]
DEFAULT_THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare official vCache, NeighborCache, and cosine as online caches.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", default="results/vcache_comparison")
    ap.add_argument("--methods", nargs="+", default=["vcache", "ours", "cosine"], choices=["vcache", "ours", "cosine"])
    ap.add_argument("--delta_values", nargs="+", type=float, default=DEFAULT_DELTAS)
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_size", type=int, default=4096)
    ap.add_argument("--eviction_policy", choices=["mru", "lru", "fifo"], default="mru")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle_stream", action="store_true")

    ap.add_argument("--prompt_key", default=None)
    ap.add_argument("--response_key", default=None)
    ap.add_argument("--id_key", default=None)
    ap.add_argument("--cluster_key", default=None)
    ap.add_argument("--embedding_key", default=None)
    ap.add_argument("--embedding_model_name", default="")

    ap.add_argument("--correctness", choices=["exact", "cluster", "callable"], default="exact")
    ap.add_argument("--judge_function", default=None)

    ap.add_argument("--vcache_repo_path", default=None)
    ap.add_argument("--vcache_policy", choices=["local", "global"], default="local")
    ap.add_argument("--vcache_async_updates", action="store_true", help="Use official async update path instead of deterministic synchronous instrumentation.")

    ap.add_argument("--ours_pairwise_data", default=None)
    ap.add_argument("--ours_method", default="WeightedEnsemble")
    ap.add_argument("--ours_pairwise_feature_key", default="emb")
    ap.add_argument("--ours_pairwise_alt_feature_key", default="cosine_to_anchor")
    ap.add_argument("--ours_label_key", default="label")
    ap.add_argument("--ours_n_train", type=int, default=1200)
    ap.add_argument("--ours_n_calib", type=int, default=1200)
    ap.add_argument("--ours_pair_feature", choices=["hadamard", "absdiff", "concat", "cosine"], default="hadamard")
    return ap.parse_args(argv)


def stream_hash(stream: Sequence[StreamExample]) -> str:
    h = hashlib.sha256()
    for ex in stream:
        h.update(str(ex.id).encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(ex.prompt.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def _validate_stream_for_embedding_methods(stream: Sequence[StreamExample], methods: Sequence[str]) -> None:
    if any(m in methods for m in ["vcache", "ours", "cosine"]):
        missing = [ex.id for ex in stream if ex.embedding is None]
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                "Embedding-based methods require an embedding for every evaluated example. "
                f"Missing {len(missing)} embeddings; sample ids: {sample}"
            )


def _fairness_warnings(
    *,
    info: DatasetInfo,
    stream: Sequence[StreamExample],
    methods: Sequence[str],
    correctness: str,
    cache_size: int,
    embedding_model: str,
) -> List[str]:
    warnings = list(info.warnings)
    if correctness == "exact":
        missing_gold = sum(1 for ex in stream if not ex.gold_response)
        if missing_gold:
            warnings.append(f"missing correctness labels/gold responses for {missing_gold} examples")
    if correctness == "cluster":
        missing_cluster = sum(1 for ex in stream if ex.cluster is None)
        if missing_cluster:
            warnings.append(f"missing cluster labels for {missing_cluster} examples")
    if not embedding_model:
        warnings.append("embedding model name not provided; using dataset embedding key as proxy")
    if cache_size <= 0:
        warnings.append("cache capacity is non-positive; all methods will miss")
    if "vcache" in methods:
        warnings.append("vCache API does not expose prompt text in metadata; adapter attaches prompt ids/text via metadata storage wrapper")
    return warnings


def _pairwise_embedding_model_hint(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or p.suffix != ".npz":
        return None
    try:
        with np.load(str(p), allow_pickle=True) as ds:
            if "model" in ds:
                arr = np.asarray(ds["model"]).reshape(-1)
                return str(arr[0]) if arr.size else None
            if "meta" in ds:
                arr = np.asarray(ds["meta"]).reshape(-1)
                if arr.size and isinstance(arr[0], dict):
                    meta = arr[0]
                    for key in ["embed_model", "model", "embedding_model"]:
                        if key in meta:
                            return str(meta[key])
    except Exception:
        return None
    return None


def _write_config(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    info: DatasetInfo,
    stream_digest: str,
    warnings: List[str],
) -> None:
    obj: Dict[str, Any] = {
        "args": vars(args),
        "dataset": info.__dict__,
        "stream_hash": stream_digest,
        "warnings": warnings,
        "protocol_notes": [
            "vCache is evaluated as an online cache policy, not as a pairwise classifier.",
            "All methods receive the same deterministic stream order and cache capacity.",
            "Correctness and nearest-candidate labels are computed with the configured equivalence function.",
        ],
    }
    write_json(output_dir / "config.json", obj)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stream, info = load_dataset_stream(
        args.dataset,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        id_key=args.id_key,
        cluster_key=args.cluster_key,
        embedding_key=args.embedding_key,
        seed=args.seed,
        shuffle=args.shuffle_stream,
        limit=args.limit,
    )
    if not stream:
        raise ValueError("Dataset stream is empty")
    _validate_stream_for_embedding_methods(stream, args.methods)

    digest = stream_hash(stream)
    embedding_model = args.embedding_model_name or (info.embedding_key or "unknown")
    warnings = _fairness_warnings(
        info=info,
        stream=stream,
        methods=args.methods,
        correctness=args.correctness,
        cache_size=args.cache_size,
        embedding_model=embedding_model,
    )
    pairwise_model = _pairwise_embedding_model_hint(args.ours_pairwise_data)
    if pairwise_model and args.embedding_model_name and pairwise_model != args.embedding_model_name:
        warnings.append(
            "embedding model mismatch warning: "
            f"stream={args.embedding_model_name!r}, ours_pairwise_data={pairwise_model!r}"
        )
    _write_config(output_dir, args=args, info=info, stream_digest=digest, warnings=warnings)

    raw_records: List[Dict[str, Any]] = []

    if "vcache" in args.methods:
        for delta in args.delta_values:
            judge = EquivalenceJudge(mode=args.correctness, function_spec=args.judge_function)
            try:
                raw_records.extend(
                    run_vcache_policy(
                        stream,
                        config=VCacheAdapterConfig(
                            delta=float(delta),
                            repo_path=args.vcache_repo_path,
                            policy=args.vcache_policy,
                            sync_updates=not args.vcache_async_updates,
                            cache_size=args.cache_size,
                            eviction_policy=args.eviction_policy,
                        ),
                        dataset=info.name,
                        seed=args.seed,
                        judge=judge,
                        embedding_model=embedding_model,
                        stream_hash=digest,
                    )
                )
            except VCacheUnavailableError as exc:
                raise SystemExit(f"vCache baseline cannot run: {exc}") from exc

    if "ours" in args.methods:
        if not args.ours_pairwise_data:
            raise SystemExit(
                "ours baseline cannot run: --ours_pairwise_data is required so the existing "
                "train/calibration protocol can fit a pairwise reuse scorer without seeing the evaluation stream."
            )
        for alpha in args.delta_values:
            model = train_ours_model(
                pairwise_data=args.ours_pairwise_data,
                method_name=args.ours_method,
                feature_key=args.ours_pairwise_feature_key,
                label_key=args.ours_label_key,
                alt_feature_key=args.ours_pairwise_alt_feature_key,
                n_train=args.ours_n_train,
                n_calib=args.ours_n_calib,
                seed=args.seed,
                alpha=float(alpha),
                pair_feature=args.ours_pair_feature,
            )
            judge = EquivalenceJudge(mode=args.correctness, function_spec=args.judge_function)
            raw_records.extend(
                run_ours_policy(
                    stream,
                    model=model,
                    dataset=info.name,
                    seed=args.seed,
                    cache_size=args.cache_size,
                    eviction_policy=args.eviction_policy,
                    judge=judge,
                    embedding_model=embedding_model,
                    stream_hash=digest,
                )
            )

    if "cosine" in args.methods:
        for threshold in args.thresholds:
            judge = EquivalenceJudge(mode=args.correctness, function_spec=args.judge_function)
            raw_records.extend(
                run_cosine_policy(
                    stream,
                    threshold=float(threshold),
                    dataset=info.name,
                    seed=args.seed,
                    cache_size=args.cache_size,
                    eviction_policy=args.eviction_policy,
                    judge=judge,
                    embedding_model=embedding_model,
                    stream_hash=digest,
                )
            )

    if not raw_records:
        raise ValueError("No raw decision records produced")

    expected_n = len(stream)
    run_hashes = {}
    for key in sorted({(r["method"], r["delta_or_threshold"]) for r in raw_records}):
        run_rows = [r for r in raw_records if (r["method"], r["delta_or_threshold"]) == key]
        n = len(run_rows)
        if n != expected_n:
            warnings.append(f"different number of evaluated examples for {key}: {n} != {expected_n}")
        run_hashes[key] = {r.get("stream_hash") for r in run_rows}
    if any(hashes != {digest} for hashes in run_hashes.values()):
        warnings.append("different stream order/hash detected across methods")
    if any(int(r.get("cache_size") or -1) != int(args.cache_size) for r in raw_records):
        warnings.append("cache capacity mismatch detected in raw records")
    if len({r.get("judge") for r in raw_records}) > 1:
        warnings.append("judge function mismatch detected across methods")
    if len({r.get("embedding_model") for r in raw_records}) > 1:
        warnings.append("embedding model mismatch detected across methods")
    if any(not r.get("decision") for r in raw_records):
        warnings.append("missing hit/miss decision logs in raw records")

    summary = summarize_all(raw_records)
    write_csv(output_dir / "raw_decisions.csv", raw_records, RAW_DECISION_FIELDS)
    write_csv(output_dir / "summary_metrics.csv", summary, SUMMARY_FIELDS)
    write_json(output_dir / "summary_metrics.json", summary)

    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config_obj = json.load(f)
    config_obj["warnings"] = warnings
    write_json(config_path, config_obj)

    generate_plots(summary, raw_records, output_dir / "plots")


if __name__ == "__main__":
    main(sys.argv[1:])
