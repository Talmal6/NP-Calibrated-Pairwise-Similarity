from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compare_vcache_real_datasets import SPECS, _embedding_choice, load_real_stream
from .equivalence import EquivalenceJudge
from .online_policies import ExactVectorCache
from .vcache_metrics import write_csv, write_json


STATS_FIELDS = [
    "dataset",
    "split_name",
    "n_stream_examples",
    "n_candidate_pairs",
    "n_H1",
    "n_H0",
    "H1_ratio",
    "cosine_min",
    "cosine_p50",
    "cosine_p95",
    "cosine_max",
    "warning",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mine online nearest-neighbor candidate pairs for learned cache admission.")
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("--embedding_model", default="GTE")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_size", type=int, default=4096)
    ap.add_argument("--eviction_policy", choices=["mru", "lru", "fifo"], default="mru")
    ap.add_argument("--split_ratios", nargs=3, type=float, default=[0.4, 0.2, 0.4])
    ap.add_argument("--equivalence_mode", choices=["cluster", "exact"], default="cluster")
    ap.add_argument("--max_examples", type=int, default=None)
    return ap.parse_args(argv)


def split_stream(stream: Sequence[Any], ratios: Sequence[float]) -> Tuple[List[Any], List[Any], List[Any], Dict[str, Any]]:
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("--split_ratios must sum to a positive value")
    r = [float(x) / total for x in ratios]
    n = len(stream)
    n_train = int(round(n * r[0]))
    n_calib = int(round(n * r[1]))
    n_train = min(max(1, n_train), n)
    n_calib = min(max(1, n_calib), max(0, n - n_train))
    n_eval = n - n_train - n_calib
    if n_eval <= 0:
        raise ValueError("Split ratios leave no eval examples")
    train = list(stream[:n_train])
    calib = list(stream[n_train : n_train + n_calib])
    eval_stream = list(stream[n_train + n_calib :])
    return train, calib, eval_stream, {
        "n_total": n,
        "n_train": len(train),
        "n_calib": len(calib),
        "n_eval": len(eval_stream),
        "train_start": 0,
        "calib_start": len(train),
        "eval_start": len(train) + len(calib),
    }


def mine_split(
    stream: Sequence[Any],
    *,
    split_name: str,
    dataset: str,
    cache_size: int,
    eviction_policy: str,
    judge: EquivalenceJudge,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[str]]:
    cache = ExactVectorCache(capacity=cache_size, eviction_policy=eviction_policy)
    query_emb: List[np.ndarray] = []
    candidate_emb: List[np.ndarray] = []
    labels: List[int] = []
    cosines: List[float] = []
    query_ids: List[str] = []
    candidate_ids: List[str] = []
    query_text: List[str] = []
    candidate_text: List[str] = []
    warnings: List[str] = []

    for ex in stream:
        nearest, sim = cache.nearest(ex.embedding)
        if nearest is not None and sim is not None and ex.embedding is not None and nearest.example.embedding is not None:
            eq = judge.equivalent_examples(ex, nearest.example)
            if eq is None:
                warnings.append(f"{split_name}: missing equivalence label for query_id={ex.id}")
            else:
                q = np.asarray(ex.embedding, dtype=np.float32).reshape(-1)
                c = np.asarray(nearest.example.embedding, dtype=np.float32).reshape(-1)
                query_emb.append(q)
                candidate_emb.append(c)
                labels.append(1 if bool(eq) else 0)
                cosines.append(float(sim))
                query_ids.append(str(ex.id))
                candidate_ids.append(str(nearest.example.id))
                query_text.append(ex.prompt)
                candidate_text.append(nearest.example.prompt)
        cache.add(ex)

    if query_emb:
        q_arr = np.vstack(query_emb).astype(np.float32, copy=False)
        c_arr = np.vstack(candidate_emb).astype(np.float32, copy=False)
        x_arr = (q_arr * c_arr).astype(np.float32, copy=False)
    else:
        q_arr = np.empty((0, 0), dtype=np.float32)
        c_arr = np.empty((0, 0), dtype=np.float32)
        x_arr = np.empty((0, 0), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    cos = np.asarray(cosines, dtype=np.float32)
    n_h1 = int(np.sum(y == 1))
    n_h0 = int(np.sum(y == 0))
    ratio = n_h1 / max(1, int(y.size))
    warning = ""
    if y.size < 100:
        warning = "too few mined candidate pairs"
    if y.size and (ratio <= 0.05 or ratio >= 0.95):
        warning = (warning + "; " if warning else "") + "weak H0/H1 mixture"
    arrays = {
        "query_emb": q_arr,
        "candidate_emb": c_arr,
        "X_hadamard": x_arr,
        "label": y,
        "cosine_similarity": cos,
        "query_id": np.asarray(query_ids, dtype=object),
        "candidate_id": np.asarray(candidate_ids, dtype=object),
        "query_text": np.asarray(query_text, dtype=object),
        "candidate_text": np.asarray(candidate_text, dtype=object),
        "split_name": np.asarray([split_name], dtype=object),
        "dataset": np.asarray([dataset], dtype=object),
    }
    stats = {
        "dataset": dataset,
        "split_name": split_name,
        "n_stream_examples": len(stream),
        "n_candidate_pairs": int(y.size),
        "n_H1": n_h1,
        "n_H0": n_h0,
        "H1_ratio": ratio if y.size else None,
        "cosine_min": float(np.min(cos)) if cos.size else None,
        "cosine_p50": float(np.quantile(cos, 0.50)) if cos.size else None,
        "cosine_p95": float(np.quantile(cos, 0.95)) if cos.size else None,
        "cosine_max": float(np.max(cos)) if cos.size else None,
        "warning": warning,
    }
    if warning:
        warnings.append(f"{split_name}: {warning}")
    return arrays, stats, warnings


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_key, embedding_model_name = _embedding_choice(args.embedding_model)
    spec = SPECS[args.dataset]
    stream, info = load_real_stream(
        spec,
        embedding_key=embedding_key,
        response_key=spec.default_response_key,
        seed=args.seed,
        limit=args.max_examples,
    )
    if not stream:
        raise ValueError("Loaded stream is empty")
    if any(ex.embedding is None for ex in stream):
        raise ValueError(f"{args.dataset} has missing embeddings for {embedding_key}")

    train, calib, eval_stream, split_info = split_stream(stream, args.split_ratios)
    judge = EquivalenceJudge(mode=args.equivalence_mode)

    stats_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for split_name, split_examples in [
        ("online_train", train),
        ("online_calib", calib),
        ("online_eval", eval_stream),
    ]:
        arrays, stats, split_warnings = mine_split(
            split_examples,
            split_name=split_name,
            dataset=args.dataset,
            cache_size=args.cache_size,
            eviction_policy=args.eviction_policy,
            judge=judge,
        )
        save_npz(out_dir / f"{split_name}_pairs.npz", arrays)
        stats_rows.append(stats)
        warnings.extend(split_warnings)
        print(
            f"[mine_online_candidate_pairs] {split_name}: "
            f"stream={stats['n_stream_examples']} pairs={stats['n_candidate_pairs']} "
            f"H1={stats['n_H1']} H0={stats['n_H0']} H1_ratio={stats['H1_ratio']}",
            flush=True,
        )

    write_csv(out_dir / "online_candidate_pair_stats.csv", stats_rows, STATS_FIELDS)
    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "dataset": args.dataset,
            "embedding_key": embedding_key,
            "embedding_model": embedding_model_name,
            "response_key": spec.default_response_key,
            "equivalence_mode": args.equivalence_mode,
            "cache_size": args.cache_size,
            "eviction_policy": args.eviction_policy,
            "split_ratios": list(args.split_ratios),
            "split_info": split_info,
            "dataset_info": info.__dict__,
            "feature_storage": "raw query_emb/candidate_emb and precomputed X_hadamard are saved; X_hadamard = query_emb * candidate_emb",
        },
    )
    with (out_dir / "warnings.txt").open("w", encoding="utf-8") as f:
        for warning in warnings:
            f.write(warning + "\n")
    if warnings:
        print("[mine_online_candidate_pairs] warnings:", flush=True)
        for warning in warnings:
            print(f"- {warning}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
