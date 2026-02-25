#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def as_vec(x) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f"Embedding must be 1D, got shape {v.shape}")
    return v


def load_all_pairs(pkl_path: Path) -> List[dict]:
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            if isinstance(v, list):
                out.extend(v)
        return out
    raise ValueError(f"Unsupported PKL type: {type(obj)}")


def build_qid_embeddings_mean(data: List[dict], model: str, report_every: int = 200_000) -> Dict[int, np.ndarray]:
    q1k = f"{model}_q1_emb"
    q2k = f"{model}_q2_emb"

    sum_map: Dict[int, np.ndarray] = {}
    cnt_map: Dict[int, int] = defaultdict(int)

    dim: Optional[int] = None

    for i, s in enumerate(data):
        if report_every and i and (i % report_every == 0):
            print(f"[INFO] processed {i}/{len(data)} pkl rows", flush=True)

        q1 = int(s["qid1"]); q2 = int(s["qid2"])
        e1 = as_vec(s[q1k]); e2 = as_vec(s[q2k])

        if dim is None:
            dim = int(e1.shape[0])
            print(f"[INFO] inferred emb dim={dim}", flush=True)

        if q1 not in sum_map:
            sum_map[q1] = e1.copy()
        else:
            sum_map[q1] += e1
        cnt_map[q1] += 1

        if q2 not in sum_map:
            sum_map[q2] = e2.copy()
        else:
            sum_map[q2] += e2
        cnt_map[q2] += 1

    emb_map = {q: (sum_map[q] / float(cnt_map[q])).astype(np.float32, copy=False) for q in sum_map.keys()}
    print(f"[INFO] built embeddings for {len(emb_map)} qids", flush=True)
    return emb_map


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def pair_features(e1: np.ndarray, e2: np.ndarray, feat: str) -> np.ndarray:
    if feat == "diff":
        return np.abs(e1 - e2)
    if feat == "hadamard":
        return e1 * e2
    if feat == "concat":
        return np.concatenate([e1, e2, np.abs(e1 - e2), e1 * e2], axis=0)
    raise ValueError("feat must be diff|hadamard|concat")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build separation-caching dataset: (q vs cluster-rep) labeled member(1)/orphan(0).")
    ap.add_argument("--groups_npz", type=Path, default=Path("NeighborCache/data/quora_groups.npz"))
    ap.add_argument("--pkl", type=Path, required=True)
    ap.add_argument("--model", type=str, default="BAAI_bge_large_en_v1.5")
    ap.add_argument("--feat", type=str, choices=["diff", "hadamard", "concat"], default="hadamard")
    ap.add_argument("--clusters", type=str, required=True, help="Comma-separated cluster ids (regions)")
    ap.add_argument("--out_npz", type=Path, required=True)
    args = ap.parse_args()

    clusters = [int(x.strip()) for x in args.clusters.split(",") if x.strip()]
    clusters_set = set(clusters)

    ds = np.load(args.groups_npz, allow_pickle=True)
    h1_nodes = ds["h1_nodes"].astype(np.int64)
    group_of_h1 = ds["group_of_h1"].astype(np.int64)
    orphans = ds["orphans"].astype(np.int64)
    assigned_group = ds["assigned_group"].astype(np.int64)

    # Build cluster -> members (H1 nodes)
    members: Dict[int, List[int]] = defaultdict(list)
    for q, g in zip(h1_nodes, group_of_h1):
        g = int(g)
        if g in clusters_set:
            members[g].append(int(q))

    # Build cluster -> orphans assigned
    orph_by_g: Dict[int, List[int]] = defaultdict(list)
    for q, g in zip(orphans, assigned_group):
        g = int(g)
        if g in clusters_set:
            orph_by_g[g].append(int(q))

    # Representative qid per cluster = first member (same semantics as your generation)
    reps: Dict[int, int] = {}
    for g in clusters:
        if len(members[g]) == 0:
            raise RuntimeError(f"Cluster {g} has 0 H1 members in file (cannot define representative).")
        reps[g] = members[g][0]

    print("[INFO] loading PKL embeddings...", flush=True)
    data = load_all_pairs(args.pkl)
    emb_map = build_qid_embeddings_mean(data, args.model)

    X_rows = []
    cos_rows = []
    y_rows = []
    rid_rows = []
    qid_rows = []
    rep_rows = []
    missing = 0

    for g in clusters:
        rep = reps[g]
        rep_emb = emb_map.get(rep)
        if rep_emb is None:
            raise RuntimeError(f"Missing embedding for representative qid={rep} cluster={g}")

        # Positive rows: every member vs rep (label=1)
        for q in members[g]:
            e = emb_map.get(q)
            if e is None:
                missing += 1
                continue
            X_rows.append(pair_features(e, rep_emb, args.feat))
            cos_rows.append(cosine(e, rep_emb))
            y_rows.append(1)
            rid_rows.append(g)
            qid_rows.append(q)
            rep_rows.append(rep)

        # Negative rows: every orphan assigned vs rep (label=0)
        for q in orph_by_g[g]:
            e = emb_map.get(q)
            if e is None:
                missing += 1
                continue
            X_rows.append(pair_features(e, rep_emb, args.feat))
            cos_rows.append(cosine(e, rep_emb))
            y_rows.append(0)
            rid_rows.append(g)
            qid_rows.append(q)
            rep_rows.append(rep)

        print(f"[INFO] cluster {g}: +members={len(members[g])} -orphans={len(orph_by_g[g])}", flush=True)

    if not X_rows:
        raise RuntimeError("No rows produced.")

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    rid = np.asarray(rid_rows, dtype=np.int64)
    cosv = np.asarray(cos_rows, dtype=np.float32)
    qids = np.asarray(qid_rows, dtype=np.int64)
    reps_arr = np.asarray(rep_rows, dtype=np.int64)

    print(f"[INFO] built rows={X.shape[0]} dim={X.shape[1]} missing_qids={missing}", flush=True)
    print(f"[INFO] label balance: n1={int(np.sum(y==1))} n0={int(np.sum(y==0))}", flush=True)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        emb=X,
        cosine_to_anchor=cosv,
        label=y,
        cluster_id=rid,
        qid=qids,
        rep_qid=reps_arr,
        feat=np.array([args.feat], dtype=object),
        model=np.array([args.model], dtype=object),
        clusters=np.asarray(clusters, dtype=np.int64),
    )
    print(f"[INFO] saved -> {args.out_npz}", flush=True)


if __name__ == "__main__":
    main()
