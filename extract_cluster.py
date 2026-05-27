#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_cluster.py

Extract all data for a single cluster from a labeled NPZ file.

Output JSON structure:
{
  "cluster_id": <int>,
  "cluster_meta": {           # from dominant_intents sidecar (if found)
    "dominant_intent": ...,
    "dominant_confidence": ...,
    "gated_to_unassigned": ...,
    "size": ...
  },
  "run_meta": { ... },        # from NPZ meta[0] dict (global run params)
  "label_counts": {"H1": N, "H0": N, "U": N},
  "items": [
    {"idx": <int>, "label": <int>, "label_name": "H1"|"H0"|"U", "text": <str>},
    ...
  ]
}

Usage:
  python extract_cluster.py --npz wildchat_h1h0_by_cluster_fixed.npz --cluster_id 459
  python extract_cluster.py --npz wildchat_h1h0_by_cluster_fixed.npz --cluster_id 459 --out cluster_459.json
  python extract_cluster.py --npz wildchat_h1h0_by_cluster_fixed.npz --cluster_id 459 --label H1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CLUSTER_ALIASES = ("global_cluster", "cluster", "cluster_id", "clusters", "label_cluster")
LABEL_NAMES = {1: "H1", 0: "H0", -1: "U"}
LABEL_MAP = {"H1": 1, "H0": 0, "U": -1}


def _pick_cluster_key(data: np.lib.npyio.NpzFile, user_key: str) -> str:
    if user_key != "auto":
        if user_key not in data:
            raise KeyError(f"cluster_key='{user_key}' not in NPZ. available={data.files}")
        return user_key
    for k in CLUSTER_ALIASES:
        if k in data:
            return k
    raise KeyError(
        f"No cluster key found. Tried {CLUSTER_ALIASES}. available={data.files}\n"
        f"Pass --cluster_key <key> explicitly."
    )


def _load_sidecar(npz_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Try to find dominant_intents sidecar JSON next to or near the NPZ."""
    candidates = [
        npz_path.with_suffix(".dominant_intents.json"),
        # e.g. wildchat_h1h0_by_cluster_fixed.npz -> wildchat_h1h0_by_cluster.dominant_intents.json
        npz_path.parent / (npz_path.stem.replace("_fixed", "") + ".dominant_intents.json"),
    ]
    for p in candidates:
        if p.exists():
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            # handle dict keyed by cluster_id
            if isinstance(data, dict):
                return list(data.values())
    return None


def _run_meta(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "meta" not in data:
        return {}
    try:
        m = data["meta"]
        obj = m.item() if hasattr(m, "item") else m[0]
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def extract_cluster(
    npz_path: Path,
    cluster_id: int,
    cluster_key: str = "auto",
    label_filter: Optional[int] = None,
) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)

    ck = _pick_cluster_key(data, cluster_key)
    clusters = np.asarray(data[ck], dtype=np.int32)

    if cluster_id not in clusters:
        raise ValueError(
            f"cluster_id={cluster_id} not found in '{ck}'. "
            f"Available ids: {sorted(set(clusters.tolist()))[:20]} ..."
        )

    mask = clusters == cluster_id
    indices = np.where(mask)[0]

    text_arr = data["prompt"]
    label_arr = np.asarray(data["label"], dtype=np.int32) if "label" in data else None

    # Apply optional label filter before building items
    if label_filter is not None and label_arr is not None:
        indices = indices[label_arr[indices] == label_filter]

    items: List[Dict[str, Any]] = []
    for idx in indices:
        item: Dict[str, Any] = {
            "idx": int(idx),
            "text": str(text_arr[idx]),
        }
        if label_arr is not None:
            lv = int(label_arr[idx])
            item["label"] = lv
            item["label_name"] = LABEL_NAMES.get(lv, str(lv))
        items.append(item)

    # Label counts over the full (unfiltered) cluster
    full_indices = np.where(mask)[0]
    label_counts: Dict[str, int] = {}
    if label_arr is not None:
        y = label_arr[full_indices]
        label_counts = {
            "H1": int(np.sum(y == 1)),
            "H0": int(np.sum(y == 0)),
            "U": int(np.sum(y == -1)),
        }

    # Sidecar dominant intent
    sidecar = _load_sidecar(npz_path)
    cluster_meta: Dict[str, Any] = {}
    if sidecar is not None:
        for entry in sidecar:
            if entry.get("cluster_id") == cluster_id:
                cluster_meta = {k: v for k, v in entry.items() if k != "cluster_id"}
                break

    return {
        "cluster_id": cluster_id,
        "cluster_key": ck,
        "cluster_meta": cluster_meta,
        "run_meta": _run_meta(data),
        "label_counts": label_counts,
        "n_items": len(items),
        "label_filter": LABEL_NAMES.get(label_filter, None) if label_filter is not None else None,
        "items": items,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract a single cluster's texts, labels, and metadata from a labeled NPZ."
    )
    ap.add_argument("--npz", required=True, help="Input NPZ file")
    ap.add_argument("--cluster_id", type=int, required=True, help="Cluster ID to extract")
    ap.add_argument(
        "--cluster_key", default="auto",
        help="NPZ key holding cluster ids (default: auto-detect)"
    )
    ap.add_argument(
        "--label", choices=["H1", "H0", "U"], default=None,
        help="Only include items with this label (default: include all)"
    )
    ap.add_argument(
        "--out", default=None,
        help="Output JSON file (default: print to stdout)"
    )
    ap.add_argument(
        "--indent", type=int, default=2,
        help="JSON indent (default: 2; use 0 for compact)"
    )
    args = ap.parse_args()

    label_filter = LABEL_MAP[args.label] if args.label else None

    result = extract_cluster(
        npz_path=Path(args.npz),
        cluster_id=args.cluster_id,
        cluster_key=args.cluster_key,
        label_filter=label_filter,
    )

    indent = args.indent if args.indent > 0 else None
    serialized = json.dumps(result, indent=indent, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(serialized, encoding="utf-8")
        print(f"Wrote {result['n_items']} items to {out_path}", file=sys.stderr)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
