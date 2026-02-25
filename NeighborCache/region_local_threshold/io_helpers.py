"""IO helpers: NPZ loading, feature resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def resolve_npz_path(data_path: str) -> Path:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    if p.suffix != ".npz":
        raise ValueError("Expected an .npz path for --data")
    return p


def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=True) as ds:
        return {k: ds[k] for k in ds.files}


def _is_numeric_2d(a: Any) -> bool:
    if not isinstance(a, np.ndarray):
        return False
    if a.ndim != 2 or a.shape[1] < 1:
        return False
    return a.dtype.kind in "fc"


def _is_numeric_1d(a: Any) -> bool:
    if not isinstance(a, np.ndarray):
        return False
    if a.ndim != 1 or a.shape[0] < 1:
        return False
    return a.dtype.kind in "fc"


def resolve_features(ds: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      feat_key: string describing the main feature matrix used by most methods
      X_main:   (N,D) float32 feature matrix
      X_cos:    (N,1) float32 cosine baseline feature if available (cosine_to_anchor), else None

    Priority for X_main:
      1) 'emb' if numeric 2D
      2) 'feat' if numeric 2D (rare; in your file it's object)
      3) fallback: 'cosine_to_anchor' as (N,1)

    NOTE:
      We keep 'cosine_to_anchor' separately so the Cosine method can use it directly,
      rather than abusing X_main.
    """
    X_cos: Optional[np.ndarray] = None
    if "cosine_to_anchor" in ds and _is_numeric_1d(ds["cosine_to_anchor"]):
        X_cos = ds["cosine_to_anchor"].astype(np.float32, copy=False).reshape(-1, 1)

    if "emb" in ds and _is_numeric_2d(ds["emb"]):
        X = ds["emb"].astype(np.float32, copy=False)
        print(f"[INFO] Using features key='emb' dim={X.shape[1]}")
        return "emb", X, X_cos

    if "feat" in ds and _is_numeric_2d(ds["feat"]):
        X = ds["feat"].astype(np.float32, copy=False)
        print(f"[INFO] Using features key='feat' dim={X.shape[1]}")
        return "feat", X, X_cos

    if X_cos is not None:
        print("[WARN] Using 1D features key='cosine_to_anchor' as main features")
        return "cosine_to_anchor", X_cos, X_cos

    feat_dtype = getattr(ds.get("feat", None), "dtype", None)
    feat_shape = getattr(ds.get("feat", None), "shape", None)
    emb_dtype = getattr(ds.get("emb", None), "dtype", None)
    emb_shape = getattr(ds.get("emb", None), "shape", None)
    raise ValueError(
        "No usable numeric features found. "
        f"Keys={sorted(ds.keys())}. "
        f"feat(dtype={feat_dtype},shape={feat_shape}) "
        f"emb(dtype={emb_dtype},shape={emb_shape})"
    )
