from __future__ import annotations
import os
import pickle
from typing import Tuple, Dict, Any, Optional

import numpy as np


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """
    Row-wise L2 normalization.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms


def _infer_keys(first_sample: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Infer embedding and label keys from the first dict sample.
    """
    emb1_key = next((k for k in first_sample.keys() if ("q1" in k and "emb" in k)), None)
    emb2_key = next((k for k in first_sample.keys() if ("q2" in k and "emb" in k)), None)

    if emb1_key is None:
        emb1_key = "q1_emb"
    if emb2_key is None:
        emb2_key = "q2_emb"

    label_key = next((k for k in first_sample.keys() if k in ("is_duplicate", "label", "y")), None)
    if label_key is None:
        label_key = "is_duplicate"

    return emb1_key, emb2_key, label_key


def load_quora_hadamard(
    pkl_path: str,
    split_key: str = "train",
    emb1_key: Optional[str] = None,
    emb2_key: Optional[str] = None,
    label_key: Optional[str] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Quora-like pickle and return:
      X: hadamard(U,V) with U,V L2-normalized (so sum(X) == cosine(U,V))
      y: {0,1}

    Supports:
      - raw list of dict samples
      - dict with a 'train' (or split_key) list

    Each sample dict must contain:
      - embedding for q1
      - embedding for q2
      - label in {0,1} or boolean/int
    """
    # Check for cached numpy files
    cache_base = f"{pkl_path}.{split_key}"
    path_X = f"{cache_base}.X.npy"
    path_y = f"{cache_base}.y.npy"

    if os.path.exists(path_X) and os.path.exists(path_y):
        print(f"Loading cached numpy data from {path_X}...")
        try:
            X = np.load(path_X, mmap_mode="r")
            y = np.load(path_y)
            if X.dtype != dtype:
                X = X.astype(dtype)
            return X, y
        except Exception as e:
            print(f"Failed to load cache: {e}. Falling back to pickle.")

    print(f"Loading data from {pkl_path}... (This might take a while, but will be cached)")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict) and split_key in raw:
        data = raw[split_key]
    else:
        data = raw

    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise ValueError("PKL does not contain a non-empty list of samples (or raw[split_key]).")

    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("Expected each sample to be a dict with q1/q2 embeddings and label.")

    if emb1_key is None or emb2_key is None or label_key is None:
        k1, k2, ky = _infer_keys(first)
        emb1_key = emb1_key or k1
        emb2_key = emb2_key or k2
        label_key = label_key or ky

    # load arrays
    print("Parsing embeddings...")
    try:
        U = np.asarray([d[emb1_key] for d in data], dtype=dtype)
        V = np.asarray([d[emb2_key] for d in data], dtype=dtype)
    except KeyError as e:
        raise KeyError(
            f"Missing embedding key {e} in samples. "
            f"Detected keys include: {list(first.keys())}"
        )

    try:
        y = np.asarray([d[label_key] for d in data], dtype=np.int32)
    except KeyError as e:
        raise KeyError(
            f"Missing label key {e} in samples. "
            f"Detected keys include: {list(first.keys())}"
        )

    y = (y > 0).astype(np.int32)

    # normalize + hadamard
    print("Normalizing and computing interaction features...")
    U = normalize_l2(U)
    V = normalize_l2(V)
    X = (U * V).astype(dtype)

    print(f"Caching numpy data to {path_X} and {path_y}...")
    try:
        np.save(path_X, X)
        np.save(path_y, y)
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return X, y


def subsample_by_class(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample balanced sets from X by y:
      returns (H0, H1) where:
        H0 are samples with y==0 (size <= n_samples)
        H1 are samples with y==1 (size <= n_samples)

    No replacement.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows != y size: {X.shape[0]} != {y.shape[0]}")

    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        # degenerate
        return X[idx0], X[idx1]

    n0 = min(idx0.size, n_samples)
    n1 = min(idx1.size, n_samples)

    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)

    return X[sel0], X[sel1]
