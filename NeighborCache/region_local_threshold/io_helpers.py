"""IO helpers: NPZ loading, feature resolution."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_RAW_PAIR_KEY_CANDIDATES: List[Tuple[str, str]] = [
    ("x", "y"),
    ("x_emb", "y_emb"),
    ("x_embedding", "y_embedding"),
    ("query_emb", "anchor_emb"),
    ("query_embedding", "anchor_embedding"),
    ("q1_emb", "q2_emb"),
    ("emb_x", "emb_y"),
]


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


def _pairwise_cosine_rows(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Xv = np.asarray(X, dtype=np.float32)
    Yv = np.asarray(Y, dtype=np.float32)
    if Xv.ndim != 2 or Yv.ndim != 2:
        raise ValueError(f"Pair embeddings must be 2D, got X={Xv.shape} Y={Yv.shape}")
    if Xv.shape != Yv.shape:
        raise ValueError(f"Pair embeddings shape mismatch: X={Xv.shape} Y={Yv.shape}")

    num = np.sum(Xv * Yv, axis=1, dtype=np.float64)
    den = np.linalg.norm(Xv, axis=1) * np.linalg.norm(Yv, axis=1)
    cos = num / np.maximum(den, float(eps))
    cos = np.nan_to_num(cos, nan=0.0, posinf=1.0, neginf=-1.0)
    cos = np.clip(cos, -1.0, 1.0)
    return cos.astype(np.float32, copy=False)


def resolve_train_pairwise_cosine(
    ds: Dict[str, np.ndarray],
    *,
    x_key: Optional[str] = None,
    y_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Resolve raw pair embeddings and compute per-row cosine(x, y).

    Returns:
      source_key_pair: "x_key+y_key" string when resolved, else None
      pair_cosine:      float32 (N,) cosine values when resolved, else None

    If explicit keys are provided, strict validation is applied and a ValueError
    is raised on mismatch.
    """
    if (x_key is None) != (y_key is None):
        raise ValueError("train pair cosine requires both --train_pair_x_key and --train_pair_y_key")

    n_rows = None
    if "label" in ds:
        n_rows = int(np.asarray(ds["label"]).shape[0])

    explicit = x_key is not None and y_key is not None
    candidates = [(str(x_key), str(y_key))] if explicit else list(_RAW_PAIR_KEY_CANDIDATES)

    for kx, ky in candidates:
        if kx not in ds or ky not in ds:
            continue

        X = np.asarray(ds[kx])
        Y = np.asarray(ds[ky])

        if not _is_numeric_2d(X) or not _is_numeric_2d(Y):
            if explicit:
                raise ValueError(
                    f"train pair keys must be numeric 2D arrays, got {kx}:{X.dtype}/{X.shape}, {ky}:{Y.dtype}/{Y.shape}"
                )
            continue

        if X.shape != Y.shape:
            if explicit:
                raise ValueError(f"train pair key shape mismatch: {kx}:{X.shape} vs {ky}:{Y.shape}")
            continue

        if n_rows is not None and int(X.shape[0]) != n_rows:
            if explicit:
                raise ValueError(
                    f"train pair key row mismatch with labels: {kx}:{X.shape[0]} vs label:{n_rows}"
                )
            continue

        return f"{kx}+{ky}", _pairwise_cosine_rows(X, Y)

    if explicit:
        keys = sorted(ds.keys())
        raise ValueError(
            "Could not resolve train pair embeddings from explicit keys "
            f"({x_key}, {y_key}). Available keys={keys}"
        )
    return None, None


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


def _infer_text_pair_keys(ds: Dict[str, np.ndarray]) -> Optional[Tuple[str, str]]:
    candidates = [
        ("query_text", "anchor_text"),
        ("q_text", "rep_text"),
        ("q1_text", "q2_text"),
        ("question", "anchor_question"),
        ("text", "anchor_text"),
    ]
    for k1, k2 in candidates:
        if k1 in ds and k2 in ds:
            return k1, k2
    return None


def _as_object_text_array(a: np.ndarray, *, key: str) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError(f"Text key '{key}' must be 1D, got shape={x.shape}")
    return x.astype(object, copy=False)


def _first_non_empty_str(*vals: Any) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v)
        if s.strip() != "":
            return s
    return None


def _load_qid_text_map(pkl_path: Path) -> Dict[int, str]:
    with pkl_path.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        if "train" in raw and isinstance(raw["train"], list):
            rows = raw["train"]
        else:
            rows = []
            for v in raw.values():
                if isinstance(v, list):
                    rows.extend(v)
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError(f"Unsupported PKL format for text mapping: {type(raw)}")

    qid_to_text: Dict[int, str] = {}
    qid_keys_1 = ["qid1", "q1_id", "question1_id"]
    qid_keys_2 = ["qid2", "q2_id", "question2_id"]
    txt_keys_1 = ["q1_text", "question1", "q1", "text1", "query"]
    txt_keys_2 = ["q2_text", "question2", "q2", "text2", "candidate", "doc"]

    for row in rows:
        if not isinstance(row, dict):
            continue

        qid1 = next((row.get(k) for k in qid_keys_1 if k in row), None)
        qid2 = next((row.get(k) for k in qid_keys_2 if k in row), None)

        txt1 = _first_non_empty_str(*(row.get(k) for k in txt_keys_1))
        txt2 = _first_non_empty_str(*(row.get(k) for k in txt_keys_2))

        if qid1 is not None and txt1 is not None:
            qid_i = int(qid1)
            if qid_i not in qid_to_text:
                qid_to_text[qid_i] = txt1
        if qid2 is not None and txt2 is not None:
            qid_i = int(qid2)
            if qid_i not in qid_to_text:
                qid_to_text[qid_i] = txt2

    return qid_to_text


def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(denom, eps)).astype(np.float32, copy=False)


def _build_text_pairs_from_single_text(
    text: np.ndarray,
    region_id: np.ndarray,
    *,
    anchor_features: Optional[np.ndarray],
    anchor_strategy: str,
    seed: int,
) -> np.ndarray:
    """Construct (query_text, anchor_text) by selecting one anchor per region.

    This mirrors the anchor-vs-others setup used by embedding methods.
    """
    t = np.asarray(text, dtype=object).reshape(-1)
    r = np.asarray(region_id, dtype=np.int64).reshape(-1)
    if t.shape[0] != r.shape[0]:
        raise ValueError(f"text/region row mismatch: text={t.shape[0]} region={r.shape[0]}")

    rng = np.random.default_rng(seed)
    anchor_idx_by_region: Dict[int, int] = {}
    feat = None
    if anchor_features is not None:
        feat = np.asarray(anchor_features)
        if feat.ndim != 2 or feat.shape[0] != t.shape[0]:
            feat = None
        elif feat.shape[1] > 1:
            feat = _normalize_rows(feat)
        else:
            feat = None

    for rid in np.unique(r):
        idx = np.flatnonzero(r == rid)
        if idx.size == 0:
            continue

        if anchor_strategy == "random":
            aidx = int(rng.choice(idx))
        elif anchor_strategy == "centroid_nearest" and feat is not None:
            Xr = feat[idx]
            c = np.mean(Xr, axis=0)
            c_norm = float(np.linalg.norm(c))
            if c_norm > 1e-12:
                c = c / c_norm
            sims = Xr @ c
            aidx = int(idx[int(np.argmax(sims))])
        else:
            aidx = int(idx[0])

        anchor_idx_by_region[int(rid)] = aidx

    anchor_text = np.empty_like(t, dtype=object)
    for i, rid in enumerate(r):
        aidx = anchor_idx_by_region.get(int(rid), i)
        anchor_text[i] = "" if t[aidx] is None else str(t[aidx])

    query_text = np.array(["" if v is None else str(v) for v in t], dtype=object)
    return np.column_stack([query_text, anchor_text]).astype(object, copy=False)


def resolve_text_pairs(
    ds: Dict[str, np.ndarray],
    *,
    text_pair_keys: Optional[Tuple[str, str]] = None,
    text_source_pkl: Optional[str] = None,
    region_id: Optional[np.ndarray] = None,
    anchor_features: Optional[np.ndarray] = None,
    anchor_strategy: str = "centroid_nearest",
    seed: int = 42,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Resolve text pairs matrix (N,2) from NPZ or qid mapping.

    Priority:
      1) Direct NPZ text keys (explicit --text_pair_keys or inferred keys)
      2) qid/anchor_qid + --text_source_pkl mapping
    """
    n_rows = None
    if "label" in ds:
        n_rows = int(np.asarray(ds["label"]).shape[0])

    resolved_keys = text_pair_keys or _infer_text_pair_keys(ds)
    if resolved_keys is not None:
        k1, k2 = resolved_keys
        if k1 in ds and k2 in ds:
            t1 = _as_object_text_array(ds[k1], key=k1)
            t2 = _as_object_text_array(ds[k2], key=k2)
            if t1.shape[0] != t2.shape[0]:
                raise ValueError(
                    f"text keys length mismatch: {k1}={t1.shape[0]} vs {k2}={t2.shape[0]}"
                )
            if n_rows is not None and t1.shape[0] != n_rows:
                raise ValueError(
                    f"text keys rows mismatch label rows: text={t1.shape[0]} label={n_rows}"
                )
            X_text = np.column_stack([t1, t2]).astype(object, copy=False)
            return f"{k1}+{k2}", X_text

    if text_source_pkl is not None:
        if "qid" not in ds or "anchor_qid" not in ds:
            raise ValueError(
                "--text_source_pkl requires NPZ keys 'qid' and 'anchor_qid'"
            )
        p = Path(text_source_pkl)
        if not p.exists():
            raise FileNotFoundError(f"text source pkl not found: {p}")

        qid_to_text = _load_qid_text_map(p)
        if len(qid_to_text) == 0:
            raise RuntimeError(f"No qid->text mappings found in {p}")

        qid = np.asarray(ds["qid"]).reshape(-1)
        aqid = np.asarray(ds["anchor_qid"]).reshape(-1)
        if qid.shape[0] != aqid.shape[0]:
            raise ValueError(f"qid/anchor_qid shape mismatch: {qid.shape} vs {aqid.shape}")
        if n_rows is not None and qid.shape[0] != n_rows:
            raise ValueError(
                f"qid rows mismatch label rows: qid={qid.shape[0]} label={n_rows}"
            )

        q_txt = np.array([qid_to_text.get(int(v), "") for v in qid], dtype=object)
        a_txt = np.array([qid_to_text.get(int(v), "") for v in aqid], dtype=object)
        X_text = np.column_stack([q_txt, a_txt]).astype(object, copy=False)
        return f"qid+anchor_qid<-{p.name}", X_text

    # Fallback: if only one text column exists, build anchor-vs-others text pairs by region.
    if "text" in ds and region_id is not None:
        t = _as_object_text_array(ds["text"], key="text")
        X_text = _build_text_pairs_from_single_text(
            t,
            np.asarray(region_id, dtype=np.int64),
            anchor_features=anchor_features,
            anchor_strategy=anchor_strategy,
            seed=seed,
        )
        return f"text+region_anchor[{anchor_strategy}]", X_text

    return None, None
