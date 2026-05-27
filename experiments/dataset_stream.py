from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StreamExample:
    id: str
    prompt: str
    gold_response: str = ""
    cluster: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class DatasetInfo:
    path: str
    name: str
    n_examples: int
    prompt_key: Optional[str]
    response_key: Optional[str]
    id_key: Optional[str]
    cluster_key: Optional[str]
    embedding_key: Optional[str]
    warnings: List[str] = field(default_factory=list)


def _first_present(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    keyset = set(keys)
    for key in candidates:
        if key in keyset:
            return key
    return None


def _json_loads_maybe(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return value


def _to_embedding(value: Any, *, key: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    value = _json_loads_maybe(value)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"Embedding key '{key}' produced scalar value")
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return None
    return arr.astype(np.float32, copy=False)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _cluster_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value)
    return text if text != "" else None


def _choose_embedding_key(
    keys: Sequence[str],
    arrays: Optional[Dict[str, np.ndarray]],
    explicit: Optional[str],
) -> Optional[str]:
    if explicit:
        return explicit

    preferred = ["embedding", "emb", "emb_gte", "emb_e5_large_v2", "emb_text-embedding-3-small"]
    hit = _first_present(keys, preferred)
    if hit is not None:
        return hit

    emb_like = [k for k in keys if k.lower().startswith("emb")]
    if arrays is not None:
        numeric_2d = [
            k
            for k in emb_like
            if k in arrays
            and getattr(arrays[k], "ndim", 0) == 2
            and getattr(arrays[k], "dtype", None) is not None
            and arrays[k].dtype.kind in "fci"
        ]
        if numeric_2d:
            return numeric_2d[0]
    return emb_like[0] if emb_like else None


def _choose_response_key(keys: Sequence[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    hit = _first_present(
        keys,
        [
            "gold_response",
            "label_response",
            "response",
            "answer",
            "target",
            "completion",
            "response_llama_3_8b",
            "response_gpt-4o-mini",
            "response_gpt-4.1-nano",
        ],
    )
    if hit is not None:
        return hit
    response_like = [k for k in keys if k.lower().startswith("response")]
    return response_like[0] if response_like else None


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise ValueError(f"{path} JSON list must contain objects")
        return list(obj)
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return list(obj["data"])
    raise ValueError(f"Unsupported JSON dataset structure in {path}")


def _load_csv_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_parquet_records(path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError(
            "Parquet loading requires pandas with a parquet engine such as pyarrow or fastparquet"
        ) from exc
    try:
        return pd.read_parquet(path).to_dict("records")
    except ImportError as exc:
        raise ImportError(
            "Parquet loading requires pyarrow or fastparquet. Install one, or convert the dataset to JSONL/CSV."
        ) from exc


def _examples_from_records(
    records: Iterable[Dict[str, Any]],
    *,
    path: Path,
    prompt_key: Optional[str],
    response_key: Optional[str],
    id_key: Optional[str],
    cluster_key: Optional[str],
    embedding_key: Optional[str],
) -> Tuple[List[StreamExample], DatasetInfo]:
    records = list(records)
    keys = list(records[0].keys()) if records else []
    prompt_key = prompt_key or _first_present(keys, ["prompt", "text", "query", "question", "input"])
    response_key = _choose_response_key(keys, response_key)
    id_key = id_key or _first_present(keys, ["id", "prompt_id", "qid", "query_id"])
    cluster_key = cluster_key or _first_present(
        keys, ["cluster", "cluster_id", "id_set", "ID_Set", "global_cluster", "label"]
    )
    embedding_key = _choose_embedding_key(keys, None, embedding_key)

    if prompt_key is None:
        raise ValueError(f"Could not infer prompt key from columns: {keys}")

    warnings: List[str] = []
    if response_key is None:
        warnings.append("missing gold/response column; exact correctness cannot be computed")
    if embedding_key is None:
        warnings.append("missing embedding column; embedding-based methods cannot run")

    examples: List[StreamExample] = []
    for i, row in enumerate(records):
        ex_id = _string_or_empty(row.get(id_key)) if id_key else str(i)
        prompt = _string_or_empty(row.get(prompt_key))
        gold = _string_or_empty(row.get(response_key)) if response_key else ""
        cluster = _cluster_or_none(row.get(cluster_key)) if cluster_key else None
        emb = _to_embedding(row.get(embedding_key), key=embedding_key) if embedding_key else None
        metadata = dict(row)
        examples.append(
            StreamExample(
                id=ex_id,
                prompt=prompt,
                gold_response=gold,
                cluster=cluster,
                metadata=metadata,
                embedding=emb,
            )
        )

    return examples, DatasetInfo(
        path=str(path),
        name=path.stem,
        n_examples=len(examples),
        prompt_key=prompt_key,
        response_key=response_key,
        id_key=id_key,
        cluster_key=cluster_key,
        embedding_key=embedding_key,
        warnings=warnings,
    )


def _npz_array_to_rows(
    ds: Dict[str, np.ndarray],
    *,
    prompt_key: str,
    response_key: Optional[str],
    id_key: Optional[str],
    cluster_key: Optional[str],
    embedding_key: Optional[str],
    indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    n = int(np.asarray(ds[prompt_key]).shape[0])
    rows: List[Dict[str, Any]] = []
    selected = [prompt_key]
    for key in [response_key, id_key, cluster_key, embedding_key]:
        if key and key not in selected:
            selected.append(key)

    row_indices: Sequence[int] = indices if indices is not None else range(n)
    for raw_i in row_indices:
        i = int(raw_i)
        row = {key: ds[key][i] for key in selected if key in ds}
        rows.append(row)
    return rows


def _load_npz(
    path: Path,
    *,
    prompt_key: Optional[str],
    response_key: Optional[str],
    id_key: Optional[str],
    cluster_key: Optional[str],
    embedding_key: Optional[str],
    seed: int = 42,
    shuffle: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[StreamExample], DatasetInfo]:
    with np.load(str(path), allow_pickle=True) as loaded:
        ds = {k: loaded[k] for k in loaded.files}

    keys = list(ds.keys())
    prompt_key = prompt_key or _first_present(keys, ["prompt", "text", "query", "question", "input"])
    if prompt_key is None:
        raise ValueError(f"Could not infer prompt key from NPZ keys: {keys}")
    response_key = _choose_response_key(keys, response_key)
    id_key = id_key or _first_present(keys, ["id", "prompt_id", "qid", "query_id"])
    cluster_key = cluster_key or _first_present(
        keys, ["cluster", "cluster_id", "id_set", "ID_Set", "global_cluster", "label"]
    )
    embedding_key = _choose_embedding_key(keys, ds, embedding_key)
    n = int(np.asarray(ds[prompt_key]).shape[0])
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        indices = rng.permutation(indices)
    if limit is not None:
        indices = indices[: int(limit)]

    rows = _npz_array_to_rows(
        ds,
        prompt_key=prompt_key,
        response_key=response_key,
        id_key=id_key,
        cluster_key=cluster_key,
        embedding_key=embedding_key,
        indices=indices,
    )
    return _examples_from_records(
        rows,
        path=path,
        prompt_key=prompt_key,
        response_key=response_key,
        id_key=id_key,
        cluster_key=cluster_key,
        embedding_key=embedding_key,
    )


def load_dataset_stream(
    dataset_path: str,
    *,
    prompt_key: Optional[str] = None,
    response_key: Optional[str] = None,
    id_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    embedding_key: Optional[str] = None,
    seed: int = 42,
    shuffle: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[StreamExample], DatasetInfo]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        examples, info = _load_npz(
            path,
            prompt_key=prompt_key,
            response_key=response_key,
            id_key=id_key,
            cluster_key=cluster_key,
            embedding_key=embedding_key,
            seed=seed,
            shuffle=shuffle,
            limit=limit,
        )
        return examples, info
    elif suffix in {".json", ".jsonl"}:
        examples, info = _examples_from_records(
            _load_json_records(path),
            path=path,
            prompt_key=prompt_key,
            response_key=response_key,
            id_key=id_key,
            cluster_key=cluster_key,
            embedding_key=embedding_key,
        )
    elif suffix == ".csv":
        examples, info = _examples_from_records(
            _load_csv_records(path),
            path=path,
            prompt_key=prompt_key,
            response_key=response_key,
            id_key=id_key,
            cluster_key=cluster_key,
            embedding_key=embedding_key,
        )
    elif suffix == ".parquet":
        examples, info = _examples_from_records(
            _load_parquet_records(path),
            path=path,
            prompt_key=prompt_key,
            response_key=response_key,
            id_key=id_key,
            cluster_key=cluster_key,
            embedding_key=embedding_key,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if shuffle:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(examples))
        examples = [examples[int(i)] for i in order]

    if limit is not None:
        examples = examples[: int(limit)]
        info.n_examples = len(examples)

    return examples, info
