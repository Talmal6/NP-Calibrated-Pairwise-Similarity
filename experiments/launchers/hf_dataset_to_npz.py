"""Convert a Hugging Face dataset snapshot to the benchmark NPZ format."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq


ALIASES = {
    "emb": ("emb", "embedding", "embeddings", "unormalized_embedding"),
    "label": ("label", "labels", "y"),
    "global_cluster": ("global_cluster", "cluster", "cluster_id", "clusters", "label_cluster"),
    "text": ("text", "prompt", "query", "question", "request"),
}


class _StreamCache:
    def __init__(self, capacity: int, eviction_policy: str) -> None:
        self.capacity = max(1, int(capacity))
        self.eviction_policy = str(eviction_policy).lower()
        self._matrix: np.ndarray | None = None
        self._ids: list[int] = []
        self._created_at: list[int] = []
        self._last_accessed: list[int] = []
        self._clock = 0

    @staticmethod
    def _normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= eps:
            return arr
        return (arr / norm).astype(np.float32, copy=False)

    def nearest(self, embedding: np.ndarray) -> tuple[int | None, float | None]:
        if not self._ids or self._matrix is None:
            return None, None
        q = self._normalize(embedding)
        sims = self._matrix[: len(self._ids)] @ q
        idx = int(np.argmax(sims))
        self._clock += 1
        self._last_accessed[idx] = self._clock
        return int(self._ids[idx]), float(sims[idx])

    def add(self, embedding: np.ndarray, row_id: int) -> None:
        vector = self._normalize(embedding)
        if self._matrix is None:
            self._matrix = np.empty((self.capacity, int(vector.size)), dtype=np.float32)
        elif self._matrix.shape[1] != int(vector.size):
            raise ValueError(f"Embedding dimension mismatch: {self._matrix.shape[1]} vs {vector.size}")

        while len(self._ids) >= self.capacity:
            self._evict_one()

        self._clock += 1
        row = len(self._ids)
        self._ids.append(int(row_id))
        self._created_at.append(self._clock)
        self._last_accessed.append(self._clock)
        self._matrix[row] = vector

    def _evict_one(self) -> None:
        if not self._ids:
            return
        if self.eviction_policy == "lru":
            idx = int(np.argmin(self._last_accessed))
        elif self.eviction_policy == "mru":
            idx = int(np.argmax(self._last_accessed))
        else:
            idx = int(np.argmin(self._created_at))

        last = len(self._ids) - 1
        if idx != last and self._matrix is not None:
            self._matrix[idx] = self._matrix[last]
            self._ids[idx] = self._ids[last]
            self._created_at[idx] = self._created_at[last]
            self._last_accessed[idx] = self._last_accessed[last]
        self._ids.pop()
        self._created_at.pop()
        self._last_accessed.pop()


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token is None and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    return token or None


def _hf_hub_download_cached_first(dataset_id: str, data_file: str) -> str:
    try:
        return hf_hub_download(
            repo_id=dataset_id,
            filename=data_file,
            repo_type="dataset",
            token=_hf_token(),
            local_files_only=True,
        )
    except Exception:
        return hf_hub_download(repo_id=dataset_id, filename=data_file, repo_type="dataset", token=_hf_token())


def _flatten_dataset(obj: Dataset | DatasetDict, split: str | None) -> Dataset:
    if isinstance(obj, Dataset):
        return obj
    if split is not None:
        return obj[split]
    parts = [obj[name] for name in obj.keys()]
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _to_array(values: Any) -> np.ndarray | None:
    arr = np.asarray(values)
    if arr.dtype == object and arr.ndim == 1 and len(arr) and isinstance(arr[0], (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(list(values))
        except (TypeError, ValueError):
            return None
    if arr.dtype == object:
        return arr
    if arr.dtype.kind in "bifcUS":
        return arr
    return None


def _parse_embedding(value: Any, *, key: str) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = ast.literal_eval(text)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"Embedding key {key!r} produced a scalar")
    return arr.reshape(-1).astype(np.float32, copy=False)


def _to_object_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(-1)


def _column_arrays(ds: Dataset) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    columns = list(ds.column_names)
    for target, aliases in ALIASES.items():
        column = _find_column(columns, aliases)
        if column is None:
            continue
        arr = _to_array(ds[column])
        if arr is not None:
            arrays[target] = arr

    for column in columns:
        if column in arrays:
            continue
        arr = _to_array(ds[column])
        if arr is not None and arr.ndim <= 2:
            arrays.setdefault(column, arr)
    return arrays


def _arrow_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        values: list[Any] = []
        for chunk in column.chunks:
            if pa.types.is_string(chunk.type):
                chunk = chunk.cast(pa.large_string())
            values.extend(chunk.to_pylist())
        return np.asarray(values, dtype=object)

    arr = column.combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        values = arr.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(arr), arr.type.list_size)
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        return np.asarray(arr.to_pylist())
    return arr.to_numpy(zero_copy_only=False)


def _parquet_arrays(dataset_id: str, data_file: str) -> dict[str, np.ndarray]:
    path = _hf_hub_download_cached_first(dataset_id, data_file)
    schema = pq.read_schema(path)
    columns = list(schema.names)
    resolved = {
        target: column
        for target, aliases in ALIASES.items()
        if (column := _find_column(columns, aliases)) is not None
    }
    table = pq.read_table(path, columns=sorted(set(resolved.values())))
    arrays = {target: _arrow_to_numpy(table[column]) for target, column in resolved.items()}
    if "emb" in arrays:
        arrays["emb"] = arrays["emb"].astype(np.float32, copy=False)
    for key in ["label", "global_cluster"]:
        if key in arrays:
            arrays[key] = arrays[key].astype(np.int32, copy=False)
    return arrays


def _parquet_selected_arrays(dataset_id: str, data_file: str, columns: list[str]) -> dict[str, np.ndarray]:
    path = _hf_hub_download_cached_first(dataset_id, data_file)
    schema = pq.read_schema(path)
    available = set(schema.names)
    selected = list(dict.fromkeys(column for column in columns if column in available))
    missing_required = [column for column in columns[:3] if column not in available]
    if missing_required:
        raise ValueError(
            f"{dataset_id} missing required columns {missing_required}. "
            f"Available columns: {sorted(available)}"
        )
    table = pq.read_table(path, columns=selected)
    return {column: _arrow_to_numpy(table[column]) for column in selected}


def _hf_selected_arrays(dataset_id: str, split: str | None, columns: list[str]) -> dict[str, np.ndarray]:
    token = _hf_token()
    loaded = load_dataset(dataset_id, split=split, token=token) if split else load_dataset(dataset_id, token=token)
    ds = _flatten_dataset(loaded, split=None)
    available = set(ds.column_names)
    missing_required = [column for column in columns[:3] if column not in available]
    if missing_required:
        raise ValueError(
            f"{dataset_id} missing required columns {missing_required}. "
            f"Available columns: {sorted(available)}"
        )
    selected = list(dict.fromkeys(column for column in columns if column in available))
    return {column: np.asarray(ds[column], dtype=object) for column in selected}


def _stable_int_ids(values: np.ndarray) -> np.ndarray:
    mapping: dict[str, int] = {}
    out = np.empty(values.shape[0], dtype=np.int32)
    for i, value in enumerate(values):
        text = str(value)
        if text not in mapping:
            mapping[text] = len(mapping)
        out[i] = mapping[text]
    return out


def _semantic_cache_pair_arrays(
    arrays: dict[str, np.ndarray],
    *,
    embedding_key: str,
    response_key: str,
    cluster_key: str | None,
    prompt_key: str,
    id_key: str | None,
    cache_size: int,
    eviction_policy: str,
    max_rows: int | None,
) -> dict[str, np.ndarray]:
    embeddings_raw = _to_object_array(arrays[embedding_key])
    responses = _to_object_array(arrays[response_key])
    prompts = _to_object_array(arrays[prompt_key])
    ids = _to_object_array(arrays[id_key]) if id_key and id_key in arrays else np.arange(prompts.shape[0], dtype=object)
    if cluster_key and cluster_key in arrays:
        clusters_raw = _to_object_array(arrays[cluster_key])
        cluster_source = cluster_key
    else:
        clusters_raw = responses
        cluster_source = response_key

    n = int(prompts.shape[0])
    if max_rows is not None:
        n = min(n, int(max_rows))

    embeddings = [_parse_embedding(embeddings_raw[i], key=embedding_key) for i in range(n)]
    cache = _StreamCache(capacity=cache_size, eviction_policy=eviction_policy)

    feats: list[np.ndarray] = []
    labels: list[int] = []
    cosines: list[float] = []
    global_clusters: list[Any] = []
    query_ids: list[Any] = []
    anchor_ids: list[Any] = []
    query_text: list[Any] = []
    anchor_text: list[Any] = []
    query_emb: list[np.ndarray] = []
    anchor_emb: list[np.ndarray] = []

    for i, q in enumerate(embeddings):
        nearest_idx, sim = cache.nearest(q)
        if nearest_idx is not None and sim is not None:
            a = embeddings[int(nearest_idx)]
            same_cluster = str(clusters_raw[i]) == str(clusters_raw[int(nearest_idx)])
            feats.append((q * a).astype(np.float32, copy=False))
            labels.append(1 if same_cluster else 0)
            cosines.append(float(sim))
            global_clusters.append(clusters_raw[i])
            query_ids.append(ids[i])
            anchor_ids.append(ids[int(nearest_idx)])
            query_text.append(prompts[i])
            anchor_text.append(prompts[int(nearest_idx)])
            query_emb.append(q)
            anchor_emb.append(a)
        cache.add(q, i)

    if not feats:
        raise ValueError(f"{prompt_key}/{embedding_key} stream produced no nearest-neighbor pairs")

    label_arr = np.asarray(labels, dtype=np.int32)
    h0 = int(np.sum(label_arr == 0))
    h1 = int(np.sum(label_arr == 1))
    if h0 == 0 or h1 == 0:
        raise ValueError(
            "Semantic-cache pair conversion produced only one class: "
            f"H0={h0}, H1={h1}. Check cluster_key={cluster_source!r}."
        )

    cluster_obj = np.asarray(global_clusters, dtype=object)
    return {
        "emb": np.vstack(feats).astype(np.float32, copy=False),
        "label": label_arr,
        "cosine_to_anchor": np.asarray(cosines, dtype=np.float32),
        "global_cluster": _stable_int_ids(cluster_obj),
        "global_cluster_raw": cluster_obj,
        "query_emb": np.vstack(query_emb).astype(np.float32, copy=False),
        "anchor_emb": np.vstack(anchor_emb).astype(np.float32, copy=False),
        "qid": np.asarray(query_ids, dtype=object),
        "anchor_qid": np.asarray(anchor_ids, dtype=object),
        "query_text": np.asarray(query_text, dtype=object),
        "anchor_text": np.asarray(anchor_text, dtype=object),
        "source_embedding_key": np.asarray([embedding_key], dtype=object),
        "source_response_key": np.asarray([response_key], dtype=object),
        "source_cluster_key": np.asarray([cluster_source], dtype=object),
        "source_prompt_key": np.asarray([prompt_key], dtype=object),
        "cache_size": np.asarray([int(cache_size)], dtype=np.int32),
        "eviction_policy": np.asarray([eviction_policy], dtype=object),
    }


def convert_semantic_cache_pairs(
    dataset_id: str,
    output: Path,
    *,
    split: str | None,
    data_file: str | None,
    embedding_key: str,
    response_key: str,
    cluster_key: str | None,
    prompt_key: str,
    id_key: str | None,
    cache_size: int,
    eviction_policy: str,
    max_rows: int | None,
) -> None:
    columns = [embedding_key, response_key, prompt_key]
    if cluster_key:
        columns.append(cluster_key)
    if id_key:
        columns.append(id_key)

    if data_file is not None:
        arrays = _parquet_selected_arrays(dataset_id, data_file, columns)
    else:
        arrays = _hf_selected_arrays(dataset_id, split, columns)

    converted = _semantic_cache_pair_arrays(
        arrays,
        embedding_key=embedding_key,
        response_key=response_key,
        cluster_key=cluster_key,
        prompt_key=prompt_key,
        id_key=id_key,
        cache_size=cache_size,
        eviction_policy=eviction_policy,
        max_rows=max_rows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **converted)
    print(f"Saved {output}")
    print(f"Rows: {int(converted['label'].shape[0])}")
    print(f"H0: {int(np.sum(converted['label'] == 0))}")
    print(f"H1: {int(np.sum(converted['label'] == 1))}")
    print(f"Keys: {sorted(converted)}")


def convert(dataset_id: str, output: Path, split: str | None, data_file: str | None) -> None:
    if data_file is not None:
        arrays = _parquet_arrays(dataset_id, data_file)
        column_names = sorted(arrays)
    else:
        token = _hf_token()
        loaded = load_dataset(dataset_id, split=split, token=token) if split else load_dataset(dataset_id, token=token)
        ds = _flatten_dataset(loaded, split=None)
        arrays = _column_arrays(ds)
        column_names = list(ds.column_names)
    required = {"emb", "label", "global_cluster"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(
            f"{dataset_id} cannot be converted to benchmark NPZ; missing arrays {missing}. "
            f"Columns: {column_names}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"Saved {output}")
    print(f"Rows: {int(arrays['label'].shape[0])}")
    print(f"Keys: {sorted(arrays)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face dataset to NeighborCache NPZ.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--mode", choices=["auto", "semantic-cache-pairs"], default="auto")
    parser.add_argument("--embedding-key", default="emb_gte")
    parser.add_argument("--response-key", default="response_llama_3_8b")
    parser.add_argument("--cluster-key", default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--cache-size", type=int, default=4096)
    parser.add_argument("--eviction-policy", choices=["mru", "lru", "fifo"], default="mru")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "semantic-cache-pairs":
        convert_semantic_cache_pairs(
            args.dataset_id,
            args.output,
            split=args.split,
            data_file=args.data_file,
            embedding_key=args.embedding_key,
            response_key=args.response_key,
            cluster_key=args.cluster_key,
            prompt_key=args.prompt_key,
            id_key=args.id_key,
            cache_size=args.cache_size,
            eviction_policy=args.eviction_policy,
            max_rows=args.max_rows,
        )
    else:
        convert(args.dataset_id, args.output, args.split, args.data_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
