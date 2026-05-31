"""Convert a Hugging Face dataset snapshot to the benchmark NPZ format."""
from __future__ import annotations

import argparse
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


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token is None and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    return token or None


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
    arr = column.combine_chunks()
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return np.asarray(arr.to_pylist(), dtype=object)
    if pa.types.is_fixed_size_list(arr.type):
        values = arr.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(arr), arr.type.list_size)
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        return np.asarray(arr.to_pylist())
    return arr.to_numpy(zero_copy_only=False)


def _parquet_arrays(dataset_id: str, data_file: str) -> dict[str, np.ndarray]:
    path = hf_hub_download(repo_id=dataset_id, filename=data_file, repo_type="dataset", token=_hf_token())
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
    args = parser.parse_args()
    convert(args.dataset_id, args.output, args.split, args.data_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
