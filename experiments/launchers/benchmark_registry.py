"""Registry for benchmark datasets, embedders, and NPZ inputs."""
from __future__ import annotations

import shutil
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DatasetMeta:
    display_name: str
    region_key: str
    hadamard_default: bool


@dataclass(frozen=True)
class EmbedderMeta:
    display_name: str
    dim: int | None
    model_id: str | None
    paper_reference: str


DATASETS: dict[str, DatasetMeta] = {
    "wildchat_final": DatasetMeta(
        display_name="WildChat final",
        region_key="global_cluster",
        hadamard_default=True,
    ),
    "lmsys_cluster": DatasetMeta(
        display_name="LMSYS cluster",
        region_key="global_cluster",
        hadamard_default=True,
    ),
}

EMBEDDERS: dict[str, EmbedderMeta] = {
    "default": EmbedderMeta(
        display_name="Default baked-in embedder",
        dim=None,
        model_id=None,
        paper_reference="Existing NPZ files",
    ),
    "langcache": EmbedderMeta(
        display_name="LangCache Embed v2",
        dim=None,
        model_id="redis/langcache-embed-v2",
        paper_reference="Section 6.4 robustness experiment",
    ),
    "gte": EmbedderMeta(
        display_name="GTE large",
        dim=None,
        model_id="thenlper/gte-large",
        paper_reference="vCache datasets",
    ),
    "bge": EmbedderMeta(
        display_name="BGE large English v1.5",
        dim=None,
        model_id="BAAI/bge-large-en-v1.5",
        paper_reference="Embedding robustness candidate",
    ),
}

NPZ_PATHS: dict[tuple[str, str], Path] = {
    ("wildchat_final", "default"): Path("NeighborCache/data/h1h0_final.npz"),
    ("wildchat_final", "langcache"): Path("NeighborCache/data/h1h0_final_langcache.npz"),
    (
        "lmsys_cluster",
        "default",
    ): Path("NeighborCache/data/lmsys_h1h0_by_cluster_h1keep_with_unormalized_embedding_fixed.npz"),
}

NPZ_DOWNLOAD_URLS: dict[tuple[str, str], str] = {
    (
        "lmsys_cluster",
        "default",
    ): (
        "https://huggingface.co/datasets/talmal6/"
        "lmsys_lmsys_h1h0_by_cluster_h1keep_with_unormalized_embedding/"
        "resolve/main/lmsys_h1h0_by_cluster_h1keep_with_unormalized_embedding_fixed.npz"
    ),
}

HF_DATASET_IDS: dict[tuple[str, str], str] = {
    ("wildchat_final", "default"): "talmal6/h1h0-dataset-judges",
}

HF_DATASET_FILES: dict[tuple[str, str], str] = {
    ("wildchat_final", "default"): "h1h0_final.parquet",
}


def _known_dataset_message(dataset: str) -> str:
    known = ", ".join(sorted(DATASETS))
    return f"Unknown dataset {dataset!r}. Known datasets: {known}"


def _known_embedder_message(embedder: str) -> str:
    known = ", ".join(sorted(EMBEDDERS))
    return f"Unknown embedder {embedder!r}. Known embedders: {known}"


def registered_embedders_for_dataset(dataset: str) -> list[str]:
    return sorted(embedder for ds, embedder in NPZ_PATHS if ds == dataset)


def resolve_npz(dataset: str, embedder: str, repo_root: Path) -> Path:
    """Resolve a registered dataset/embedder NPZ path under repo_root."""
    if dataset not in DATASETS:
        raise KeyError(_known_dataset_message(dataset))
    if embedder not in EMBEDDERS:
        raise KeyError(_known_embedder_message(embedder))

    rel_path = NPZ_PATHS.get((dataset, embedder))
    if rel_path is None:
        available = registered_embedders_for_dataset(dataset)
        suffix = ", ".join(available) if available else "none"
        raise KeyError(
            f"No registered NPZ path for ({dataset!r}, {embedder!r}). "
            f"Registered embedders for {dataset!r}: {suffix}"
        )

    return (Path(repo_root) / rel_path).resolve()


def available_combinations(repo_root: Path) -> list[tuple[str, str]]:
    """Return registered combinations whose NPZ exists on disk."""
    root = Path(repo_root)
    out: list[tuple[str, str]] = []
    for combo, rel_path in sorted(NPZ_PATHS.items()):
        if (root / rel_path).exists():
            out.append(combo)
    return out


def validate_combination(dataset: str, embedder: str, repo_root: Path) -> None:
    """Raise if the combination is unregistered or its registered NPZ is absent."""
    npz_path = resolve_npz(dataset, embedder, repo_root)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Registered NPZ for ({dataset!r}, {embedder!r}) is missing: {npz_path}"
        )


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token is None and token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
    return token


def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "neighborcache-launcher"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download_npz(dataset: str, embedder: str, repo_root: Path) -> Path:
    """Download a registered NPZ if it is missing, then return its local path."""
    npz_path = resolve_npz(dataset, embedder, repo_root)
    if npz_path.exists():
        return npz_path

    url = NPZ_DOWNLOAD_URLS.get((dataset, embedder))
    if url is None:
        raise FileNotFoundError(
            f"Registered NPZ for ({dataset!r}, {embedder!r}) is missing: {npz_path}. "
            "No download URL is registered for this combination."
        )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = npz_path.with_name(npz_path.name + ".download")
    print(f"Downloading ({dataset}, {embedder}) from {url}", flush=True)
    print(f"Saving to {npz_path}", flush=True)
    try:
        request = Request(url, headers=_hf_headers())
        with urlopen(request) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(npz_path)
    except (HTTPError, URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        hint = (
            " For gated datasets, set HF_TOKEN or run `huggingface-cli login`; "
            "the launcher sends the same `Authorization: Bearer $HF_TOKEN` header "
            "shown in the Hugging Face API panel."
        )
        raise FileNotFoundError(
            f"Could not download ({dataset!r}, {embedder!r}) from {url}: {exc}.{hint}"
        ) from exc
    return npz_path


def ensure_npz(dataset: str, embedder: str, repo_root: Path, python_bin: str) -> Path:
    """Ensure a registered NPZ exists, downloading or converting HF data if needed."""
    npz_path = resolve_npz(dataset, embedder, repo_root)
    if npz_path.exists():
        return npz_path
    dataset_id = HF_DATASET_IDS.get((dataset, embedder))
    if dataset_id is not None:
        data_file = HF_DATASET_FILES.get((dataset, embedder))
        print(f"Materializing ({dataset}, {embedder}) from Hugging Face dataset {dataset_id}")
        cmd = [
            python_bin,
            "-m",
            "experiments.launchers.hf_dataset_to_npz",
            "--dataset-id",
            dataset_id,
            "--output",
            str(npz_path),
        ]
        if data_file is not None:
            cmd.extend(["--data-file", data_file])
        subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
        )
        return npz_path
    return download_npz(dataset, embedder, repo_root)
