from __future__ import annotations

import importlib.util
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset_stream import StreamExample
from .equivalence import EquivalenceJudge
from .online_policies import _base_record, _evaluate_decision
from .vcache_metrics import add_cumulative_fields


class VCacheUnavailableError(RuntimeError):
    pass


class _Registry:
    def __init__(self, stream: Sequence[StreamExample], judge_mode: str) -> None:
        self.prompt_to_example: Dict[str, StreamExample] = {}
        self.embedding_id_to_example: Dict[int, StreamExample] = {}
        self.last_prompt: Optional[str] = None
        self.last_example: Optional[StreamExample] = None
        self.cluster_to_int: Dict[str, int] = {}
        self.judge_mode = judge_mode
        for i, ex in enumerate(stream):
            wrapped = self.wrap_prompt(ex, i)
            self.prompt_to_example[wrapped] = ex
            if ex.cluster is not None and str(ex.cluster) not in self.cluster_to_int:
                self.cluster_to_int[str(ex.cluster)] = len(self.cluster_to_int)

    @staticmethod
    def wrap_prompt(example: StreamExample, index: int) -> str:
        return f"__neighborcache_prompt_id__={index}\n{example.prompt}"

    def id_set_for(self, example: StreamExample, index: int) -> int:
        if self.judge_mode == "cluster" and example.cluster is not None:
            return int(self.cluster_to_int[str(example.cluster)])
        return int(index)

    def set_last_prompt(self, prompt: str) -> None:
        self.last_prompt = prompt
        self.last_example = self.prompt_to_example.get(prompt)

    def attach_metadata(self, embedding_id: int, metadata: Any) -> None:
        example = self.last_example
        if example is None:
            return
        if metadata is not None:
            setattr(metadata, "prompt_id", example.id)
            setattr(metadata, "prompt_text", example.prompt)
            setattr(metadata, "gold_response", example.gold_response)
            setattr(metadata, "cluster", example.cluster)
        self.embedding_id_to_example[int(embedding_id)] = example

    def example_for_metadata(self, metadata: Any) -> Optional[StreamExample]:
        if metadata is None:
            return None
        embedding_id = getattr(metadata, "embedding_id", -1)
        if embedding_id is None or int(embedding_id) < 0:
            return None
        return self.embedding_id_to_example.get(int(embedding_id))

    def example_for_embedding_id(self, embedding_id: Optional[int]) -> Optional[StreamExample]:
        if embedding_id is None or int(embedding_id) < 0:
            return None
        return self.embedding_id_to_example.get(int(embedding_id))


class _PrecomputedEmbeddingEngine:
    def __init__(self, registry: _Registry) -> None:
        self.registry = registry

    def get_embedding(self, text: str) -> List[float]:
        self.registry.set_last_prompt(text)
        example = self.registry.prompt_to_example.get(text)
        if example is None:
            raise ValueError("vCache requested embedding for an unknown prompt")
        if example.embedding is None:
            raise ValueError(f"Missing embedding for prompt id={example.id}")
        return np.asarray(example.embedding, dtype=np.float32).reshape(-1).tolist()


class _DatasetInferenceEngine:
    def __init__(self, registry: _Registry) -> None:
        self.registry = registry
        self.calls = 0

    def create(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        del system_prompt
        self.calls += 1
        example = self.registry.prompt_to_example.get(prompt)
        if example is None:
            raise ValueError("vCache requested inference for an unknown prompt")
        return example.gold_response


class _LoggingMetadataStorage:
    def __init__(self, registry: _Registry) -> None:
        self.registry = registry
        self.metadata_storage: Dict[int, Any] = {}

    def add_metadata(self, embedding_id: int, metadata: Optional[Any] = None) -> int:
        self.registry.attach_metadata(int(embedding_id), metadata)
        self.metadata_storage[int(embedding_id)] = metadata
        return int(embedding_id)

    def get_metadata(self, embedding_id: int) -> Any:
        embedding_id = int(embedding_id)
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Embedding metadata for embedding id {embedding_id} not found")
        return self.metadata_storage[embedding_id]

    def update_metadata(self, embedding_id: int, metadata: Optional[Any] = None) -> Any:
        embedding_id = int(embedding_id)
        if embedding_id not in self.metadata_storage:
            raise ValueError(f"Embedding metadata for embedding id {embedding_id} not found")
        self.registry.attach_metadata(embedding_id, metadata)
        self.metadata_storage[embedding_id] = metadata
        return metadata

    def remove_metadata(self, embedding_id: int) -> bool:
        embedding_id = int(embedding_id)
        if embedding_id in self.metadata_storage:
            del self.metadata_storage[embedding_id]
            self.registry.embedding_id_to_example.pop(embedding_id, None)
            return True
        return False

    def flush(self) -> None:
        self.metadata_storage = {}
        self.registry.embedding_id_to_example = {}

    def get_all_embedding_metadata_objects(self) -> List[Any]:
        return list(self.metadata_storage.values())


class _ExactVectorDB:
    def __init__(self, max_capacity: int) -> None:
        self.max_capacity = int(max_capacity)
        self._matrix: Optional[np.ndarray] = None
        self._ids: List[int] = []
        self._id_to_row: Dict[int, int] = {}
        self._next_id = 0
        self.last_knn: List[Tuple[float, int]] = []

    @staticmethod
    def _norm(vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            return arr
        return (arr / norm).astype(np.float32, copy=False)

    def add(self, embedding: List[float]) -> int:
        if len(self._ids) >= self.max_capacity:
            raise RuntimeError(
                "ExactVectorDB reached max_capacity before eviction ran; "
                "increase --cache_size or use a lower eviction watermark."
            )
        vector = self._norm(embedding)
        if self._matrix is None:
            self._matrix = np.empty((self.max_capacity, int(vector.size)), dtype=np.float32)
        elif self._matrix.shape[1] != int(vector.size):
            raise ValueError(f"Embedding dimension mismatch in vCache vector DB: {self._matrix.shape[1]} vs {vector.size}")
        embedding_id = self._next_id
        self._next_id += 1
        row = len(self._ids)
        self._ids.append(embedding_id)
        self._id_to_row[embedding_id] = row
        self._matrix[row] = vector
        return embedding_id

    def remove(self, embedding_id: int) -> int:
        embedding_id = int(embedding_id)
        row = self._id_to_row.pop(embedding_id, None)
        if row is None:
            return embedding_id
        last_row = len(self._ids) - 1
        last_id = self._ids[last_row]
        if row != last_row and self._matrix is not None:
            self._matrix[row] = self._matrix[last_row]
            self._ids[row] = last_id
            self._id_to_row[last_id] = row
        self._ids.pop()
        return int(embedding_id)

    def get_knn(self, embedding: List[float], k: int) -> List[Tuple[float, int]]:
        if not self._ids or self._matrix is None:
            self.last_knn = []
            return []
        q = self._norm(embedding)
        mat = self._matrix[: len(self._ids)]
        sims = mat @ q
        order = np.argsort(-sims)[: int(k)]
        self.last_knn = [(float(sims[int(i)]), int(self._ids[int(i)])) for i in order]
        return list(self.last_knn)

    def reset(self) -> None:
        self._matrix = None
        self._ids = []
        self._id_to_row = {}
        self._next_id = 0
        self.last_knn = []

    def _init_vector_store(self, embedding_dim: int) -> None:
        del embedding_dim
        self.reset()

    def is_empty(self) -> bool:
        return len(self._ids) == 0

    def size(self) -> int:
        return len(self._ids)


class _JudgeSimilarityEvaluator:
    def __init__(self, judge: EquivalenceJudge) -> None:
        self.judge = judge
        self.calls = 0

    def answers_similar(self, a: str, b: str, id_set_a: int, id_set_b: int) -> bool:
        before = self.judge.calls
        out = self.judge.equivalent_responses(a, b, id_set_a=id_set_a, id_set_b=id_set_b)
        self.calls += self.judge.calls - before
        return bool(out)


class _SynchronousFuture:
    def __init__(self, result: Any = None, exception: Optional[BaseException] = None) -> None:
        self._result = result
        self._exception = exception

    def result(self, timeout: Optional[float] = None) -> Any:
        del timeout
        if self._exception is not None:
            raise self._exception
        return self._result


class _SynchronousExecutor:
    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> _SynchronousFuture:
        try:
            return _SynchronousFuture(result=fn(*args, **kwargs))
        except BaseException as exc:
            return _SynchronousFuture(exception=exc)

    def shutdown(self, wait: bool = True) -> None:
        del wait


class _SynchronousCallbackQueue:
    def __init__(self, callback_function: Any) -> None:
        self.callback_function = callback_function

    def put(self, item: Any) -> None:
        if item is not None:
            self.callback_function(item)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _load_official_vcache(repo_path: Optional[str]) -> Dict[str, Any]:
    path: Optional[Path] = None
    if repo_path:
        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            raise VCacheUnavailableError(f"--vcache_repo_path does not exist: {path}")
        sys.path.insert(0, str(path))

    try:
        from vcache.config import VCacheConfig
        from vcache.main import VCache
        from vcache.vcache_core.cache.eviction_policy.strategies.fifo import FIFOEvictionPolicy
        from vcache.vcache_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy
        from vcache.vcache_core.cache.eviction_policy.strategies.mru import MRUEvictionPolicy
        from vcache.vcache_policy.strategies.benchmark_verified_global import (
            BenchmarkVerifiedGlobalDecisionPolicy,
        )
        from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy
    except SyntaxError as exc:
        raise VCacheUnavailableError(
            "Official vCache could not be imported because the active Python cannot parse it. "
            "vCache currently requires Python >=3.10; this environment may be older."
        ) from exc
    except Exception as exc:
        if path is not None:
            try:
                return _load_minimal_official_vcache(path, import_error=exc)
            except SyntaxError as fallback_exc:
                raise VCacheUnavailableError(
                    "Official vCache could not be imported because the active Python cannot parse it. "
                    "vCache currently requires Python >=3.10; this environment may be older."
                ) from fallback_exc
            except Exception as fallback_exc:
                raise VCacheUnavailableError(
                    "Official vCache could not be imported, and the source adapter could not load "
                    "the official core/policy modules. Direct import error: "
                    f"{type(exc).__name__}: {exc}. Source-adapter error: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc
        raise VCacheUnavailableError(
            "Official vCache could not be imported. Install the official repo and dependencies, "
            "or pass --vcache_repo_path. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return {
        "VCacheConfig": VCacheConfig,
        "VCache": VCache,
        "VerifiedDecisionPolicy": VerifiedDecisionPolicy,
        "BenchmarkVerifiedGlobalDecisionPolicy": BenchmarkVerifiedGlobalDecisionPolicy,
        "FIFOEvictionPolicy": FIFOEvictionPolicy,
        "LRUEvictionPolicy": LRUEvictionPolicy,
        "MRUEvictionPolicy": MRUEvictionPolicy,
    }


def _ensure_package(name: str, path: Optional[Path] = None) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)] if path is not None else []  # type: ignore[attr-defined]
        sys.modules[name] = module
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent = _ensure_package(parent_name)
        setattr(parent, attr, module)
    return module


def _load_source_module(name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent = _ensure_package(parent_name)
        setattr(parent, attr, module)
    spec.loader.exec_module(module)
    return module


def _set_pkg_attr(package: str, name: str, value: Any) -> None:
    setattr(_ensure_package(package), name, value)


def _make_minimal_vcache_config_module() -> ModuleType:
    module = ModuleType("vcache.config")

    class VCacheConfig:
        def __init__(
            self,
            inference_engine: Any = None,
            embedding_engine: Any = None,
            vector_db: Any = None,
            embedding_metadata_storage: Any = None,
            eviction_policy: Any = None,
            similarity_evaluator: Any = None,
            system_prompt: Optional[str] = None,
        ) -> None:
            self.inference_engine = inference_engine
            self.embedding_engine = embedding_engine
            self.vector_db = vector_db
            self.eviction_policy = eviction_policy
            self.embedding_metadata_storage = embedding_metadata_storage
            self.similarity_evaluator = similarity_evaluator
            self.system_prompt = system_prompt

    module.VCacheConfig = VCacheConfig  # type: ignore[attr-defined]
    sys.modules["vcache.config"] = module
    setattr(_ensure_package("vcache"), "config", module)
    return module


def _load_minimal_official_vcache(repo_path: Path, *, import_error: BaseException) -> Dict[str, Any]:
    """Load official vCache core/policy source while bypassing optional top-level backends.

    The official package imports all concrete engines/vector DBs from its package
    initializers. That requires OpenAI/LangChain/vLLM/Chroma/etc. even when the
    experiment injects precomputed embeddings and dataset responses. This loader
    imports only the official VCache, cache, eviction, and verified-policy source
    files needed for the baseline.
    """
    root = repo_path / "vcache"
    if not root.exists():
        raise FileNotFoundError(f"Expected official vCache source at {root}")

    direct_error = f"{type(import_error).__name__}: {import_error}"
    for module_name in list(sys.modules):
        if module_name == "vcache" or module_name.startswith("vcache."):
            sys.modules.pop(module_name, None)

    packages = {
        "vcache": root,
        "vcache.inference_engine": root / "inference_engine",
        "vcache.vcache_core": root / "vcache_core",
        "vcache.vcache_core.cache": root / "vcache_core" / "cache",
        "vcache.vcache_core.cache.embedding_engine": root / "vcache_core" / "cache" / "embedding_engine",
        "vcache.vcache_core.cache.embedding_store": root / "vcache_core" / "cache" / "embedding_store",
        "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage": root
        / "vcache_core"
        / "cache"
        / "embedding_store"
        / "embedding_metadata_storage",
        "vcache.vcache_core.cache.embedding_store.vector_db": root
        / "vcache_core"
        / "cache"
        / "embedding_store"
        / "vector_db",
        "vcache.vcache_core.cache.eviction_policy": root / "vcache_core" / "cache" / "eviction_policy",
        "vcache.vcache_core.cache.eviction_policy.strategies": root
        / "vcache_core"
        / "cache"
        / "eviction_policy"
        / "strategies",
        "vcache.vcache_core.similarity_evaluator": root / "vcache_core" / "similarity_evaluator",
        "vcache.vcache_policy": root / "vcache_policy",
        "vcache.vcache_policy.strategies": root / "vcache_policy" / "strategies",
    }
    for package_name, package_path in packages.items():
        _ensure_package(package_name, package_path)

    inference_mod = _load_source_module(
        "vcache.inference_engine.inference_engine", root / "inference_engine" / "inference_engine.py"
    )
    _set_pkg_attr("vcache.inference_engine", "InferenceEngine", inference_mod.InferenceEngine)

    emb_engine_mod = _load_source_module(
        "vcache.vcache_core.cache.embedding_engine.embedding_engine",
        root / "vcache_core" / "cache" / "embedding_engine" / "embedding_engine.py",
    )
    _set_pkg_attr("vcache.vcache_core.cache.embedding_engine", "EmbeddingEngine", emb_engine_mod.EmbeddingEngine)

    metadata_obj_mod = _load_source_module(
        "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj",
        root
        / "vcache_core"
        / "cache"
        / "embedding_store"
        / "embedding_metadata_storage"
        / "embedding_metadata_obj.py",
    )
    metadata_storage_mod = _load_source_module(
        "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_storage",
        root
        / "vcache_core"
        / "cache"
        / "embedding_store"
        / "embedding_metadata_storage"
        / "embedding_metadata_storage.py",
    )
    _set_pkg_attr(
        "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage",
        "EmbeddingMetadataObj",
        metadata_obj_mod.EmbeddingMetadataObj,
    )
    _set_pkg_attr(
        "vcache.vcache_core.cache.embedding_store.embedding_metadata_storage",
        "EmbeddingMetadataStorage",
        metadata_storage_mod.EmbeddingMetadataStorage,
    )

    vector_db_mod = _load_source_module(
        "vcache.vcache_core.cache.embedding_store.vector_db.vector_db",
        root / "vcache_core" / "cache" / "embedding_store" / "vector_db" / "vector_db.py",
    )
    _set_pkg_attr("vcache.vcache_core.cache.embedding_store.vector_db", "VectorDB", vector_db_mod.VectorDB)
    _set_pkg_attr(
        "vcache.vcache_core.cache.embedding_store.vector_db",
        "SimilarityMetricType",
        vector_db_mod.SimilarityMetricType,
    )

    eviction_mod = _load_source_module(
        "vcache.vcache_core.cache.eviction_policy.eviction_policy",
        root / "vcache_core" / "cache" / "eviction_policy" / "eviction_policy.py",
    )
    _set_pkg_attr("vcache.vcache_core.cache.eviction_policy", "EvictionPolicy", eviction_mod.EvictionPolicy)

    embedding_store_mod = _load_source_module(
        "vcache.vcache_core.cache.embedding_store.embedding_store",
        root / "vcache_core" / "cache" / "embedding_store" / "embedding_store.py",
    )
    _set_pkg_attr("vcache.vcache_core.cache.embedding_store", "EmbeddingStore", embedding_store_mod.EmbeddingStore)

    cache_mod = _load_source_module(
        "vcache.vcache_core.cache.cache",
        root / "vcache_core" / "cache" / "cache.py",
    )
    _set_pkg_attr("vcache.vcache_core.cache", "Cache", cache_mod.Cache)

    similarity_mod = _load_source_module(
        "vcache.vcache_core.similarity_evaluator.similarity_evaluator",
        root / "vcache_core" / "similarity_evaluator" / "similarity_evaluator.py",
    )
    _set_pkg_attr("vcache.vcache_core.similarity_evaluator", "SimilarityEvaluator", similarity_mod.SimilarityEvaluator)

    config_mod = _make_minimal_vcache_config_module()

    policy_mod = _load_source_module(
        "vcache.vcache_policy.vcache_policy",
        root / "vcache_policy" / "vcache_policy.py",
    )
    _set_pkg_attr("vcache.vcache_policy", "VCachePolicy", policy_mod.VCachePolicy)

    fifo_mod = _load_source_module(
        "vcache.vcache_core.cache.eviction_policy.strategies.fifo",
        root / "vcache_core" / "cache" / "eviction_policy" / "strategies" / "fifo.py",
    )
    lru_mod = _load_source_module(
        "vcache.vcache_core.cache.eviction_policy.strategies.lru",
        root / "vcache_core" / "cache" / "eviction_policy" / "strategies" / "lru.py",
    )
    mru_mod = _load_source_module(
        "vcache.vcache_core.cache.eviction_policy.strategies.mru",
        root / "vcache_core" / "cache" / "eviction_policy" / "strategies" / "mru.py",
    )

    verified_mod = _load_source_module(
        "vcache.vcache_policy.strategies.verified",
        root / "vcache_policy" / "strategies" / "verified.py",
    )
    benchmark_mod = _load_source_module(
        "vcache.vcache_policy.strategies.benchmark_verified_global",
        root / "vcache_policy" / "strategies" / "benchmark_verified_global.py",
    )
    _set_pkg_attr(
        "vcache.vcache_policy.strategies",
        "VerifiedDecisionPolicy",
        verified_mod.VerifiedDecisionPolicy,
    )
    _set_pkg_attr(
        "vcache.vcache_policy.strategies",
        "BenchmarkVerifiedGlobalDecisionPolicy",
        benchmark_mod.BenchmarkVerifiedGlobalDecisionPolicy,
    )

    main_mod = _load_source_module("vcache.main", root / "main.py")
    _set_pkg_attr("vcache", "VCache", main_mod.VCache)
    _set_pkg_attr("vcache", "VCacheConfig", config_mod.VCacheConfig)

    classes = {
        "VCacheConfig": config_mod.VCacheConfig,
        "VCache": main_mod.VCache,
        "VerifiedDecisionPolicy": verified_mod.VerifiedDecisionPolicy,
        "BenchmarkVerifiedGlobalDecisionPolicy": benchmark_mod.BenchmarkVerifiedGlobalDecisionPolicy,
        "FIFOEvictionPolicy": fifo_mod.FIFOEvictionPolicy,
        "LRUEvictionPolicy": lru_mod.LRUEvictionPolicy,
        "MRUEvictionPolicy": mru_mod.MRUEvictionPolicy,
    }
    classes["_adapter_note"] = (
        "Loaded official vCache core/policy source through a minimal adapter because direct import failed: "
        f"{direct_error}"
    )
    return classes


def _make_eviction_policy(classes: Dict[str, Any], name: str, cache_size: int) -> Any:
    policy = name.lower()
    kwargs = {"max_size": int(cache_size), "watermark": 1.0, "eviction_percentage": 1.0 / max(1, int(cache_size))}
    if policy == "fifo":
        return classes["FIFOEvictionPolicy"](**kwargs)
    if policy == "lru":
        return classes["LRUEvictionPolicy"](**kwargs)
    if policy == "mru":
        return classes["MRUEvictionPolicy"](**kwargs)
    raise ValueError("vCache adapter supports eviction_policy fifo, lru, or mru")


def _make_policy(classes: Dict[str, Any], policy_name: str, delta: float) -> Any:
    if policy_name == "local":
        return classes["VerifiedDecisionPolicy"](delta=float(delta))
    if policy_name == "global":
        return classes["BenchmarkVerifiedGlobalDecisionPolicy"](delta=float(delta))
    raise ValueError("--vcache_policy must be local or global")


def _synchronize_vcache_policy(policy: Any) -> None:
    callback_queue = getattr(policy, "callback_queue", None)
    if callback_queue is not None:
        callback_function = getattr(callback_queue, "callback_function", None)
        try:
            callback_queue.stop()
        except Exception:
            pass
        if callback_function is not None:
            policy.callback_queue = _SynchronousCallbackQueue(callback_function)
    if hasattr(policy, "executor"):
        policy.executor = _SynchronousExecutor()


@dataclass
class VCacheAdapterConfig:
    delta: float
    repo_path: Optional[str]
    policy: str
    sync_updates: bool
    cache_size: int
    eviction_policy: str


def run_vcache_policy(
    stream: Sequence[StreamExample],
    *,
    config: VCacheAdapterConfig,
    dataset: str,
    seed: int,
    judge: EquivalenceJudge,
    embedding_model: str,
    stream_hash: str,
) -> List[Dict[str, Any]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    classes = _load_official_vcache(config.repo_path)
    registry = _Registry(stream, judge.mode)
    embedding_engine = _PrecomputedEmbeddingEngine(registry)
    inference_engine = _DatasetInferenceEngine(registry)
    metadata_storage = _LoggingMetadataStorage(registry)
    vector_db = _ExactVectorDB(max_capacity=max(1, int(config.cache_size) + 1))
    similarity_evaluator = _JudgeSimilarityEvaluator(judge)
    eviction_policy = _make_eviction_policy(classes, config.eviction_policy, config.cache_size)

    vcache_config = classes["VCacheConfig"](
        inference_engine=inference_engine,
        embedding_engine=embedding_engine,
        vector_db=vector_db,
        embedding_metadata_storage=metadata_storage,
        similarity_evaluator=similarity_evaluator,
        eviction_policy=eviction_policy,
    )
    policy = _make_policy(classes, config.policy, config.delta)
    vcache = classes["VCache"](vcache_config, policy)
    if config.sync_updates:
        _synchronize_vcache_policy(vcache.vcache_policy)

    records: List[Dict[str, Any]] = []
    for i, example in enumerate(stream):
        wrapped_prompt = registry.wrap_prompt(example, i)
        id_set = registry.id_set_for(example, i)
        llm_before = inference_engine.calls
        online_judge_before = similarity_evaluator.calls
        t0 = time.perf_counter()
        is_hit, returned, _response_metadata, nn_metadata = vcache.infer_with_cache_info(
            prompt=wrapped_prompt,
            system_prompt=None,
            id_set=id_set,
        )
        latency = time.perf_counter() - t0
        sim = None
        nearest_embedding_id = None
        if getattr(vector_db, "last_knn", None):
            sim = float(vector_db.last_knn[0][0])
            nearest_embedding_id = int(vector_db.last_knn[0][1])
        nearest_example = registry.example_for_metadata(nn_metadata)
        if nearest_example is None:
            nearest_example = registry.example_for_embedding_id(nearest_embedding_id)
        nearest_entry = None
        if nearest_example is not None:
            nearest_entry = type(
                "NearestEntry",
                (),
                {"example": nearest_example},
            )()
        correctness, would_correct, eval_calls = _evaluate_decision(
            judge,
            example,
            nearest_entry,
            is_hit=bool(is_hit),
            returned_response=returned,
        )
        records.append(
            _base_record(
                dataset=dataset,
                method="vcache",
                seed=seed,
                param=float(config.delta),
                request_index=i,
                example=example,
                nearest=nearest_entry,
                decision="exploit" if is_hit else "explore",
                returned_response=returned,
                correctness=correctness,
                would_correct=would_correct,
                similarity_score=sim,
                method_score=sim,
                latency=latency,
                llm_calls=inference_engine.calls - llm_before,
                online_judge_calls=similarity_evaluator.calls - online_judge_before,
                evaluation_judge_calls=eval_calls,
                cache_size=config.cache_size,
                embedding_model=embedding_model,
                judge_name=judge.name,
                stream_hash=stream_hash,
            )
        )

    shutdown = getattr(vcache.vcache_policy, "shutdown", None)
    if callable(shutdown):
        shutdown()
    evict_shutdown = getattr(eviction_policy, "shutdown", None)
    if callable(evict_shutdown):
        evict_shutdown()
    add_cumulative_fields(records)
    return records
