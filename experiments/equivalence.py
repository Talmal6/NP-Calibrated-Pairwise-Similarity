from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .dataset_stream import StreamExample


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _load_callable(spec: Optional[str]) -> Optional[Callable[..., bool]]:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("--judge_function must be in module:function form")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"Judge function is not callable: {spec}")
    return func


@dataclass
class EquivalenceJudge:
    mode: str = "exact"
    function_spec: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in {"exact", "cluster", "callable"}:
            raise ValueError("correctness mode must be exact, cluster, or callable")
        self.calls = 0
        self._callable = _load_callable(self.function_spec)
        if self.mode == "callable" and self._callable is None:
            raise ValueError("correctness=callable requires --judge_function")

    @property
    def name(self) -> str:
        if self.mode == "callable":
            return f"callable:{self.function_spec}"
        return self.mode

    def reset(self) -> None:
        self.calls = 0

    def equivalent_examples(
        self,
        current: StreamExample,
        cached: Optional[StreamExample],
        *,
        returned_response: Optional[str] = None,
    ) -> Optional[bool]:
        self.calls += 1
        if cached is None:
            return None

        if self.mode == "cluster":
            if current.cluster is not None and cached.cluster is not None:
                return str(current.cluster) == str(cached.cluster)
            if not current.gold_response or not cached.gold_response:
                return None
            return _norm_text(current.gold_response) == _norm_text(cached.gold_response)

        if self.mode == "exact":
            left = current.gold_response
            right = returned_response if returned_response is not None else cached.gold_response
            if not left or not right:
                return None
            return _norm_text(left) == _norm_text(right)

        assert self._callable is not None
        return bool(
            self._callable(
                current.gold_response,
                returned_response if returned_response is not None else cached.gold_response,
                current.metadata,
                cached.metadata,
            )
        )

    def equivalent_responses(
        self,
        a: str,
        b: str,
        *,
        id_set_a: Any = None,
        id_set_b: Any = None,
    ) -> bool:
        self.calls += 1
        if self.mode == "cluster":
            return id_set_a is not None and id_set_b is not None and id_set_a == id_set_b
        if self.mode == "exact":
            return bool(a and b and _norm_text(a) == _norm_text(b))
        assert self._callable is not None
        return bool(self._callable(a, b, {}, {}))

