from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


FAISS_SUFFIX = " [faiss]"
FAISS_SCORE_TOL = 1e-4


@dataclass(frozen=True)
class PairContext:
    query: Optional[np.ndarray]
    anchor: Optional[np.ndarray]
    anchor_ids: Optional[np.ndarray] = None
    features_are_hadamard: bool = False
    reason: str = ""
    source: str = ""

    @property
    def available(self) -> bool:
        return self.query is not None and self.anchor is not None

    @property
    def n_rows(self) -> int:
        if self.query is None:
            return 0
        return int(self.query.shape[0])


@dataclass(frozen=True)
class PairScoreForm:
    kind: str
    weights: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    bias: float = 0.0
    reason: str = ""


@dataclass
class FaissRegistrationResult:
    name: str
    faiss_name: str
    eligible: bool
    reason: str


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(denom, eps)).astype(np.float32, copy=False)


def _as_2d_float(name: str, X: Optional[np.ndarray]) -> np.ndarray:
    if X is None:
        raise ValueError(f"{name} is unavailable")
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        raise ValueError(f"{name} must be a 2D float matrix, got shape={arr.shape}")
    return arr


def _parse_linear_form(method: Any) -> tuple[Optional[PairScoreForm], str]:
    linear_form = getattr(method, "linear_form", None)
    if not callable(linear_form):
        return None, "method exposes no linear_form() capability"

    try:
        raw = linear_form()
    except Exception as exc:
        return None, f"linear_form() failed: {exc}"

    if raw is None:
        return None, "linear_form() returned None"
    if not isinstance(raw, tuple) or not raw:
        return None, f"linear_form() returned unsupported value {raw!r}"

    kind = str(raw[0])
    if kind == "pair_cosine":
        return PairScoreForm(kind=kind, bias=0.0), "pairwise normalized cosine"

    if kind == "hadamard_linear":
        if len(raw) < 2:
            return None, "hadamard_linear form missing weights"
        w = np.asarray(raw[1], dtype=np.float64).reshape(-1)
        b = float(raw[2]) if len(raw) >= 3 else 0.0
        return PairScoreForm(kind=kind, weights=w, bias=b), "linear scorer on Hadamard pair features"

    if kind == "bilinear":
        if len(raw) < 2:
            return None, "bilinear form missing matrix"
        M = np.asarray(raw[1], dtype=np.float64)
        if M.ndim != 2:
            return None, f"bilinear matrix must be 2D, got shape={M.shape}"
        b = float(raw[2]) if len(raw) >= 3 else 0.0
        return PairScoreForm(kind=kind, matrix=M, bias=b), "general bilinear scorer"

    return None, f"unsupported linear_form kind={kind!r}"


class FaissBackedScorer:
    """Exact pair scorer backed by faiss.IndexFlatIP.

    The scorer indexes transformed anchor vectors and scores each requested row
    against that row's assigned anchor via IndexFlatIP.compute_distance_subset.
    It deliberately refuses to fall back to scoring merged pair-feature rows.
    """

    input_space = "embedding"
    needs_weights = False
    needs_seed = False

    def __init__(
        self,
        *,
        name: str,
        source_name: str,
        source_method: Any,
        form: PairScoreForm,
        pair_context: PairContext,
        X_main: np.ndarray,
        X_cos: Optional[np.ndarray] = None,
    ) -> None:
        self.name = str(name)
        self.source_name = str(source_name)
        self.source_method = source_method
        self.form = form
        self.pair_context = pair_context
        self.X_main = np.asarray(X_main)
        self.X_cos = None if X_cos is None else np.asarray(X_cos)
        self.diagnostic_max_abs_diff = 0.0
        self.diagnostic_count = 0
        self.diagnostic_rank_agreement = True

        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise ImportError(f"faiss is unavailable: {exc}") from exc
        self._faiss = faiss

        q = _as_2d_float("pair_context.query", pair_context.query)
        a = _as_2d_float("pair_context.anchor", pair_context.anchor)
        if q.shape != a.shape:
            raise ValueError(f"query/anchor shape mismatch: query={q.shape} anchor={a.shape}")
        if q.shape[0] != self.X_main.shape[0]:
            raise ValueError(
                "pair context row count does not match X_main: "
                f"{q.shape[0]} vs {self.X_main.shape[0]}"
            )

        self._query = q
        self._anchor = a
        self._row_to_anchor_pos, unique_anchors = self._dedupe_anchors(a, pair_context.anchor_ids)
        self._index_anchor = self._transform_anchor(unique_anchors).astype(np.float32, copy=False)
        if self._index_anchor.ndim != 2 or self._index_anchor.shape[1] <= 0:
            raise ValueError(f"transformed anchors must be 2D, got shape={self._index_anchor.shape}")

        self._index = faiss.IndexFlatIP(int(self._index_anchor.shape[1]))
        self._index.add(np.ascontiguousarray(self._index_anchor, dtype=np.float32))
        self._assert_subset_supported()

    @staticmethod
    def _dedupe_anchors(
        anchors: np.ndarray,
        anchor_ids: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(anchors.shape[0])
        if anchor_ids is None:
            return np.arange(n, dtype=np.int64), anchors

        ids = np.asarray(anchor_ids, dtype=object).reshape(-1)
        if ids.shape[0] != n:
            return np.arange(n, dtype=np.int64), anchors

        first_pos: dict[str, int] = {}
        first_row: dict[str, int] = {}
        row_to_pos = np.empty(n, dtype=np.int64)
        unique_rows: list[int] = []
        for i, value in enumerate(ids.tolist()):
            key = str(value)
            pos = first_pos.get(key)
            if pos is None:
                pos = len(unique_rows)
                first_pos[key] = pos
                first_row[key] = i
                unique_rows.append(i)
            else:
                j = int(first_row[key])
                if not np.allclose(anchors[i], anchors[j], rtol=1e-5, atol=1e-6):
                    return np.arange(n, dtype=np.int64), anchors
            row_to_pos[i] = int(pos)
        return row_to_pos, anchors[np.asarray(unique_rows, dtype=np.int64)]

    def _transform_anchor(self, A: np.ndarray) -> np.ndarray:
        if self.form.kind == "pair_cosine":
            return _l2_normalize_rows(A)
        if self.form.kind == "hadamard_linear":
            return np.asarray(A, dtype=np.float64)
        if self.form.kind == "bilinear":
            return np.asarray(A, dtype=np.float64)
        raise ValueError(f"unsupported form kind={self.form.kind!r}")

    def _transform_query(self, Q: np.ndarray) -> np.ndarray:
        if self.form.kind == "pair_cosine":
            return _l2_normalize_rows(Q)
        if self.form.kind == "hadamard_linear":
            assert self.form.weights is not None
            w = np.asarray(self.form.weights, dtype=np.float64).reshape(1, -1)
            if Q.shape[1] != w.shape[1]:
                raise ValueError(
                    f"query dim mismatch for Hadamard weights: query={Q.shape[1]} weights={w.shape[1]}"
                )
            return np.asarray(Q, dtype=np.float64) * w
        if self.form.kind == "bilinear":
            assert self.form.matrix is not None
            M = np.asarray(self.form.matrix, dtype=np.float64)
            if Q.shape[1] != M.shape[0]:
                raise ValueError(
                    f"query dim mismatch for bilinear matrix: query={Q.shape[1]} matrix={M.shape}"
                )
            if self._anchor.shape[1] != M.shape[1]:
                raise ValueError(
                    f"anchor dim mismatch for bilinear matrix: anchor={self._anchor.shape[1]} matrix={M.shape}"
                )
            return np.asarray(Q, dtype=np.float64) @ M
        raise ValueError(f"unsupported form kind={self.form.kind!r}")

    def _assert_subset_supported(self) -> None:
        if not hasattr(self._index, "compute_distance_subset"):
            raise RuntimeError("faiss.IndexFlatIP lacks compute_distance_subset; exact paired scoring unavailable")
        labels = np.zeros((1, 1), dtype=np.int64)
        distances = np.empty((1, 1), dtype=np.float32)
        x = np.zeros((1, self._index.d), dtype=np.float32)
        try:
            self._index.compute_distance_subset(1, x, 1, distances, labels)
        except TypeError:
            try:
                self._index.compute_distance_subset(
                    1,
                    self._faiss.swig_ptr(x),
                    1,
                    self._faiss.swig_ptr(distances),
                    self._faiss.swig_ptr(labels),
                )
            except Exception:
                # Some FAISS builds expose a friendlier wrapper with ndarray args.
                try:
                    self._index.compute_distance_subset(x, labels, distances)
                except Exception as exc:
                    raise RuntimeError(
                        "faiss.IndexFlatIP.compute_distance_subset is present but not callable "
                        f"with ndarray buffers: {exc}"
                    ) from exc
        except Exception as exc:
            raise RuntimeError(f"faiss subset scoring check failed: {exc}") from exc

    def _compute_subset_ip(self, queries: np.ndarray, labels: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
        lab = np.ascontiguousarray(labels.reshape(-1, 1).astype(np.int64, copy=False))
        distances = np.empty((q.shape[0], 1), dtype=np.float32)
        try:
            self._index.compute_distance_subset(int(q.shape[0]), q, 1, distances, lab)
        except TypeError:
            try:
                self._index.compute_distance_subset(
                    int(q.shape[0]),
                    self._faiss.swig_ptr(q),
                    1,
                    self._faiss.swig_ptr(distances),
                    self._faiss.swig_ptr(lab),
                )
            except Exception:
                self._index.compute_distance_subset(q, lab, distances)
        return distances[:, 0].astype(np.float64, copy=False)

    def _direct_scores(self, row_indices: np.ndarray) -> np.ndarray:
        if self.form.kind == "pair_cosine":
            q = _l2_normalize_rows(self._query[row_indices])
            a = _l2_normalize_rows(self._anchor[row_indices])
            return np.sum(q * a, axis=1, dtype=np.float64)

        if self.form.kind in {"hadamard_linear", "bilinear"}:
            return np.asarray(self.source_method.score(self.X_main[row_indices]), dtype=np.float64).reshape(-1)

        return np.asarray(self.source_method.score(self.X_main[row_indices]), dtype=np.float64).reshape(-1)

    def _record_diagnostics(self, row_indices: np.ndarray, faiss_scores: np.ndarray) -> None:
        try:
            direct = self._direct_scores(row_indices)
        except Exception:
            return
        n = min(int(direct.size), int(faiss_scores.size))
        if n == 0:
            return
        d = direct[:n].astype(np.float64, copy=False)
        f = faiss_scores[:n].astype(np.float64, copy=False)
        diff = np.abs(d - f)
        self.diagnostic_max_abs_diff = max(self.diagnostic_max_abs_diff, float(np.max(diff)))
        self.diagnostic_count += n
        if n > 1:
            self.diagnostic_rank_agreement = bool(
                self.diagnostic_rank_agreement
                and np.array_equal(np.argsort(d, kind="mergesort"), np.argsort(f, kind="mergesort"))
            )

    def score_with_indices(self, row_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return np.zeros(0, dtype=np.float32)
        if int(np.min(idx)) < 0 or int(np.max(idx)) >= self._query.shape[0]:
            raise IndexError("row index out of range for FAISS pair context")

        q = self._transform_query(self._query[idx])
        labels = self._row_to_anchor_pos[idx]
        scores = self._compute_subset_ip(q, labels) + float(self.form.bias)
        self._record_diagnostics(idx, scores)
        return scores.astype(np.float32, copy=False)

    def score(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("FaissBackedScorer requires score_with_indices(); row indices were not provided")


def build_faiss_variant(
    *,
    source_name: str,
    source_method: Any,
    pair_context: PairContext,
    X_main: np.ndarray,
    X_cos: Optional[np.ndarray],
) -> tuple[Optional[FaissBackedScorer], FaissRegistrationResult]:
    faiss_name = f"{source_name}{FAISS_SUFFIX}"
    if source_name.endswith(FAISS_SUFFIX):
        return None, FaissRegistrationResult(source_name, faiss_name, False, "already a FAISS variant")
    if not pair_context.available:
        return None, FaissRegistrationResult(source_name, faiss_name, False, pair_context.reason or "pair context unavailable")

    form, form_reason = _parse_linear_form(source_method)
    if form is None:
        return None, FaissRegistrationResult(source_name, faiss_name, False, form_reason)

    if form.kind in {"hadamard_linear", "bilinear"} and not pair_context.features_are_hadamard:
        return None, FaissRegistrationResult(
            source_name,
            faiss_name,
            False,
            "fitted scorer is linear over pair features, but current X_main is not a clean q*anchor Hadamard product",
        )
    if form.kind == "pair_cosine":
        if X_cos is None:
            return None, FaissRegistrationResult(
                source_name,
                faiss_name,
                False,
                "pair_cosine source requires X_cos, but X_cos is unavailable",
            )
        xcos = np.asarray(X_cos, dtype=np.float32)
        if xcos.ndim != 2 or xcos.shape[0] != pair_context.n_rows or xcos.shape[1] != 1:
            return None, FaissRegistrationResult(
                source_name,
                faiss_name,
                False,
                f"pair_cosine source X_cos has incompatible shape {xcos.shape}",
            )
        q = _l2_normalize_rows(_as_2d_float("pair_context.query", pair_context.query))
        a = _l2_normalize_rows(_as_2d_float("pair_context.anchor", pair_context.anchor))
        n = int(q.shape[0])
        if n > 4096:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, size=4096, replace=False))
        else:
            idx = np.arange(n, dtype=np.int64)
        direct_cos = np.sum(q[idx] * a[idx], axis=1)
        if not np.allclose(direct_cos, xcos[idx, 0], rtol=1e-4, atol=1e-4):
            return None, FaissRegistrationResult(
                source_name,
                faiss_name,
                False,
                "X_cos does not match normalized query-anchor cosine for this pair context",
            )

    try:
        scorer = FaissBackedScorer(
            name=faiss_name,
            source_name=source_name,
            source_method=source_method,
            form=form,
            pair_context=pair_context,
            X_main=X_main,
            X_cos=X_cos,
        )
    except Exception as exc:
        return None, FaissRegistrationResult(source_name, faiss_name, False, str(exc))

    return scorer, FaissRegistrationResult(source_name, faiss_name, True, form_reason)


def is_faiss_variant_name(name: str) -> bool:
    return str(name).endswith(FAISS_SUFFIX)


def source_name_for_faiss(name: str) -> str:
    text = str(name)
    return text[: -len(FAISS_SUFFIX)] if text.endswith(FAISS_SUFFIX) else text
