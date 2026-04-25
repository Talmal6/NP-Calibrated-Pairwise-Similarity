"""OCaTS-style online teacher-student baselines.

This module is intentionally isolated from existing threshold methods so the
current default behavior is unchanged unless explicitly enabled from CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    q = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), eps, 1.0)
    q = q / np.sum(q)
    return float(-np.sum(q * np.log(q)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)


class OCaTSCache:
    """Simple dynamic cache with KNN neighborhood queries."""

    def __init__(
        self,
        *,
        k: int = 8,
        d_thresh: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        self.k = int(max(1, k))
        self.d_thresh = float(d_thresh)
        self.eps = float(eps)
        self.X: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.y: np.ndarray = np.empty((0,), dtype=np.int32)
        self._x_sq_norm: np.ndarray = np.empty((0,), dtype=np.float32)
        self._initial_size: int = 0

    def fit(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        X = np.asarray(vectors, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"cache.fit expects 2D vectors, got shape={X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"cache.fit row mismatch: X={X.shape[0]} y={y.shape[0]}"
            )
        self.X = X.copy()
        self.y = y.copy()
        self._x_sq_norm = np.sum(self.X * self.X, axis=1, dtype=np.float32)
        self._initial_size = int(self.y.size)

    def add(self, vector: np.ndarray, label: int) -> None:
        x = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        y = np.asarray([int(label)], dtype=np.int32)

        if self.X.size == 0:
            self.X = x.copy()
            self.y = y.copy()
            self._x_sq_norm = np.sum(self.X * self.X, axis=1, dtype=np.float32)
            if self._initial_size == 0:
                self._initial_size = 0
            return

        if x.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"cache.add dimension mismatch: cache_dim={self.X.shape[1]} x_dim={x.shape[1]}"
            )
        self.X = np.concatenate([self.X, x], axis=0)
        self.y = np.concatenate([self.y, y], axis=0)
        x_sq = np.sum(x * x, axis=1, dtype=np.float32)
        self._x_sq_norm = np.concatenate([self._x_sq_norm, x_sq], axis=0)

    def set_threshold(self, d_thresh: float) -> None:
        self.d_thresh = float(d_thresh)

    def _topk(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.X.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        x = np.asarray(vector, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.X.shape[1]:
            raise ValueError(
                f"cache query dimension mismatch: cache_dim={self.X.shape[1]} x_dim={x.shape[0]}"
            )

        # Compute squared L2 distances via dot-products to avoid materializing
        # a large (N,D) difference matrix for every query.
        x_sq = np.float32(np.dot(x, x))
        d2 = self._x_sq_norm + x_sq - (2.0 * (self.X @ x))
        d2 = np.maximum(d2, 0.0)

        n = d2.shape[0]
        k_eff = int(min(self.k, n))
        if k_eff <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        if k_eff == n:
            idx = np.arange(n, dtype=np.int64)
        else:
            idx = np.argpartition(d2, k_eff - 1)[:k_eff].astype(np.int64)
        ord_idx = idx[np.argsort(d2[idx])]
        d = np.sqrt(np.maximum(d2[ord_idx], 0.0)).astype(np.float64, copy=False)
        return ord_idx, d

    def centroid_distance_from_topk(
        self,
        vector: np.ndarray,
        idx: np.ndarray,
        d: np.ndarray,
    ) -> float:
        if idx.size == 0:
            return float("inf")

        x = np.asarray(vector, dtype=np.float32).reshape(-1)
        nbr = self.X[idx]
        w = 1.0 / np.maximum(np.asarray(d, dtype=np.float64), self.eps)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            return float("inf")

        centroid = np.sum(nbr * w[:, None], axis=0) / w_sum
        return float(np.linalg.norm(x - centroid))

    def centroid_distance(self, vector: np.ndarray) -> float:
        idx, d = self._topk(vector)
        return self.centroid_distance_from_topk(vector, idx, d)

    def is_near_from_topk(self, vector: np.ndarray, idx: np.ndarray, d: np.ndarray) -> bool:
        return bool(self.centroid_distance_from_topk(vector, idx, d) <= self.d_thresh)

    def is_near(self, vector: np.ndarray) -> bool:
        return bool(self.centroid_distance(vector) <= self.d_thresh)

    def get_last_p_added(self, p: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.X.size == 0 or self.y.size == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)

        p_eff = int(max(0, p))
        if p_eff == 0:
            return np.empty((0, self.X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

        lo = int(max(self._initial_size, self.y.size - p_eff))
        if lo >= self.y.size:
            return np.empty((0, self.X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return self.X[lo:].copy(), self.y[lo:].copy()


class WeightedKNNStudent:
    """Weighted KNN student where class probabilities come from weighted votes."""

    def __init__(
        self,
        cache: OCaTSCache,
        *,
        n_classes: int = 2,
        power: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        self.cache = cache
        self.n_classes = int(max(2, n_classes))
        self.power = float(max(1e-6, power))
        self.eps = float(eps)

    def predict_proba(self, vector: np.ndarray) -> np.ndarray:
        idx, d = self.cache._topk(vector)
        return self.predict_proba_from_topk(idx, d)

    def predict_proba_from_topk(self, idx: np.ndarray, d: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.full((self.n_classes,), 1.0 / self.n_classes, dtype=np.float64)

        yk = self.cache.y[idx]
        w = 1.0 / np.power(np.maximum(d, self.eps), self.power)
        scores = np.zeros((self.n_classes,), dtype=np.float64)
        for cls in range(self.n_classes):
            scores[cls] = float(np.sum(w[yk == cls]))

        s = float(np.sum(scores))
        if s <= 0.0:
            return np.full((self.n_classes,), 1.0 / self.n_classes, dtype=np.float64)
        return scores / s


class SimpleMLPStudent:
    """Small one-hidden-layer MLP with dropout trained in NumPy."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(max(2, hidden_dim))
        self.output_dim = 2
        self.dropout = float(np.clip(dropout, 0.0, 0.95))
        self.lr = float(max(1e-6, lr))
        self.epochs = int(max(1, epochs))
        self.batch_size = int(max(1, batch_size))
        self.weight_decay = float(max(0.0, weight_decay))
        self.rng = np.random.default_rng(int(seed))

        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: np.ndarray | None = None

        self._adam_t = 0
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._fallback_class: int = 0

    def _init_params(self) -> None:
        s1 = np.sqrt(2.0 / max(1, self.input_dim))
        s2 = np.sqrt(2.0 / max(1, self.hidden_dim))
        self.W1 = (self.rng.standard_normal((self.input_dim, self.hidden_dim)) * s1).astype(np.float64)
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.W2 = (self.rng.standard_normal((self.hidden_dim, self.output_dim)) * s2).astype(np.float64)
        self.b2 = np.zeros((self.output_dim,), dtype=np.float64)

        self._adam_t = 0
        self._adam_m = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }
        self._adam_v = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }

    def _forward(self, X: np.ndarray, *, training: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.W1 is None or self.b1 is None or self.W2 is None or self.b2 is None:
            raise RuntimeError("MLP parameters are not initialized")

        h_pre = X @ self.W1 + self.b1[None, :]
        h = np.maximum(h_pre, 0.0)

        if training and self.dropout > 0.0:
            keep = 1.0 - self.dropout
            mask = (self.rng.random(h.shape) < keep).astype(np.float64) / max(keep, 1e-12)
            h_do = h * mask
        else:
            mask = np.ones_like(h, dtype=np.float64)
            h_do = h

        logits = h_do @ self.W2 + self.b2[None, :]
        probs = _softmax(logits)
        return h_pre, h, mask, probs

    def _adam_update(self, grads: Dict[str, np.ndarray]) -> None:
        if self.W1 is None or self.b1 is None or self.W2 is None or self.b2 is None:
            raise RuntimeError("MLP parameters are not initialized")

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self._adam_t += 1

        for name in ("W1", "b1", "W2", "b2"):
            g = grads[name]
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * g
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (g * g)

            m_hat = self._adam_m[name] / (1.0 - beta1 ** self._adam_t)
            v_hat = self._adam_v[name] / (1.0 - beta2 ** self._adam_t)
            step = self.lr * m_hat / (np.sqrt(v_hat) + eps)

            if name == "W1":
                self.W1 -= step
            elif name == "b1":
                self.b1 -= step
            elif name == "W2":
                self.W2 -= step
            else:
                self.b2 -= step

    def fit(self, X: np.ndarray, y: np.ndarray, *, reset: bool = False) -> None:
        Xf = np.asarray(X, dtype=np.float64)
        yi = np.asarray(y, dtype=np.int32).reshape(-1)
        if Xf.ndim != 2:
            raise ValueError(f"MLP.fit expects 2D X, got shape={Xf.shape}")
        if Xf.shape[0] != yi.shape[0]:
            raise ValueError(f"MLP.fit row mismatch: X={Xf.shape[0]} y={yi.shape[0]}")
        if Xf.shape[1] != self.input_dim:
            raise ValueError(
                f"MLP.fit dimension mismatch: expected={self.input_dim} got={Xf.shape[1]}"
            )

        n = int(yi.size)
        if n == 0:
            return

        binc = np.bincount(np.clip(yi, 0, 1), minlength=2)
        self._fallback_class = int(np.argmax(binc))

        if np.unique(yi).size < 2:
            return

        if reset or self.W1 is None:
            self._init_params()

        for _ in range(self.epochs):
            order = np.arange(n)
            self.rng.shuffle(order)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                bi = order[start:end]
                xb = Xf[bi]
                yb = yi[bi]
                m = xb.shape[0]
                if m <= 0:
                    continue

                h_pre, h, mask, probs = self._forward(xb, training=True)

                dlogits = probs
                dlogits[np.arange(m), yb] -= 1.0
                dlogits /= float(m)

                if self.W2 is None:
                    raise RuntimeError("MLP parameters are not initialized")

                h_do = h * mask
                dW2 = h_do.T @ dlogits + self.weight_decay * self.W2
                db2 = np.sum(dlogits, axis=0)

                dh = dlogits @ self.W2.T
                dh = dh * mask
                dh[h_pre <= 0.0] = 0.0

                if self.W1 is None:
                    raise RuntimeError("MLP parameters are not initialized")
                dW1 = xb.T @ dh + self.weight_decay * self.W1
                db1 = np.sum(dh, axis=0)

                grads = {
                    "W1": dW1,
                    "b1": db1,
                    "W2": dW2,
                    "b2": db2,
                }
                self._adam_update(grads)

    def predict_proba(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"MLP.predict_proba dimension mismatch: expected={self.input_dim} got={x.shape[1]}"
            )

        if self.W1 is None or self.W2 is None or self.b1 is None or self.b2 is None:
            p = np.full((2,), 0.0, dtype=np.float64)
            p[self._fallback_class] = 1.0
            p[1 - self._fallback_class] = 0.0
            return p

        _, _, _, probs = self._forward(x, training=False)
        return probs.reshape(-1)


@dataclass
class OCaTSRunResult:
    method: str
    n_stream: int
    accuracy: float
    calls: int
    call_rate: float
    tp: int
    tn: int
    fp: int
    fn: int
    tpr: float
    fpr: float
    precision: float
    recall: float
    e_thresh: float
    d_thresh: float
    online_retrains: int
    curve_rows: List[Dict[str, Any]]


def _run_stream_once(
    *,
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    seed: int,
    cache_k: int,
    e_thresh: float,
    d_thresh: float,
    knn_weight_power: float,
    mlp_hidden_dim: int,
    mlp_dropout: float,
    mlp_lr: float,
    mlp_epochs: int,
    mlp_batch_size: int,
    mlp_weight_decay: float,
    online_retrain_interval: int,
    online_retrain_last_p: int,
    record_curve: bool,
    progress_every: int = 0,
) -> OCaTSRunResult:
    cache = OCaTSCache(k=cache_k, d_thresh=d_thresh)
    cache.fit(X_train, y_train)

    method_id = str(method)
    if method_id == "ocats_knn":
        student: Any = WeightedKNNStudent(
            cache,
            n_classes=2,
            power=knn_weight_power,
        )
    elif method_id == "ocats_mlp":
        student = SimpleMLPStudent(
            input_dim=int(X_train.shape[1]),
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
            lr=mlp_lr,
            epochs=mlp_epochs,
            batch_size=mlp_batch_size,
            weight_decay=mlp_weight_decay,
            seed=seed,
        )
        student.fit(X_train, y_train, reset=True)
    else:
        raise ValueError(f"unknown OCaTS method: {method_id}")

    n_stream = int(y_stream.shape[0])
    calls = 0
    correct = 0
    tp = tn = fp = fn = 0
    since_retrain_calls = 0
    online_retrains = 0
    curve_rows: List[Dict[str, Any]] = []

    for i in range(n_stream):
        x = X_stream[i]
        y_true = int(y_stream[i])

        if method_id == "ocats_knn":
            idx, d = cache._topk(x)
            p = np.asarray(student.predict_proba_from_topk(idx, d), dtype=np.float64).reshape(-1)
            near = cache.is_near_from_topk(x, idx, d)
        else:
            p = np.asarray(student.predict_proba(x), dtype=np.float64).reshape(-1)
            near = cache.is_near(x)

        if p.size != 2:
            if p.size == 1:
                p = np.array([1.0 - float(p[0]), float(p[0])], dtype=np.float64)
            else:
                p = np.array([0.5, 0.5], dtype=np.float64)
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p)

        ent = _entropy_from_probs(p)

        if ent <= float(e_thresh) and near:
            y_pred = int(np.argmax(p))
            used_teacher = False
        else:
            y_pred = y_true
            used_teacher = True
            calls += 1
            since_retrain_calls += 1
            cache.add(x, y_true)

            if (
                method_id == "ocats_mlp"
                and online_retrain_interval > 0
                and since_retrain_calls >= int(online_retrain_interval)
            ):
                last_p = int(max(1, online_retrain_last_p))
                X_new, y_new = cache.get_last_p_added(last_p)
                if X_new.shape[0] > 1 and np.unique(y_new).size >= 2:
                    student.fit(X_new, y_new, reset=False)
                    online_retrains += 1
                since_retrain_calls = 0

        correct += int(y_pred == y_true)

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

        if record_curve:
            step = i + 1
            curve_rows.append(
                {
                    "step": int(step),
                    "cum_accuracy": float(correct / max(1, step)),
                    "teacher_calls": int(calls),
                    "call_rate": float(calls / max(1, step)),
                    "entropy": float(ent),
                    "is_near": bool(near),
                    "used_teacher": bool(used_teacher),
                    "pred": int(y_pred),
                    "label": int(y_true),
                }
            )

        step = i + 1
        if progress_every > 0 and (step % int(progress_every) == 0 or step == n_stream):
            print(
                f"[OCATS:{method_id}] step={step}/{n_stream} "
                f"acc={float(correct / max(1, step)):.4f} "
                f"call_rate={float(calls / max(1, step)):.4f}"
            )

    accuracy = float(correct / max(1, n_stream))
    call_rate = float(calls / max(1, n_stream))
    tpr = float(tp / max(1, tp + fn))
    fpr = float(fp / max(1, fp + tn))
    precision = float(tp / max(1, tp + fp))
    recall = tpr

    return OCaTSRunResult(
        method=method_id,
        n_stream=n_stream,
        accuracy=accuracy,
        calls=int(calls),
        call_rate=call_rate,
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        recall=recall,
        e_thresh=float(e_thresh),
        d_thresh=float(d_thresh),
        online_retrains=int(online_retrains),
        curve_rows=curve_rows,
    )


def _auto_d_grid(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    cache_k: int,
) -> List[float]:
    cache = OCaTSCache(k=cache_k, d_thresh=0.0)
    cache.fit(X_train, y_train)

    if X_calib.shape[0] == 0:
        return [0.25, 0.5, 0.75, 1.0]

    probe_n = int(min(512, X_calib.shape[0]))
    probe = X_calib[:probe_n]
    vals = np.array([cache.centroid_distance(x) for x in probe], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [0.25, 0.5, 0.75, 1.0]

    qs = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
    out = sorted({float(max(1e-8, q)) for q in qs})
    return out if out else [float(np.median(vals))]


def _tune_thresholds(
    *,
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    seed: int,
    lambdas: Sequence[float],
    e_grid: Sequence[float],
    d_grid: Sequence[float],
    cache_k: int,
    knn_weight_power: float,
    mlp_hidden_dim: int,
    mlp_dropout: float,
    mlp_lr: float,
    mlp_epochs: int,
    mlp_batch_size: int,
    mlp_weight_decay: float,
    online_retrain_interval: int,
    online_retrain_last_p: int,
    alpha: float | None,
    progress_every: int,
) -> Tuple[Dict[float, Dict[str, float]], List[Dict[str, Any]]]:
    def _select_candidate_row(
        rows: Sequence[Dict[str, float]],
        *,
        alpha_bound: float | None,
    ) -> Tuple[Dict[str, float] | None, bool]:
        if not rows:
            return None, False

        use_constraint = alpha_bound is not None and np.isfinite(float(alpha_bound))
        alpha_eps = float(alpha_bound) + 1e-12 if use_constraint else float("inf")

        best_row: Dict[str, float] | None = None
        best_key: Tuple[float, float, float, float] | None = None
        has_feasible = False

        for row in rows:
            discounted = float(row.get("discounted_score", float("-inf")))
            calls_term = float(-int(row.get("calls", 0)))

            if use_constraint:
                fpr = float(row.get("fpr", float("inf")))
                feasible = np.isfinite(fpr) and (fpr <= alpha_eps)
                if feasible:
                    has_feasible = True
                    key = (1.0, discounted, 0.0, calls_term)
                else:
                    key = (0.0, -fpr, discounted, calls_term)
            else:
                key = (1.0, discounted, 0.0, calls_term)

            if best_key is None or key > best_key:
                best_key = key
                best_row = row

        return best_row, has_feasible

    tuning_rows: List[Dict[str, Any]] = []
    best_by_lambda: Dict[float, Dict[str, float]] = {}
    alpha_eff = float(alpha) if alpha is not None else None

    for lam in lambdas:
        candidates: List[Dict[str, float]] = []

        for e in e_grid:
            for d in d_grid:
                run = _run_stream_once(
                    method=method,
                    X_train=X_train,
                    y_train=y_train,
                    X_stream=X_calib,
                    y_stream=y_calib,
                    seed=seed,
                    cache_k=cache_k,
                    e_thresh=float(e),
                    d_thresh=float(d),
                    knn_weight_power=knn_weight_power,
                    mlp_hidden_dim=mlp_hidden_dim,
                    mlp_dropout=mlp_dropout,
                    mlp_lr=mlp_lr,
                    mlp_epochs=mlp_epochs,
                    mlp_batch_size=mlp_batch_size,
                    mlp_weight_decay=mlp_weight_decay,
                    online_retrain_interval=online_retrain_interval,
                    online_retrain_last_p=online_retrain_last_p,
                    record_curve=False,
                    progress_every=0,
                )

                discounted = float(run.accuracy - float(lam) * run.call_rate)
                row = {
                    "method": str(method),
                    "lambda": float(lam),
                    "e_thresh": float(e),
                    "d_thresh": float(d),
                    "accuracy": float(run.accuracy),
                    "calls": int(run.calls),
                    "call_rate": float(run.call_rate),
                    "discounted_score": float(discounted),
                    "fpr": float(run.fpr),
                    "tpr": float(run.tpr),
                }
                candidates.append(dict(row))
                tuning_rows.append(
                    {
                        "method": str(method),
                        "lambda": float(lam),
                        "e_thresh": float(e),
                        "d_thresh": float(d),
                        "accuracy": float(run.accuracy),
                        "calls": int(run.calls),
                        "call_rate": float(run.call_rate),
                        "discounted_score": float(discounted),
                    }
                )

        best_row, has_feasible = _select_candidate_row(candidates, alpha_bound=alpha_eff)

        if best_row is None:
            best_row = {
                "method": str(method),
                "lambda": float(lam),
                "e_thresh": 0.25,
                "d_thresh": 0.5,
                "accuracy": 0.0,
                "calls": 0,
                "call_rate": 0.0,
                "discounted_score": -float("inf"),
                "fpr": float("inf"),
            }
        elif alpha_eff is not None and not has_feasible:
            print(
                f"[WARN][OCATS:{method}] lambda={float(lam):.4g} "
                f"no calibration candidate satisfied FPR<=alpha={float(alpha_eff):.4f}; "
                "using lowest-FPR fallback"
            )

        best_by_lambda[float(lam)] = {
            "e_thresh": float(best_row["e_thresh"]),
            "d_thresh": float(best_row["d_thresh"]),
            "discounted_score": float(best_row["discounted_score"]),
            "accuracy": float(best_row["accuracy"]),
            "calls": float(best_row["calls"]),
        }

    return best_by_lambda, tuning_rows


def run_ocats_baselines_for_split(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int,
    methods: Sequence[str],
    lambdas: Sequence[float],
    tune_ocats: bool,
    cache_k: int,
    e_thresh: float,
    d_thresh: float,
    e_thresh_grid: Sequence[float],
    d_thresh_grid: Sequence[float],
    knn_weight_power: float,
    mlp_hidden_dim: int,
    mlp_dropout: float,
    mlp_lr: float,
    mlp_epochs: int,
    mlp_batch_size: int,
    mlp_weight_decay: float,
    online_retrain_interval: int,
    online_retrain_last_p: int,
    alpha: float | None = None,
    record_curve: bool = False,
    progress_every: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    X_all = np.asarray(X, dtype=np.float32)
    y_all = np.asarray(y, dtype=np.int32).reshape(-1)

    tr = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    ca = np.asarray(calib_idx, dtype=np.int64).reshape(-1)
    ev = np.asarray(eval_idx, dtype=np.int64).reshape(-1)

    if tr.size == 0:
        return {
            "trial_rows": [],
            "tuning_rows": [],
            "curve_rows": [],
        }

    X_train = X_all[tr]
    y_train = y_all[tr]

    X_calib = X_all[ca] if ca.size > 0 else np.empty((0, X_all.shape[1]), dtype=np.float32)
    y_calib = y_all[ca] if ca.size > 0 else np.empty((0,), dtype=np.int32)

    X_eval = X_all[ev]
    y_eval = y_all[ev]

    # Build deterministic mixed streams for online gating decisions.
    rng_stream = np.random.default_rng(int(seed))
    if X_calib.shape[0] > 0:
        ord_cal = rng_stream.permutation(X_calib.shape[0])
        X_calib = X_calib[ord_cal]
        y_calib = y_calib[ord_cal]
    if X_eval.shape[0] > 0:
        ord_eval = rng_stream.permutation(X_eval.shape[0])
        X_eval = X_eval[ord_eval]
        y_eval = y_eval[ord_eval]

    lam_list = [float(l) for l in lambdas] if len(lambdas) > 0 else [0.0]
    e_grid = [float(v) for v in e_thresh_grid] if len(e_thresh_grid) > 0 else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    if len(d_thresh_grid) > 0:
        d_grid = [float(v) for v in d_thresh_grid]
    else:
        d_grid = _auto_d_grid(
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            cache_k=cache_k,
        )

    trial_rows: List[Dict[str, Any]] = []
    tuning_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for method in methods:
        m = str(method)
        if m not in {"ocats_knn", "ocats_mlp"}:
            continue

        alpha_eff = float(alpha) if alpha is not None else None

        if tune_ocats and X_calib.shape[0] > 0:
            best_by_lambda, tuning_rows_m = _tune_thresholds(
                method=m,
                X_train=X_train,
                y_train=y_train,
                X_calib=X_calib,
                y_calib=y_calib,
                seed=seed,
                lambdas=lam_list,
                e_grid=e_grid,
                d_grid=d_grid,
                cache_k=cache_k,
                knn_weight_power=knn_weight_power,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_dropout=mlp_dropout,
                mlp_lr=mlp_lr,
                mlp_epochs=mlp_epochs,
                mlp_batch_size=mlp_batch_size,
                mlp_weight_decay=mlp_weight_decay,
                online_retrain_interval=online_retrain_interval,
                online_retrain_last_p=online_retrain_last_p,
                alpha=alpha_eff,
                progress_every=0,
            )
            tuning_rows.extend(tuning_rows_m)
        elif alpha_eff is not None and X_calib.shape[0] > 0:
            # Lightweight alpha guardrail for fixed-threshold mode: adjust
            # confidence/near thresholds on calibration to satisfy FPR when possible.
            e_candidates = sorted(
                {
                    float(e_thresh),
                    0.30,
                    0.20,
                    0.10,
                    0.05,
                    0.02,
                    0.00,
                }
            )
            d_base = float(max(0.0, d_thresh))
            d_candidates = sorted({d_base, 0.0})

            calib_cache: Dict[Tuple[float, float], OCaTSRunResult] = {}
            candidates_by_lambda: Dict[float, List[Dict[str, float]]] = {
                float(lam): [] for lam in lam_list
            }

            for e_c in e_candidates:
                for d_c in d_candidates:
                    k_cal = (float(e_c), float(d_c))
                    run_cal = calib_cache.get(k_cal)
                    if run_cal is None:
                        run_cal = _run_stream_once(
                            method=m,
                            X_train=X_train,
                            y_train=y_train,
                            X_stream=X_calib,
                            y_stream=y_calib,
                            seed=seed,
                            cache_k=cache_k,
                            e_thresh=float(e_c),
                            d_thresh=float(d_c),
                            knn_weight_power=knn_weight_power,
                            mlp_hidden_dim=mlp_hidden_dim,
                            mlp_dropout=mlp_dropout,
                            mlp_lr=mlp_lr,
                            mlp_epochs=mlp_epochs,
                            mlp_batch_size=mlp_batch_size,
                            mlp_weight_decay=mlp_weight_decay,
                            online_retrain_interval=online_retrain_interval,
                            online_retrain_last_p=online_retrain_last_p,
                            record_curve=False,
                            progress_every=0,
                        )
                        calib_cache[k_cal] = run_cal

                    for lam in lam_list:
                        discounted = float(run_cal.accuracy - float(lam) * run_cal.call_rate)
                        candidates_by_lambda[float(lam)].append(
                            {
                                "method": str(m),
                                "lambda": float(lam),
                                "e_thresh": float(e_c),
                                "d_thresh": float(d_c),
                                "accuracy": float(run_cal.accuracy),
                                "calls": int(run_cal.calls),
                                "call_rate": float(run_cal.call_rate),
                                "discounted_score": float(discounted),
                                "fpr": float(run_cal.fpr),
                                "tpr": float(run_cal.tpr),
                            }
                        )

            best_by_lambda = {}
            alpha_eps = float(alpha_eff) + 1e-12
            for lam in lam_list:
                rows_l = candidates_by_lambda[float(lam)]
                feasible = [r for r in rows_l if np.isfinite(float(r.get("fpr", float("inf")))) and float(r.get("fpr", float("inf")) ) <= alpha_eps]
                if feasible:
                    best_row = max(
                        feasible,
                        key=lambda r: (float(r.get("discounted_score", float("-inf"))), -int(r.get("calls", 0))),
                    )
                elif rows_l:
                    best_row = max(
                        rows_l,
                        key=lambda r: (-float(r.get("fpr", float("inf"))), float(r.get("discounted_score", float("-inf"))), -int(r.get("calls", 0))),
                    )
                    print(
                        f"[WARN][OCATS:{m}] lambda={float(lam):.4g} "
                        f"fixed-threshold guardrail could not satisfy FPR<=alpha={float(alpha_eff):.4f}; "
                        "using lowest-FPR fallback"
                    )
                else:
                    best_row = {
                        "e_thresh": float(e_thresh),
                        "d_thresh": float(d_thresh),
                        "discounted_score": -float("inf"),
                        "accuracy": 0.0,
                        "calls": 0,
                    }

                best_by_lambda[float(lam)] = {
                    "e_thresh": float(best_row["e_thresh"]),
                    "d_thresh": float(best_row["d_thresh"]),
                    "discounted_score": float(best_row.get("discounted_score", float("nan"))),
                    "accuracy": float(best_row.get("accuracy", float("nan"))),
                    "calls": float(best_row.get("calls", float("nan"))),
                }
        else:
            best_by_lambda = {
                float(lam): {
                    "e_thresh": float(e_thresh),
                    "d_thresh": float(d_thresh),
                    "discounted_score": float("nan"),
                    "accuracy": float("nan"),
                    "calls": float("nan"),
                }
                for lam in lam_list
            }

        # Evaluate each unique (e_thresh, d_thresh) pair only once and reuse
        # results across lambdas. This avoids redundant full stream passes when
        # thresholds are shared (common case when tune_ocats is disabled).
        run_cache: Dict[Tuple[float, float], OCaTSRunResult] = {}

        for lam in lam_list:
            best = best_by_lambda[float(lam)]
            e_star = float(best["e_thresh"])
            d_star = float(best["d_thresh"])

            k_run = (e_star, d_star)
            run = run_cache.get(k_run)
            if run is None:
                run = _run_stream_once(
                    method=m,
                    X_train=X_train,
                    y_train=y_train,
                    X_stream=X_eval,
                    y_stream=y_eval,
                    seed=seed,
                    cache_k=cache_k,
                    e_thresh=e_star,
                    d_thresh=d_star,
                    knn_weight_power=knn_weight_power,
                    mlp_hidden_dim=mlp_hidden_dim,
                    mlp_dropout=mlp_dropout,
                    mlp_lr=mlp_lr,
                    mlp_epochs=mlp_epochs,
                    mlp_batch_size=mlp_batch_size,
                    mlp_weight_decay=mlp_weight_decay,
                    online_retrain_interval=online_retrain_interval,
                    online_retrain_last_p=online_retrain_last_p,
                    record_curve=record_curve,
                    progress_every=progress_every,
                )
                run_cache[k_run] = run

            discounted = float(run.accuracy - float(lam) * run.call_rate)
            trial_rows.append(
                {
                    "method": str(m),
                    "lambda": float(lam),
                    "n_stream": int(run.n_stream),
                    "accuracy": float(run.accuracy),
                    "calls": int(run.calls),
                    "call_rate": float(run.call_rate),
                    "discounted_score": float(discounted),
                    "tpr": float(run.tpr),
                    "fpr": float(run.fpr),
                    "precision": float(run.precision),
                    "recall": float(run.recall),
                    "e_thresh": float(e_star),
                    "d_thresh": float(d_star),
                    "cache_k": int(cache_k),
                    "online_retrains": int(run.online_retrains),
                    "tune_ocats": bool(tune_ocats),
                }
            )

            if record_curve and run.curve_rows:
                for r in run.curve_rows:
                    curve_rows.append(
                        {
                            "method": str(m),
                            "lambda": float(lam),
                            "e_thresh": float(e_star),
                            "d_thresh": float(d_star),
                            **r,
                        }
                    )

    return {
        "trial_rows": trial_rows,
        "tuning_rows": tuning_rows,
        "curve_rows": curve_rows,
    }
