from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseMethod


class BGERerankerMethod(BaseMethod):
    """Cross-encoder scoring baseline using a BGE reranker model.

    Expects input matrix with shape (N, 2):
      X[:, 0] -> query text
      X[:, 1] -> anchor/candidate text
    """

    name = "BGE Reranker"
    needs_weights = False
    needs_seed = False
    input_space = "text_pair"

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 32,
        max_length: int = 512,
        normalize_scores: bool = False,
        device: Optional[str] = None,
        backend: str = "auto",
    ) -> None:
        self.model_name = str(model_name)
        self.batch_size = int(max(1, batch_size))
        self.max_length = int(max(8, max_length))
        self.normalize_scores = bool(normalize_scores)
        self.device_arg = device
        self.backend = str(backend)

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._backend_kind: Optional[str] = None
        self._fallback_reason: Optional[str] = None

    @staticmethod
    def _validate_text_pair_matrix(X: np.ndarray, *, where: str) -> np.ndarray:
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                f"BGEReranker expects text-pair matrix (N,2) in {where}, got shape={X.shape}"
            )
        return X

    def _ensure_backend(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "BGERerankerMethod requires 'torch' and 'transformers'. "
                "Install them to enable this method."
            ) from exc

        if self.backend not in {"auto", "cross", "bi"}:
            raise ValueError("backend must be one of {'auto','cross','bi'}")

        if self.device_arg:
            device = self.device_arg
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = None
        backend_kind: Optional[str] = None
        if self.backend in {"auto", "cross"}:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                backend_kind = "cross"
            except Exception:
                if self.backend == "cross":
                    raise

        if model is None and self.backend in {"auto", "bi"}:
            model = AutoModel.from_pretrained(self.model_name)
            backend_kind = "bi"

        if model is None or backend_kind is None:
            raise RuntimeError(f"Could not initialize backend for model={self.model_name}")

        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._backend_kind = backend_kind
        self._fallback_reason = None

    @property
    def using_fallback(self) -> bool:
        return self._fallback_reason is not None

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @staticmethod
    def _tokenize_simple(s: str) -> set[str]:
        return {w for w in s.lower().split() if w}

    def _score_fallback(self, Xp: np.ndarray) -> np.ndarray:
        # Jaccard token overlap as a deterministic no-model fallback.
        out = np.zeros((Xp.shape[0],), dtype=np.float32)
        for i, (q_raw, a_raw) in enumerate(Xp):
            q = "" if q_raw is None else str(q_raw)
            a = "" if a_raw is None else str(a_raw)
            tq = self._tokenize_simple(q)
            ta = self._tokenize_simple(a)
            if not tq and not ta:
                out[i] = 0.0
                continue
            inter = len(tq & ta)
            union = len(tq | ta)
            out[i] = float(inter / max(1, union))
        return out

    def _encode_texts_bi(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        device = self._device

        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            b = texts[i : i + self.batch_size]
            enc = tokenizer(
                b,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                out = model(**enc)
                h = out.last_hidden_state
                m = enc.get("attention_mask")
                if m is None:
                    v = h[:, 0, :]
                else:
                    m = m.unsqueeze(-1)
                    s = (h * m).sum(dim=1)
                    d = m.sum(dim=1).clamp(min=1)
                    v = s / d
                v = torch.nn.functional.normalize(v, p=2, dim=1)
            vecs.append(v.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(vecs, axis=0)

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights=None,
        seed=None,
        alpha: float = 0.05,
    ) -> "BGERerankerMethod":
        del weights, seed, alpha
        self._validate_text_pair_matrix(H0_train, where="fit(H0_train)")
        self._validate_text_pair_matrix(H1_train, where="fit(H1_train)")
        try:
            self._ensure_backend()
        except Exception as exc:
            self._fallback_reason = str(exc)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xp = self._validate_text_pair_matrix(X, where="score")
        if Xp.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        if self._fallback_reason is not None:
            return self._score_fallback(Xp)

        try:
            self._ensure_backend()
        except Exception as exc:
            self._fallback_reason = str(exc)
            return self._score_fallback(Xp)

        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        device = self._device
        backend_kind = self._backend_kind or "cross"

        if backend_kind == "bi":
            q = ["" if v is None else str(v) for v in Xp[:, 0]]
            a = ["" if v is None else str(v) for v in Xp[:, 1]]
            qv = self._encode_texts_bi(q)
            av = self._encode_texts_bi(a)
            s = np.sum(qv * av, axis=1)
            if self.normalize_scores:
                s = 0.5 * (s + 1.0)
            return s.astype(np.float32, copy=False)

        scores: list[np.ndarray] = []
        for i in range(0, Xp.shape[0], self.batch_size):
            b = Xp[i : i + self.batch_size]
            q = ["" if v is None else str(v) for v in b[:, 0]]
            a = ["" if v is None else str(v) for v in b[:, 1]]

            enc = tokenizer(
                q,
                a,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.inference_mode():
                out = model(**enc)
                logits = out.logits
                if logits.ndim == 2:
                    if logits.shape[1] == 1:
                        s = logits[:, 0]
                    else:
                        s = logits[:, -1]
                else:
                    s = logits.reshape(-1)
                if self.normalize_scores:
                    s = torch.sigmoid(s)

            scores.append(s.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(scores, axis=0).astype(np.float32, copy=False)
