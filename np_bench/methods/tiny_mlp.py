from __future__ import annotations

import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from .base import BaseMethod
from typing import Optional


BEST_TINY_MLP_PARAMS = {
    "tiny_mlp_activation": "relu",
    "tiny_mlp_alpha": 0.004276973521202226,
    "tiny_mlp_batch_size": 64,
    "tiny_mlp_early_stopping": True,
    "tiny_mlp_hidden_dim": 16,
    "tiny_mlp_learning_rate_init": 0.005360555220684653,
    "tiny_mlp_max_iter": 200,
    "tiny_mlp_n_layers": 2,
    "tiny_mlp_validation_fraction": 0.057694220332308754,
}


class TinyMLPMethod(BaseMethod):
    name = "Tiny MLP"
    needs_weights = False
    needs_seed = True

    def __init__(
        self,
        *,
        # Exact Optuna best-trial Tiny MLP params
        tiny_mlp_activation: str = "relu",
        tiny_mlp_alpha: float = 0.004276973521202226,
        tiny_mlp_batch_size: int | str = 64,
        tiny_mlp_early_stopping: bool = True,
        tiny_mlp_hidden_dim: int = 16,
        tiny_mlp_learning_rate_init: float = 0.005360555220684653,
        tiny_mlp_max_iter: int = 200,
        tiny_mlp_n_layers: int = 2,
        tiny_mlp_validation_fraction: float = 0.057694220332308754,

        # Backward-compatible aliases
        hidden_layer_sizes: Optional[tuple[int, ...]] = None,
        activation: Optional[str] = None,
        alpha: Optional[float] = None,
        learning_rate_init: Optional[float] = None,
        max_iter: Optional[int] = None,
        batch_size: Optional[int | str] = None,
        early_stopping: Optional[bool] = None,
        validation_fraction: Optional[float] = None,
    ):
        self.clf = None

        # Old names override Optuna-style defaults only when explicitly provided.
        if activation is not None:
            tiny_mlp_activation = activation
        if alpha is not None:
            tiny_mlp_alpha = alpha
        if learning_rate_init is not None:
            tiny_mlp_learning_rate_init = learning_rate_init
        if max_iter is not None:
            tiny_mlp_max_iter = max_iter
        if batch_size is not None:
            tiny_mlp_batch_size = batch_size
        if early_stopping is not None:
            tiny_mlp_early_stopping = early_stopping
        if validation_fraction is not None:
            tiny_mlp_validation_fraction = validation_fraction

        if hidden_layer_sizes is not None:
            normalized_hidden_layer_sizes = tuple(
                int(max(1, h)) for h in hidden_layer_sizes
            )
            tiny_mlp_hidden_dim = normalized_hidden_layer_sizes[0]
            tiny_mlp_n_layers = len(normalized_hidden_layer_sizes)
        else:
            tiny_mlp_hidden_dim = int(max(1, tiny_mlp_hidden_dim))
            tiny_mlp_n_layers = int(max(1, tiny_mlp_n_layers))
            normalized_hidden_layer_sizes = tuple(
                tiny_mlp_hidden_dim for _ in range(tiny_mlp_n_layers)
            )

        if tiny_mlp_activation not in {"identity", "logistic", "tanh", "relu"}:
            raise ValueError(
                "TinyMLPMethod activation must be one of "
                "{'identity', 'logistic', 'tanh', 'relu'}"
            )

        if not (
            isinstance(tiny_mlp_batch_size, str)
            and tiny_mlp_batch_size == "auto"
        ):
            tiny_mlp_batch_size = int(max(1, int(tiny_mlp_batch_size)))

        self.tiny_mlp_activation = str(tiny_mlp_activation)
        self.tiny_mlp_alpha = float(max(0.0, tiny_mlp_alpha))
        self.tiny_mlp_batch_size = tiny_mlp_batch_size
        self.tiny_mlp_early_stopping = bool(tiny_mlp_early_stopping)
        self.tiny_mlp_hidden_dim = int(max(1, tiny_mlp_hidden_dim))
        self.tiny_mlp_learning_rate_init = float(
            max(1e-8, tiny_mlp_learning_rate_init)
        )
        self.tiny_mlp_max_iter = int(max(1, tiny_mlp_max_iter))
        self.tiny_mlp_n_layers = int(max(1, tiny_mlp_n_layers))
        self.tiny_mlp_validation_fraction = float(
            np.clip(tiny_mlp_validation_fraction, 0.05, 0.4)
        )

        self.hidden_layer_sizes = normalized_hidden_layer_sizes
        self.activation = self.tiny_mlp_activation
        self.alpha = self.tiny_mlp_alpha
        self.learning_rate_init = self.tiny_mlp_learning_rate_init
        self.max_iter = self.tiny_mlp_max_iter
        self.batch_size = self.tiny_mlp_batch_size
        self.early_stopping = self.tiny_mlp_early_stopping
        self.validation_fraction = self.tiny_mlp_validation_fraction

    def fit(
        self,
        H0_train: np.ndarray,
        H1_train: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> "TinyMLPMethod":
        del weights

        X_tr = np.vstack([H0_train, H1_train])
        y_tr = np.hstack([
            np.zeros(len(H0_train), dtype=np.int32),
            np.ones(len(H1_train), dtype=np.int32),
        ])

        self.clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.tiny_mlp_activation,
            solver="adam",
            max_iter=self.tiny_mlp_max_iter,
            alpha=self.tiny_mlp_alpha,
            learning_rate_init=self.tiny_mlp_learning_rate_init,
            batch_size=self.tiny_mlp_batch_size,
            early_stopping=self.tiny_mlp_early_stopping,
            validation_fraction=self.tiny_mlp_validation_fraction,
            random_state=int(seed if seed is not None else 42),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.clf.fit(X_tr, y_tr)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("TinyMLPMethod.score() called before fit().")

        return self.clf.predict_proba(X)[:, 1].astype(np.float32)