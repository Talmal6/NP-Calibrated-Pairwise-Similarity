from __future__ import annotations
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from .base import BaseMethod
from typing import Optional


class TinyMLPMethod(BaseMethod):
    name = "Tiny MLP"
    needs_weights = False
    needs_seed = True

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (16,),
        activation: str = "relu",
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        max_iter: int = 800,
        batch_size: int | str = "auto",
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
    ):
        self.clf = None
        self.hidden_layer_sizes = tuple(int(max(1, h)) for h in hidden_layer_sizes)
        self.activation = str(activation)
        self.alpha = float(max(0.0, alpha))
        self.learning_rate_init = float(max(1e-8, learning_rate_init))
        self.max_iter = int(max(1, max_iter))
        self.batch_size = batch_size
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(np.clip(validation_fraction, 0.05, 0.4))

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
        y_tr = np.hstack([np.zeros(len(H0_train)), np.ones(len(H1_train))])

        self.clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            max_iter=self.max_iter,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=int(seed if seed is not None else 42),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.clf.fit(X_tr, y_tr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)
