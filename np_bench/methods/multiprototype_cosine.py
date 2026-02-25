import numpy as np
from .base import OnlineBaseMethod

def _l2_normalize(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def _kmeans(X, k, n_iter=20, seed=0):
    rng = np.random.default_rng(seed)
    Xn = _l2_normalize(X)
    # init
    C = Xn[rng.choice(len(Xn), size=k, replace=False)]
    for _ in range(n_iter):
        sims = Xn @ C.T              # cosine because normalized
        assign = np.argmax(sims, axis=1)
        newC = []
        for j in range(k):
            pts = Xn[assign == j]
            if len(pts) == 0:
                newC.append(C[j])
            else:
                c = pts.mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-12)
                newC.append(c)
        C = np.stack(newC, axis=0)
    return C  # (k,d)

class MultiPrototypeCosineMethod(OnlineBaseMethod):
    def __init__(self, name="MultiProtoCosine", k=4, seed=0):
        super().__init__()
        self.name = name
        self.k = k
        self.seed = seed
        self.C = None  # (k,d)

    def fit(self, H0_train: np.ndarray, H1_train: np.ndarray, *,
            weights=None, seed=None) -> "MultiPrototypeCosineMethod":
        # Learn prototypes from H1 (positive class)
        self.C = _kmeans(H1_train, k=min(self.k, H1_train.shape[0]),
                         seed=self.seed)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.C is None:
            return np.zeros(X.shape[0], dtype=float)
        X_norm = _l2_normalize(X)
        C_norm = _l2_normalize(self.C)
        sims = X_norm @ C_norm.T  # (n, k)
        return sims.max(axis=1)

    def score_pairs(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # fallback to max cosine between normalized pair midpoints and prototypes.
        M = _l2_normalize((A + B) * 0.5)
        sims = M @ self.C.T
        return np.max(sims, axis=1)