import numpy as np
from sklearn.metrics import roc_auc_score
from .base import OnlineBaseMethod

class AUCWeightedDiffMethod(OnlineBaseMethod):
    def __init__(self, name="AUCWeightedL1", eps=1e-12, drop_frac=0.1):
        super().__init__()
        self.name = name
        self.eps = eps
        self.drop_frac = drop_frac
        self.w = None  # (d,)

    def fit(self, A, B, y):
        D = np.abs(A - B)  # (n,d)
        d = D.shape[1]
        aucs = np.zeros(d, dtype=np.float64)
        for j in range(d):
            # smaller diff => more likely H1, so invert sign for AUC
            aucs[j] = roc_auc_score(y, -D[:, j])
        # convert to weight: how far from random
        w = np.maximum(aucs - 0.5, 0.0)
        # drop bottom fraction (optional)
        if self.drop_frac > 0:
            k = int(np.floor((1.0 - self.drop_frac) * d))
            idx = np.argsort(w)[::-1][:max(1, k)]
            mask = np.zeros(d, dtype=np.float64)
            mask[idx] = 1.0
            w = w * mask
        # normalize weights
        s = w.sum()
        self.w = w / (s + self.eps)
        return self

    def score_pairs(self, A, B):
        D = np.abs(A - B)
        return -(D @ self.w)