import numpy as np
from sklearn.linear_model import LogisticRegression
from .base import OnlineBaseMethod

def _l2(x, eps=1e-12):
    return np.sqrt(np.sum(x*x, axis=1) + eps)

def _cos(A, B, eps=1e-12):
    return np.sum(A*B, axis=1) / (_l2(A,eps)*_l2(B,eps) + eps)

def pair_features(A, B):
    d = A - B
    ad = np.abs(d)
    d2 = d * d
    feats = np.stack([
        _cos(A,B),
        np.sum(A*B, axis=1),
        _l2(A), _l2(B),
        _l2(d),
        ad.mean(axis=1),
        d2.mean(axis=1),
    ], axis=1)
    return feats

class PairFeatureLogRegMethod(OnlineBaseMethod):
    def __init__(self, name="PairFeatLogReg", C=1.0):
        super().__init__()
        self.name = name
        self.C = C
        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, A, B, y):
        X = pair_features(A, B)
        self.clf.fit(X, y)
        return self

    def score_pairs(self, A, B):
        X = pair_features(A, B)
        # score as logit / prob
        p = self.clf.predict_proba(X)[:, 1]
        return p