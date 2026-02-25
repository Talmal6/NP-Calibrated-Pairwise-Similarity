import numpy as np
from sklearn.isotonic import IsotonicRegression
from .base import OnlineBaseMethod

def _l2(x, eps=1e-12):
    return np.sqrt(np.sum(x*x, axis=1) + eps)

def _cos(A, B, eps=1e-12):
    return np.sum(A*B, axis=1) / (_l2(A,eps)*_l2(B,eps) + eps)

class IsotonicCalibratedCosineMethod(OnlineBaseMethod):
    def __init__(self, name="IsoCosine"):
        super().__init__()
        self.name = name
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, A, B, y):
        s = _cos(A,B)
        self.iso.fit(s, y)
        return self

    def score_pairs(self, A, B):
        s = _cos(A,B)
        return self.iso.predict(s)  # calibrated prob-like score