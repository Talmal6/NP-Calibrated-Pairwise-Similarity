from sklearn.ensemble import HistGradientBoostingClassifier
from .base import OnlineBaseMethod
from .pair_logreg import pair_features

class HistGBDTMethod(OnlineBaseMethod):
    def __init__(self, name="HistGBDT", max_depth=3):
        super().__init__()
        self.name = name
        self.clf = HistGradientBoostingClassifier(max_depth=max_depth)

    def fit(self, A, B, y):
        X = pair_features(A,B)
        self.clf.fit(X, y)
        return self

    def score_pairs(self, A, B):
        X = pair_features(A,B)
        return self.clf.predict_proba(X)[:, 1]