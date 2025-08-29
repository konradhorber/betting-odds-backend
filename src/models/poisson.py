from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import PoissonRegressor as SkPoissonRegressor
from models.model import Model
import joblib
from pathlib import Path


# =========================
# Poisson goals wrapper
# =========================
class PoissonGoalsRegressor(Model):
    """
    Multi-output wrapper around sklearn's PoissonRegressor to predict λ_home, λ_away.
    """
    def __init__(self, max_iter: int = 2000):
        self.model = MultiOutputRegressor(
            SkPoissonRegressor(max_iter=max_iter)
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        # returns means (lambdas) for [home, away]
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)