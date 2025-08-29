from abc import ABC, abstractmethod
from pathlib import Path


# =========================
# Base interface
# =========================
class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def save(self, path: Path):
        ...

    @abstractmethod
    def load(self, path: Path):
        ...