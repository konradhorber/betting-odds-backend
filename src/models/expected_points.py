from pathlib import Path
import numpy as np

# --- torch (Expected Points model) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model



# =========================
# Utilities for Expected Points
# =========================
def score_pairs(K: int):
    """List all (home, away) pairs 0..K inclusive."""
    return [(h, a) for h in range(K + 1) for a in range(K + 1)]

def pair_to_index(h: int, a: int, K: int):
    return h * (K + 1) + a

def clamp_goals(y: np.ndarray, K: int) -> np.ndarray:
    """
    Clamp actual goals to [0..K] (anything above K becomes K).
    y: shape (n, 2) with [FTHG, FTAG]
    """
    y = np.asarray(y, dtype=np.int64)
    y = np.clip(y, 0, K)
    return y

def build_reward_matrix(K: int) -> torch.Tensor:
    """
    R[a, s] = points if we *predict* a (scoreline index), and the *actual* is s.
    a,s are indices in [0 .. (K+1)^2 - 1]
    """
    pairs = score_pairs(K)
    C = (K + 1) ** 2
    R = torch.zeros((C, C), dtype=torch.float32)

    def points(pred, actual):
        ph, pa = pred; ah, aa = actual
        if (ph, pa) == (ah, aa):
            return 7
        if ph - pa == ah - aa:
            return 5 if ph != pa else 4
        if (ph > pa and ah > aa) or (ph < pa and ah < aa) or (ph == pa and ah == aa):
            return 3
        return 0

    for a_idx, a in enumerate(pairs):
        for s_idx, s in enumerate(pairs):
            R[a_idx, s_idx] = points(a, s)
    return R  # [C, C]


# =========================
# Expected Points model (PyTorch)
# =========================
class _ExpectedPointsNet(nn.Module):
    def __init__(self, in_dim: int, K: int, hidden: int = 128):
        super().__init__()
        self.K = K
        self.C = (K + 1) ** 2
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.C)
        )

    def forward(self, x):
        logits = self.ff(x)                 # [B, C]
        probs = F.softmax(logits, dim=-1)   # p(actual_score | x)
        return logits, probs


class ExpectedPointsModel(Model):
    """
    Trains a softmax classifier over scorelines (0..K goals per team),
    optimizing a blended loss: alpha * CrossEntropy + (1-alpha) * (-ExpectedPoints).

    During prediction, returns the scoreline that maximizes expected points:
        argmax_a sum_s p(s|x) * R[a,s]
    """
    def __init__(
        self,
        K: int = 5,
        hidden: int = 128,
        alpha: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 256,
        device: str | None = None,
        seed: int = 42,
    ):
        """
        K: max goals per team included in the class grid (0..K)
        alpha: CE-EP blend (0.5 is a good start)
        """
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.alpha = float(alpha)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self.model: _ExpectedPointsNet | None = None
        self.R: torch.Tensor | None = None  # [C, C]
        self._in_dim: int | None = None

        # for saving
        self._state = {}

    # --- internal helpers
    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self._in_dim is None:
            self._in_dim = X.shape[1]
            torch.manual_seed(self.seed)
            self.model = _ExpectedPointsNet(self._in_dim, self.K, self.hidden).to(self.device)
            self.R = build_reward_matrix(self.K).to(self.device)  # [C, C]
        return X

    def _y_to_class_indices(self, y: np.ndarray) -> np.ndarray:
        """
        Map actual [FTHG, FTAG] to class index in [0 .. C-1].
        Any goals > K are clamped to K (tail bucket).
        """
        y = clamp_goals(y, self.K)
        return np.array([pair_to_index(h, a, self.K) for h, a in y], dtype=np.int64)

    # --- public API
    def fit(self, X, y):
        """
        X: (n, d) features
        y: (n, 2) actual goals [FTHG, FTAG]
        """
        X = self._prepare(X)
        y_idx = self._y_to_class_indices(y)

        X_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y_idx).to(self.device)

        model = self.model
        R = self.R
        assert model is not None and R is not None

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        # simple mini-batch loop
        n = X_t.shape[0]
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            X_shuf = X_t[perm]
            y_shuf = y_t[perm]

            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                logits, probs = model(xb)  # logits: [B,C], probs: [B,C]

                # Cross-entropy with true class (actual score)
                loss_ce = F.cross_entropy(logits, yb)

                # Expected points term: EP(p, s*) = sum_a p(a|x)*R[a, s*]
                # R[:, s*] is the reward if actual is s*
                # Gather R columns for the batch true classes -> [C,B]
                R_cols = R[:, yb]              # [C, B]
                EP = (probs.T * R_cols).sum(dim=0)   # [B]
                loss_ep = -EP.mean()

                loss = self.alpha * loss_ce + (1 - self.alpha) * loss_ep

                opt.zero_grad()
                loss.backward()
                opt.step()

        # cache for save
        self._state = dict(
            K=self.K, hidden=self.hidden, alpha=self.alpha, lr=self.lr,
            epochs=self.epochs, batch_size=self.batch_size, seed=self.seed,
            in_dim=self._in_dim, device=self.device
        )

    @torch.no_grad()
    def predict(self, X):
        """
        Returns an array of shape (n, 2): predicted (home_goals, away_goals)
        chosen to maximize expected points under p(s|x).
        """
        assert self.model is not None and self.R is not None
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)

        logits, probs = self.model(X_t)      # [B, C]
        # EP(a|x) = sum_s p(s|x) * R[a,s]  -> matrix multiply with R^T
        EP_actions = probs @ self.R.T        # [B, C]
        a_idx = EP_actions.argmax(dim=1).tolist()

        # map class index back to (h, a)
        preds = []
        K = self.K
        for idx in a_idx:
            h = idx // (K + 1)
            a = idx %  (K + 1)
            preds.append((h, a))
        return np.array(preds, dtype=np.int64)

    def save(self, path: Path):
        assert self.model is not None
        payload = {
            "state": self._state,
            "model_state": self.model.state_dict(),
            # store reward matrix size, can rebuild; no need to store R tensor
        }
        torch.save(payload, path)

    def load(self, path: Path):
        payload = torch.load(path, map_location=self.device)
        self._state = payload["state"]
        self.K = self._state["K"]
        self.hidden = self._state["hidden"]
        self.alpha = self._state["alpha"]
        self.lr = self._state["lr"]
        self.epochs = self._state["epochs"]
        self.batch_size = self._state["batch_size"]
        self.seed = self._state["seed"]
        self._in_dim = self._state["in_dim"]
        self.device = self._state["device"]

        torch.manual_seed(self.seed)
        self.model = _ExpectedPointsNet(self._in_dim, self.K, self.hidden).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.R = build_reward_matrix(self.K).to(self.device)
