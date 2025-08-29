from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import Model


# ------------------------
# Utilities
# ------------------------
def clamp_goals(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    return np.clip(y, 0, K)


def score_pairs(K: int) -> List[Tuple[int, int]]:
    return [(h, a) for h in range(K + 1) for a in range(K + 1)]


def build_reward_matrix(K: int) -> torch.Tensor:
    pairs = score_pairs(K)
    C = (K + 1) ** 2
    R = torch.zeros((C, C), dtype=torch.float32)

    def points(pred, actual):
        ph, pa = pred
        ah, aa = actual
        if (ph, pa) == (ah, aa):
            return 7
        if ph - pa == ah - aa:
            return 5 if ph != pa else 4
        if (
            (ph > pa and ah > aa)
            or (ph < pa and ah < aa)
            or (ph == pa and ah == aa)
        ):
            return 3
        return 0

    for a_idx, a in enumerate(pairs):
        for s_idx, s in enumerate(pairs):
            R[a_idx, s_idx] = points(a, s)
    return R  # [C, C]


def nbinom_logpmf_vector(
    mu: torch.Tensor, r: torch.Tensor, K: int
) -> torch.Tensor:
    """
    mu, r: 0-D or 1-D tensors on any device (require grad is fine).
    Returns log pmf over y=0..K as a tensor on the same device.
    """
    device, dtype = mu.device, mu.dtype
    y = torch.arange(0, K + 1, device=device, dtype=dtype)

    mu = mu.clamp_min(torch.tensor(1e-8, device=device, dtype=dtype))
    r = r.clamp_min(torch.tensor(1e-6, device=device, dtype=dtype))

    logp = (
        torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
        + r * torch.log(r / (r + mu))
        + y * torch.log(mu / (r + mu))
    )
    return logp  # (K+1,)


# ------------------------
# Net
# ------------------------
class _NBEPNet(nn.Module):
    """
    Shared trunk -> heads for:
      - log_mu_h, log_r_h
      - log_mu_a, log_r_a
      - draw_bias (real, applied on diagonal cells as exp(bias))
    """
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_mu_h = nn.Linear(hidden, 1)
        self.head_r_h = nn.Linear(hidden, 1)
        self.head_mu_a = nn.Linear(hidden, 1)
        self.head_r_a = nn.Linear(hidden, 1)
        self.head_draw = nn.Linear(hidden, 1)

        # sensible initializations
        nn.init.constant_(self.head_mu_h.bias, math.log(1.3))
        nn.init.constant_(self.head_mu_a.bias, math.log(1.2))
        # r ~ 2 (mild overdispersion):
        nn.init.constant_(self.head_r_h.bias,  math.log(2.0))
        nn.init.constant_(self.head_r_a.bias,  math.log(2.0))
        # no draw bias initially:
        nn.init.constant_(self.head_draw.bias, 0.0)

    def forward(self, x):
        z = self.trunk(x)
        log_mu_h = self.head_mu_h(z).squeeze(-1)
        log_r_h = self.head_r_h(z).squeeze(-1)
        log_mu_a = self.head_mu_a(z).squeeze(-1)
        log_r_a = self.head_r_a(z).squeeze(-1)
        draw_raw = self.head_draw(z).squeeze(-1)
        return log_mu_h, log_r_h, log_mu_a, log_r_a, draw_raw


# ------------------------
# Model
# ------------------------
class NBExpectedPointsModel(Model):
    """
    Negative-Binomial goals for home and away (independent),
    plus a differentiable draw-bias factor on the diagonal cells.
    Training: maximize joint log-likelihood of observed (FTHG, FTAG).
    Inference: build P(h,a) on 0..K, then pick argmax_a sum_s P(s) * R[a,s].
    """
    def __init__(
        self,
        K: int = 7,
        hidden: int = 128,
        lr: float = 2e-3,
        epochs: int = 40,
        batch_size: int = 512,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        seed: int = 42,
        draw_clamp: float = 3.0,  # limits exp(draw_bias) to ~[e^-3, e^3]
    ):
        self.K = int(K)
        self.hidden = int(hidden)
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = weight_decay
        self.seed = int(seed)
        self.draw_clamp = float(draw_clamp)

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._in_dim: Optional[int] = None
        self.net: Optional[_NBEPNet] = None
        self.R: Optional[torch.Tensor] = None  # [C, C]

        self._state: Dict[str, Any] = {}

    # --- helpers
    def _prepare(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if self._in_dim is None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self._in_dim = X.shape[1]
            self.net = _NBEPNet(self._in_dim, self.hidden).to(self.device)
            self.R = build_reward_matrix(self.K).to(self.device)
        return X

    def _build_joint(self, mu_h, r_h, mu_a, r_a, draw_bias) -> torch.Tensor:
        """
        Accepts scalar Tensors (preferred) or Python floats.
        Returns P(h,a) on 0..K as a torch Tensor on self.device.
        """
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(
                x, device=self.device, dtype=torch.float32
            )

        mu_h = _t(mu_h)
        r_h = _t(r_h)
        mu_a = _t(mu_a)
        r_a = _t(r_a)
        draw_bias = _t(draw_bias)

        K = self.K
        device, dtype = mu_h.device, mu_h.dtype

        logph = nbinom_logpmf_vector(mu_h, r_h, K)   # (K+1,)
        logpa = nbinom_logpmf_vector(mu_a, r_a, K)   # (K+1,)
        ph = torch.exp(logph)                        # (K+1,)
        pa = torch.exp(logpa)                        # (K+1,)
        P = torch.outer(ph, pa)                      # (K+1, K+1)

        # clamp draw bias on diagonal
        bias = draw_bias.clamp(-self.draw_clamp, self.draw_clamp)
        idx = torch.arange(K + 1, device=device)
        P[idx, idx] = P[idx, idx] * torch.exp(bias)

        # renormalize
        P = P / (P.sum() + torch.tensor(1e-12, device=device, dtype=dtype))
        return P

    # --- API
    def fit(self, X, y):
        """
        X: (n, d) float32 features
        y: (n, 2) int goals [FTHG, FTAG]
        """
        X = self._prepare(X)
        y = clamp_goals(y, self.K)

        net = self.net
        R = self.R
        assert net is not None and R is not None

        X_t = torch.from_numpy(X).to(self.device)
        yh_t = torch.from_numpy(y[:, 0].astype(np.int64)).to(self.device)
        ya_t = torch.from_numpy(y[:, 1].astype(np.int64)).to(self.device)

        opt = torch.optim.Adam(
            net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        n = X_t.shape[0]
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            Xs = X_t[perm]
            yh = yh_t[perm]
            ya = ya_t[perm]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                xb = Xs[start:end]
                yhb = yh[start:end]
                yab = ya[start:end]

                log_mu_h, log_r_h, log_mu_a, log_r_a, draw_raw = net(xb)
                mu_h = F.softplus(log_mu_h) + 1e-6   # tensors [B]
                r_h = F.softplus(log_r_h) + 1e-6
                mu_a = F.softplus(log_mu_a) + 1e-6
                r_a = F.softplus(log_r_a) + 1e-6
                draw_bias = draw_raw                 # tensor [B]

                batch_ll = []
                for i in range(xb.shape[0]):
                    P = self._build_joint(
                        mu_h[i], r_h[i], mu_a[i], r_a[i], draw_bias[i]
                    )  # tensors in
                    # index with Long tensors:
                    ll = torch.log(P[yhb[i], yab[i]] + 1e-12)
                    batch_ll.append(ll)
                batch_ll = torch.stack(batch_ll).mean()

                loss = -batch_ll
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def predict(self, X):
        assert self.net is not None and self.R is not None
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)

        log_mu_h, log_r_h, log_mu_a, log_r_a, draw_raw = self.net(X_t)
        mu_h = F.softplus(log_mu_h) + 1e-6   # tensors [B]
        r_h = F.softplus(log_r_h) + 1e-6
        mu_a = F.softplus(log_mu_a) + 1e-6
        r_a = F.softplus(log_r_a) + 1e-6
        draw_bias = draw_raw                 # tensor [B]

        K = self.K
        C = (K + 1) ** 2
        R = self.R  # ensure this was created with .to(self.device)
        preds = []

        for i in range(X_t.shape[0]):
            P = self._build_joint(
                mu_h[i], r_h[i], mu_a[i], r_a[i], draw_bias[i]
            )  # pass tensors
            P_vec = P.reshape(C)                      # (C,)
            EP_actions = torch.mv(R, P_vec)          # (C,)
            a_idx = int(EP_actions.argmax().item())

            h = a_idx // (K + 1)
            a = a_idx % (K + 1)
            preds.append((h, a))

        return np.array(preds, dtype=np.int64)

    def save(self, path: Path):
        assert self.net is not None
        payload = {
            "state": dict(
                K=self.K, hidden=self.hidden, lr=self.lr, epochs=self.epochs,
                batch_size=self.batch_size, weight_decay=self.weight_decay,
                seed=self.seed, in_dim=self._in_dim, device=self.device,
                draw_clamp=self.draw_clamp,
            ),
            "model_state": self.net.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: Path):
        payload = torch.load(
            path,
            map_location=self.device,
            weights_only=False
        )
        st = payload["state"]
        self.K = int(st["K"])
        self.hidden = int(st["hidden"])
        self.lr = float(st["lr"])
        self.epochs = int(st["epochs"])
        self.batch_size = int(st["batch_size"])
        self.weight_decay = float(st["weight_decay"])
        self.seed = int(st["seed"])
        self._in_dim = int(st["in_dim"])
        self.device = st["device"]
        self.draw_clamp = float(st["draw_clamp"])

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = _NBEPNet(self._in_dim, self.hidden).to(self.device)
        self.net.load_state_dict(payload["model_state"])
        self.R = build_reward_matrix(self.K).to(self.device)
