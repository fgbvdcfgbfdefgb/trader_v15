"""The three agents (v10 - scalable sizes).

- Predictor: price-forecast agent. Stacked GRU (+LayerNorm) over the day's
  features; at every minute it forecasts the vol-normalized cumulative return
  over the next 5/15/60/240 minutes for each coin, using only information
  available up to that minute.
- Analyzer: market-analysis agent. Stacked GRU encoder producing a
  market-state embedding, with volatility-regime / trend heads as its
  training signal.
- Trader: trade-maker agent. Gaussian policy over per-coin target portfolio
  weights + a value head, trained with PPO. Two of these (A and B) train in
  parallel on separate GPUs and compete; the better one on held-out days wins.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class Predictor(nn.Module):
    def __init__(self, n_features, n_out, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):  # (B,T,F) -> (B,T,n_out)
        h, _ = self.gru(x)
        return self.head(self.norm(h))


class Analyzer(nn.Module):
    def __init__(self, n_features, n_pairs=3, hidden=64, layers=1, emb_dim=16,
                 n_classes=3, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.emb = nn.Sequential(nn.Linear(hidden, emb_dim), nn.Tanh())
        self.vol_head = nn.Linear(emb_dim, n_pairs * n_classes)
        self.trend_head = nn.Linear(emb_dim, n_pairs * n_classes)
        self.n_pairs = n_pairs
        self.n_classes = n_classes

    def forward(self, x):  # -> emb (B,T,E), vol (B,T,P,C), trend (B,T,P,C)
        h, _ = self.gru(x)
        e = self.emb(self.norm(h))
        v = self.vol_head(e).reshape(*e.shape[:-1], self.n_pairs, self.n_classes)
        t = self.trend_head(e).reshape(*e.shape[:-1], self.n_pairs, self.n_classes)
        return e, v, t


class Trader(nn.Module):
    def __init__(self, obs_dim, act_dim=3, hidden=(512, 512)):
        super().__init__()
        hidden = list(hidden)
        layers = []
        in_d = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(in_d, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.25))
        self.vf = nn.Linear(in_d, 1)
        self.act_dim = act_dim

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mu(h)
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(mu)
        value = self.vf(h).squeeze(-1)
        return mu, std, value

    def act(self, obs, deterministic=False):
        mu, std, value = self.forward(obs)
        if deterministic:
            return mu.clamp(-1.0, 1.0), None, value
        dist = Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a.clamp(-1.0, 1.0), logp, value
