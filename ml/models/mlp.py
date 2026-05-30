"""One-step MLP surrogate for Lorenz 63."""

import torch
import torch.nn as nn


class Normalizer:
    """Per-component standardisation computed from training data."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean  # (3,)
        self.std  = std   # (3,)

    @classmethod
    def from_data(cls, data: torch.Tensor) -> "Normalizer":
        """data: (N, T, 3) or (N, 3)."""
        flat = data.reshape(-1, data.shape[-1]).float()
        return cls(flat.mean(0), flat.std(0))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["mean"], d["std"])


class L63MLP(nn.Module):
    """
    One-step MLP surrogate for L63.

    Predicts the normalised residual Δx_norm = norm(x_{t+1}) - norm(x_t).
    Calling .step(x, normalizer) returns x_{t+1} in physical coordinates.

    Architecture: Linear → [Tanh → Linear] × (n_layers-1)
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: (..., 3) normalised → delta_norm: (..., 3)."""
        return self.net(x_norm)

    @torch.no_grad()
    def step(self, x: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
        """Single step in physical coordinates. x: (..., 3) → x_next: (..., 3)."""
        x_norm     = normalizer.normalize(x)
        delta_norm = self(x_norm)
        x_next_norm = x_norm + delta_norm
        return normalizer.denormalize(x_next_norm)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, nsteps: int,
                normalizer: Normalizer) -> torch.Tensor:
        """Free rollout from x0. Returns (nsteps+1, 3) trajectory."""
        traj = [x0]
        x = x0
        for _ in range(nsteps):
            x = self.step(x, normalizer)
            traj.append(x)
        return torch.stack(traj)
