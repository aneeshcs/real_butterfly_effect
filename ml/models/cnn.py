"""CNN surrogate for 2D QG on a doubly-periodic N×N grid.

A 2D CNN with circular padding is the natural and efficient analog of the
roll-based 1D GNN used for L96: circular padding enforces periodic boundaries
and Conv2d uses optimised CUDA kernels rather than explicit message passing.

State contract: flat vector (B, N*N).  Internally reshaped to (B, 1, N, N)
for CNN processing, then flattened back.  This keeps gtf_loss unchanged.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.models.mlp import Normalizer


class ResBlock(nn.Module):
    """Conv(3×3, circular) + Tanh + Conv(1×1), with residual skip."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                               padding=1, padding_mode='circular')
        self.act   = nn.Tanh()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.conv2(self.act(self.conv1(h)))


class QG2DCNN(nn.Module):
    """Encode-Process-Decode CNN surrogate for 2D QG.

    Predicts the normalised residual Δq_norm = norm(q_{t+1}) − norm(q_t).

    Architecture:
      Encoder:    Conv2d(1 → H, k=1) → Tanh
      Processor:  n_layers × ResBlock(H)
      Decoder:    Conv2d(H → 1, k=1)
    """

    def __init__(self, N: int = 64, hidden_dim: int = 64, n_layers: int = 8):
        super().__init__()
        self.N = N
        H = hidden_dim
        self.encoder   = nn.Sequential(nn.Conv2d(1, H, kernel_size=1), nn.Tanh())
        self.processor = nn.Sequential(*[ResBlock(H) for _ in range(n_layers)])
        self.decoder   = nn.Conv2d(H, 1, kernel_size=1)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: (B, N*N) or (N*N,) → delta_norm: same shape."""
        batched = x_norm.dim() == 2
        if not batched:
            x_norm = x_norm.unsqueeze(0)
        B  = x_norm.shape[0]
        N  = self.N
        h  = x_norm.view(B, 1, N, N)
        h  = self.encoder(h)
        h  = self.processor(h)
        delta = self.decoder(h).view(B, N * N)
        if not batched:
            delta = delta.squeeze(0)
        return delta

    @torch.no_grad()
    def step(self, x: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
        """Single step in physical coords.  x: (B, N*N) → x_next: (B, N*N)."""
        x_norm      = normalizer.normalize(x)
        delta_norm  = self(x_norm)
        x_next_norm = x_norm + delta_norm
        return normalizer.denormalize(x_next_norm)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, nsteps: int,
                normalizer: Normalizer) -> torch.Tensor:
        """Free rollout from x0 (N*N,). Returns (nsteps+1, N*N)."""
        traj = [x0]
        x    = x0.unsqueeze(0)
        for _ in range(nsteps):
            x = self.step(x, normalizer)
            traj.append(x.squeeze(0))
        return torch.stack(traj)
