"""GNN surrogate for Lorenz 96 — Encode-Process-Decode on the periodic ring graph."""

import torch
import torch.nn as nn

from ml.models.mlp import Normalizer  # noqa: reused unchanged


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class GNNBlock(nn.Module):
    """One round of message passing on the L96 ring graph.

    For each node i, gathers features from neighbours i−2, i−1, i+1 via
    torch.roll (exploits periodic ring topology without a graph library).
    Edge type is distinguished by a 3-dim one-hot offset embedding.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        H = hidden_dim
        # Shared edge MLP: [h_src (H), h_tgt (H), offset_onehot (3)] → message (H)
        self.edge_mlp = _make_mlp(2 * H + 3, H, H)
        # Node update MLP: [h (H), aggregated_messages (H)] → delta_h (H)
        self.node_mlp = _make_mlp(2 * H, H, H)

        # One-hot offset embeddings — constant, no grad (registered as buffers)
        # offsets: -2 → [1,0,0], -1 → [0,1,0], +1 → [0,0,1]
        self.register_buffer('e_im2', torch.tensor([1., 0., 0.]))
        self.register_buffer('e_im1', torch.tensor([0., 1., 0.]))
        self.register_buffer('e_ip1', torch.tensor([0., 0., 1.]))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, K, H) → h_new: (B, K, H)."""
        B, K, H = h.shape

        h_im2 = torch.roll(h, shifts=+2, dims=1)   # neighbour at i−2
        h_im1 = torch.roll(h, shifts=+1, dims=1)   # neighbour at i−1
        h_ip1 = torch.roll(h, shifts=-1, dims=1)   # neighbour at i+1

        # Broadcast offset one-hots: (3,) → (B, K, 3)
        def expand(e):
            return e.expand(B, K, -1)

        msg_im2 = self.edge_mlp(torch.cat([h_im2, h, expand(self.e_im2)], dim=-1))
        msg_im1 = self.edge_mlp(torch.cat([h_im1, h, expand(self.e_im1)], dim=-1))
        msg_ip1 = self.edge_mlp(torch.cat([h_ip1, h, expand(self.e_ip1)], dim=-1))

        agg   = msg_im2 + msg_im1 + msg_ip1                        # sum aggregation
        delta = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h + delta                                            # residual update


class L96GNN(nn.Module):
    """
    Encode-Process-Decode GNN surrogate for Lorenz 96.

    Predicts the normalised residual Δx_norm for each node.
    Calling .step(x, normalizer) returns x_{t+1} in physical coordinates.

    Input/output shapes use (B, K) for batched operation; (K,) for single states.
    """

    def __init__(self, hidden_dim: int = 128, n_mp_layers: int = 6):
        super().__init__()
        H = hidden_dim
        self.hidden_dim  = hidden_dim
        self.n_mp_layers = n_mp_layers

        self.encoder   = nn.Sequential(nn.Linear(1, H), nn.Tanh())
        self.processor = nn.ModuleList([GNNBlock(H) for _ in range(n_mp_layers)])
        self.decoder   = nn.Linear(H, 1)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: (B, K) or (K,) normalised → delta_norm: same shape."""
        squeeze = x_norm.dim() == 1
        if squeeze:
            x_norm = x_norm.unsqueeze(0)   # (1, K)

        h = self.encoder(x_norm.unsqueeze(-1))   # (B, K, H)
        for block in self.processor:
            h = block(h)
        delta = self.decoder(h).squeeze(-1)       # (B, K)

        return delta.squeeze(0) if squeeze else delta

    @torch.no_grad()
    def step(self, x: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
        """Single step in physical coordinates. x: (..., K) → x_next: (..., K)."""
        x_norm      = normalizer.normalize(x)
        delta_norm  = self(x_norm)
        x_next_norm = x_norm + delta_norm
        return normalizer.denormalize(x_next_norm)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, nsteps: int,
                normalizer: Normalizer) -> torch.Tensor:
        """Free rollout from x0. Returns (nsteps+1, K) trajectory."""
        traj = [x0]
        x = x0
        for _ in range(nsteps):
            x = self.step(x, normalizer)
            traj.append(x)
        return torch.stack(traj)
