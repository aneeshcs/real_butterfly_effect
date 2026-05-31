"""Train the L96 GNN surrogate with Generalized Teacher Forcing (GTF)."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.models.gnn import L96GNN
from ml.models.mlp import Normalizer
from ml.train.train_l63 import gtf_loss, get_K   # dimension-agnostic, reused directly

# ── defaults ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data',
                          'l96_trajectories.npz')
CKPT_PATH  = os.path.join(os.path.dirname(__file__), '..', 'checkpoints',
                          'l96_gnn_best.pt')
HIDDEN_DIM   = 128
N_MP_LAYERS  = 6
BATCH_SIZE   = 256
LR           = 1e-3
N_EPOCHS     = 200
NOISE_STD    = 0.01
GRAD_CLIP    = 1.0
SEED         = 0
# ──────────────────────────────────────────────────────────────────────────────


class L96Dataset(Dataset):
    """Yields (x_start_noisy, x_target) pairs from a trajectory array.

    x_start: (K,)   — state at random time t, with Gaussian noise
    x_target: (K, K) — wait, (steps, K) — true states at t+1, ..., t+K
    """

    def __init__(self, data: np.ndarray, K_steps: int, noise_std: float = NOISE_STD,
                 seed: int = 0):
        # data: (N_traj, T+1, K_vars)
        self.data      = torch.from_numpy(data).float()
        self.K_steps   = K_steps
        self.noise_std = noise_std
        self.rng       = torch.Generator().manual_seed(seed)
        N, T1, _ = data.shape
        self.pairs = [(i, t) for i in range(N) for t in range(T1 - K_steps)]

    def update_K(self, K_steps: int):
        N, T1, _ = self.data.shape
        self.K_steps = K_steps
        self.pairs = [(i, t) for i in range(N) for t in range(T1 - K_steps)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, t    = self.pairs[idx]
        x_start = self.data[i, t].clone()
        if self.noise_std > 0:
            x_start = x_start + torch.randn(x_start.shape, generator=self.rng) * self.noise_std
        x_target = self.data[i, t + 1 : t + 1 + self.K_steps]   # (K_steps, K_vars)
        return x_start, x_target


def main():
    parser = argparse.ArgumentParser(description="Train L96 GNN surrogate")
    parser.add_argument("--data",         default=DATA_PATH)
    parser.add_argument("--ckpt",         default=CKPT_PATH)
    parser.add_argument("--hidden_dim",   type=int,   default=HIDDEN_DIM)
    parser.add_argument("--n_mp_layers",  type=int,   default=N_MP_LAYERS)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--n_epochs",     type=int,   default=N_EPOCHS)
    parser.add_argument("--seed",         type=int,   default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    npz        = np.load(args.data)
    train_data = npz["train"]   # (500, 1001, 36)
    val_data   = npz["val"]     # (60, 1001, 36)

    normalizer = Normalizer.from_data(torch.from_numpy(train_data))
    normalizer.mean = normalizer.mean.to(device)
    normalizer.std  = normalizer.std.to(device)

    model     = L96GNN(hidden_dim=args.hidden_dim,
                       n_mp_layers=args.n_mp_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    best_val_loss = float("inf")
    current_K     = 0

    train_ds = L96Dataset(train_data, K_steps=1, seed=args.seed)
    val_ds   = L96Dataset(val_data,   K_steps=1, seed=args.seed + 1)

    for epoch in range(args.n_epochs):
        K = get_K(epoch)
        if K != current_K:
            train_ds.update_K(K)
            val_ds.update_K(K)
            current_K = K
            best_val_loss = float("inf")   # reset so each K phase saves its own best
            print(f"  [epoch {epoch}] K → {K}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        # ── train ──
        model.train()
        train_loss = 0.0
        for x_start, x_target in train_loader:
            x_start  = x_start.to(device)
            x_target = x_target.to(device)
            optimizer.zero_grad()
            loss = gtf_loss(model, normalizer, x_start, x_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_start, x_target in val_loader:
                x_start  = x_start.to(device)
                x_target = x_target.to(device)
                val_loss += gtf_loss(model, normalizer, x_start, x_target).item()
        val_loss /= len(val_loader)
        scheduler.step()

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "normalizer":  normalizer.state_dict(),
                "config":      {"hidden_dim":  args.hidden_dim,
                                "n_mp_layers": args.n_mp_layers},
                "epoch":       epoch,
                "val_loss":    val_loss,
            }, args.ckpt)
            saved = "  ← saved best"

        print(f"Epoch {epoch+1:3d}/{args.n_epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}{saved}")

    # Save final epoch model regardless of val loss (captures full K=10 training)
    final_ckpt = args.ckpt.replace("_best.pt", "_final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "normalizer":  normalizer.state_dict(),
        "config":      {"hidden_dim":  args.hidden_dim,
                        "n_mp_layers": args.n_mp_layers},
        "epoch":       args.n_epochs - 1,
        "val_loss":    val_loss,
    }, final_ckpt)
    print(f"Saved final checkpoint: {final_ckpt}")
    print(f"\nDone. Best val loss: {best_val_loss:.6f}  checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
