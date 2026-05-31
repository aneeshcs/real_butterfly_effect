"""Evaluate the L96 GNN surrogate: Hovmöller, VPT, PSD, FTLE."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.models.gnn import L96GNN
from ml.models.mlp import Normalizer
from ml.utils.l96_physics import integrate

CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints',
                         'l96_gnn_final.pt')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                         'l96_trajectories.npz')
FIG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'figures')
LAMBDA1       = 1.68           # L96 leading Lyapunov exponent (F=8, K=36)
LYAPUNOV_TIME = 1.0 / LAMBDA1  # ≈ 0.60 t.u.


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = L96GNN(hidden_dim=cfg["hidden_dim"],
                   n_mp_layers=cfg["n_mp_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    nd = ckpt["normalizer"]
    normalizer = Normalizer(nd["mean"].to(device), nd["std"].to(device))
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    return model, normalizer


def model_rollout_np(model, normalizer, x0_np, nsteps, device):
    """Free rollout from numpy IC. Returns (nsteps+1, K) numpy array."""
    x = torch.from_numpy(x0_np.astype(np.float32)).to(device)
    traj = model.rollout(x, nsteps, normalizer)
    return traj.cpu().numpy()


# ── 1. Hovmöller diagram ──────────────────────────────────────────────────────

def plot_hovmoller(model, normalizer, test_data, dt, device, fig_dir):
    x0     = test_data[0, 0]
    nsteps = 500

    traj_model = model_rollout_np(model, normalizer, x0, nsteps, device)
    traj_truth = integrate(x0.astype(np.float64), nsteps, dt)

    times = np.arange(nsteps + 1) * dt
    K     = x0.shape[0]
    vmax  = max(np.abs(traj_truth).max(), np.abs(traj_model).max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, traj, title in zip(axes,
                                [traj_truth, traj_model],
                                ["Truth (RK4)", "Model (GNN)"]):
        im = ax.imshow(traj.T, origin="lower", aspect="auto",
                       extent=[times[0], times[-1], 0, K],
                       vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_xlabel("Time (t.u.)")
        ax.set_ylabel("Node index i")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="$x_i$", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "l96_hovmoller.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── 2. Valid prediction time ──────────────────────────────────────────────────

def plot_vpt(model, normalizer, test_data, dt, device, fig_dir, n_ics=60):
    sigma_att = test_data.reshape(-1, test_data.shape[-1]).std(axis=0)  # (K,)
    T         = test_data.shape[1] - 1
    n_ics     = min(n_ics, len(test_data))

    all_E = []
    vpts  = []
    for i in range(n_ics):
        x0    = test_data[i, 0]
        truth = test_data[i].astype(np.float64)
        pred  = model_rollout_np(model, normalizer, x0, T, device)

        E = np.sqrt(np.mean(((pred - truth) / sigma_att)**2, axis=1))
        all_E.append(E)
        above = np.where(E > 0.5)[0]
        vpts.append(above[0] * dt if len(above) > 0 else T * dt)

    all_E    = np.stack(all_E)
    times    = np.arange(T + 1) * dt
    mean_E   = all_E.mean(0)
    std_E    = all_E.std(0)
    mean_vpt = float(np.mean(vpts))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(times, mean_E - std_E, mean_E + std_E,
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.plot(times, mean_E, color="steelblue", label="Mean norm. RMSE")
    ax.axhline(0.5, color="k",       ls="--", lw=0.8, label="VPT threshold (0.5)")
    ax.axvline(mean_vpt,       color="firebrick", ls="-",
               label=f"Mean VPT = {mean_vpt:.2f} t.u.")
    ax.axvline(LYAPUNOV_TIME, color="gray",      ls=":",
               label=f"Lyapunov time = {LYAPUNOV_TIME:.2f} t.u.")
    ax.set_xlabel("Time (t.u.)")
    ax.set_ylabel("Normalised RMSE")
    ax.set_title("L96 GNN — Valid Prediction Time")
    ax.set_xlim(0, min(8 * LYAPUNOV_TIME, times[-1]))
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "l96_vpt.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}  (mean VPT = {mean_vpt:.2f} t.u., "
          f"Lyapunov time = {LYAPUNOV_TIME:.2f} t.u.)")


# ── 3. Power spectral density ─────────────────────────────────────────────────

def plot_psd(model, normalizer, test_data, dt, device, fig_dir):
    x0     = test_data[0, 0]
    nsteps = 50_000
    pred   = model_rollout_np(model, normalizer, x0, nsteps, device)
    truth  = integrate(x0.astype(np.float64), nsteps, dt)
    K      = x0.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Temporal PSD at node 0
    fs = 1.0 / dt
    f_t, p_t = welch(truth[:, 0], fs=fs, nperseg=2048)
    f_m, p_m = welch(pred[:,  0], fs=fs, nperseg=2048)
    axes[0].loglog(f_t, p_t, color="steelblue", lw=1.2, label="Truth")
    axes[0].loglog(f_m, p_m, color="firebrick", lw=1.2, ls="--", label="Model")
    axes[0].set_xlabel("Frequency (1/t.u.)")
    axes[0].set_ylabel("PSD")
    axes[0].set_title("Temporal PSD  ($x_0(t)$)")
    axes[0].legend(fontsize=8)

    # Spatial spectrum (averaged over all snapshots)
    sp_truth = 0.5 * np.abs(np.fft.rfft(truth, axis=1) / np.sqrt(K))**2
    sp_model = 0.5 * np.abs(np.fft.rfft(pred,  axis=1) / np.sqrt(K))**2
    ks = np.arange(sp_truth.shape[1])
    axes[1].semilogy(ks, sp_truth.mean(0), color="steelblue", lw=1.2, label="Truth")
    axes[1].semilogy(ks, sp_model.mean(0), color="firebrick", lw=1.2, ls="--",
                     label="Model")
    axes[1].set_xlabel("Wavenumber k")
    axes[1].set_ylabel("Energy E(k)")
    axes[1].set_title("Spatial Energy Spectrum")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(fig_dir, "l96_psd.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── 4. FTLE recovery ─────────────────────────────────────────────────────────

def plot_ftle(model, normalizer, test_data, dt, device, fig_dir,
              n_pairs=50, eps=0.01):
    rng   = np.random.default_rng(7)
    T     = test_data.shape[1] - 1
    times = np.arange(T + 1) * dt

    all_ftle = []
    for i in range(min(n_pairs, len(test_data))):
        x0  = test_data[i, 0]
        dv  = rng.standard_normal(x0.shape)
        dv /= np.linalg.norm(dv)
        x0p = x0 + eps * dv

        ref  = model_rollout_np(model, normalizer, x0,  T, device)
        pert = model_rollout_np(model, normalizer, x0p, T, device)

        Eerr = np.mean((pert - ref)**2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ftle = np.where(times > 0,
                            np.log(Eerr / eps**2) / (2 * np.maximum(times, 1e-12)),
                            np.nan)
        all_ftle.append(ftle)

    mean_ftle = np.nanmean(np.stack(all_ftle), axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, mean_ftle, color="steelblue", lw=1.2, label="Model FTLE")
    ax.axhline(LAMBDA1, color="firebrick", ls="--",
               label=f"Reference λ₁ = {LAMBDA1}")
    ax.set_xlabel("Time (t.u.)")
    ax.set_ylabel("λ(t)  [FTLE]")
    ax.set_title("L96 GNN — FTLE Recovery")
    ax.set_xlim(0, min(10 * LYAPUNOV_TIME, times[-1]))
    ax.set_ylim(-1, 5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(fig_dir, "l96_ftle.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(FIG_DIR, exist_ok=True)

    model, normalizer = load_model(CKPT_PATH, device)

    npz       = np.load(DATA_PATH)
    test_data = npz["test"].astype(np.float32)   # (60, 1001, 36)
    dt        = float(npz["dt"])

    plot_hovmoller(model, normalizer, test_data, dt, device, FIG_DIR)
    plot_vpt(      model, normalizer, test_data, dt, device, FIG_DIR)
    plot_psd(      model, normalizer, test_data, dt, device, FIG_DIR)
    plot_ftle(     model, normalizer, test_data, dt, device, FIG_DIR)

    print("\nAll evaluation figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
