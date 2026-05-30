"""Evaluate the L63 MLP surrogate: attractor, VPT, PSD, FTLE."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.models.mlp import L63MLP, Normalizer
from ml.utils.l63_physics import integrate

CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints',
                         'l63_mlp_best.pt')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                         'l63_trajectories.npz')
FIG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'figures')
LYAPUNOV_TIME = 1.0 / 0.906   # ≈ 1.10 t.u.
LAMBDA1       = 0.906


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = L63MLP(hidden_dim=cfg["hidden_dim"],
                   n_layers=cfg["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    nd = ckpt["normalizer"]
    normalizer = Normalizer(nd["mean"].to(device), nd["std"].to(device))
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    return model, normalizer


def model_rollout_np(model, normalizer, x0_np, nsteps, device):
    """Free rollout from numpy IC. Returns (nsteps+1, 3) numpy array."""
    x = torch.from_numpy(x0_np.astype(np.float32)).to(device)
    traj = model.rollout(x, nsteps, normalizer)
    return traj.cpu().numpy()


# ── 1. Attractor geometry ─────────────────────────────────────────────────────

def plot_attractor(model, normalizer, test_data, dt, device, fig_dir):
    x0 = test_data[0, 0]  # first test IC
    nsteps = 20_000

    traj_model = model_rollout_np(model, normalizer, x0, nsteps, device)
    traj_truth = integrate(x0.astype(np.float64), nsteps, dt)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.5))
    for col, (xi, zi, label) in enumerate(
            [(0, 2, "x–z"), (0, 1, "x–y")]):
        axes[col*2].set_title(f"Truth ({label})")
        axes[col*2].plot(traj_truth[:, xi], traj_truth[:, zi],
                         lw=0.3, alpha=0.6, color="steelblue")
        axes[col*2].set_xlabel(["x","x"][col])
        axes[col*2].set_ylabel(["z","y"][col])

        axes[col*2+1].set_title(f"Model ({label})")
        axes[col*2+1].plot(traj_model[:, xi], traj_model[:, zi],
                           lw=0.3, alpha=0.6, color="firebrick")
        axes[col*2+1].set_xlabel(["x","x"][col])
        axes[col*2+1].set_ylabel(["z","y"][col])

    plt.tight_layout()
    path = os.path.join(fig_dir, "l63_attractor.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── 2. Valid prediction time ──────────────────────────────────────────────────

def plot_vpt(model, normalizer, test_data, dt, device, fig_dir, n_ics=100):
    # σ_att: per-component std of training data used as scale
    sigma_att = test_data.reshape(-1, 3).std(axis=0)  # (3,)
    T = test_data.shape[1] - 1
    n_ics = min(n_ics, len(test_data))

    all_E = []
    vpts  = []
    for i in range(n_ics):
        x0     = test_data[i, 0]
        truth  = test_data[i].astype(np.float64)           # (T+1, 3)
        pred   = model_rollout_np(model, normalizer, x0, T, device)

        E = np.sqrt(np.mean(((pred - truth) / sigma_att)**2, axis=1))  # (T+1,)
        all_E.append(E)
        above = np.where(E > 0.5)[0]
        vpts.append(above[0] * dt if len(above) > 0 else T * dt)

    all_E  = np.stack(all_E)   # (n_ics, T+1)
    times  = np.arange(T + 1) * dt
    mean_E = all_E.mean(0)
    std_E  = all_E.std(0)
    mean_vpt = float(np.mean(vpts))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(times, mean_E - std_E, mean_E + std_E, alpha=0.25,
                    color="steelblue", label="±1 std")
    ax.plot(times, mean_E, color="steelblue", label="Mean norm. RMSE")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="VPT threshold (0.5)")
    ax.axvline(mean_vpt, color="firebrick", ls="-",
               label=f"Mean VPT = {mean_vpt:.2f} t.u.")
    ax.axvline(LYAPUNOV_TIME, color="gray", ls=":",
               label=f"Lyapunov time = {LYAPUNOV_TIME:.2f} t.u.")
    ax.set_xlabel("Time (t.u.)")
    ax.set_ylabel("Normalised RMSE")
    ax.set_title("L63 MLP — Valid Prediction Time")
    ax.set_xlim(0, min(5 * LYAPUNOV_TIME, times[-1]))
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "l63_vpt.png")
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

    labels = ["x", "y", "z"]
    colors = ["steelblue", "firebrick"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for j, lbl in enumerate(labels):
        fs = 1.0 / dt
        f_t, p_t = welch(truth[:, j], fs=fs, nperseg=2048)
        f_m, p_m = welch(pred[:,  j], fs=fs, nperseg=2048)
        axes[j].loglog(f_t, p_t, color=colors[0], lw=1.2, label="Truth")
        axes[j].loglog(f_m, p_m, color=colors[1], lw=1.2, ls="--", label="Model")
        axes[j].set_xlabel("Frequency (1/t.u.)")
        axes[j].set_ylabel("PSD")
        axes[j].set_title(f"L63 PSD — {lbl}(t)")
        axes[j].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "l63_psd.png")
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
        x0   = test_data[i, 0]
        dv   = rng.standard_normal(3)
        dv  /= np.linalg.norm(dv)
        x0p  = x0 + eps * dv

        ref  = model_rollout_np(model, normalizer, x0,  T, device)
        pert = model_rollout_np(model, normalizer, x0p, T, device)

        Eerr = np.sum((pert - ref)**2, axis=1)  # (T+1,)
        # λ(t) = (1/2t) ln(Eerr / eps²); skip t=0
        with np.errstate(divide="ignore", invalid="ignore"):
            ftle = np.where(times > 0,
                            np.log(Eerr / eps**2) / (2 * np.maximum(times, 1e-12)),
                            np.nan)
        all_ftle.append(ftle)

    all_ftle = np.stack(all_ftle)   # (n_pairs, T+1)
    mean_ftle = np.nanmean(all_ftle, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, mean_ftle, color="steelblue", lw=1.2, label="Model FTLE")
    ax.axhline(LAMBDA1, color="firebrick", ls="--",
               label=f"Reference λ₁ = {LAMBDA1}")
    ax.set_xlabel("Time (t.u.)")
    ax.set_ylabel("λ(t)  [FTLE]")
    ax.set_title("L63 MLP — FTLE Recovery")
    ax.set_xlim(0, min(10 * LYAPUNOV_TIME, times[-1]))
    ax.set_ylim(-1, 3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(fig_dir, "l63_ftle.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(FIG_DIR, exist_ok=True)

    model, normalizer = load_model(CKPT_PATH, device)

    npz       = np.load(DATA_PATH)
    test_data = npz["test"].astype(np.float32)   # (80, 2001, 3)
    dt        = float(npz["dt"])

    plot_attractor(model, normalizer, test_data, dt, device, FIG_DIR)
    plot_vpt(     model, normalizer, test_data, dt, device, FIG_DIR)
    plot_psd(     model, normalizer, test_data, dt, device, FIG_DIR)
    plot_ftle(    model, normalizer, test_data, dt, device, FIG_DIR)

    print("\nAll evaluation figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
