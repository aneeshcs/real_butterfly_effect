"""Evaluate the QG2D CNN surrogate: snapshots, VPT, energy spectrum, FTLE."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.models.cnn import QG2DCNN
from ml.models.mlp import Normalizer
from ml.utils.qg2d_physics import integrate, make_ic

CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints',
                         'qg2d_cnn_best.pt')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                         'qg2d_trajectories.npz')
FIG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'figures')
LAMBDA1       = 0.5           # placeholder; FTLE from truth establishes reference
LYAPUNOV_TIME = 1.0 / LAMBDA1


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = QG2DCNN(N=cfg["N"], hidden_dim=cfg["hidden_dim"],
                    n_layers=cfg["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    nd = ckpt["normalizer"]
    normalizer = Normalizer(nd["mean"].to(device), nd["std"].to(device))
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    return model, normalizer


def model_rollout_np(model, normalizer, q0_flat_np, nsteps, device):
    """Free rollout from flat numpy IC. Returns (nsteps+1, N*N) numpy array."""
    x = torch.from_numpy(q0_flat_np.astype(np.float32)).to(device)
    traj = model.rollout(x, nsteps, normalizer)
    return traj.cpu().numpy()


def _isotropic_spectrum(q, N):
    """Isotropic kinetic energy spectrum E(k) from physical PV field q (N,N)."""
    from numpy.fft import fft2, fftfreq
    kx = fftfreq(N) * N
    ky = fftfreq(N) * N
    KX, KY   = np.meshgrid(kx, ky)
    K2       = KX**2 + KY**2
    K2_safe  = K2.copy(); K2_safe[0, 0] = 1.0

    q_hat    = fft2(q)
    psi_hat  = -q_hat / K2_safe
    psi_hat[0, 0] = 0.0
    E        = 0.5 * K2 * np.abs(psi_hat)**2 / N**4

    k_mag  = np.sqrt(K2)
    k_bins = np.arange(0, N // 2 + 1)
    Ek     = np.zeros(len(k_bins))
    k_int  = np.round(k_mag).astype(int)
    for ki, k in enumerate(k_bins):
        Ek[ki] = E[k_int == k].sum()
    return k_bins, Ek


# ── 1. Snapshot comparison ────────────────────────────────────────────────────

def plot_snapshots(model, normalizer, test_data, dt, device, fig_dir, N):
    q0_flat = test_data[0, 0].flatten()
    snap_steps = [0, int(0.5 / dt), int(1.0 / dt), int(2.0 / dt)]
    nsteps = max(snap_steps)

    pred_flat = model_rollout_np(model, normalizer, q0_flat, nsteps, device)
    truth_flat = test_data[0, :nsteps + 1].reshape(nsteps + 1, N * N)

    pred  = pred_flat.reshape(-1, N, N)
    truth = truth_flat.reshape(-1, N, N)

    titles = [f"t = {s * dt:.1f} t.u." for s in snap_steps]
    vmax   = max(np.abs(truth[snap_steps]).max(), np.abs(pred[snap_steps]).max())

    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True)
    for col, (s, title) in enumerate(zip(snap_steps, titles)):
        for row, (field, label) in enumerate([(truth[s], "Truth (RK4)"),
                                               (pred[s],  "Model (CNN)")]):
            im = axes[row, col].imshow(field, origin="lower", aspect="equal",
                                       vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            axes[row, col].set_title(f"{label}\n{title}" if row == 0 else title)
            axes[row, col].axis("off")
    fig.colorbar(im, ax=axes, label="q", shrink=0.6)
    plt.tight_layout()
    path = os.path.join(fig_dir, "qg2d_snapshots.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── 2. Valid prediction time ──────────────────────────────────────────────────

def plot_vpt(model, normalizer, test_data, dt, device, fig_dir, N, n_ics=30):
    n_ics    = min(n_ics, len(test_data))
    T        = test_data.shape[1] - 1
    state_dim = N * N
    sigma_att = test_data.reshape(-1, state_dim).std(axis=0)

    all_E = []
    vpts  = []
    for i in range(n_ics):
        q0_flat = test_data[i, 0].flatten()
        truth   = test_data[i].reshape(T + 1, state_dim).astype(np.float64)
        pred    = model_rollout_np(model, normalizer, q0_flat, T, device)

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
               label=f"Lyapunov time ≈ {LYAPUNOV_TIME:.2f} t.u.")
    ax.set_xlabel("Time (t.u.)")
    ax.set_ylabel("Normalised RMSE")
    ax.set_title("QG2D CNN — Valid Prediction Time")
    ax.set_xlim(0, min(8 * LYAPUNOV_TIME, times[-1]))
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "qg2d_vpt.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}  (mean VPT = {mean_vpt:.2f} t.u., "
          f"Lyapunov time ≈ {LYAPUNOV_TIME:.2f} t.u.)")


# ── 3. Isotropic energy spectrum ──────────────────────────────────────────────

def plot_spectrum(model, normalizer, test_data, dt, device, fig_dir, N):
    q0_flat = test_data[0, 0].flatten()
    nsteps  = 2000
    pred_flat  = model_rollout_np(model, normalizer, q0_flat, nsteps, device)
    truth_flat = integrate(test_data[0, 0].astype(np.float64), nsteps, dt, N=N)
    truth_flat = truth_flat.reshape(nsteps + 1, N * N)

    pred  = pred_flat.reshape(-1, N, N)
    truth = truth_flat.reshape(-1, N, N)

    # Average spectrum over all snapshots
    k_bins_p, Ek_pred  = zip(*[_isotropic_spectrum(pred[i],  N) for i in range(nsteps + 1)])
    k_bins_t, Ek_truth = zip(*[_isotropic_spectrum(truth[i], N) for i in range(nsteps + 1)])
    k_bins = k_bins_p[0]
    Ek_pred_mean  = np.stack(Ek_pred).mean(0)
    Ek_truth_mean = np.stack(Ek_truth).mean(0)

    # k^-3 reference
    k_ref = k_bins[2:].astype(float)
    E_ref = k_ref[0]**3 * Ek_truth_mean[2] * k_ref**(-3)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(k_bins[1:], Ek_truth_mean[1:], color="steelblue", lw=1.5, label="Truth")
    ax.loglog(k_bins[1:], Ek_pred_mean[1:],  color="firebrick", lw=1.5, ls="--", label="Model")
    ax.loglog(k_ref, E_ref, color="gray", lw=1.0, ls=":", label="$k^{-3}$")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("QG2D CNN — Isotropic Energy Spectrum")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(fig_dir, "qg2d_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── 4. FTLE recovery ─────────────────────────────────────────────────────────

def plot_ftle(model, normalizer, test_data, dt, device, fig_dir, N,
              n_pairs=30, eps=0.01):
    rng   = np.random.default_rng(7)
    T     = test_data.shape[1] - 1
    times = np.arange(T + 1) * dt
    state_dim = N * N

    all_ftle_model = []
    all_ftle_truth = []
    for i in range(min(n_pairs, len(test_data))):
        q0_flat = test_data[i, 0].flatten()
        dv      = rng.standard_normal(q0_flat.shape)
        dv     /= np.linalg.norm(dv)
        q0p_flat = q0_flat + eps * dv

        # model FTLE
        ref_m  = model_rollout_np(model, normalizer, q0_flat,  T, device)
        pert_m = model_rollout_np(model, normalizer, q0p_flat, T, device)
        Eerr_m = np.mean((pert_m - ref_m)**2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ftle_m = np.where(times > 0,
                              np.log(Eerr_m / eps**2) / (2 * np.maximum(times, 1e-12)),
                              np.nan)
        all_ftle_model.append(ftle_m)

        # truth FTLE
        ref_t  = test_data[i].reshape(T + 1, state_dim).astype(np.float64)
        pert_t = integrate(q0p_flat.reshape(N, N).astype(np.float64), T, dt, N=N
                           ).reshape(T + 1, state_dim)
        Eerr_t = np.mean((pert_t - ref_t)**2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ftle_t = np.where(times > 0,
                              np.log(Eerr_t / eps**2) / (2 * np.maximum(times, 1e-12)),
                              np.nan)
        all_ftle_truth.append(ftle_t)

    mean_ftle_m = np.nanmean(np.stack(all_ftle_model), axis=0)
    mean_ftle_t = np.nanmean(np.stack(all_ftle_truth), axis=0)
    lambda1_est = float(np.nanmean(mean_ftle_t[len(times) // 4 : len(times) // 2]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, mean_ftle_m, color="steelblue", lw=1.2, label="Model FTLE")
    ax.plot(times, mean_ftle_t, color="steelblue", lw=1.2, ls=":", label="Truth FTLE")
    ax.axhline(lambda1_est, color="firebrick", ls="--",
               label=f"Truth λ₁ ≈ {lambda1_est:.2f}")
    ax.set_xlabel("Time (t.u.)")
    ax.set_ylabel("λ(t)  [FTLE]")
    ax.set_title("QG2D CNN — FTLE Recovery")
    ax.set_xlim(0, min(10 * LYAPUNOV_TIME, times[-1]))
    ax.set_ylim(-2, 6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(fig_dir, "qg2d_ftle.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}  (truth λ₁ ≈ {lambda1_est:.2f})")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(FIG_DIR, exist_ok=True)

    model, normalizer = load_model(CKPT_PATH, device)

    npz       = np.load(DATA_PATH)
    test_data = npz["test"].astype(np.float32)   # (30, 501, 64, 64)
    dt        = float(npz["dt"])
    N         = int(npz["N"])

    plot_snapshots(model, normalizer, test_data, dt, device, FIG_DIR, N)
    plot_vpt(      model, normalizer, test_data, dt, device, FIG_DIR, N)
    plot_spectrum( model, normalizer, test_data, dt, device, FIG_DIR, N)
    plot_ftle(     model, normalizer, test_data, dt, device, FIG_DIR, N)

    print("\nAll evaluation figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
