"""Generate Lorenz 63 trajectory dataset and save to ml/data/l63_trajectories.npz."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.utils.l63_physics import integrate

# ── defaults ──────────────────────────────────────────────────────────────────
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT        = 0.01
TRAJ_LEN  = 2000       # steps per trajectory (20 time units)
BURNIN    = 1000       # steps to discard before sampling ICs
N_TRAIN   = 640
N_VAL     = 80
N_TEST    = 80
SEED      = 42
# ──────────────────────────────────────────────────────────────────────────────


def generate_dataset(sigma=SIGMA, rho=RHO, beta=BETA, dt=DT,
                     traj_len=TRAJ_LEN, burnin=BURNIN,
                     n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                     seed=SEED):
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test

    # Single long reference trajectory to sample on-attractor ICs from.
    # Each IC is separated by at least traj_len steps to keep them independent.
    ref_len = burnin + n_total * traj_len
    x0_ref  = np.array([1.0, 0.0, 0.0])
    ref     = integrate(x0_ref, ref_len, dt, sigma, rho, beta)

    # Sample n_total starting indices from the post-burnin portion.
    # Stride by traj_len so trajectories don't overlap.
    starts = burnin + np.arange(n_total) * traj_len
    # Shuffle which trajectory goes to which split.
    idx = rng.permutation(n_total)

    data = np.stack([
        ref[s : s + traj_len + 1].astype(np.float32)
        for s in starts[idx]
    ])  # (n_total, traj_len+1, 3)

    train = data[:n_train]
    val   = data[n_train : n_train + n_val]
    test  = data[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Generate L63 dataset")
    parser.add_argument("--dt",       type=float, default=DT)
    parser.add_argument("--traj_len", type=int,   default=TRAJ_LEN)
    parser.add_argument("--n_train",  type=int,   default=N_TRAIN)
    parser.add_argument("--n_val",    type=int,   default=N_VAL)
    parser.add_argument("--n_test",   type=int,   default=N_TEST)
    parser.add_argument("--seed",     type=int,   default=SEED)
    parser.add_argument("--out",      type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "l63_trajectories.npz"))
    args = parser.parse_args()

    print("Generating L63 dataset...")
    train, val, test = generate_dataset(
        dt=args.dt, traj_len=args.traj_len,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seed=args.seed,
    )

    np.savez(args.out,
             train=train, val=val, test=test,
             dt=args.dt, sigma=SIGMA, rho=RHO, beta=BETA)
    print(f"Saved {args.out}")
    print(f"  train={train.shape}  val={val.shape}  test={test.shape}")


if __name__ == "__main__":
    main()
