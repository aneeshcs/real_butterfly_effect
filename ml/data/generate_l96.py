"""Generate Lorenz 96 trajectory dataset and save to ml/data/l96_trajectories.npz."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.utils.l96_physics import integrate

# ── defaults ──────────────────────────────────────────────────────────────────
K         = 36
F         = 8.0
DT        = 0.01
TRAJ_LEN  = 1000       # steps per trajectory (10 time units)
BURNIN    = 2000       # steps to discard before sampling ICs
N_TRAIN   = 500
N_VAL     = 60
N_TEST    = 60
SEED      = 42
# ──────────────────────────────────────────────────────────────────────────────


def generate_dataset(K=K, F=F, dt=DT, traj_len=TRAJ_LEN, burnin=BURNIN,
                     n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED):
    rng     = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test

    # Single long reference trajectory; sample on-attractor ICs from it.
    ref_len = burnin + n_total * traj_len
    x0_ref  = np.full(K, F, dtype=np.float64)
    x0_ref[K // 4] += 0.01  # standard L96 perturbation to break symmetry
    ref     = integrate(x0_ref, ref_len, dt, F)

    starts = burnin + np.arange(n_total) * traj_len
    idx    = rng.permutation(n_total)

    data = np.stack([
        ref[s : s + traj_len + 1].astype(np.float32)
        for s in starts[idx]
    ])  # (n_total, traj_len+1, K)

    train = data[:n_train]
    val   = data[n_train : n_train + n_val]
    test  = data[n_train + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Generate L96 dataset")
    parser.add_argument("--K",        type=int,   default=K)
    parser.add_argument("--F",        type=float, default=F)
    parser.add_argument("--dt",       type=float, default=DT)
    parser.add_argument("--traj_len", type=int,   default=TRAJ_LEN)
    parser.add_argument("--n_train",  type=int,   default=N_TRAIN)
    parser.add_argument("--n_val",    type=int,   default=N_VAL)
    parser.add_argument("--n_test",   type=int,   default=N_TEST)
    parser.add_argument("--seed",     type=int,   default=SEED)
    parser.add_argument("--out",      type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "l96_trajectories.npz"))
    args = parser.parse_args()

    print("Generating L96 dataset...")
    train, val, test = generate_dataset(
        K=args.K, F=args.F, dt=args.dt, traj_len=args.traj_len,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seed=args.seed,
    )

    np.savez(args.out,
             train=train, val=val, test=test,
             dt=args.dt, K=args.K, F=args.F)
    print(f"Saved {args.out}")
    print(f"  train={train.shape}  val={val.shape}  test={test.shape}")


if __name__ == "__main__":
    main()
