"""Generate NS2D trajectory dataset for ML surrogate training.

Uses a single long trajectory from a spun-up attractor state:
  1. Burn in from a random IC for BURNIN steps
  2. Integrate (N_TRAIN+N_VAL+N_TEST)*TRAJ_LEN more steps
  3. Slice into non-overlapping windows of length TRAJ_LEN+1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.utils.ns2d_physics import make_ic, make_forcing, integrate

# ── defaults ──────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(__file__), 'ns2d_trajectories.npz')
N        = 64
DT       = 0.005
BURNIN   = 2000
TRAJ_LEN = 500
NU       = 1e-6
ORDER    = 2
MU       = 0.1
K_F      = 4
F0       = 1.0
N_TRAIN  = 200
N_VAL    = 30
N_TEST   = 30
SEED     = 42
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate NS2D trajectories")
    parser.add_argument("--out",      default=OUT_PATH)
    parser.add_argument("--N",        type=int,   default=N)
    parser.add_argument("--dt",       type=float, default=DT)
    parser.add_argument("--burnin",   type=int,   default=BURNIN)
    parser.add_argument("--traj_len", type=int,   default=TRAJ_LEN)
    parser.add_argument("--nu",       type=float, default=NU)
    parser.add_argument("--order",    type=int,   default=ORDER)
    parser.add_argument("--mu",       type=float, default=MU)
    parser.add_argument("--k_f",      type=int,   default=K_F)
    parser.add_argument("--F0",       type=float, default=F0)
    parser.add_argument("--n_train",  type=int,   default=N_TRAIN)
    parser.add_argument("--n_val",    type=int,   default=N_VAL)
    parser.add_argument("--n_test",   type=int,   default=N_TEST)
    parser.add_argument("--seed",     type=int,   default=SEED)
    args = parser.parse_args()

    n_total = args.n_train + args.n_val + args.n_test
    print(f"NS2D data generation  "
          f"N={args.N}, dt={args.dt}, nu={args.nu}, order={args.order}, "
          f"mu={args.mu}, k_f={args.k_f}, F0={args.F0}, seed={args.seed}")

    F_hat  = make_forcing(args.N, k_f=args.k_f, F0=args.F0, seed=args.seed)
    omega0 = make_ic(args.N, k0=4, seed=args.seed)

    # ── Burnin ────────────────────────────────────────────────────────────────
    print(f"Burning in for {args.burnin} steps...")
    burnin_traj  = integrate(omega0, args.burnin, args.dt, F_hat,
                             N=args.N, nu=args.nu, order=args.order, mu=args.mu)
    omega_start  = burnin_traj[-1].copy()

    # ── Single long trajectory ────────────────────────────────────────────────
    n_steps_total = n_total * args.traj_len
    print(f"Integrating {n_steps_total} steps ({n_total} × {args.traj_len})...")
    long_traj = integrate(omega_start, n_steps_total, args.dt, F_hat,
                          N=args.N, nu=args.nu, order=args.order, mu=args.mu)
    # long_traj: (n_steps_total+1, N, N)

    # ── Slice into non-overlapping windows ────────────────────────────────────
    trajs = []
    for i in range(n_total):
        start = i * args.traj_len
        end   = start + args.traj_len + 1
        trajs.append(long_traj[start:end].astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"  Sliced {i+1}/{n_total}")

    trajs = np.stack(trajs)   # (n_total, traj_len+1, N, N)
    train = trajs[:args.n_train]
    val   = trajs[args.n_train : args.n_train + args.n_val]
    test  = trajs[args.n_train + args.n_val :]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        train=train, val=val, test=test,
                        dt=args.dt, N=args.N, nu=args.nu,
                        order=args.order, mu=args.mu,
                        k_f=args.k_f, F0=args.F0)
    print(f"\nSaved {args.out}")
    print(f"  train={train.shape}  val={val.shape}  test={test.shape}")


if __name__ == "__main__":
    main()
