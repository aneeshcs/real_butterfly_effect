"""Generate QG2D trajectory dataset for ML surrogate training."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ml.utils.qg2d_physics import make_ic, integrate

# ── defaults ──────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(__file__), 'qg2d_trajectories.npz')
N        = 64
DT       = 0.005
TRAJ_LEN = 500
NU       = 1e-7
ORDER    = 2
BETA     = 0.0
N_TRAIN  = 200
N_VAL    = 30
N_TEST   = 30
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate QG2D trajectories")
    parser.add_argument("--out",      default=OUT_PATH)
    parser.add_argument("--N",        type=int,   default=N)
    parser.add_argument("--dt",       type=float, default=DT)
    parser.add_argument("--traj_len", type=int,   default=TRAJ_LEN)
    parser.add_argument("--nu",       type=float, default=NU)
    parser.add_argument("--order",    type=int,   default=ORDER)
    parser.add_argument("--beta",     type=float, default=BETA)
    parser.add_argument("--n_train",  type=int,   default=N_TRAIN)
    parser.add_argument("--n_val",    type=int,   default=N_VAL)
    parser.add_argument("--n_test",   type=int,   default=N_TEST)
    args = parser.parse_args()

    n_total = args.n_train + args.n_val + args.n_test
    print(f"Generating {n_total} trajectories  "
          f"(N={args.N}, dt={args.dt}, traj_len={args.traj_len}, "
          f"nu={args.nu}, order={args.order}, beta={args.beta})")

    trajs = []
    for i in range(n_total):
        q0   = make_ic(args.N, k0=4, seed=i)
        traj = integrate(q0, args.traj_len, args.dt,
                         N=args.N, nu=args.nu, order=args.order, beta=args.beta)
        trajs.append(traj.astype(np.float32))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_total}")

    trajs = np.stack(trajs)   # (n_total, traj_len+1, N, N)
    train = trajs[:args.n_train]
    val   = trajs[args.n_train : args.n_train + args.n_val]
    test  = trajs[args.n_train + args.n_val :]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        train=train, val=val, test=test,
                        dt=args.dt, N=args.N, nu=args.nu,
                        order=args.order, beta=args.beta)
    print(f"\nSaved {args.out}")
    print(f"  train={train.shape}  val={val.shape}  test={test.shape}")


if __name__ == "__main__":
    main()
