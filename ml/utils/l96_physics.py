import numpy as np


def l96_rhs(x, F):
    """L96 right-hand side: (x_{i+1} − x_{i-2}) x_{i-1} − x_i + F."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def rk4_step(x, dt, F):
    """4th-order Runge-Kutta step for L96."""
    k1 = l96_rhs(x,              F)
    k2 = l96_rhs(x + 0.5*dt*k1, F)
    k3 = l96_rhs(x + 0.5*dt*k2, F)
    k4 = l96_rhs(x +     dt*k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(x0, nsteps, dt, F=8.0):
    """Integrate L96 from x0 for nsteps. Returns trajectory of shape (nsteps+1, K)."""
    K = len(x0)
    traj = np.empty((nsteps + 1, K), dtype=np.float64)
    traj[0] = x0
    for i in range(nsteps):
        traj[i + 1] = rk4_step(traj[i], dt, F)
    return traj
