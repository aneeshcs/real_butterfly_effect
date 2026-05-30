import numpy as np


def l63_rhs(state, sigma, rho, beta):
    """L63 right-hand side: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz."""
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def rk4_step(state, dt, sigma, rho, beta):
    """4th-order Runge-Kutta step for L63."""
    k1 = l63_rhs(state,              sigma, rho, beta)
    k2 = l63_rhs(state + 0.5*dt*k1, sigma, rho, beta)
    k3 = l63_rhs(state + 0.5*dt*k2, sigma, rho, beta)
    k4 = l63_rhs(state +     dt*k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(x0, nsteps, dt, sigma=10.0, rho=28.0, beta=8/3):
    """Integrate L63 from x0 for nsteps. Returns trajectory of shape (nsteps+1, 3)."""
    traj = np.empty((nsteps + 1, 3), dtype=np.float64)
    traj[0] = x0
    for i in range(nsteps):
        traj[i + 1] = rk4_step(traj[i], dt, sigma, rho, beta)
    return traj
