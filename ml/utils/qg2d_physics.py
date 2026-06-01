"""2D QG pseudospectral physics: PV inversion, Jacobian, RK4, IC generator."""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

# Cache of pre-computed spectral arrays keyed by N.
_ARRAY_CACHE = {}


def _get_arrays(N):
    if N not in _ARRAY_CACHE:
        kx = fftfreq(N, d=1.0 / N)
        ky = fftfreq(N, d=1.0 / N)
        KX, KY = np.meshgrid(kx, ky)
        K2      = KX**2 + KY**2
        K2_safe = K2.copy()
        K2_safe[0, 0] = 1.0          # avoid division by zero at k=0
        kmax    = N // 3             # 2/3-rule dealiasing cutoff
        dealias = ((np.abs(KX) < kmax) & (np.abs(KY) < kmax)).astype(float)
        _ARRAY_CACHE[N] = (KX, KY, K2, K2_safe, dealias)
    return _ARRAY_CACHE[N]


def invert_pv(q_hat, K2_safe):
    """ψ̂ = −q̂ / K²  (mean streamfunction = 0)."""
    psi_hat = -q_hat / K2_safe
    psi_hat[0, 0] = 0.0
    return psi_hat


def jacobian_hat(psi_hat, q_hat, KX, KY, dealias):
    """Pseudospectral J(ψ,q) = ψ_x q_y − ψ_y q_x with 2/3-rule dealiasing."""
    psi_x = np.real(ifft2(1j * KX * psi_hat))
    psi_y = np.real(ifft2(1j * KY * psi_hat))
    q_x   = np.real(ifft2(1j * KX * q_hat))
    q_y   = np.real(ifft2(1j * KY * q_hat))
    return fft2(psi_x * q_y - psi_y * q_x) * dealias


def qg2d_rhs(q_hat, nu, order, beta, KX, KY, K2, K2_safe, dealias):
    """dq/dt = −J(ψ,q) − β ∂ψ/∂x + ν (−1)^(p+1) ∇^(2p) q."""
    psi_hat  = invert_pv(q_hat, K2_safe)
    beta_hat = -beta * (1j * KX * psi_hat)
    jac      = jacobian_hat(psi_hat, q_hat, KX, KY, dealias)
    diss     = nu * ((-1.0) ** (order + 1)) * (K2 ** order) * q_hat
    return -jac + beta_hat + diss


def rk4_step(q_hat, dt, nu, order, beta, KX, KY, K2, K2_safe, dealias):
    """4th-order Runge-Kutta step."""
    def f(q): return qg2d_rhs(q, nu, order, beta, KX, KY, K2, K2_safe, dealias)
    k1 = f(q_hat)
    k2 = f(q_hat + 0.5 * dt * k1)
    k3 = f(q_hat + 0.5 * dt * k2)
    k4 = f(q_hat +       dt * k3)
    return q_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def make_ic(N, k0=4, seed=None):
    """Band-limited random PV field, Hermitian-symmetrised, unit enstrophy.

    Returns q0: (N, N) float64.
    """
    KX, KY, K2, K2_safe, dealias = _get_arrays(N)
    rng   = np.random.default_rng(seed)
    k_mag = np.sqrt(K2)
    band  = np.exp(-0.5 * ((k_mag - k0) / 2.0)**2)

    phases = rng.uniform(0, 2 * np.pi, (N, N))
    q0_hat = band * np.exp(1j * phases) * dealias
    q0_hat = q0_hat + np.conj(q0_hat[::-1, ::-1])
    q0_hat[0, 0] = 0.0

    q0   = np.real(ifft2(q0_hat))
    norm = np.sqrt(np.mean(q0**2))
    q0_hat = q0_hat / (norm + 1e-30)
    return np.real(ifft2(q0_hat))


def integrate(q0, nsteps, dt, N=64, nu=1e-7, order=2, beta=0.0):
    """Integrate QG2D from physical PV q0.

    q0: (N, N) float64
    Returns: (nsteps+1, N, N) float64 — physical PV at each step.
    """
    KX, KY, K2, K2_safe, dealias = _get_arrays(N)
    q_hat = fft2(q0.astype(np.float64))

    traj = np.empty((nsteps + 1, N, N), dtype=np.float64)
    traj[0] = q0
    for i in range(nsteps):
        q_hat  = rk4_step(q_hat, dt, nu, order, beta, KX, KY, K2, K2_safe, dealias)
        traj[i + 1] = np.real(ifft2(q_hat))
    return traj
