"""2D Navier-Stokes pseudospectral physics: vorticity inversion, Jacobian, RK4, IC/forcing."""

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


def invert_vorticity(omega_hat, K2_safe):
    """ψ̂ = −ω̂ / K²  (mean streamfunction = 0)."""
    psi_hat = -omega_hat / K2_safe
    psi_hat[0, 0] = 0.0
    return psi_hat


def jacobian_hat(psi_hat, omega_hat, KX, KY, dealias):
    """Pseudospectral J(ψ,ω) = ψ_x ω_y − ψ_y ω_x with 2/3-rule dealiasing."""
    psi_x   = np.real(ifft2(1j * KX * psi_hat))
    psi_y   = np.real(ifft2(1j * KY * psi_hat))
    omega_x = np.real(ifft2(1j * KX * omega_hat))
    omega_y = np.real(ifft2(1j * KY * omega_hat))
    return fft2(psi_x * omega_y - psi_y * omega_x) * dealias


def make_forcing(N, k_f=4, F0=1.0, seed=42):
    """Generate a fixed stochastic forcing pattern in spectral space.

    Returns F_hat: (N, N) complex128, Hermitian-symmetric, normalised so that
    the RMS of the physical forcing field equals F0.
    """
    KX, KY, K2, K2_safe, dealias = _get_arrays(N)
    rng   = np.random.default_rng(seed)
    k_mag = np.sqrt(K2)

    # Band-pass filter centred on k_f with width 1
    band = np.exp(-0.5 * ((k_mag - k_f) / 1.0)**2)

    phases = rng.uniform(0, 2 * np.pi, (N, N))
    F_hat  = band * np.exp(1j * phases) * dealias
    # Enforce Hermitian symmetry → real physical field
    F_hat  = F_hat + np.conj(F_hat[::-1, ::-1])
    F_hat[0, 0] = 0.0

    # Normalise to target RMS amplitude F0
    F_phys = np.real(ifft2(F_hat))
    rms    = np.sqrt(np.mean(F_phys**2))
    if rms > 1e-30:
        F_hat = F_hat * (F0 / rms)
    return F_hat


def ns2d_rhs(omega_hat, F_hat, nu, order, mu, KX, KY, K2, K2_safe, dealias):
    """dω/dt = −J(ψ,ω) + ν(−1)^(p+1) ∇^(2p) ω − μω + F."""
    psi_hat = invert_vorticity(omega_hat, K2_safe)
    jac     = jacobian_hat(psi_hat, omega_hat, KX, KY, dealias)
    visc    = nu * ((-1.0) ** (order + 1)) * (K2 ** order) * omega_hat
    drag    = -mu * omega_hat
    return -jac + visc + drag + F_hat


def rk4_step(omega_hat, F_hat, dt, nu, order, mu, KX, KY, K2, K2_safe, dealias):
    """4th-order Runge-Kutta step; F_hat is passed unchanged to each stage."""
    def f(w): return ns2d_rhs(w, F_hat, nu, order, mu, KX, KY, K2, K2_safe, dealias)
    k1 = f(omega_hat)
    k2 = f(omega_hat + 0.5 * dt * k1)
    k3 = f(omega_hat + 0.5 * dt * k2)
    k4 = f(omega_hat +       dt * k3)
    return omega_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def make_ic(N, k0=4, seed=0):
    """Band-limited random vorticity field, Hermitian-symmetrised, unit RMS.

    Returns omega0: (N, N) float64.
    """
    KX, KY, K2, K2_safe, dealias = _get_arrays(N)
    rng   = np.random.default_rng(seed)
    k_mag = np.sqrt(K2)
    band  = np.exp(-0.5 * ((k_mag - k0) / 2.0)**2)

    phases    = rng.uniform(0, 2 * np.pi, (N, N))
    omega_hat = band * np.exp(1j * phases) * dealias
    omega_hat = omega_hat + np.conj(omega_hat[::-1, ::-1])
    omega_hat[0, 0] = 0.0

    omega0 = np.real(ifft2(omega_hat))
    rms    = np.sqrt(np.mean(omega0**2))
    omega_hat = omega_hat / (rms + 1e-30)
    return np.real(ifft2(omega_hat))


def integrate(omega0, nsteps, dt, F_hat, N=64, nu=1e-6, order=2, mu=0.1):
    """Integrate NS2D from physical vorticity omega0.

    omega0: (N, N) float64
    F_hat:  (N, N) complex128  — fixed spectral forcing pattern
    Returns: (nsteps+1, N, N) float64 — physical ω at each step.
    """
    KX, KY, K2, K2_safe, dealias = _get_arrays(N)
    omega_hat = fft2(omega0.astype(np.float64))

    traj = np.empty((nsteps + 1, N, N), dtype=np.float64)
    traj[0] = omega0
    for i in range(nsteps):
        omega_hat  = rk4_step(omega_hat, F_hat, dt, nu, order, mu,
                               KX, KY, K2, K2_safe, dealias)
        traj[i + 1] = np.real(ifft2(omega_hat))
    return traj
