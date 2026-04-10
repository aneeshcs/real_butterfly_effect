import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from numpy.fft import fft2, ifft2, fftfreq
    import matplotlib.pyplot as plt
    return fft2, fftfreq, ifft2, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Surface Quasi-Geostrophic (SQG) Turbulence

        Pseudospectral solver on a doubly-periodic domain $[0, 2\pi)^2$.

        **Governing equation** (surface buoyancy $b$):
        $$\frac{\partial b}{\partial t} + J(\psi, b) = \nu (-1)^{p+1} \nabla^{2p} b$$

        **SQG inversion** (replaces Laplacian inversion in 2D/3D QG):
        $$\hat{\psi}(\mathbf{k}) = \frac{\hat{b}(\mathbf{k})}{|\mathbf{k}|}$$

        The surface buoyancy plays the role of PV, but the Green's function is $|\mathbf{k}|^{-1}$
        rather than $|\mathbf{k}|^{-2}$ as in 2D QG. This produces smaller-scale, more filamentary
        structures and a shallower kinetic energy spectrum $E(k) \sim k^{-5/3}$, identical in slope
        to the 3D Kolmogorov inertial range.

        A key identity: in SQG, **surface KE equals buoyancy variance**:
        $$\frac{1}{2}\sum_\mathbf{k} k^2 |\hat\psi_\mathbf{k}|^2
          = \frac{1}{2}\sum_\mathbf{k} k^2 \frac{|\hat b_\mathbf{k}|^2}{k^2}
          = \frac{1}{2}\sum_\mathbf{k} |\hat b_\mathbf{k}|^2$$

        **Numerics:** pseudospectral with 2/3-rule dealiasing, RK4 time stepping.
        """
    )
    return


@app.cell
def _(mo):
    N_ctrl      = mo.ui.slider(64, 512, step=64, value=128, label="Resolution N")
    nu_ctrl     = mo.ui.number(start=1e-10, stop=1e-3, step=1e-9, value=1e-7,
                               label="Hyperviscosity ν")
    order_ctrl  = mo.ui.slider(1, 4, step=1, value=2, label="Order p")
    dt_ctrl     = mo.ui.number(start=1e-4, stop=0.05, step=1e-4, value=0.005,
                               label="Time step dt")
    nsteps_ctrl = mo.ui.slider(200, 10000, step=200, value=500,
                               label="Steps to run")
    seed_ctrl   = mo.ui.number(start=0, stop=9999, step=1, value=42,
                               label="Random seed")

    mo.vstack([
        mo.md("## Model Parameters"),
        mo.hstack([N_ctrl, nu_ctrl, order_ctrl]),
        mo.hstack([dt_ctrl, nsteps_ctrl, seed_ctrl]),
    ])
    return N_ctrl, dt_ctrl, nsteps_ctrl, nu_ctrl, order_ctrl, seed_ctrl


@app.cell
def _(N_ctrl, fftfreq, np):
    # ── Grid and spectral operators ───────────────────────────────────────────
    N = N_ctrl.value
    L = 2.0 * np.pi

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    kx = fftfreq(N, d=1.0 / N)
    ky = fftfreq(N, d=1.0 / N)
    KX, KY   = np.meshgrid(kx, ky)
    K2        = KX**2 + KY**2
    K_mag     = np.sqrt(K2)
    K_mag_safe = K_mag.copy()
    K_mag_safe[0, 0] = 1.0   # avoid division by zero at k=0

    kmax    = N // 3          # 2/3 dealiasing cutoff
    dealias = ((np.abs(KX) < kmax) & (np.abs(KY) < kmax)).astype(float)

    return K2, KX, KY, K_mag, K_mag_safe, L, N, X, Y, dealias, kmax, x, y


@app.cell
def _(K2, K_mag_safe, KX, KY, dealias, fft2, ifft2, np):
    # ── Core SQG operators ────────────────────────────────────────────────────

    def invert_sqg(b_hat):
        """SQG inversion: ψ̂ = b̂ / |k|  (zero mean streamfunction)."""
        psi_hat = b_hat / K_mag_safe
        psi_hat[0, 0] = 0.0
        return psi_hat

    def jacobian_hat(psi_hat, b_hat):
        """Dealiased spectral Jacobian J(ψ, b) = ψ_x b_y − ψ_y b_x."""
        psi_x = np.real(ifft2(1j * KX * psi_hat))
        psi_y = np.real(ifft2(1j * KY * psi_hat))
        b_x   = np.real(ifft2(1j * KX * b_hat))
        b_y   = np.real(ifft2(1j * KY * b_hat))
        return fft2(psi_x * b_y - psi_y * b_x) * dealias

    def rhs(b_hat, nu, order):
        """RHS of ∂b/∂t = −J(ψ,b) + ν(−1)^(p+1) ∇^(2p) b."""
        psi_hat = invert_sqg(b_hat)
        jac     = jacobian_hat(psi_hat, b_hat)
        diss    = nu * ((-1.0) ** (order + 1)) * (K2 ** order) * b_hat
        return -jac + diss

    def rk4_step(b_hat, dt, nu, order):
        """4th-order Runge-Kutta step."""
        k1 = rhs(b_hat,              nu, order)
        k2 = rhs(b_hat + 0.5*dt*k1, nu, order)
        k3 = rhs(b_hat + 0.5*dt*k2, nu, order)
        k4 = rhs(b_hat +     dt*k3, nu, order)
        return b_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return invert_sqg, jacobian_hat, rhs, rk4_step


@app.cell
def _(K_mag, N, dealias, ifft2, np, seed_ctrl):
    # ── Initial condition: band-limited random surface buoyancy ───────────────
    rng = np.random.default_rng(seed_ctrl.value)

    k0   = 4
    band = np.exp(-0.5 * ((K_mag - k0) / 2.0)**2)

    phases = rng.uniform(0, 2*np.pi, (N, N))
    b0_hat = band * np.exp(1j * phases) * dealias
    # Hermitian symmetry → real-valued buoyancy field
    b0_hat += np.conj(b0_hat[::-1, ::-1])
    b0_hat[0, 0] = 0.0

    # Normalise to unit buoyancy variance (= unit surface KE)
    b0   = np.real(ifft2(b0_hat))
    norm = np.sqrt(np.mean(b0**2))
    b0_hat = b0_hat / (norm + 1e-30)

    return b0, b0_hat, band, k0, norm, phases, rng


@app.cell
def _(b0_hat, dt_ctrl, mo, np, nsteps_ctrl, nu_ctrl, order_ctrl, rk4_step):
    # ── Time integration ─────────────────────────────────────────────────────
    dt     = dt_ctrl.value
    nu     = nu_ctrl.value
    order  = order_ctrl.value
    nsteps = nsteps_ctrl.value

    b_hat = b0_hat.copy()
    t     = 0.0

    save_every = max(1, nsteps // 100)
    times, bvar_series = [], []

    for _step in range(nsteps):
        b_hat = rk4_step(b_hat, dt, nu, order)
        t    += dt
        if _step % save_every == 0:
            # Surface energy = buoyancy variance = KE (SQG identity)
            BV = 0.5 * np.sum(np.abs(b_hat)**2) / b_hat.size**2
            times.append(t)
            bvar_series.append(BV)

    times       = np.array(times)
    bvar_series = np.array(bvar_series)

    mo.md(f"**Integration complete.** t = {t:.3f},  steps = {nsteps},  dt = {dt}")
    return BV, b_hat, bvar_series, dt, nsteps, nu, order, save_every, t, times


@app.cell
def _(K_mag, b_hat, invert_sqg, np, plt):
    # ── Snapshot visualisation ────────────────────────────────────────────────
    from numpy.fft import ifft2 as _ifft2

    b_phys   = np.real(_ifft2(b_hat))
    psi_hat_ = invert_sqg(b_hat)
    psi_phys = np.real(_ifft2(psi_hat_))

    # Buoyancy variance spectrum: E(k) = ½ |b̂|²  (= KE spectrum in SQG)
    k_int    = K_mag.ravel().astype(int)
    E_flat   = 0.5 * (np.abs(b_hat)**2 / b_hat.size**2).ravel()
    kmax_int = k_int.max()
    E_spec   = np.bincount(k_int, weights=E_flat, minlength=kmax_int + 1)
    k_bins   = np.arange(kmax_int + 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    im0 = axes[0].imshow(b_phys, cmap='RdBu_r', origin='lower')
    axes[0].set_title('Surface Buoyancy $b$')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(psi_phys, cmap='RdBu_r', origin='lower')
    axes[1].set_title('Streamfunction $\\psi$')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    k_plot = k_bins[1:kmax_int//2]
    axes[2].loglog(k_plot, E_spec[1:kmax_int//2], 'b-', linewidth=1.5, label='$E(k)$')
    k_ref = k_plot[len(k_plot)//4 : 3*len(k_plot)//4]
    A     = E_spec[k_plot[len(k_plot)//3]] + 1e-30
    axes[2].loglog(k_ref, A * (k_ref / k_ref[0])**(-5/3), 'r--', label='$k^{-5/3}$')
    axes[2].loglog(k_ref, A * (k_ref / k_ref[0])**(-3),   'k--', label='$k^{-3}$')
    axes[2].set_xlabel('Wavenumber $k$')
    axes[2].set_ylabel('$E(k)$')
    axes[2].set_title('Buoyancy Variance Spectrum (= KE)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fig
    return (
        A, E_flat, E_spec, fig, im0, im1, k_bins, k_int, k_plot, k_ref,
        kmax_int, psi_hat_, psi_phys, b_phys,
    )


@app.cell
def _(bvar_series, plt, times):
    fig_ts, ax_ts = plt.subplots(figsize=(7, 3.5))
    ax_ts.plot(times, bvar_series, 'b-', linewidth=1.5)
    ax_ts.set_xlabel('Time')
    ax_ts.set_ylabel(r'$E = \frac{1}{2}\langle b^2 \rangle$')
    ax_ts.set_title('Surface Energy (Buoyancy Variance = KE) vs Time')
    ax_ts.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_ts
    return ax_ts, fig_ts


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Predictability Experiment

        Two SQG runs from nearly identical initial surface buoyancy fields
        $b_0$ (reference) and $b_0 + \varepsilon\,\delta b$ (perturbed).

        We track the **error surface energy** (= error buoyancy variance = error KE):
        $$E_{\rm err}(t) = \frac{1}{2N^2} \sum_\mathbf{k} |\delta\hat{b}_\mathbf{k}|^2$$

        and the **finite-time Lyapunov exponent (FTLE)**:
        $$\lambda(t) = \frac{1}{2t} \ln \frac{E_{\rm err}(t)}{\varepsilon^2}$$

        SQG errors grow faster than in 2D QG because the shallower $k^{-5/3}$ spectrum
        implies more energetic small scales, which amplify perturbations more rapidly.
        This connects directly to Lorenz's (1969) theory of predictability limits in
        flows with $k^{-5/3}$ spectra.
        """
    )
    return


@app.cell
def _(mo):
    eps_ctrl   = mo.ui.number(start=1e-10, stop=1e-2, step=1e-10, value=1e-5,
                               label="Perturbation amplitude ε")
    pseed_ctrl = mo.ui.number(start=0, stop=9999, step=1, value=137,
                               label="Perturbation seed")
    mo.hstack([eps_ctrl, pseed_ctrl])
    return eps_ctrl, pseed_ctrl


@app.cell
def _(
    b0_hat,
    dealias,
    dt,
    eps_ctrl,
    np,
    nsteps,
    nu,
    order,
    pseed_ctrl,
    rk4_step,
    save_every,
):
    # ── Predictability integration ────────────────────────────────────────────
    eps  = eps_ctrl.value
    prng = np.random.default_rng(pseed_ctrl.value)

    # Unit-normalised random perturbation in spectral space
    dph    = prng.uniform(0, 2*np.pi, b0_hat.shape)
    db_hat = dealias * np.exp(1j * dph)
    db_hat += np.conj(db_hat[::-1, ::-1])
    db_hat[0, 0] = 0.0
    db_hat /= (np.sqrt(np.sum(np.abs(db_hat)**2)) + 1e-30)

    bref_hat  = b0_hat.copy()
    bpert_hat = b0_hat + eps * db_hat

    pred_times, err_energy, ftle = [], [], []

    for _s in range(nsteps):
        bref_hat  = rk4_step(bref_hat,  dt, nu, order)
        bpert_hat = rk4_step(bpert_hat, dt, nu, order)
        _t = (_s + 1) * dt

        if _s % save_every == 0 and _t > 0:
            delta_hat = bpert_hat - bref_hat
            Eerr = 0.5 * np.sum(np.abs(delta_hat)**2) / b0_hat.size**2
            pred_times.append(_t)
            err_energy.append(Eerr)
            ftle.append(
                np.log(Eerr / (eps**2 + 1e-30)) / (2.0 * _t) if Eerr > 0 else np.nan
            )

    pred_times = np.array(pred_times)
    err_energy = np.array(err_energy)
    ftle       = np.array(ftle)

    return Eerr, bpert_hat, bref_hat, db_hat, dph, eps, err_energy, ftle, pred_times, prng


@app.cell
def _(err_energy, ftle, plt, pred_times):
    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(10, 4))

    axes_pred[0].semilogy(pred_times, err_energy, 'b-', linewidth=1.5)
    axes_pred[0].set_xlabel('Time')
    axes_pred[0].set_ylabel('Error Energy $E_{\\rm err}$')
    axes_pred[0].set_title('Error Buoyancy Variance Growth')
    axes_pred[0].grid(True, which='both', alpha=0.3)

    axes_pred[1].plot(pred_times, ftle, 'r-', linewidth=1.5)
    axes_pred[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes_pred[1].set_xlabel('Time')
    axes_pred[1].set_ylabel('FTLE $\\lambda(t)$')
    axes_pred[1].set_title('Finite-Time Lyapunov Exponent')
    axes_pred[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_pred
    return axes_pred, fig_pred


@app.cell
def _(err_energy, ftle, mo, np, pred_times):
    _tail      = ftle[~np.isnan(ftle)]
    lambda_inf = _tail[-10:].mean() if len(_tail) > 10 else np.nan
    t_pred     = (
        pred_times[np.argmax(err_energy > 0.1 * err_energy[-1])]
        if err_energy[-1] > 0 else np.nan
    )

    mo.md(
        f"""
        ### Summary

        | Quantity | Value |
        |---|---|
        | Asymptotic FTLE $\\lambda$ | `{lambda_inf:.4f}` |
        | Predictability time ($E_{{\\rm err}} > 0.1\\,E_{{\\rm err,max}}$) | `{t_pred:.3f}` |
        | Final error energy | `{err_energy[-1]:.4e}` |
        """
    )
    return lambda_inf, t_pred


if __name__ == "__main__":
    app.run()
