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
        # 2D Quasi-Geostrophic Turbulence

        Pseudospectral solver on a doubly-periodic domain $[0, 2\pi)^2$.

        **Governing equation:**
        $$\frac{\partial q}{\partial t} + J(\psi, q) = \nu (-1)^{p+1} \nabla^{2p} q$$

        where $q = \nabla^2 \psi + \beta y$ is the potential vorticity (PV),
        $J(\psi, q) = \psi_x q_y - \psi_y q_x$ is the Jacobian,
        and $\nu, p$ are the hyperviscosity coefficient and order.

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
    beta_ctrl   = mo.ui.number(start=0.0, stop=20.0, step=0.5, value=0.0,
                               label="β (planetary vorticity gradient)")
    dt_ctrl     = mo.ui.number(start=1e-4, stop=0.05, step=1e-4, value=0.005,
                               label="Time step dt")
    nsteps_ctrl = mo.ui.slider(200, 10000, step=200, value=500,
                               label="Steps to run")
    seed_ctrl   = mo.ui.number(start=0, stop=9999, step=1, value=42,
                               label="Random seed")

    mo.vstack([
        mo.md("## Model Parameters"),
        mo.hstack([N_ctrl, nu_ctrl, order_ctrl, beta_ctrl]),
        mo.hstack([dt_ctrl, nsteps_ctrl, seed_ctrl]),
    ])
    return (
        N_ctrl,
        beta_ctrl,
        dt_ctrl,
        nsteps_ctrl,
        nu_ctrl,
        order_ctrl,
        seed_ctrl,
    )


@app.cell
def _(N_ctrl, beta_ctrl, fftfreq, np):
    # ── Grid and spectral operators ───────────────────────────────────────────
    N    = N_ctrl.value
    L    = 2.0 * np.pi
    beta = beta_ctrl.value

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    kx = fftfreq(N, d=1.0 / N)
    ky = fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky)
    K2      = KX**2 + KY**2
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0          # avoid division by zero at k=0

    kmax   = N // 3              # 2/3 dealiasing cutoff
    dealias = ((np.abs(KX) < kmax) & (np.abs(KY) < kmax)).astype(float)
    return (
        K2,
        K2_safe,
        KX,
        KY,
        L,
        N,
        X,
        Y,
        beta,
        dealias,
        kmax,
        x,
        y,
    )


@app.cell
def _(K2_safe, KX, KY, dealias, fft2, ifft2, np):
    # ── Core spectral operators ───────────────────────────────────────────────

    def invert_pv(q_hat):
        """Solve ∇²ψ = q  →  ψ̂ = -q̂ / K²  (f-plane, mean streamfunction = 0)."""
        psi_hat = -q_hat / K2_safe
        psi_hat[0, 0] = 0.0
        return psi_hat

    def jacobian_hat(psi_hat, q_hat):
        """Spectral Jacobian J(ψ,q) = ψ_x q_y − ψ_y q_x with dealiasing."""
        psi_x = np.real(ifft2(1j * KX * psi_hat))
        psi_y = np.real(ifft2(1j * KY * psi_hat))
        q_x   = np.real(ifft2(1j * KX * q_hat))
        q_y   = np.real(ifft2(1j * KY * q_hat))
        return fft2(psi_x * q_y - psi_y * q_x) * dealias

    def rhs(q_hat, nu, order, beta, KX, K2):
        """RHS of dq/dt = -J(ψ,q) + ν(-1)^(p+1) ∇^(2p) q."""
        psi_hat  = invert_pv(q_hat)
        # beta term: -β ∂ψ/∂x  (contribution to dq/dt from beta advection of f)
        beta_hat = -beta * (1j * KX * psi_hat)
        jac      = jacobian_hat(psi_hat, q_hat)
        diss     = nu * ((-1.0) ** (order + 1)) * (K2 ** order) * q_hat
        return -jac + beta_hat + diss

    def rk4_step(q_hat, dt, nu, order, beta, KX, K2):
        """4th-order Runge-Kutta step."""
        k1 = rhs(q_hat,              nu, order, beta, KX, K2)
        k2 = rhs(q_hat + 0.5*dt*k1, nu, order, beta, KX, K2)
        k3 = rhs(q_hat + 0.5*dt*k2, nu, order, beta, KX, K2)
        k4 = rhs(q_hat +     dt*k3, nu, order, beta, KX, K2)
        return q_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return invert_pv, jacobian_hat, rhs, rk4_step


@app.cell
def _(K2, N, dealias, fft2, ifft2, np, seed_ctrl):
    # ── Initial condition: band-limited random vorticity ─────────────────────
    rng = np.random.default_rng(seed_ctrl.value)

    # Excite a ring of wavenumbers around k0
    k0   = 4
    k_mag = np.sqrt(K2)
    band  = np.exp(-0.5 * ((k_mag - k0) / 2.0)**2)

    phases = rng.uniform(0, 2*np.pi, (N, N))
    q0_hat = band * np.exp(1j * phases) * dealias
    # Make it Hermitian (real-valued vorticity)
    q0_hat = q0_hat + np.conj(q0_hat[::-1, ::-1])
    q0_hat[0, 0] = 0.0

    # Normalise to unit enstrophy
    q0   = np.real(ifft2(q0_hat))
    norm = np.sqrt(np.mean(q0**2))
    q0_hat = q0_hat / (norm + 1e-30)

    return band, k0, k_mag, norm, phases, q0_hat, rng


@app.cell
def _(
    K2,
    KX,
    beta,
    dt_ctrl,
    fft2,
    invert_pv,
    mo,
    np,
    nsteps_ctrl,
    nu_ctrl,
    order_ctrl,
    q0_hat,
    rk4_step,
):
    # ── Time integration ─────────────────────────────────────────────────────
    dt     = dt_ctrl.value
    nu     = nu_ctrl.value
    order  = order_ctrl.value
    nsteps = nsteps_ctrl.value

    q_hat = q0_hat.copy()
    t     = 0.0

    save_every = max(1, nsteps // 100)
    times, energy_series, enstrophy_series = [], [], []

    for step in range(nsteps):
        q_hat = rk4_step(q_hat, dt, nu, order, beta, KX, K2)
        t    += dt
        if step % save_every == 0:
            psi_hat = invert_pv(q_hat)
            Ek  = 0.5 * np.sum(K2 * np.abs(psi_hat)**2) / q_hat.size**2
            Enst = 0.5 * np.sum(np.abs(q_hat)**2) / q_hat.size**2
            times.append(t)
            energy_series.append(Ek)
            enstrophy_series.append(Enst)

    times            = np.array(times)
    energy_series    = np.array(energy_series)
    enstrophy_series = np.array(enstrophy_series)

    mo.md(f"**Integration complete.** t = {t:.3f},  steps = {nsteps},  dt = {dt}")
    return (
        Ek,
        Enst,
        dt,
        energy_series,
        enstrophy_series,
        nsteps,
        nu,
        order,
        q_hat,
        save_every,
        step,
        t,
        times,
    )


@app.cell
def _(K2, KX, KY, invert_pv, np, plt, q_hat):
    # ── Snapshot visualisation ───────────────────────────────────────────────
    from numpy.fft import ifft2 as _ifft2

    q_phys   = np.real(_ifft2(q_hat))
    psi_hat_ = invert_pv(q_hat)
    psi_phys = np.real(_ifft2(psi_hat_))

    # Azimuthally-averaged energy spectrum
    k_mag_    = np.sqrt(K2).ravel().astype(int)
    E_k_flat  = 0.5 * (K2 * np.abs(psi_hat_)**2 / q_hat.size**2).ravel()
    kmax_int  = k_mag_.max()
    E_spec    = np.bincount(k_mag_, weights=E_k_flat, minlength=kmax_int + 1)
    k_bins    = np.arange(kmax_int + 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    im0 = axes[0].imshow(q_phys, cmap='RdBu_r', origin='lower')
    axes[0].set_title('Potential Vorticity $q$')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(psi_phys, cmap='RdBu_r', origin='lower')
    axes[1].set_title('Streamfunction $\\psi$')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    k_plot = k_bins[1:kmax_int//2]
    axes[2].loglog(k_plot, E_spec[1:kmax_int//2], 'b-', linewidth=1.5, label='KE spectrum')
    # Reference slopes
    k_ref = k_plot[len(k_plot)//4 : 3*len(k_plot)//4]
    A = E_spec[k_plot[len(k_plot)//3]]
    axes[2].loglog(k_ref, A * (k_ref / k_ref[0])**(-3), 'k--', label='$k^{-3}$')
    axes[2].loglog(k_ref, A * (k_ref / k_ref[0])**(-5/3), 'r--', label='$k^{-5/3}$')
    axes[2].set_xlabel('Wavenumber $k$')
    axes[2].set_ylabel('$E(k)$')
    axes[2].set_title('KE Spectrum')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fig
    return (
        A,
        E_k_flat,
        E_spec,
        fig,
        im0,
        im1,
        k_bins,
        k_mag_,
        k_plot,
        k_ref,
        kmax_int,
        psi_hat_,
        psi_phys,
        q_phys,
    )


@app.cell
def _(energy_series, enstrophy_series, plt, times):
    fig_ts, axes_ts = plt.subplots(1, 2, figsize=(11, 3.5))

    axes_ts[0].plot(times, energy_series, 'b-', linewidth=1.5)
    axes_ts[0].set_xlabel('Time')
    axes_ts[0].set_ylabel('Kinetic Energy')
    axes_ts[0].set_title('KE vs Time')
    axes_ts[0].grid(True, alpha=0.3)

    axes_ts[1].plot(times, enstrophy_series, 'r-', linewidth=1.5)
    axes_ts[1].set_xlabel('Time')
    axes_ts[1].set_ylabel('Enstrophy')
    axes_ts[1].set_title('Enstrophy vs Time')
    axes_ts[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_ts
    return axes_ts, fig_ts


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Predictability Experiment

        Two runs are initialised from nearly identical states:
        $q_0$ (reference) and $q_0 + \varepsilon \delta q$ (perturbed),
        where $\delta q$ is a small-amplitude random perturbation.

        We track the **error energy**:
        $$E_{\text{err}}(t) = \frac{1}{2N^2} \sum_{\mathbf{k}} K^2 \, |\delta\hat{\psi}_{\mathbf{k}}|^2$$

        and estimate the **finite-time Lyapunov exponent (FTLE)**:
        $$\lambda(t) = \frac{1}{t} \ln \frac{E_{\text{err}}(t)}{E_{\text{err}}(0)}$$
        """
    )
    return


@app.cell
def _(mo):
    eps_ctrl    = mo.ui.number(start=1e-10, stop=1e-2, step=1e-10, value=1e-5,
                               label="Perturbation amplitude ε")
    pseed_ctrl  = mo.ui.number(start=0, stop=9999, step=1, value=137,
                               label="Perturbation seed")
    mo.hstack([eps_ctrl, pseed_ctrl])
    return eps_ctrl, pseed_ctrl


@app.cell
def _(
    K2,
    KX,
    dealias,
    dt,
    eps_ctrl,
    fft2,
    invert_pv,
    np,
    nsteps,
    nu,
    order,
    beta,
    pseed_ctrl,
    q0_hat,
    rk4_step,
    save_every,
):
    # ── Predictability integration ────────────────────────────────────────────
    eps   = eps_ctrl.value
    prng  = np.random.default_rng(pseed_ctrl.value)

    # Small random perturbation in spectral space
    dq_phases = prng.uniform(0, 2*np.pi, q0_hat.shape)
    dq_hat    = dealias * np.exp(1j * dq_phases)
    dq_hat   += np.conj(dq_hat[::-1, ::-1])
    dq_hat[0, 0] = 0.0
    dq_hat   /= (np.sqrt(np.sum(np.abs(dq_hat)**2)) + 1e-30)

    qref_hat  = q0_hat.copy()
    qpert_hat = q0_hat + eps * dq_hat

    pred_times, err_energy, ftle = [], [], []

    for _step in range(nsteps):
        qref_hat  = rk4_step(qref_hat,  dt, nu, order, beta, KX, K2)
        qpert_hat = rk4_step(qpert_hat, dt, nu, order, beta, KX, K2)
        _t = (_step + 1) * dt

        if _step % save_every == 0 and _t > 0:
            delta_hat = qpert_hat - qref_hat
            dpsi_hat  = invert_pv(delta_hat)
            Eerr = 0.5 * np.sum(K2 * np.abs(dpsi_hat)**2) / q0_hat.size**2
            pred_times.append(_t)
            err_energy.append(Eerr)
            if Eerr > 0:
                ftle.append(np.log(Eerr / (eps**2 + 1e-30)) / (2.0 * _t))
            else:
                ftle.append(np.nan)

    pred_times = np.array(pred_times)
    err_energy = np.array(err_energy)
    ftle       = np.array(ftle)

    return (
        Eerr,
        delta_hat,
        dq_hat,
        dq_phases,
        dpsi_hat,
        eps,
        err_energy,
        ftle,
        pred_times,
        prng,
        qpert_hat,
        qref_hat,
    )


@app.cell
def _(err_energy, ftle, plt, pred_times):
    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(10, 4))

    axes_pred[0].semilogy(pred_times, err_energy, 'b-', linewidth=1.5)
    axes_pred[0].set_xlabel('Time')
    axes_pred[0].set_ylabel('Error Energy $E_{\\rm err}$')
    axes_pred[0].set_title('Error Energy Growth')
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
def _(K2, err_energy, ftle, mo, np, pred_times):
    # Summary statistics
    lambda_inf = ftle[~np.isnan(ftle)][-10:].mean() if len(ftle) > 10 else np.nan
    t_pred     = pred_times[np.argmax(err_energy > 0.1 * err_energy[-1])] if err_energy[-1] > 0 else np.nan

    mo.md(
        f"""
        ### Summary

        | Quantity | Value |
        |---|---|
        | Asymptotic FTLE $\\lambda$ | `{lambda_inf:.4f}` |
        | Predictability time ($E_{{\\rm err}} > 0.1 E_{{\\rm err,max}}$) | `{t_pred:.3f}` |
        | Final error energy | `{err_energy[-1]:.4e}` |
        """
    )
    return lambda_inf, t_pred


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Wavenumber-Targeted Perturbation

        A perturbation is injected at a single wavenumber shell $|\mathbf{k}| = k_{\rm inj}$
        with a chosen signed amplitude. Positive amplitude adds cyclonic vorticity at that scale;
        negative adds anticyclonic. For finite amplitudes these can evolve differently due to
        nonlinearity, even though the linearised dynamics are amplitude-sign symmetric.

        We track the **spectral error KE** at multiple times:
        $$E_{\rm err}(k, t) = \frac{1}{2N^2} \sum_{|\mathbf{k}'| = k}
          K'^2 \, |\delta\hat{\psi}_{\mathbf{k}'}(t)|^2$$

        Curves are coloured from early (blue/purple) to late (yellow), showing how
        error energy spreads upscale and downscale from the injection wavenumber.
        """
    )
    return


@app.cell
def _(kmax, mo):
    k_inj_ctrl = mo.ui.slider(1, kmax - 1, step=1, value=8,
                              label="Injection wavenumber $k_{\\rm inj}$")
    wamp_ctrl  = mo.ui.number(start=-0.1, stop=0.1, step=0.001, value=1e-3,
                              label="Perturbation amplitude (signed)")
    wseed_ctrl = mo.ui.number(start=0, stop=9999, step=1, value=99,
                              label="Perturbation seed")
    mo.hstack([k_inj_ctrl, wamp_ctrl, wseed_ctrl])
    return k_inj_ctrl, wamp_ctrl, wseed_ctrl


@app.cell
def _(
    K2,
    KX,
    beta,
    dealias,
    dt,
    invert_pv,
    k_inj_ctrl,
    np,
    nsteps,
    nu,
    order,
    q0_hat,
    rk4_step,
    wamp_ctrl,
    wseed_ctrl,
):
    # ── Wavenumber-targeted perturbation integration ──────────────────────────
    k_inj  = k_inj_ctrl.value
    wamp   = wamp_ctrl.value
    K_mag_w = np.sqrt(K2)

    # Build ring-localised perturbation at wavenumber shell k_inj
    ring    = (np.abs(K_mag_w - k_inj) < 0.5).astype(float) * dealias
    wrng    = np.random.default_rng(wseed_ctrl.value)
    wph     = wrng.uniform(0, 2 * np.pi, q0_hat.shape)
    dqw_hat = ring * np.exp(1j * wph)
    dqw_hat += np.conj(dqw_hat[::-1, ::-1])
    dqw_hat[0, 0] = 0.0
    wnorm   = np.sqrt(np.sum(np.abs(dqw_hat)**2)) + 1e-30
    dqw_hat = (wamp / wnorm) * dqw_hat

    qref_w  = q0_hat.copy()
    qpert_w = q0_hat + dqw_hat

    # Save spectral error at n_snaps evenly spaced times
    n_snaps  = 8
    snap_idx = set(np.linspace(0, nsteps - 1, n_snaps, dtype=int))
    k_int_w  = K_mag_w.ravel().astype(int)
    kmax_w   = k_int_w.max()

    wsnap_times, wsnap_spectra = [], []

    for _s in range(nsteps):
        qref_w  = rk4_step(qref_w,  dt, nu, order, beta, KX, K2)
        qpert_w = rk4_step(qpert_w, dt, nu, order, beta, KX, K2)
        if _s in snap_idx:
            _delta  = qpert_w - qref_w
            _dpsi   = invert_pv(_delta)
            _Ek     = 0.5 * (K2 * np.abs(_dpsi)**2 / _delta.size**2).ravel()
            _spec   = np.bincount(k_int_w, weights=_Ek, minlength=kmax_w + 1)
            wsnap_times.append((_s + 1) * dt)
            wsnap_spectra.append(_spec)

    wsnap_times   = np.array(wsnap_times)
    wsnap_spectra = np.array(wsnap_spectra)   # shape: (n_snaps, kmax_w+1)
    k_bins_w      = np.arange(kmax_w + 1)

    return K_mag_w, dqw_hat, k_bins_w, k_inj, kmax_w, ring, wamp, wsnap_spectra, wsnap_times


@app.cell
def _(k_bins_w, k_inj, kmax_w, np, plt, wamp, wsnap_spectra, wsnap_times):
    import matplotlib.cm as _cm

    fig_wn, ax_wn = plt.subplots(figsize=(9, 4.5))
    _cmap   = _cm.plasma
    _colors = _cmap(np.linspace(0.1, 0.9, len(wsnap_times)))
    k_plot_w = k_bins_w[1:kmax_w // 2]

    for _i, (_spec, _tt) in enumerate(zip(wsnap_spectra, wsnap_times)):
        ax_wn.semilogy(
            k_plot_w, np.maximum(_spec[1:kmax_w // 2], 1e-30),
            color=_colors[_i], linewidth=1.5, label=f't = {_tt:.2f}'
        )

    ax_wn.axvline(k_inj, color='k', linestyle='--', linewidth=1.2,
                  label=f'$k_{{\\rm inj}} = {k_inj}$')
    ax_wn.set_xlabel('Wavenumber $k$')
    ax_wn.set_ylabel('Error KE Spectrum  $E_{\\rm err}(k, t)$')
    ax_wn.set_title(
        f'Spectral Error Growth  '
        f'(injection: $k = {k_inj}$,  amplitude = {wamp:.2e})'
    )
    ax_wn.legend(fontsize=7, ncol=2, loc='lower left')
    ax_wn.grid(True, which='both', alpha=0.3)

    _sm = plt.cm.ScalarMappable(
        cmap=_cmap,
        norm=plt.Normalize(wsnap_times.min(), wsnap_times.max())
    )
    plt.colorbar(_sm, ax=ax_wn, label='Time')
    plt.tight_layout()
    fig_wn
    return ax_wn, fig_wn, k_plot_w


if __name__ == "__main__":
    app.run()
