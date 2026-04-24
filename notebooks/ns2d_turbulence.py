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
        # 2D Navier-Stokes Turbulence

        Pseudospectral solver for the **incompressible 2D Navier-Stokes equations**
        in vorticity-streamfunction form on a doubly-periodic domain $[0, 2\pi)^2$:

        $$\frac{\partial \omega}{\partial t} + J(\psi, \omega)
          = \nu (-1)^{p+1} \nabla^{2p} \omega - \mu\omega + F$$

        where $\omega = \nabla^2\psi$ is the vorticity, $\mu$ is a linear drag coefficient
        (large-scale energy sink), and $F$ is a fixed stochastic forcing pattern
        at wavenumber $k_f$ that maintains a statistically steady turbulent state.

        Unlike the QG models there is **no geostrophic constraint**, **no $\beta$-plane**,
        and **no stratification** — this is the fundamental equation for 2D incompressible flow.

        **Key physics:**
        - Enstrophy cascades **forward** (small scales), spectrum $E(k) \sim k^{-3}$
        - Energy cascades **inversely** (large scales), spectrum $E(k) \sim k^{-5/3}$
        - Setting $F_0 = 0$ switches to freely **decaying** 2D turbulence

        **Numerics:** pseudospectral, 2/3-rule dealiasing, RK4 time stepping.
        """
    )
    return


@app.cell
def _(mo):
    N_ctrl      = mo.ui.slider(64, 512, step=64, value=128,
                               label="Resolution N")
    nu_ctrl     = mo.ui.number(start=1e-10, stop=1e-2, step=1e-9, value=1e-6,
                               label="Viscosity ν")
    order_ctrl  = mo.ui.slider(1, 4, step=1, value=2,
                               label="Viscosity order p")
    mu_ctrl     = mo.ui.number(start=0.0, stop=2.0, step=0.01, value=0.1,
                               label="Linear drag μ")
    kf_ctrl     = mo.ui.slider(2, 20, step=1, value=4,
                               label="Forcing wavenumber $k_f$")
    F0_ctrl     = mo.ui.number(start=0.0, stop=10.0, step=0.1, value=1.0,
                               label="Forcing amplitude $F_0$  (0 = decaying)")
    dt_ctrl     = mo.ui.number(start=1e-4, stop=0.05, step=1e-4, value=0.005,
                               label="Time step dt")
    nsteps_ctrl = mo.ui.slider(200, 10000, step=200, value=500,
                               label="Steps to run")
    seed_ctrl   = mo.ui.number(start=0, stop=9999, step=1, value=42,
                               label="Random seed")

    mo.vstack([
        mo.md("## Model Parameters"),
        mo.hstack([N_ctrl, nu_ctrl, order_ctrl, mu_ctrl]),
        mo.hstack([kf_ctrl, F0_ctrl, dt_ctrl, nsteps_ctrl, seed_ctrl]),
    ])
    return (
        F0_ctrl, N_ctrl, dt_ctrl, kf_ctrl, mu_ctrl,
        nsteps_ctrl, nu_ctrl, order_ctrl, seed_ctrl,
    )


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
    KX, KY  = np.meshgrid(kx, ky)
    K2      = KX**2 + KY**2
    K_mag   = np.sqrt(K2)
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0

    kmax    = N // 3
    dealias = ((np.abs(KX) < kmax) & (np.abs(KY) < kmax)).astype(float)

    return K2, K2_safe, KX, KY, K_mag, L, N, X, Y, dealias, kmax, x, y


@app.cell
def _(K2, K2_safe, KX, KY, dealias, fft2, ifft2, np):
    # ── Core 2D NS operators ──────────────────────────────────────────────────

    def invert_vorticity(omega_hat):
        """ω = ∇²ψ  →  ψ̂ = −ω̂ / K²  (zero mean streamfunction)."""
        psi_hat = -omega_hat / K2_safe
        psi_hat[0, 0] = 0.0
        return psi_hat

    def jacobian_hat(psi_hat, omega_hat):
        """Dealiased spectral Jacobian J(ψ, ω) = ψ_x ω_y − ψ_y ω_x."""
        psi_x = np.real(ifft2(1j * KX * psi_hat))
        psi_y = np.real(ifft2(1j * KY * psi_hat))
        om_x  = np.real(ifft2(1j * KX * omega_hat))
        om_y  = np.real(ifft2(1j * KY * omega_hat))
        return fft2(psi_x * om_y - psi_y * om_x) * dealias

    def rhs(omega_hat, F_hat, nu, order, mu):
        """RHS: −J(ψ,ω) + ν(−1)^(p+1) ∇^(2p) ω − μω + F."""
        psi_hat = invert_vorticity(omega_hat)
        jac     = jacobian_hat(psi_hat, omega_hat)
        visc    = nu * ((-1.0) ** (order + 1)) * (K2 ** order) * omega_hat
        drag    = -mu * omega_hat
        return -jac + visc + drag + F_hat

    def rk4_step(omega_hat, F_hat, dt, nu, order, mu):
        """4th-order Runge-Kutta step."""
        k1 = rhs(omega_hat,              F_hat, nu, order, mu)
        k2 = rhs(omega_hat + 0.5*dt*k1, F_hat, nu, order, mu)
        k3 = rhs(omega_hat + 0.5*dt*k2, F_hat, nu, order, mu)
        k4 = rhs(omega_hat +     dt*k3, F_hat, nu, order, mu)
        return omega_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return invert_vorticity, jacobian_hat, rhs, rk4_step


@app.cell
def _(F0_ctrl, K_mag, N, dealias, ifft2, kf_ctrl, np, seed_ctrl):
    # ── Initial condition + fixed stochastic forcing pattern ──────────────────
    rng = np.random.default_rng(seed_ctrl.value)
    k_f = kf_ctrl.value
    F0  = F0_ctrl.value

    # Initial vorticity: band-limited random field around k0 = 4
    k0   = 4
    band = np.exp(-0.5 * ((K_mag - k0) / 2.0)**2)
    ph0  = rng.uniform(0, 2 * np.pi, (N, N))
    om0_hat  = band * np.exp(1j * ph0) * dealias
    om0_hat += np.conj(om0_hat[::-1, ::-1])
    om0_hat[0, 0] = 0.0
    om0_phys = np.real(ifft2(om0_hat))
    om0_hat /= (np.sqrt(np.mean(om0_phys**2)) + 1e-30)   # unit RMS vorticity

    # Fixed forcing pattern (generated once, held constant during integration)
    f_band  = np.exp(-0.5 * ((K_mag - k_f) / 1.0)**2)
    ph_f    = rng.uniform(0, 2 * np.pi, (N, N))
    F_hat   = f_band * np.exp(1j * ph_f) * dealias
    F_hat  += np.conj(F_hat[::-1, ::-1])
    F_hat[0, 0] = 0.0
    F_hat = (F0 / (np.sqrt(np.sum(np.abs(F_hat)**2)) + 1e-30)) * F_hat

    return F_hat, band, k0, k_f, om0_hat, rng


@app.cell
def _(
    F_hat,
    K2,
    dt_ctrl,
    invert_vorticity,
    mo,
    mu_ctrl,
    np,
    nsteps_ctrl,
    nu_ctrl,
    om0_hat,
    order_ctrl,
    rk4_step,
):
    # ── Time integration ─────────────────────────────────────────────────────
    dt     = dt_ctrl.value
    nu     = nu_ctrl.value
    order  = order_ctrl.value
    mu     = mu_ctrl.value
    nsteps = nsteps_ctrl.value

    omega_hat = om0_hat.copy()
    t         = 0.0

    save_every = max(1, nsteps // 100)
    times, KE_series, Z_series = [], [], []

    for _step in range(nsteps):
        omega_hat = rk4_step(omega_hat, F_hat, dt, nu, order, mu)
        t        += dt
        if _step % save_every == 0:
            psi_hat_diag = invert_vorticity(omega_hat)
            KE = 0.5 * np.sum(K2 * np.abs(psi_hat_diag)**2) / omega_hat.size**2
            Z  = 0.5 * np.sum(np.abs(omega_hat)**2) / omega_hat.size**2
            times.append(t)
            KE_series.append(KE)
            Z_series.append(Z)

    times     = np.array(times)
    KE_series = np.array(KE_series)
    Z_series  = np.array(Z_series)

    mo.md(f"**Integration complete.** t = {t:.3f},  steps = {nsteps},  dt = {dt}")
    return (
        KE, KE_series, Z, Z_series, dt, mu, nsteps,
        nu, omega_hat, order, save_every, t, times,
    )


@app.cell
def _(K2, K_mag, invert_vorticity, np, omega_hat, plt):
    # ── Snapshot visualisation ────────────────────────────────────────────────
    from numpy.fft import ifft2 as _ifft2

    om_phys  = np.real(_ifft2(omega_hat))
    psi_hat_ = invert_vorticity(omega_hat)
    psi_phys = np.real(_ifft2(psi_hat_))

    # Azimuthally-averaged KE spectrum: E(k) = ½ k² |ψ̂|²
    k_int    = K_mag.ravel().astype(int)
    E_flat   = 0.5 * (K2 * np.abs(psi_hat_)**2 / omega_hat.size**2).ravel()
    kmax_int = k_int.max()
    E_spec   = np.bincount(k_int, weights=E_flat, minlength=kmax_int + 1)
    k_bins   = np.arange(kmax_int + 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    im0 = axes[0].imshow(om_phys, cmap='RdBu_r', origin='lower')
    axes[0].set_title('Vorticity $\\omega$')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(psi_phys, cmap='RdBu_r', origin='lower')
    axes[1].set_title('Streamfunction $\\psi$')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    k_plot = k_bins[1:kmax_int // 2]
    axes[2].loglog(k_plot, E_spec[1:kmax_int // 2], 'b-', linewidth=1.5, label='KE spectrum')
    k_ref = k_plot[len(k_plot)//4 : 3*len(k_plot)//4]
    A = E_spec[k_plot[len(k_plot)//3]] + 1e-30
    axes[2].loglog(k_ref, A * (k_ref / k_ref[0])**(-3),   'k--', label='$k^{-3}$')
    axes[2].loglog(k_ref, A * (k_ref / k_ref[0])**(-5/3), 'r--', label='$k^{-5/3}$')
    axes[2].set_xlabel('Wavenumber $k$'); axes[2].set_ylabel('$E(k)$')
    axes[2].set_title('KE Spectrum')
    axes[2].legend(fontsize=8); axes[2].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fig
    return (
        A, E_flat, E_spec, fig, im0, im1, k_bins, k_int, k_plot, k_ref,
        kmax_int, om_phys, psi_hat_, psi_phys,
    )


@app.cell
def _(KE_series, Z_series, plt, times):
    fig_ts, axes_ts = plt.subplots(1, 2, figsize=(10, 3.5))

    axes_ts[0].plot(times, KE_series, 'b-', linewidth=1.5)
    axes_ts[0].set_xlabel('Time'); axes_ts[0].set_ylabel('Kinetic Energy')
    axes_ts[0].set_title('KE vs Time'); axes_ts[0].grid(True, alpha=0.3)

    axes_ts[1].plot(times, Z_series, 'r-', linewidth=1.5)
    axes_ts[1].set_xlabel('Time'); axes_ts[1].set_ylabel('Enstrophy')
    axes_ts[1].set_title('Enstrophy vs Time'); axes_ts[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_ts
    return axes_ts, fig_ts


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Predictability Experiment

        Two 2D NS runs with the **same forcing** $F$ but slightly different initial conditions:
        $\omega_0$ (reference) and $\omega_0 + \varepsilon\,\delta\omega$ (perturbed).

        Using the same forcing in both runs isolates sensitivity to initial conditions
        from sensitivity to external input — the classical definition of predictability.

        **Error KE:**
        $$E_{\rm err}(t) = \frac{1}{2N^2} \sum_\mathbf{k} K^2 |\delta\hat{\psi}_\mathbf{k}|^2$$

        **FTLE:** $\lambda(t) = \frac{1}{2t} \ln \frac{E_{\rm err}(t)}{\varepsilon^2}$

        In forced 2D turbulence, the predictability horizon is set by the
        enstrophy injection rate at forcing scale $k_f$.
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
    F_hat,
    K2,
    dealias,
    dt,
    eps_ctrl,
    invert_vorticity,
    mo,
    mu,
    np,
    nsteps,
    nu,
    om0_hat,
    order,
    pseed_ctrl,
    rk4_step,
    save_every,
):
    # ── Predictability integration ────────────────────────────────────────────
    eps  = eps_ctrl.value
    prng = np.random.default_rng(pseed_ctrl.value)

    dph     = prng.uniform(0, 2 * np.pi, om0_hat.shape)
    dom_hat = dealias * np.exp(1j * dph)
    dom_hat += np.conj(dom_hat[::-1, ::-1])
    dom_hat[0, 0] = 0.0
    dom_hat /= (np.sqrt(np.sum(np.abs(dom_hat)**2)) + 1e-30)

    omref_hat  = om0_hat.copy()
    ompert_hat = om0_hat + eps * dom_hat

    pred_times, err_energy, ftle = [], [], []

    for _s in range(nsteps):
        omref_hat  = rk4_step(omref_hat,  F_hat, dt, nu, order, mu)
        ompert_hat = rk4_step(ompert_hat, F_hat, dt, nu, order, mu)
        _t = (_s + 1) * dt

        if _s % save_every == 0 and _t > 0:
            _delta  = ompert_hat - omref_hat
            _dpsi   = invert_vorticity(_delta)
            Eerr    = 0.5 * np.sum(K2 * np.abs(_dpsi)**2) / om0_hat.size**2
            pred_times.append(_t)
            err_energy.append(Eerr)
            ftle.append(
                np.log(Eerr / (eps**2 + 1e-30)) / (2.0 * _t) if Eerr > 0 else np.nan
            )

    pred_times = np.array(pred_times)
    err_energy = np.array(err_energy)
    ftle       = np.array(ftle)

    return (
        Eerr, dom_hat, dph, eps, err_energy, ftle,
        ompert_hat, omref_hat, pred_times, prng,
    )


@app.cell
def _(err_energy, ftle, plt, pred_times):
    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(10, 4))

    axes_pred[0].semilogy(pred_times, err_energy, 'b-', linewidth=1.5)
    axes_pred[0].set_xlabel('Time')
    axes_pred[0].set_ylabel('Error KE  $E_{\\rm err}$')
    axes_pred[0].set_title('Error Energy Growth')
    axes_pred[0].grid(True, which='both', alpha=0.3)

    axes_pred[1].plot(pred_times, ftle, 'r-', linewidth=1.5)
    axes_pred[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes_pred[1].set_xlabel('Time')
    axes_pred[1].set_ylabel('FTLE  $\\lambda(t)$')
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Wavenumber-Targeted Perturbation

        A perturbation is injected at a single wavenumber shell $|\mathbf{k}| = k_{\rm inj}$
        with a chosen signed amplitude. Positive amplitude adds cyclonic vorticity
        at that scale; negative adds anticyclonic. Both runs use the same forcing $F$.

        We track the **spectral error KE** at multiple times:
        $$E_{\rm err}(k, t) = \frac{1}{2N^2} \sum_{|\mathbf{k}'| = k}
          K'^2 \, |\delta\hat{\psi}_{\mathbf{k}'}(t)|^2$$

        Curves coloured blue/purple (early) → yellow (late) show whether errors
        spread preferentially toward large scales (inverse cascade) or
        small scales (enstrophy cascade) from the injection point.
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
    F_hat,
    K2,
    K_mag,
    dealias,
    dt,
    invert_vorticity,
    k_inj_ctrl,
    mu,
    np,
    nsteps,
    nu,
    om0_hat,
    order,
    rk4_step,
    wamp_ctrl,
    wseed_ctrl,
):
    # ── Wavenumber-targeted perturbation integration ──────────────────────────
    k_inj  = k_inj_ctrl.value
    wamp   = wamp_ctrl.value

    ring    = (np.abs(K_mag - k_inj) < 0.5).astype(float) * dealias
    wrng    = np.random.default_rng(wseed_ctrl.value)
    wph     = wrng.uniform(0, 2 * np.pi, om0_hat.shape)
    dom_w   = ring * np.exp(1j * wph)
    dom_w  += np.conj(dom_w[::-1, ::-1])
    dom_w[0, 0] = 0.0
    dom_w = (wamp / (np.sqrt(np.sum(np.abs(dom_w)**2)) + 1e-30)) * dom_w

    omref_w  = om0_hat.copy()
    ompert_w = om0_hat + dom_w

    n_snaps  = 8
    snap_idx = set(np.linspace(0, nsteps - 1, n_snaps, dtype=int))
    k_int_w  = K_mag.ravel().astype(int)
    kmax_w   = k_int_w.max()

    wsnap_times, wsnap_spectra = [], []

    for _s in range(nsteps):
        omref_w  = rk4_step(omref_w,  F_hat, dt, nu, order, mu)
        ompert_w = rk4_step(ompert_w, F_hat, dt, nu, order, mu)
        if _s in snap_idx:
            _delta = ompert_w - omref_w
            _dpsi  = invert_vorticity(_delta)
            _Ek    = 0.5 * (K2 * np.abs(_dpsi)**2 / _delta.size**2).ravel()
            _spec  = np.bincount(k_int_w, weights=_Ek, minlength=kmax_w + 1)
            wsnap_times.append((_s + 1) * dt)
            wsnap_spectra.append(_spec)

    wsnap_times   = np.array(wsnap_times)
    wsnap_spectra = np.array(wsnap_spectra)
    k_bins_w      = np.arange(kmax_w + 1)

    return dom_w, k_bins_w, k_inj, kmax_w, ring, wamp, wsnap_spectra, wsnap_times


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
