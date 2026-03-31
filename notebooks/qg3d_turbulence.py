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
        # 3D Quasi-Geostrophic Turbulence — 2-Layer Model

        Pseudospectral solver for the **2-layer quasi-geostrophic** equations
        on a doubly-periodic domain $[0, 2\pi)^2$, representing stratified 3D QG flow.

        **Governing equations** (layer $n = 1, 2$):
        $$\frac{\partial q_n}{\partial t} + J(\psi_n, q_n) = \text{dissipation}$$

        **Potential vorticity:**
        $$q_1 = \nabla^2 \psi_1 + F_1(\psi_2 - \psi_1) + \beta y$$
        $$q_2 = \nabla^2 \psi_2 + F_2(\psi_1 - \psi_2) + \beta y$$

        where $F_n = f_0^2 / (g' H_n)$ is the inverse squared Rossby radius for layer $n$,
        $g'$ is the reduced gravity, and $H_n$ is the layer depth.

        **Dissipation:** hyperviscosity $\nu(-1)^{p+1}\nabla^{2p}$ in both layers;
        linear bottom drag $-\kappa \nabla^2 \psi_2$ in layer 2.

        **Numerics:** pseudospectral with 2/3-rule dealiasing, RK4 time stepping.
        """
    )
    return


@app.cell
def _(mo):
    N_ctrl      = mo.ui.slider(64, 512, step=64, value=128, label="Resolution N")
    F1_ctrl     = mo.ui.number(start=0.0, stop=100.0, step=0.5, value=10.0,
                               label="F₁ = f₀²/(g′H₁)")
    F2_ctrl     = mo.ui.number(start=0.0, stop=100.0, step=0.5, value=10.0,
                               label="F₂ = f₀²/(g′H₂)")
    beta_ctrl   = mo.ui.number(start=0.0, stop=20.0, step=0.5, value=0.0,
                               label="β")
    nu_ctrl     = mo.ui.number(start=1e-10, stop=1e-3, step=1e-9, value=1e-6,
                               label="Hyperviscosity ν")
    order_ctrl  = mo.ui.slider(1, 4, step=1, value=2, label="Order p")
    kappa_ctrl  = mo.ui.number(start=0.0, stop=1.0, step=0.01, value=0.1,
                               label="Bottom drag κ")
    dt_ctrl     = mo.ui.number(start=1e-4, stop=0.05, step=1e-4, value=0.005,
                               label="Time step dt")
    nsteps_ctrl = mo.ui.slider(200, 10000, step=200, value=2000,
                               label="Steps to run")
    seed_ctrl   = mo.ui.number(start=0, stop=9999, step=1, value=42,
                               label="Random seed")

    mo.vstack([
        mo.md("## Model Parameters"),
        mo.hstack([N_ctrl, F1_ctrl, F2_ctrl, beta_ctrl]),
        mo.hstack([nu_ctrl, order_ctrl, kappa_ctrl]),
        mo.hstack([dt_ctrl, nsteps_ctrl, seed_ctrl]),
    ])
    return (
        F1_ctrl,
        F2_ctrl,
        N_ctrl,
        beta_ctrl,
        dt_ctrl,
        kappa_ctrl,
        nsteps_ctrl,
        nu_ctrl,
        order_ctrl,
        seed_ctrl,
    )


@app.cell
def _(F1_ctrl, F2_ctrl, N_ctrl, beta_ctrl, fftfreq, np):
    # ── Grid and spectral operators ───────────────────────────────────────────
    N    = N_ctrl.value
    L    = 2.0 * np.pi
    F1   = F1_ctrl.value
    F2   = F2_ctrl.value
    beta = beta_ctrl.value

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    kx = fftfreq(N, d=1.0 / N)
    ky = fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky)
    K2      = KX**2 + KY**2

    kmax    = N // 3
    dealias = ((np.abs(KX) < kmax) & (np.abs(KY) < kmax)).astype(float)

    return F1, F2, K2, KX, KY, L, N, X, Y, beta, dealias, kmax, x, y


@app.cell
def _(F1, F2, K2, KX, KY, dealias, fft2, ifft2, np):
    # ── Spectral PV inversion and Jacobian ───────────────────────────────────

    def invert_pv_2layer(q1_hat, q2_hat):
        """
        Solve for ψ₁, ψ₂ given q₁, q₂ via the 2-layer PV relations.

        In spectral space, for each wavenumber K²:
            [-(K²+F1)   F1 ] [ψ̂₁]   [q̂₁]
            [  F2  -(K²+F2)] [ψ̂₂] = [q̂₂]

        det = K²(K² + F1 + F2)
        """
        det = K2 * (K2 + F1 + F2)
        det_safe = det.copy()
        det_safe[0, 0] = 1.0  # handle k=0 separately

        psi1_hat = (-(K2 + F2) * q1_hat - F1 * q2_hat) / det_safe
        psi2_hat = (-F2 * q1_hat - (K2 + F1) * q2_hat) / det_safe

        # Zero mean streamfunction
        psi1_hat[0, 0] = 0.0
        psi2_hat[0, 0] = 0.0
        return psi1_hat, psi2_hat

    def jacobian_hat(psi_hat, q_hat):
        """Dealiased spectral Jacobian J(ψ,q)."""
        psi_x = np.real(ifft2(1j * KX * psi_hat))
        psi_y = np.real(ifft2(1j * KY * psi_hat))
        q_x   = np.real(ifft2(1j * KX * q_hat))
        q_y   = np.real(ifft2(1j * KY * q_hat))
        return fft2(psi_x * q_y - psi_y * q_x) * dealias

    return invert_pv_2layer, jacobian_hat


@app.cell
def _(
    F1,
    F2,
    K2,
    KX,
    beta,
    invert_pv_2layer,
    jacobian_hat,
    kappa_ctrl,
    np,
    nu_ctrl,
    order_ctrl,
):
    # ── RHS and RK4 ──────────────────────────────────────────────────────────
    nu    = nu_ctrl.value
    order = order_ctrl.value
    kappa = kappa_ctrl.value

    def rhs_2layer(q1_hat, q2_hat):
        """RHS for 2-layer QG: returns dq1/dt, dq2/dt in spectral space."""
        psi1_hat, psi2_hat = invert_pv_2layer(q1_hat, q2_hat)

        # Jacobians
        J1 = jacobian_hat(psi1_hat, q1_hat)
        J2 = jacobian_hat(psi2_hat, q2_hat)

        # Beta terms: -β ∂ψ/∂x
        beta1 = -beta * (1j * KX * psi1_hat)
        beta2 = -beta * (1j * KX * psi2_hat)

        # Hyperviscosity
        diss = nu * ((-1.0) ** (order + 1)) * K2**order
        d1   = diss * q1_hat
        d2   = diss * q2_hat

        # Bottom drag on layer 2 (acts on relative vorticity)
        drag2 = -kappa * K2 * psi2_hat

        dq1dt = -J1 + beta1 + d1
        dq2dt = -J2 + beta2 + d2 + drag2
        return dq1dt, dq2dt

    def rk4_step_2layer(q1_hat, q2_hat, dt):
        """RK4 step for 2-layer system."""
        k1a, k1b = rhs_2layer(q1_hat, q2_hat)
        k2a, k2b = rhs_2layer(q1_hat + 0.5*dt*k1a, q2_hat + 0.5*dt*k1b)
        k3a, k3b = rhs_2layer(q1_hat + 0.5*dt*k2a, q2_hat + 0.5*dt*k2b)
        k4a, k4b = rhs_2layer(q1_hat +     dt*k3a, q2_hat +     dt*k3b)
        q1_new = q1_hat + (dt/6.0)*(k1a + 2*k2a + 2*k3a + k4a)
        q2_new = q2_hat + (dt/6.0)*(k1b + 2*k2b + 2*k3b + k4b)
        return q1_new, q2_new

    return kappa, nu, order, rhs_2layer, rk4_step_2layer


@app.cell
def _(K2, N, dealias, fft2, np, seed_ctrl):
    # ── Initial conditions ───────────────────────────────────────────────────
    rng  = np.random.default_rng(seed_ctrl.value)
    k0   = 4
    k_mag = np.sqrt(K2)
    band  = np.exp(-0.5 * ((k_mag - k0) / 2.0)**2)

    def make_ic(rng_in):
        phases = rng_in.uniform(0, 2*np.pi, (N, N))
        qh     = band * np.exp(1j * phases) * dealias
        qh    += np.conj(qh[::-1, ::-1])
        qh[0, 0] = 0.0
        q_phys = np.real(fft2.__module__ and __import__('numpy.fft', fromlist=['ifft2']).ifft2(qh))
        norm   = np.sqrt(np.mean(q_phys**2)) + 1e-30
        return qh / norm

    q1_0_hat = make_ic(rng)
    q2_0_hat = make_ic(rng)

    return band, k0, k_mag, make_ic, q1_0_hat, q2_0_hat, rng


@app.cell
def _(
    K2,
    dt_ctrl,
    invert_pv_2layer,
    mo,
    np,
    nsteps_ctrl,
    q1_0_hat,
    q2_0_hat,
    rk4_step_2layer,
):
    # ── Time integration ─────────────────────────────────────────────────────
    dt     = dt_ctrl.value
    nsteps = nsteps_ctrl.value

    q1_hat = q1_0_hat.copy()
    q2_hat = q2_0_hat.copy()
    t      = 0.0

    save_every = max(1, nsteps // 100)
    times, KE1_series, KE2_series, Enst1_series, Enst2_series = [], [], [], [], []

    for _step in range(nsteps):
        q1_hat, q2_hat = rk4_step_2layer(q1_hat, q2_hat, dt)
        t += dt
        if _step % save_every == 0:
            psi1h, psi2h = invert_pv_2layer(q1_hat, q2_hat)
            KE1  = 0.5 * np.sum(K2 * np.abs(psi1h)**2) / q1_hat.size**2
            KE2  = 0.5 * np.sum(K2 * np.abs(psi2h)**2) / q1_hat.size**2
            En1  = 0.5 * np.sum(np.abs(q1_hat)**2) / q1_hat.size**2
            En2  = 0.5 * np.sum(np.abs(q2_hat)**2) / q1_hat.size**2
            times.append(t)
            KE1_series.append(KE1); KE2_series.append(KE2)
            Enst1_series.append(En1); Enst2_series.append(En2)

    times        = np.array(times)
    KE1_series   = np.array(KE1_series)
    KE2_series   = np.array(KE2_series)
    Enst1_series = np.array(Enst1_series)
    Enst2_series = np.array(Enst2_series)

    mo.md(f"**Integration complete.** t = {t:.3f},  steps = {nsteps},  dt = {dt}")
    return (
        KE1,
        KE1_series,
        KE2,
        KE2_series,
        En1,
        En2,
        Enst1_series,
        Enst2_series,
        dt,
        nsteps,
        psi1h,
        psi2h,
        q1_hat,
        q2_hat,
        save_every,
        t,
        times,
    )


@app.cell
def _(K2, invert_pv_2layer, np, plt, q1_hat, q2_hat):
    from numpy.fft import ifft2 as _ifft2

    psi1_f, psi2_f = invert_pv_2layer(q1_hat, q2_hat)
    q1_phys = np.real(_ifft2(q1_hat))
    q2_phys = np.real(_ifft2(q2_hat))

    # Barotropic / baroclinic decomposition
    psi_bt = 0.5 * (psi1_f + psi2_f)   # barotropic mode
    psi_bc = 0.5 * (psi1_f - psi2_f)   # baroclinic mode

    # Energy spectra
    k_mag_int = np.sqrt(K2).ravel().astype(int)
    kmax_int  = k_mag_int.max()

    def spectrum(psi_hat):
        E_flat = 0.5 * (K2 * np.abs(psi_hat)**2 / psi_hat.size**2).ravel()
        return np.bincount(k_mag_int, weights=E_flat, minlength=kmax_int + 1)

    E_bt   = spectrum(psi_bt)
    E_bc   = spectrum(psi_bc)
    k_bins = np.arange(kmax_int + 1)

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))

    # Layer 1 PV
    im00 = axes2[0,0].imshow(q1_phys, cmap='RdBu_r', origin='lower')
    axes2[0,0].set_title('Layer 1: PV $q_1$')
    plt.colorbar(im00, ax=axes2[0,0])

    # Layer 2 PV
    im10 = axes2[1,0].imshow(q2_phys, cmap='RdBu_r', origin='lower')
    axes2[1,0].set_title('Layer 2: PV $q_2$')
    plt.colorbar(im10, ax=axes2[1,0])

    # Barotropic streamfunction
    bt_phys = np.real(_ifft2(psi_bt))
    im01 = axes2[0,1].imshow(bt_phys, cmap='RdBu_r', origin='lower')
    axes2[0,1].set_title('Barotropic $\\psi_{bt}$')
    plt.colorbar(im01, ax=axes2[0,1])

    # Baroclinic streamfunction
    bc_phys = np.real(_ifft2(psi_bc))
    im11 = axes2[1,1].imshow(bc_phys, cmap='RdBu_r', origin='lower')
    axes2[1,1].set_title('Baroclinic $\\psi_{bc}$')
    plt.colorbar(im11, ax=axes2[1,1])

    # KE spectra
    k_plot = k_bins[1:kmax_int//2]
    axes2[0,2].loglog(k_plot, E_bt[1:kmax_int//2], 'b-', linewidth=1.5, label='Barotropic')
    axes2[0,2].loglog(k_plot, E_bc[1:kmax_int//2], 'r-', linewidth=1.5, label='Baroclinic')
    k_ref2 = k_plot[len(k_plot)//4 : 3*len(k_plot)//4]
    A2 = E_bt[k_plot[len(k_plot)//3]] + 1e-30
    axes2[0,2].loglog(k_ref2, A2 * (k_ref2/k_ref2[0])**(-3), 'k--', label='$k^{-3}$')
    axes2[0,2].set_xlabel('Wavenumber $k$')
    axes2[0,2].set_ylabel('$E(k)$')
    axes2[0,2].set_title('KE Spectra')
    axes2[0,2].legend(fontsize=8); axes2[0,2].grid(True, which='both', alpha=0.3)

    axes2[1,2].axis('off')

    plt.tight_layout()
    fig2
    return (
        A2,
        E_bc,
        E_bt,
        bc_phys,
        bt_phys,
        fig2,
        im00,
        im01,
        im10,
        im11,
        k_bins,
        k_mag_int,
        k_plot,
        k_ref2,
        kmax_int,
        psi1_f,
        psi2_f,
        psi_bc,
        psi_bt,
        q1_phys,
        q2_phys,
        spectrum,
    )


@app.cell
def _(
    KE1_series,
    KE2_series,
    Enst1_series,
    Enst2_series,
    plt,
    times,
):
    fig_ts2, axes_ts2 = plt.subplots(1, 2, figsize=(11, 3.5))

    axes_ts2[0].plot(times, KE1_series, 'b-', linewidth=1.5, label='Layer 1')
    axes_ts2[0].plot(times, KE2_series, 'r-', linewidth=1.5, label='Layer 2')
    axes_ts2[0].set_xlabel('Time')
    axes_ts2[0].set_ylabel('Kinetic Energy')
    axes_ts2[0].set_title('KE vs Time')
    axes_ts2[0].legend(); axes_ts2[0].grid(True, alpha=0.3)

    axes_ts2[1].plot(times, Enst1_series, 'b-', linewidth=1.5, label='Layer 1')
    axes_ts2[1].plot(times, Enst2_series, 'r-', linewidth=1.5, label='Layer 2')
    axes_ts2[1].set_xlabel('Time')
    axes_ts2[1].set_ylabel('Enstrophy')
    axes_ts2[1].set_title('Enstrophy vs Time')
    axes_ts2[1].legend(); axes_ts2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_ts2
    return axes_ts2, fig_ts2


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Predictability Experiment

        Two 2-layer runs from nearly identical initial states.
        We track **layer-wise and total error energy**:
        $$E_{\text{err},n}(t) = \frac{1}{2N^2}\sum_\mathbf{k} K^2 |\delta\hat{\psi}_{n,\mathbf{k}}|^2$$
        and the **barotropic/baroclinic error decomposition**:
        $$E_{\text{bt}} = \tfrac{1}{2}(E_{\text{err,1}} + E_{\text{err,2}}), \quad
          E_{\text{bc}} = \tfrac{1}{2}(E_{\text{err,1}} + E_{\text{err,2}}) - \text{cross terms}$$
        """
    )
    return


@app.cell
def _(mo):
    eps2_ctrl   = mo.ui.number(start=1e-10, stop=1e-2, step=1e-10, value=1e-5,
                               label="Perturbation amplitude ε")
    pseed2_ctrl = mo.ui.number(start=0, stop=9999, step=1, value=137,
                               label="Perturbation seed")
    mo.hstack([eps2_ctrl, pseed2_ctrl])
    return eps2_ctrl, pseed2_ctrl


@app.cell
def _(
    K2,
    dealias,
    dt,
    eps2_ctrl,
    invert_pv_2layer,
    np,
    nsteps,
    pseed2_ctrl,
    q1_0_hat,
    q2_0_hat,
    rk4_step_2layer,
    save_every,
):
    eps2  = eps2_ctrl.value
    prng2 = np.random.default_rng(pseed2_ctrl.value)

    def _rand_perturb(rng_in, shape):
        ph = rng_in.uniform(0, 2*np.pi, shape)
        dh = dealias * np.exp(1j * ph)
        dh += np.conj(dh[::-1, ::-1]); dh[0,0] = 0.0
        return dh / (np.sqrt(np.sum(np.abs(dh)**2)) + 1e-30)

    dq1_hat = _rand_perturb(prng2, q1_0_hat.shape)
    dq2_hat = _rand_perturb(prng2, q2_0_hat.shape)

    q1r_hat = q1_0_hat.copy()
    q2r_hat = q2_0_hat.copy()
    q1p_hat = q1_0_hat + eps2 * dq1_hat
    q2p_hat = q2_0_hat + eps2 * dq2_hat

    pred_times2, err1_series, err2_series, ftle2 = [], [], [], []

    for _s in range(nsteps):
        q1r_hat, q2r_hat = rk4_step_2layer(q1r_hat, q2r_hat, dt)
        q1p_hat, q2p_hat = rk4_step_2layer(q1p_hat, q2p_hat, dt)
        _t2 = (_s + 1) * dt

        if _s % save_every == 0 and _t2 > 0:
            d1h = q1p_hat - q1r_hat
            d2h = q2p_hat - q2r_hat
            dp1h, dp2h = invert_pv_2layer(d1h, d2h)
            Ee1 = 0.5 * np.sum(K2 * np.abs(dp1h)**2) / q1_0_hat.size**2
            Ee2 = 0.5 * np.sum(K2 * np.abs(dp2h)**2) / q1_0_hat.size**2
            Etot = Ee1 + Ee2
            pred_times2.append(_t2)
            err1_series.append(Ee1)
            err2_series.append(Ee2)
            if Etot > 0:
                ftle2.append(np.log(Etot / (2 * eps2**2 + 1e-30)) / (2.0 * _t2))
            else:
                ftle2.append(np.nan)

    pred_times2 = np.array(pred_times2)
    err1_series = np.array(err1_series)
    err2_series = np.array(err2_series)
    ftle2       = np.array(ftle2)

    return (
        Ee1,
        Ee2,
        Etot,
        d1h,
        d2h,
        dp1h,
        dp2h,
        dq1_hat,
        dq2_hat,
        eps2,
        err1_series,
        err2_series,
        ftle2,
        pred_times2,
        prng2,
        q1p_hat,
        q1r_hat,
        q2p_hat,
        q2r_hat,
    )


@app.cell
def _(err1_series, err2_series, ftle2, np, plt, pred_times2):
    fig_pred2, axes_pred2 = plt.subplots(1, 2, figsize=(11, 4))

    axes_pred2[0].semilogy(pred_times2, err1_series, 'b-', linewidth=1.5, label='Layer 1')
    axes_pred2[0].semilogy(pred_times2, err2_series, 'r-', linewidth=1.5, label='Layer 2')
    axes_pred2[0].semilogy(pred_times2, err1_series + err2_series, 'k-',
                           linewidth=2, label='Total')
    axes_pred2[0].set_xlabel('Time')
    axes_pred2[0].set_ylabel('Error Energy')
    axes_pred2[0].set_title('Layer-wise Error Energy Growth')
    axes_pred2[0].legend(); axes_pred2[0].grid(True, which='both', alpha=0.3)

    axes_pred2[1].plot(pred_times2, ftle2, 'r-', linewidth=1.5)
    axes_pred2[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes_pred2[1].set_xlabel('Time')
    axes_pred2[1].set_ylabel('FTLE $\\lambda(t)$')
    axes_pred2[1].set_title('Finite-Time Lyapunov Exponent')
    axes_pred2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_pred2
    return axes_pred2, fig_pred2


@app.cell
def _(err1_series, err2_series, ftle2, mo, np, pred_times2):
    _ftle_tail = ftle2[~np.isnan(ftle2)]
    lambda_inf2 = _ftle_tail[-10:].mean() if len(_ftle_tail) > 10 else np.nan
    err_tot     = err1_series + err2_series

    mo.md(
        f"""
        ### Summary

        | Quantity | Value |
        |---|---|
        | Asymptotic FTLE $\\lambda$ | `{lambda_inf2:.4f}` |
        | Final Layer 1 error energy | `{err1_series[-1]:.4e}` |
        | Final Layer 2 error energy | `{err2_series[-1]:.4e}` |
        | Final total error energy | `{err_tot[-1]:.4e}` |
        """
    )
    return err_tot, lambda_inf2


if __name__ == "__main__":
    app.run()
