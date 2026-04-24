import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Lorenz (1996) Model

        A cyclic system of $K$ variables designed by Lorenz to mimic the
        large-scale behaviour of a single atmospheric quantity around a latitude circle:

        $$\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\,x_{i-1} - x_i + F,
          \qquad i = 1, \ldots, K \quad (\text{periodic})$$

        The nonlinear term represents advection, $-x_i$ is damping, and $F$ is
        a constant large-scale forcing. The model transitions from steady ($F \lesssim 5$)
        to fully chaotic ($F = 8$, the standard benchmark) behaviour.

        **Key quantities:**
        - Total energy: $E = \tfrac{1}{2K}\sum_i x_i^2$
        - Fourier spectrum: $E(k) = \tfrac{1}{2K}|\hat{x}_k|^2$,
          revealing the spatial scales of the dominant variability
        - Leading Lyapunov exponent scales as $\lambda_1 \propto F^{2/3}$

        **Numerics:** 4th-order Runge-Kutta (RK4), fixed time step.
        """
    )
    return


@app.cell
def _(mo):
    K_ctrl      = mo.ui.slider(8, 80, step=4, value=36,
                               label="Number of variables K")
    F_ctrl      = mo.ui.number(start=0.1, stop=20.0, step=0.1, value=8.0,
                               label="Forcing F  (chaos onset ≈ 5.8)")
    dt_ctrl     = mo.ui.number(start=1e-4, stop=0.1, step=1e-4, value=0.01,
                               label="Time step dt")
    nsteps_ctrl = mo.ui.slider(200, 10000, step=200, value=2000,
                               label="Steps to run")
    seed_ctrl   = mo.ui.number(start=0, stop=9999, step=1, value=42,
                               label="Random seed")

    mo.vstack([
        mo.md("## Model Parameters"),
        mo.hstack([K_ctrl, F_ctrl, dt_ctrl, nsteps_ctrl, seed_ctrl]),
    ])
    return F_ctrl, K_ctrl, dt_ctrl, nsteps_ctrl, seed_ctrl


@app.cell
def _(np):
    # ── L96 dynamics ──────────────────────────────────────────────────────────

    def l96_rhs(x, F):
        """L96 right-hand side: (x_{i+1} − x_{i-2}) x_{i-1} − x_i + F."""
        return (
            (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
        )

    def rk4_step(x, dt, F):
        """4th-order Runge-Kutta step."""
        k1 = l96_rhs(x,              F)
        k2 = l96_rhs(x + 0.5*dt*k1, F)
        k3 = l96_rhs(x + 0.5*dt*k2, F)
        k4 = l96_rhs(x +     dt*k3, F)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return l96_rhs, rk4_step


@app.cell
def _(F_ctrl, K_ctrl, np, seed_ctrl):
    # ── Initial condition ─────────────────────────────────────────────────────
    K   = K_ctrl.value
    F   = F_ctrl.value
    rng = np.random.default_rng(seed_ctrl.value)

    # Standard L96 IC: all equal to F, one perturbed slightly
    x0      = np.full(K, F, dtype=float)
    x0[K//4] += 0.01 * rng.standard_normal()

    return F, K, rng, x0


@app.cell
def _(F, K, dt_ctrl, mo, np, nsteps_ctrl, rk4_step, x0):
    # ── Time integration with Hovmöller storage ───────────────────────────────
    dt     = dt_ctrl.value
    nsteps = nsteps_ctrl.value

    save_every  = max(1, nsteps // 200)
    n_saves     = nsteps // save_every + 1
    hover_data  = np.empty((n_saves, K))    # space-time array for Hovmöller
    hover_times = np.empty(n_saves)
    energy_ts   = []
    times_ts    = []

    x     = x0.copy()
    i_sv  = 0
    hover_data[0]  = x
    hover_times[0] = 0.0

    for _s in range(nsteps):
        x = rk4_step(x, dt, F)
        if (_s + 1) % save_every == 0 and i_sv + 1 < n_saves:
            i_sv += 1
            hover_data[i_sv]  = x
            hover_times[i_sv] = (_s + 1) * dt
        energy_ts.append(0.5 * np.mean(x**2))
        times_ts.append((_s + 1) * dt)

    x_final   = x
    energy_ts = np.array(energy_ts)
    times_ts  = np.array(times_ts)
    hover_data  = hover_data[:i_sv + 1]
    hover_times = hover_times[:i_sv + 1]

    mo.md(
        f"**Integration complete.** "
        f"t = {nsteps * dt:.2f},  steps = {nsteps},  K = {K},  F = {F}"
    )
    return (
        dt, energy_ts, hover_data, hover_times,
        i_sv, n_saves, nsteps, save_every, times_ts, x_final,
    )


@app.cell
def _(K, hover_data, hover_times, np, plt):
    # ── Hovmöller diagram ─────────────────────────────────────────────────────
    fig_hov, ax_hov = plt.subplots(figsize=(9, 4))
    im_hov = ax_hov.pcolormesh(
        np.arange(1, K + 1), hover_times,
        hover_data, cmap='RdBu_r', shading='auto'
    )
    ax_hov.set_xlabel('Variable index $i$')
    ax_hov.set_ylabel('Time')
    ax_hov.set_title('Hovmöller Diagram  $x_i(t)$')
    plt.colorbar(im_hov, ax=ax_hov, label='$x_i$')
    plt.tight_layout()
    fig_hov
    return ax_hov, fig_hov, im_hov


@app.cell
def _(K, energy_ts, np, plt, times_ts, x_final):
    # ── Energy time series + final-state Fourier spectrum ────────────────────
    fig_diag, axes_diag = plt.subplots(1, 2, figsize=(10, 3.5))

    axes_diag[0].plot(times_ts, energy_ts, 'b-', linewidth=1.0)
    axes_diag[0].set_xlabel('Time'); axes_diag[0].set_ylabel('Energy  $E$')
    axes_diag[0].set_title('Energy vs Time'); axes_diag[0].grid(True, alpha=0.3)

    # Fourier spectrum of final state
    x_hat  = np.fft.rfft(x_final) / np.sqrt(K)
    k_bins = np.arange(len(x_hat))
    E_spec = 0.5 * np.abs(x_hat)**2
    axes_diag[1].semilogy(k_bins[1:], E_spec[1:], 'b-o', markersize=3, linewidth=1.2)
    axes_diag[1].set_xlabel('Wavenumber $k$'); axes_diag[1].set_ylabel('$E(k)$')
    axes_diag[1].set_title('Fourier Energy Spectrum (final state)')
    axes_diag[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fig_diag
    return E_spec, axes_diag, fig_diag, k_bins, x_hat


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Predictability Experiment

        Two L96 runs from nearly identical initial conditions:
        $\mathbf{x}_0$ (reference) and $\mathbf{x}_0 + \varepsilon\,\delta\mathbf{x}$
        (perturbed), where $\delta\mathbf{x}$ is a broadband random perturbation.

        **Error energy:**
        $$E_{\rm err}(t) = \frac{1}{K} \sum_{i=1}^K \delta x_i(t)^2$$

        **FTLE:** $\lambda(t) = \dfrac{1}{2t} \ln \dfrac{E_{\rm err}(t)}{\varepsilon^2}$

        For $F = 8$, the leading Lyapunov exponent is $\lambda_1 \approx 1.68$,
        giving a predictability horizon of $\sim 1/\lambda_1 \approx 0.6$ time units
        — roughly 2 days in Lorenz's original atmospheric scaling.
        """
    )
    return


@app.cell
def _(mo):
    eps_ctrl   = mo.ui.number(start=1e-12, stop=1.0, step=1e-10, value=1e-8,
                               label="Perturbation amplitude ε")
    pseed_ctrl = mo.ui.number(start=0, stop=9999, step=1, value=137,
                               label="Perturbation seed")
    mo.hstack([eps_ctrl, pseed_ctrl])
    return eps_ctrl, pseed_ctrl


@app.cell
def _(
    F, K, dt, eps_ctrl, np, nsteps, pseed_ctrl, rk4_step, save_every, x0,
):
    # ── Predictability integration ────────────────────────────────────────────
    eps  = eps_ctrl.value
    prng = np.random.default_rng(pseed_ctrl.value)

    dv   = prng.standard_normal(K)
    dv  /= np.linalg.norm(dv)

    xref  = x0.copy()
    xpert = x0 + eps * dv

    pred_times, err_energy, ftle = [0.0], [eps**2], [np.nan]

    for _s in range(nsteps):
        xref  = rk4_step(xref,  dt, F)
        xpert = rk4_step(xpert, dt, F)
        if _s % save_every == 0:
            _t   = (_s + 1) * dt
            Eerr = np.mean((xpert - xref)**2)
            pred_times.append(_t)
            err_energy.append(Eerr)
            ftle.append(
                np.log(Eerr / (eps**2 + 1e-30)) / (2.0 * _t) if Eerr > 0 else np.nan
            )

    pred_times = np.array(pred_times)
    err_energy = np.array(err_energy)
    ftle       = np.array(ftle)

    return Eerr, dv, eps, err_energy, ftle, pred_times, prng, xpert, xref


@app.cell
def _(err_energy, ftle, plt, pred_times):
    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(10, 4))

    axes_pred[0].semilogy(pred_times, err_energy, 'b-', linewidth=1.2)
    axes_pred[0].set_xlabel('Time')
    axes_pred[0].set_ylabel('$E_{\\rm err}$')
    axes_pred[0].set_title('Error Energy Growth')
    axes_pred[0].grid(True, which='both', alpha=0.3)

    axes_pred[1].plot(pred_times, ftle, 'r-', linewidth=1.2)
    axes_pred[1].axhline(1.68, color='k', linestyle='--', linewidth=0.9,
                         label='$\\lambda_1 \\approx 1.68$  ($F=8$)')
    axes_pred[1].axhline(0, color='gray', linestyle=':', linewidth=0.8)
    axes_pred[1].set_xlabel('Time')
    axes_pred[1].set_ylabel('FTLE  $\\lambda(t)$')
    axes_pred[1].set_title('Finite-Time Lyapunov Exponent')
    axes_pred[1].legend(fontsize=8)
    axes_pred[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_pred
    return axes_pred, fig_pred


@app.cell
def _(err_energy, ftle, mo, np, pred_times):
    _tail      = ftle[~np.isnan(ftle)]
    lambda_inf = _tail[-20:].mean() if len(_tail) > 20 else np.nan
    t_sat      = pred_times[np.argmax(err_energy > 0.5 * err_energy[-1])]

    mo.md(
        f"""
        ### Summary

        | Quantity | Value |
        |---|---|
        | Estimated $\\lambda_1$ (asymptotic FTLE) | `{lambda_inf:.4f}` |
        | Theoretical $\\lambda_1$ ($F = 8$) | `1.68` |
        | Error saturation time | `{t_sat:.3f}` |
        | Final error energy | `{err_energy[-1]:.4e}` |
        """
    )
    return lambda_inf, t_sat


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Wavenumber-Targeted Perturbation

        Because L96 lives on a ring of $K$ variables, it has a natural Fourier
        decomposition. A perturbation is injected at a single Fourier mode $k_{\rm inj}$:

        $$\delta x_i = A \cos\!\left(\frac{2\pi\, k_{\rm inj}\, i}{K}\right)$$

        We track the **spectral error energy** at multiple times:
        $$E_{\rm err}(k, t) = \frac{1}{2K} |\widehat{\delta x}_k(t)|^2$$

        This reveals which spatial scales absorb the error energy as the system evolves —
        directly analogous to the spectral error plots in the PDE notebooks.
        """
    )
    return


@app.cell
def _(K, mo):
    k_inj_ctrl = mo.ui.slider(1, K // 2, step=1, value=4,
                              label="Injection wavenumber $k_{\\rm inj}$")
    wamp_ctrl  = mo.ui.number(start=-2.0, stop=2.0, step=0.01, value=0.01,
                              label="Perturbation amplitude (signed)")
    mo.hstack([k_inj_ctrl, wamp_ctrl])
    return k_inj_ctrl, wamp_ctrl


@app.cell
def _(F, K, dt, k_inj_ctrl, np, nsteps, rk4_step, wamp_ctrl, x0):
    # ── Wavenumber-targeted perturbation integration ──────────────────────────
    k_inj = k_inj_ctrl.value
    wamp  = wamp_ctrl.value

    # Real cosine perturbation at wavenumber k_inj
    i_arr  = np.arange(K, dtype=float)
    dx_inj = wamp * np.cos(2 * np.pi * k_inj * i_arr / K)

    xref_w  = x0.copy()
    xpert_w = x0 + dx_inj

    n_snaps  = 8
    snap_idx = set(np.linspace(0, nsteps - 1, n_snaps, dtype=int))

    wsnap_times, wsnap_spectra = [], []

    for _s in range(nsteps):
        xref_w  = rk4_step(xref_w,  dt, F)
        xpert_w = rk4_step(xpert_w, dt, F)
        if _s in snap_idx:
            _delta   = xpert_w - xref_w
            _dhat    = np.fft.rfft(_delta) / np.sqrt(K)
            _E_err_k = 0.5 * np.abs(_dhat)**2
            wsnap_times.append((_s + 1) * dt)
            wsnap_spectra.append(_E_err_k)

    wsnap_times   = np.array(wsnap_times)
    wsnap_spectra = np.array(wsnap_spectra)   # shape: (n_snaps, K//2+1)
    k_bins_w      = np.arange(wsnap_spectra.shape[1])

    return dx_inj, i_arr, k_bins_w, k_inj, wamp, wsnap_spectra, wsnap_times


@app.cell
def _(k_bins_w, k_inj, np, plt, wamp, wsnap_spectra, wsnap_times):
    import matplotlib.cm as _cm

    fig_wn, ax_wn = plt.subplots(figsize=(9, 4.5))
    _cmap   = _cm.plasma
    _colors = _cmap(np.linspace(0.1, 0.9, len(wsnap_times)))

    for _i, (_spec, _tt) in enumerate(zip(wsnap_spectra, wsnap_times)):
        ax_wn.semilogy(
            k_bins_w[1:], np.maximum(_spec[1:], 1e-30),
            color=_colors[_i], linewidth=1.5, label=f't = {_tt:.2f}'
        )

    ax_wn.axvline(k_inj, color='k', linestyle='--', linewidth=1.2,
                  label=f'$k_{{\\rm inj}} = {k_inj}$')
    ax_wn.set_xlabel('Wavenumber $k$')
    ax_wn.set_ylabel('Spectral Error Energy  $E_{\\rm err}(k, t)$')
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
    return ax_wn, fig_wn


if __name__ == "__main__":
    app.run()
