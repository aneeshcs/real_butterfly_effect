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
        # Lorenz (1963) System

        The original chaotic system derived by Edward Lorenz from a truncated model
        of Rayleigh-Bénard convection — the canonical example of sensitive dependence
        on initial conditions and the mathematical origin of the "butterfly effect":

        $$\frac{dx}{dt} = \sigma(y - x), \qquad
          \frac{dy}{dt} = x(\rho - z) - y, \qquad
          \frac{dz}{dt} = xy - \beta z$$

        The classical chaotic regime uses $\sigma = 10$, $\rho = 28$, $\beta = 8/3$,
        giving a strange attractor with leading Lyapunov exponent $\lambda_1 \approx 0.906$.

        **Numerics:** 4th-order Runge-Kutta (RK4), fixed time step.
        """
    )
    return


@app.cell
def _(mo):
    sigma_ctrl  = mo.ui.number(start=0.1, stop=50.0, step=0.1, value=10.0,
                               label="σ (Prandtl number)")
    rho_ctrl    = mo.ui.number(start=0.1, stop=100.0, step=0.1, value=28.0,
                               label="ρ (Rayleigh number)")
    beta_ctrl   = mo.ui.number(start=0.1, stop=10.0, step=0.01, value=8/3,
                               label="β")
    dt_ctrl     = mo.ui.number(start=1e-4, stop=0.05, step=1e-4, value=0.01,
                               label="Time step dt")
    nsteps_ctrl = mo.ui.slider(200, 10000, step=200, value=2000,
                               label="Steps to run")
    seed_ctrl   = mo.ui.number(start=0, stop=9999, step=1, value=42,
                               label="Random seed")

    mo.vstack([
        mo.md("## Model Parameters"),
        mo.hstack([sigma_ctrl, rho_ctrl, beta_ctrl]),
        mo.hstack([dt_ctrl, nsteps_ctrl, seed_ctrl]),
    ])
    return beta_ctrl, dt_ctrl, nsteps_ctrl, rho_ctrl, seed_ctrl, sigma_ctrl


@app.cell
def _(np):
    # ── L63 dynamics ──────────────────────────────────────────────────────────

    def l63_rhs(state, sigma, rho, beta):
        """L63 right-hand side."""
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])

    def rk4_step(state, dt, sigma, rho, beta):
        """4th-order Runge-Kutta step."""
        k1 = l63_rhs(state,              sigma, rho, beta)
        k2 = l63_rhs(state + 0.5*dt*k1, sigma, rho, beta)
        k3 = l63_rhs(state + 0.5*dt*k2, sigma, rho, beta)
        k4 = l63_rhs(state +     dt*k3, sigma, rho, beta)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return l63_rhs, rk4_step


@app.cell
def _(
    beta_ctrl,
    dt_ctrl,
    mo,
    np,
    nsteps_ctrl,
    rho_ctrl,
    rk4_step,
    seed_ctrl,
    sigma_ctrl,
):
    # ── Time integration ─────────────────────────────────────────────────────
    sigma  = sigma_ctrl.value
    rho    = rho_ctrl.value
    beta   = beta_ctrl.value
    dt     = dt_ctrl.value
    nsteps = nsteps_ctrl.value

    # Random initial condition near the attractor
    rng    = np.random.default_rng(seed_ctrl.value)
    x0     = np.array([rng.uniform(-15, 15),
                       rng.uniform(-20, 20),
                       rng.uniform(5, 45)])

    # Store full trajectory (cheap — only 3 numbers per step)
    trajectory = np.empty((nsteps + 1, 3))
    trajectory[0] = x0
    times = np.linspace(0, nsteps * dt, nsteps + 1)

    state = x0.copy()
    for _i in range(nsteps):
        state = rk4_step(state, dt, sigma, rho, beta)
        trajectory[_i + 1] = state

    save_every = max(1, nsteps // 100)

    mo.md(
        f"**Integration complete.** "
        f"t = {nsteps * dt:.2f},  steps = {nsteps},  "
        f"σ = {sigma},  ρ = {rho},  β = {beta:.3f}"
    )
    return beta, dt, nsteps, rho, rng, save_every, sigma, state, times, trajectory, x0


@app.cell
def _(plt, times, trajectory):
    # ── Attractor and time series ─────────────────────────────────────────────
    xs, ys, zs = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    fig_att, axes_att = plt.subplots(1, 3, figsize=(11, 3.8))

    # Phase portraits
    axes_att[0].plot(xs, zs, 'b-', linewidth=0.3, alpha=0.6)
    axes_att[0].set_xlabel('x'); axes_att[0].set_ylabel('z')
    axes_att[0].set_title('Lorenz Attractor ($x$–$z$)')

    axes_att[1].plot(xs, ys, 'b-', linewidth=0.3, alpha=0.6)
    axes_att[1].set_xlabel('x'); axes_att[1].set_ylabel('y')
    axes_att[1].set_title('Phase Portrait ($x$–$y$)')

    # Time series of x
    axes_att[2].plot(times, xs, 'b-', linewidth=0.6)
    axes_att[2].set_xlabel('Time'); axes_att[2].set_ylabel('$x(t)$')
    axes_att[2].set_title('$x$ vs Time')
    axes_att[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_att
    return axes_att, fig_att, xs, ys, zs


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Predictability Experiment

        Two L63 trajectories from nearly identical initial conditions:
        $\mathbf{x}_0$ (reference) and $\mathbf{x}_0 + \varepsilon\,\delta\mathbf{x}$
        (perturbed), where $\delta\mathbf{x}$ is a random unit vector.

        **Error norm:**
        $$E_{\rm err}(t) = \|\delta\mathbf{x}(t)\|^2
          = \delta x^2 + \delta y^2 + \delta z^2$$

        **FTLE:** $\lambda(t) = \dfrac{1}{2t} \ln \dfrac{E_{\rm err}(t)}{\varepsilon^2}$

        In the chaotic regime ($\rho = 28$), the FTLE converges to the leading
        Lyapunov exponent $\lambda_1 \approx 0.906$, setting the predictability horizon.
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
def _(beta, dt, eps_ctrl, np, nsteps, pseed_ctrl, rk4_step, rho, sigma, x0):
    # ── Predictability integration ────────────────────────────────────────────
    eps   = eps_ctrl.value
    prng  = np.random.default_rng(pseed_ctrl.value)

    # Random unit-vector perturbation
    dv    = prng.standard_normal(3)
    dv   /= np.linalg.norm(dv)

    xref  = x0.copy()
    xpert = x0 + eps * dv

    pred_times, err_energy, ftle = [0.0], [eps**2], [np.nan]

    for _s in range(nsteps):
        xref  = rk4_step(xref,  dt, sigma, rho, beta)
        xpert = rk4_step(xpert, dt, sigma, rho, beta)
        _t    = (_s + 1) * dt
        Eerr  = np.sum((xpert - xref)**2)
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
    axes_pred[0].set_ylabel('$E_{\\rm err} = \\|\\delta\\mathbf{x}\\|^2$')
    axes_pred[0].set_title('Error Growth')
    axes_pred[0].grid(True, which='both', alpha=0.3)

    axes_pred[1].plot(pred_times, ftle, 'r-', linewidth=1.2)
    axes_pred[1].axhline(0.906, color='k', linestyle='--', linewidth=0.9,
                         label='$\\lambda_1 \\approx 0.906$')
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
    lambda_inf = _tail[-50:].mean() if len(_tail) > 50 else np.nan
    t_sat      = pred_times[np.argmax(err_energy > 0.5 * err_energy[-1])]

    mo.md(
        f"""
        ### Summary

        | Quantity | Value |
        |---|---|
        | Estimated $\\lambda_1$ (asymptotic FTLE) | `{lambda_inf:.4f}` |
        | Theoretical $\\lambda_1$ (σ=10, ρ=28, β=8/3) | `0.9056` |
        | Error saturation time | `{t_sat:.2f}` |
        | Final error energy | `{err_energy[-1]:.4e}` |
        """
    )
    return lambda_inf, t_sat


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Variable-Targeted Perturbation

        Instead of a spectral injection, L63's three-dimensional state space lets us
        ask: **which variable is the most sensitive direction for perturbation growth?**

        Perturbations are injected along a single axis ($x$, $y$, or $z$) with a
        chosen signed amplitude. We track the squared error in each component:

        $$E_x(t) = \delta x(t)^2, \quad
          E_y(t) = \delta y(t)^2, \quad
          E_z(t) = \delta z(t)^2$$

        The fastest-growing direction reveals the local structure of the unstable
        manifold. For finite amplitudes, positive and negative perturbations can
        follow the two wings of the attractor differently.
        """
    )
    return


@app.cell
def _(mo):
    vinj_ctrl = mo.ui.dropdown(
        options={"x": 0, "y": 1, "z": 2},
        value="x",
        label="Perturb variable"
    )
    vamp_ctrl = mo.ui.number(start=-2.0, stop=2.0, step=0.01, value=0.01,
                             label="Perturbation amplitude (signed)")
    mo.hstack([vinj_ctrl, vamp_ctrl])
    return vamp_ctrl, vinj_ctrl


@app.cell
def _(beta, dt, np, nsteps, rk4_step, rho, save_every, sigma, vamp_ctrl, vinj_ctrl, x0):
    # ── Variable-targeted perturbation integration ────────────────────────────
    vinj = vinj_ctrl.value    # 0, 1, or 2
    vamp = vamp_ctrl.value

    dv_target      = np.zeros(3)
    dv_target[vinj] = vamp

    xref_v  = x0.copy()
    xpert_v = x0 + dv_target

    vtimes = [0.0]
    verr   = [[0.0, 0.0, 0.0]]   # [δx², δy², δz²] at each saved step

    for _s in range(nsteps):
        xref_v  = rk4_step(xref_v,  dt, sigma, rho, beta)
        xpert_v = rk4_step(xpert_v, dt, sigma, rho, beta)
        if _s % save_every == 0:
            _d = xpert_v - xref_v
            vtimes.append((_s + 1) * dt)
            verr.append([_d[0]**2, _d[1]**2, _d[2]**2])

    vtimes = np.array(vtimes)
    verr   = np.array(verr)    # shape: (n_saves, 3)

    return dv_target, verr, vinj, vamp, vtimes, xpert_v, xref_v


@app.cell
def _(np, plt, vamp, verr, vinj, vtimes):
    _var_names = ['x', 'y', 'z']
    _colors    = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig_vp, ax_vp = plt.subplots(figsize=(8, 4))

    for _i, (_name, _col) in enumerate(zip(_var_names, _colors)):
        ax_vp.semilogy(
            vtimes, np.maximum(verr[:, _i], 1e-30),
            color=_col, linewidth=1.5, label=f'$\\delta {_name}^2$'
        )

    ax_vp.set_xlabel('Time')
    ax_vp.set_ylabel('Squared error per component')
    ax_vp.set_title(
        f'Variable-Targeted Error Growth  '
        f'(perturbation in ${_var_names[vinj]}$,  amplitude = {vamp:.2e})'
    )
    ax_vp.legend(fontsize=9)
    ax_vp.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig_vp
    return ax_vp, fig_vp


if __name__ == "__main__":
    app.run()
