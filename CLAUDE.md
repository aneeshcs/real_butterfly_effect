# CLAUDE.md — real_butterfly_effect

Context and guidelines for AI-assisted development in this repository.

---

## Project Overview

**Goal:** Research tools for studying predictability in geophysical turbulence using
quasi-geostrophic (QG) models. The name references Lorenz's original insight that
small initial errors grow exponentially in turbulent flows — the real butterfly effect.

**Primary researcher:** Aneesh Subramanian (aneeshcs), NCAR/CGD
**GitHub:** https://github.com/aneeshcs/real_butterfly_effect
**Pages site:** https://aneeshcs.github.io/real_butterfly_effect/

---

## Environment

### HPC System
- **Cluster:** NCAR Casper (login nodes, GLADE filesystem)
- **Home:** `/glade/u/home/acsubram/`
- **Work/repo root:** `/glade/work/acsubram/GitRepos/real_butterfly_effect/`

### Python Environment
- **Conda env:** `gfd-notebooks` (in `/glade/u/home/acsubram/.conda/envs/gfd-notebooks/`)
- **Activate:** `source /glade/u/home/acsubram/miniconda3/etc/profile.d/conda.sh && conda activate gfd-notebooks`
- **Key packages:** marimo 0.21.1, numpy 2.4.3, scipy 1.17.1, matplotlib 3.10.8
- **Add packages to this env** (not others) to keep the environment reproducible

### GitHub CLI
- **Binary:** `~/.local/bin/gh` (manually installed, not a system package)
- **Auth:** `gho_****` token with scopes: `gist`, `read:org`, `repo`, `workflow`
- **`gh` is not on the default PATH** on Casper — always use the full path or ensure
  `~/.local/bin` is on PATH (added to `~/.bashrc`)
- To re-authenticate or add scopes: `~/.local/bin/gh auth refresh -h github.com -s <scope>`

### Git Remote
- The remote URL embeds a token for HTTPS pushes on Casper (no SSH agent available):
  ```
  git remote set-url origin https://aneeshcs:$(~/.local/bin/gh auth token)@github.com/aneeshcs/real_butterfly_effect.git
  ```
  Re-run this if the token expires or is refreshed.

---

## Repository Structure

```
real_butterfly_effect/
├── CLAUDE.md                        # this file
├── README.md                        # project overview and usage
├── requirements.txt                 # pip-installable dependencies
├── notebooks/
│   ├── qg2d_turbulence.py           # 2D QG pseudospectral marimo notebook
│   └── qg3d_turbulence.py           # 2-layer QG pseudospectral marimo notebook
├── docs/
│   └── index.html                   # GitHub Pages landing page (static HTML)
└── .github/
    └── workflows/
        └── deploy.yml               # CI: export notebooks to WASM → GitHub Pages
```

---

## Physics and Numerics

### 2D QG Model (`qg2d_turbulence.py`)
- **Domain:** doubly-periodic `[0, 2π)²`
- **PV equation:** `∂q/∂t + J(ψ,q) = ν(−1)^(p+1) ∇^(2p) q`
- **PV definition:** `q = ∇²ψ + βy` (β-plane; set β=0 for f-plane)
- **PV inversion:** `ψ̂ = −q̂/K²` with `ψ̂(0,0) = 0`
- **Jacobian:** computed pseudospectrally in physical space, transformed back

### 3D QG Model — 2-Layer (`qg3d_turbulence.py`)
- **Layers:** upper (1) and lower (2) with depths H₁, H₂
- **PV:** `q_n = ∇²ψ_n + F_n(ψ_{3-n} − ψ_n) + βy`
- **Stratification:** `F_n = f₀²/(g′H_n)` (inverse squared Rossby radius)
- **PV inversion:** solved as a 2×2 spectral system at each wavenumber;
  determinant `det = K²(K² + F₁ + F₂)`, handle `K=0` separately
- **Bottom drag:** `−κ∇²ψ₂` on layer 2 only
- **Diagnostics:** barotropic `ψ_bt = ½(ψ₁+ψ₂)` and baroclinic `ψ_bc = ½(ψ₁−ψ₂)`

### Numerical Methods (both models)
- **Spatial discretisation:** pseudospectral (FFT-based)
- **Dealiasing:** 2/3 rule — zero all modes with `|kx| ≥ N/3` or `|ky| ≥ N/3`
- **Time stepping:** 4th-order Runge-Kutta (RK4)
- **Initial conditions:** band-limited random vorticity centred on wavenumber k₀=4,
  normalised to unit enstrophy, Hermitian-symmetrised for real fields

### Predictability Diagnostics
- **Error energy:** `E_err(t) = ½N⁻² Σ_k K² |δψ̂_k|²`
- **FTLE:** `λ(t) = (1/2t) ln(E_err(t) / ε²)` where ε is perturbation amplitude
- Twin-run approach: reference + perturbed IC, track divergence

---

## Marimo Notebook Coding Standards

These rules are critical — violating them causes silent failures or broken WASM exports.

### Cell Dependency Rules
- **Every variable used in a cell must appear in its argument list.**
  Marimo's reactive system determines dependencies solely from function arguments.
  Do NOT rely on closures or global scope.
- **Every variable passed to downstream cells must appear in the return tuple.**
- When adding a new import (e.g. `ifft2`), check every cell that uses it and add it
  to that cell's argument list. Missing arguments are the #1 source of bugs here.

  ```python
  # WRONG — ifft2 not in args, will fail in Pyodide
  @app.cell
  def _(np, q_hat):
      q = np.real(ifft2(q_hat))   # NameError in browser
      return (q,)

  # CORRECT
  @app.cell
  def _(ifft2, np, q_hat):
      q = np.real(ifft2(q_hat))
      return (q,)
  ```

- **Never use walrus operators or inline `__import__` calls** to work around missing
  arguments. Fix the argument list instead.

### Imports
- All top-level imports go in the **first cell only**, which returns them as variables.
- Within a cell body, `from x import y as _y` (with a leading underscore) is acceptable
  for local-only use (e.g. `from numpy.fft import ifft2 as _ifft2`), but prefer passing
  the function as a cell argument.
- Remove unused imports — they generate Pyodide warnings that clutter the browser console.

### Figures
- **Maximum `figsize` width: 11 inches.** Wider figures overflow the browser viewport
  in the WASM export. Prefer `(10–11, 3.5–4)` for side-by-side plots.
- Always return `fig` from the cell so marimo can display it.
- Use `plt.tight_layout()` before returning.
- Do not use `plt.show()` — marimo handles rendering.

### UI Controls
- **Default values must be browser-safe.** For the hosted WASM version, default
  resolution N should be ≤128 (2D) or ≤64 (3D). Users on Casper can increase N via
  the slider.
- UI control cells should only create controls and return them — no computation.

### Long-running Cells
- Time integration loops run synchronously in the browser (Pyodide is single-threaded).
  Keep default `nsteps ≤ 500` for the WASM version.
- Do not add `sleep` or threading — it will break the reactive system.

---

## Deployment

### GitHub Actions (`deploy.yml`)
- Triggered on every push to `main`
- Installs marimo, exports both notebooks with `marimo export html-wasm --mode run`
- Copies `docs/index.html` as the landing page
- Deploys the `site/` directory to GitHub Pages via `actions/deploy-pages`

### GitHub Pages
- Source: **GitHub Actions** (not a branch)
- Enabled via: `gh api repos/aneeshcs/real_butterfly_effect/pages --method POST -f build_type=workflow`
- Live URL: https://aneeshcs.github.io/real_butterfly_effect/

### Adding a New Notebook
1. Create `notebooks/<name>.py` as a marimo app following the standards above
2. Add export + copy steps to `.github/workflows/deploy.yml`
3. Add a card to `docs/index.html`
4. Test the export locally first:
   ```bash
   conda activate gfd-notebooks
   marimo export html-wasm notebooks/<name>.py --output /tmp/<name>/ --mode run
   ```

---

## Common Tasks

### Run a notebook locally (interactive)
```bash
conda activate gfd-notebooks
marimo edit notebooks/qg2d_turbulence.py
```

### Run a notebook locally (read-only, as served)
```bash
conda activate gfd-notebooks
marimo run notebooks/qg2d_turbulence.py
```

### Test WASM export before pushing
```bash
conda activate gfd-notebooks
marimo export html-wasm notebooks/qg2d_turbulence.py --output /tmp/test_export/ --mode run
python -m http.server --directory /tmp/test_export   # then open localhost:8000
```

### Check Actions run status
```bash
~/.local/bin/gh run list --repo aneeshcs/real_butterfly_effect
```

### Push (with authenticated remote)
```bash
git remote set-url origin https://aneeshcs:$(~/.local/bin/gh auth token)@github.com/aneeshcs/real_butterfly_effect.git
git push
```

---

## Key Decisions and Rationale

| Decision | Rationale |
|---|---|
| Pseudospectral + RK4 | Most accurate and efficient for periodic turbulence; standard in GFD community |
| 2/3-rule dealiasing | Prevents aliasing errors in quadratic nonlinearities (Jacobian) |
| Hyperviscosity (∇²ᵖ) | Scale-selective dissipation; preserves inertial range better than Laplacian viscosity |
| 2-layer model for "3D QG" | Captures essential baroclinic dynamics without continuous stratification complexity |
| marimo over Jupyter | Reactive cells enforce clean data flow; WASM export enables browser hosting without a server |
| WASM / GitHub Pages | Zero-cost, serverless hosting; notebooks fully interactive in browser via Pyodide |
| Lower browser defaults (N=128/64, nsteps=500) | Pyodide is single-threaded; large N causes browser tab to freeze |
| `gfd-notebooks` conda env | Isolated from other NCAR/CGD environments; reproducible across sessions |
