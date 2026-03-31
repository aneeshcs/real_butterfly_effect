# real_butterfly_effect

Marimo notebooks for studying predictability in geophysical turbulence using pseudospectral quasi-geostrophic (QG) models.

## Models

### 2D QG Turbulence (`notebooks/qg2d_turbulence.py`)
Pseudospectral solver for the 2D quasi-geostrophic equations on a doubly-periodic domain:

$$\frac{\partial q}{\partial t} + J(\psi, q) = \nu (-1)^{p+1} \nabla^{2p} q$$

where $q = \nabla^2 \psi + \beta y$ is the potential vorticity, $J$ is the Jacobian, and $\nu$ is the (hyper)viscosity coefficient.

### 3D QG Turbulence — 2-Layer Model (`notebooks/qg3d_turbulence.py`)
Pseudospectral solver for the 2-layer quasi-geostrophic model:

$$\frac{\partial q_n}{\partial t} + J(\psi_n, q_n) = \text{dissipation}$$

$$q_1 = \nabla^2 \psi_1 + F_1(\psi_2 - \psi_1) + \beta y$$
$$q_2 = \nabla^2 \psi_2 + F_2(\psi_1 - \psi_2) + \beta y$$

where $F_n = f_0^2 / (g' H_n)$ are the stratification parameters (inverse Rossby radius squared).

## Predictability Diagnostics

Both notebooks include tools for studying error growth and predictability:
- Error energy evolution: $E_{\text{err}}(t) = \frac{1}{2} \int |\nabla \delta\psi|^2 \, dA$
- Kinetic energy spectra of reference and error fields
- Finite-time Lyapunov exponent (FTLE) estimation

## Setup

```bash
conda activate gfd-notebooks
marimo run notebooks/qg2d_turbulence.py
marimo run notebooks/qg3d_turbulence.py
```

Or to edit interactively:
```bash
marimo edit notebooks/qg2d_turbulence.py
```

## Requirements

See `requirements.txt`. Install via:
```bash
pip install -r requirements.txt
```
