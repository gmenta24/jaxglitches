# jaxglitches

JAX-based LISA glitch waveforms, TDI responses, and parameter-estimation
utilities. Everything is differentiable and jittable (float64 is enabled on
import).

## What's inside

- **Waveforms** (`jaxglitches.waveform`): analytic time- and frequency-domain
  TDI-1/TDI-2 responses to a test-mass glitch on link 12, for one- and
  two-exponential (integrated shapelet) templates — the LISA Pathfinder glitch
  model of [lisaglitch](https://gitlab.in2p3.fr/lisa-simulation/glitch).
- **Signal builders** (`jaxglitches.data`): `clean_signal_f/t`, the raw
  (pre-TDI) single-link glitch, and numerical TDI application via
  frequency-domain delay operators. Unequal-arm variants (`*_unequal`) use the
  six per-link light travel times frozen at the glitch epoch.
- **Orbits** (`jaxglitches.orbits`): per-link light travel times from
  [lisaorbits](https://pypi.org/project/lisaorbits/) in the standard link
  ordering `(12, 23, 31, 13, 32, 21)`.
- **Inference** (`jaxglitches.likelihood`, `jaxglitches.priors`):
  frequency-domain Gaussian likelihood, matched-filter SNR, Fisher matrices,
  and LPF-population-motivated priors with unit-hypercube and log-coordinate
  parametrisations.
- **Catalogues** (`jaxglitches.catalog_generator`): realistic glitch
  populations (Poisson arrivals at LPF rates, parameters resampled from the
  empirical LPF catalogue of Baghi et al. 2022,
  [arXiv:2112.07490](https://arxiv.org/abs/2112.07490)).

Noise PSDs used by the notebooks live in `noise.py` at the repository root.

## Install

```sh
uv sync            # or: pip install -e .
```

## Quick start

```python
import jax.numpy as jnp
import jaxglitches as jg

freq = jg.freq_grid()                        # default: 1 h window, dt = 0.25 s
params = jnp.array([400.0, 1.2e-13, 0.79])   # [t0 (s), Deltav (m/s), tau (s)]
h_fd = jg.clean_signal_f(params, freq, tdi=1)  # (F, 3) complex, columns [A, E, T]

ltt = jg.link_ltt(0.0)                       # frozen per-link travel times (s)
h_un = jg.clean_signal_f_unequal(params, freq, jnp.asarray(ltt), tdi=1)
```

## Tests

```sh
uv run pytest
```
