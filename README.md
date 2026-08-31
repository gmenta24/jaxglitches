# jaxglitches

JAX-based LISA glitch waveforms, TDI responses, and parameter-estimation
utilities. Everything is differentiable and jittable (float64 is enabled on
import).

## What's inside

- **Waveforms** (`jaxglitches.waveform`): analytic time- and frequency-domain
  TDI-1/TDI-2 responses to a test-mass glitch on link 12, for the
  one-exponential (integrated n=1 shapelet) template — the LISA Pathfinder
  glitch model of [lisaglitch](https://gitlab.in2p3.fr/lisa-simulation/glitch).
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

Noise PSDs used by the notebooks live in `notebooks/noise.py`, outside the package.

## Install

```sh
uv sync            # or: pip install -e .
```

### GPU

The package is pure JAX with no custom kernels, so the same code runs on GPU
unchanged — you only need a CUDA-enabled `jaxlib`. Check your driver with
`nvidia-smi` and install the matching extra:

```sh
uv sync --extra gpu           # CUDA 12 build, driver >= 525
uv sync --extra gpu-cuda13    # CUDA 13 build, driver >= 580
```

Verify with `python -c "import jax; print(jax.devices())"`; if it prints
`[CpuDevice(...)]` the plugin does not match the driver and JAX has silently
fallen back to CPU.

Everything stays in float64 on GPU. That matters here — the glitch phase
`exp(-2i pi f t0)` with `t0` up to a year needs the full double mantissa — and it
is not free on consumer hardware, where FP64 runs at a fraction of the FP32 rate.
It turns out not to hurt: these kernels are memory-bound rather than FLOP-bound,
and on an RTX 2000 Ada the template is ~5x faster than on 22 CPU cores for grids
above 1e4 bins. The Hessian benefits far more (14.7x one likelihood on CPU
vs 1.9x on GPU). See `paper/validation/03_benchmarks.ipynb`.

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

`notebooks/glitch_only/unequal_vs_equal_arm.ipynb` goes one step further and asks
what the equal-arm approximation *costs*: it injects with the six frozen delays of a
Keplerian constellation, fits with the equal-arm template, and measures the resulting
bias over a year of epochs. Appendix B of the paper reports the answer. Two findings
are worth knowing before reusing that code — the T channel stops being null once the
arms are unequal, so the equal-arm `S_T` of `notebooks/noise.py` must not be used with
an unequal-arm signal; and the equal-arm `T` an analyst picks matters, the epoch's mean
delay being a factor four better than the design value.

The waveforms are also checked end to end against the external LISA simulation
chain — a glitch injected with `lisaglitch`, propagated by `lisainstrument` and
combined into Michelson TDI by `pytdi` — in
`paper/validation/04_end_to_end.ipynb` and §5 of
`notebooks/glitch_only/raw_and_tdi_tests.ipynb`. With every light travel time set
to an integer number of samples nothing in the chain interpolates and the
agreement is exact to 2e-15 across all six channels and both TDI generations.
`lisainstrument` and `pytdi` are not package dependencies:

```sh
uv sync --extra notebook --extra simulation
```

### Optional extras, and what needs them

`jaxglitches` itself depends only on `jax`, `numpy`, `lisaglitch` and
`lisaorbits`. Everything else the repository uses is an extra, because nothing
under `src/jaxglitches` imports it and the test suite does not need it:

| Extra | Pulls in | Needed by |
|---|---|---|
| `notebook` | matplotlib, scipy, ipykernel, nbconvert, jexplore | every notebook |
| `joint` | `jaxgb` | `notebooks/glitch_GB/*` — the Galactic binary the glitch is fitted alongside |
| `wdm` | `wdm-transform[jax]` | the time–frequency analyses in `notebooks/glitch_GB/` |
| `simulation` | lisainstrument, pytdi | the end-to-end validation above |
| `gpu` / `gpu-cuda13` | CUDA `jaxlib` | running on a GPU |

So to reproduce the paper's joint analysis:

```sh
uv sync --extra notebook --extra joint --extra wdm
```

and to use the waveforms and the likelihood on their own, `uv sync` alone is
enough.
