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

#### Benchmarking todo

Two gaps in the numbers above, both of which need hardware or a run this machine
cannot provide:

- [ ] **A second GPU.** Every benchmark here is on one Ada-generation laptop part,
  whose FP64 rate is 1/64 of FP32. A data-centre card with a 1/2 FP64 ratio (A100,
  H100) would separate the memory-bandwidth-bound reading above from a residual FLOP
  limitation: if that reading is right, the speed-up should still saturate near the
  bandwidth ratio rather than scale with FP64 throughput. One run on a cluster node
  settles it.
- [ ] **End-to-end sampling benchmark.** The year-long joint run reaches the MAP in
  3.6 s and does 16 walkers x 22,000 iterations in 52 s, but those numbers are
  reported in passing, per run. What is missing is the scaling in walkers and in
  device, measured and reported as a benchmark in its own right.

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
bias over a year of epochs. Appendix D of the paper reports the answer. Two findings
are worth knowing before reusing that code — the T channel stops being null once the
arms are unequal, so the equal-arm `S_T` of `notebooks/noise.py` must not be used with
an unequal-arm signal; and the equal-arm `T` an analyst picks matters, the epoch's mean
delay being a factor four better than the design value.

`notebooks/glitch_only/compare_other_codes.ipynb` checks the package against
published results rather than against itself: the optimal SNRs of the seven glitches
tabulated by Muratore et al. (2025), the LISA Pathfinder population projected to LISA
by Baghi et al. (2022), the two-year population percentiles of Boumerdassi et al.
(2026), and the parameter degeneracy Sauter et al. (2025) report for sub-sample
glitches. Appendix C of the paper reports what came out. Two of the four agree
sharply; the other two turn on what a published amplitude column means, and the
notebook says so rather than picking the flattering reading.

`notebooks/glitch_only/exact_vs_hybrid.ipynb` asks whether the binned likelihood the
paper samples from gives the same posterior as the exact one over all 93,697 Fourier
bins. Same stored data, same priors, same sampler, same Newton start — the only thing
that changes is the grid. The awkward part is knowing what agreement to demand, since
two chains of any finite length disagree, so the notebook measures that too: one of its
five chains is a second hybrid run differing only in its random seed, and that control is
the yardstick. The binning comes out smaller than it, on both the posterior widths and
the medians. Section 3.7 of the paper reports the numbers.

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

## Reproducing the paper's figures

Every figure and generated table in `paper/main.tex` is produced by code in this
repository, and `paper/make_figures.py` is the index of which producer makes which:

```sh
uv run python paper/make_figures.py --list     # what makes what, what it reads, how long
uv run python paper/make_figures.py --check    # verify the figures against the manifest
uv run python paper/make_figures.py --run draw # redraw all 10 stored-chain figures, ~1 min
```

Producers are grouped into tiers by cost. `draw` redraws from the stored `.npz`
chains and computes nothing; `validation` runs the validation notebooks (~14 min);
`derived` builds the convergence table; `sample` is the MCMC that produced the stored
chains in the first place — hours, mostly on a GPU.

Those `.npz` chains are **not** in version control (`notebooks/**/*.npz` is ignored,
as is `paper/`), so the manifest below is the only record tying a published figure to
the sampler run behind it. Keep them together.

Three properties make this reproducible rather than merely scripted:

1. **Figures are byte-deterministic.** Every producer writes through one `save()`
   (`paper/validation/_style.py`), which pins the PDF timestamp to
   `SOURCE_DATE_EPOCH`. Re-running a producer on unchanged input gives a
   byte-identical file, so a changed figure in `git status` means a changed *figure*.

2. **Provenance is recorded, not remembered.** `paper/figures/MANIFEST.json` holds,
   per figure: its SHA-256, the producer that wrote it, the seeds it used, the
   SHA-256 of every data file it read, the git commit, and the versions of `jax`,
   `jaxglitches`, `lisaglitch`, `lisainstrument`, `pytdi` and the rest, plus the JAX
   backend — which matters, because Appendix B.4 of the paper quotes residuals at
   `1e-16`, where CPU and GPU differ. The commit it records is this repository's;
   the paper tree and the chains are outside it, which is precisely why their
   hashes are worth writing down.

3. **Staleness is detectable.** `--check` re-hashes outputs *and inputs*, so a figure
   drawn before its chain was re-sampled is reported rather than silently shipped.
   `.npz` inputs are hashed member by member, so a rewritten-but-identical chain does
   not raise a false alarm.

Verified by running every producer outside the `sample` tier twice from scratch:
every output comes back byte-identical except the two benchmark outputs,
`fig_benchmark.pdf` and `tab_benchmark.tex`, whose content is wall-clock timings and
which are flagged `reproducible: false` in the manifest. (`fig_wdm_tiling.pdf` comes
from a 40-minute sampler notebook and is registered in the manifest rather than
re-run.) Five producers had to give up the GPU to get there
(`05_heterodyne.ipynb`, `unequal_vs_equal_arm.ipynb`, `compare_other_codes.ipynb`,
`exact_vs_hybrid.ipynb`, `run_convergence.py`): GPU
reductions are not bit-reproducible, and each was rewriting its figure on every run
at a relative `1e-6` that changed nothing it reports. They set `JAX_PLATFORMS=cpu`,
overridable, and each says why at the top of the file. The MCMC in the `sample` tier
still runs on the GPU and is reproducible in the ordinary seeded sense.

One figure is deliberately environment-dependent and says so: panel (d) of
`fig_gradients` visualises how the installed JAX version differentiates a function
that is not differentiable at the point in question. Under jax 0.10 the Hessian comes
out identically zero where older versions returned a finite but wrong matrix; the
notebook detects which it got and draws that one, instead of crashing on
`np.linalg.inv` as it had started to do.
