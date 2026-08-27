# Joint Galactic-binary + glitch analysis

The two notebooks here carry the analyses of Secs. "Frequency-domain inference"
and "Time--frequency inference" of `../../paper/main.tex`. The scripts alongside
them are the extra studies: each is standalone, reads the shared data stream, and
persists what it produces so that figures can be redrawn without re-sampling.

## Data

`make_dataset.py` writes `dataset.npz` — one grid, one pair of injections, one
noise draw — which **everything** here reads. Run it first. Generating the data
once and reading it many times is what allows posterior *centres* to be compared
across analyses and not merely their widths; see Sec. "What the split costs".

## Notebooks

| File | Produces | Runtime |
|---|---|---|
| `glitch_and_gb.ipynb` | `fd_chains.npz`, Figs. corner / residual | ~5 min |
| `glitch_and_gb_wdm.ipynb` | `wdm_chain.npz`, `fig_wdm_tiling.pdf`, `fig_wdm_corner.pdf` | ~40 min |

## Scripts

| File | Question it answers | Produces | Runtime |
|---|---|---|---|
| `run_gb_free_sky.py` | does separating glitch from binary need the binary localised? | `gb_free_sky.npz` | ~1.5 h, GPU |
| `run_wdm_tdi2.py` | does the time--frequency split depend on the TDI generation? | `wdm_tdi2.npz` | ~1 h, GPU |
| `run_knee_scan.py` | how far from the fiducial configuration does the glitch/binary separation survive? | `knee_scan.npz`, `knee_mcmc.npz`, `fig_knee_scan.pdf` | ~40 min, GPU |
| `run_lpf_snr.py` | how loud are LPF-like glitches, seen by LISA on this grid? | `lpf_snr.npz` | ~2 min |
| `combine_knee_lpf.py` | how often does the population reach its own bias threshold? | — (prints) | seconds |
| `run_convergence.py` | are the chains converged, and has the Newton step? | `convergence.npz`, `tab_convergence.tex` | ~2 min |
| `make_fig_wdm_tdi.py` | — | `fig_wdm_tdi.pdf` | seconds |
| `make_corner_wdm_vs_fd.py` | — | `fig_wdm_corner.pdf` | seconds |
| `make_fig_fd.py` | — | `fig_corner.pdf`, `fig_residual.pdf` | seconds |

`../pp/run_ppplot.py` and `../pp/run_decimation.py` share `fd_pipeline.py` too: the
first calibrates the hybrid binned likelihood over 100 realisations, the second runs
the same realisations through the decimated likelihood of `fd_pipeline.build_decimated`
to show what a stride costs. Both are CPU-parallel and resumable; ~2 h on 20 cores.

`run_knee_scan.py` has five modes, and the ones that are not the scan matter as much
as the scan: `--check` verifies that the degradation and correlation it reports are
independent of both signal amplitudes, `--t0check` verifies that maximising the bias
over a constant arrival phase gives the same answer as walking the arrival time
explicitly, `--mcmc` tests the linearised bias against chains, and `--figure` draws.
Run the scan first; the others read `knee_scan.npz`.

`fd_pipeline.py` is the shared implementation of the hybrid binned likelihood and
the sampler, extracted from `glitch_and_gb.ipynb` so that the scripts and the
P--P study in `../pp/` do not drift apart from it. `run_wdm_tdi2.py` carries its
own copy of the WDM construction, since that one lives in the other notebook.

Chains are stored **walker-major**: a flat `(n_walker * n_iter, dim)` array whose
reshape is `(n_walker, n_iter, dim)`. Reading it the other way round makes walkers
look correlated at 0.8 and shrinks every autocorrelation time by an order of
magnitude; `run_convergence.py` checks the layout before trusting it.

## Four results worth knowing before reusing this code

**One Newton step is not enough.** The `(log f0, log fdot)` block is ill-conditioned
(cond(H) ≈ 1e17), so an undamped step from the injected values overshoots to a
log-posterior 30 below where it started. `fd_pipeline.laplace` now backtracks and
iterates, converging in five steps to a maximum that agrees with the chain median to
0.2 sigma. `gb_free_sky.npz` and the 100 P--P runs under `../pp/ppruns/` have been
regenerated with it. `wdm_chain.npz` still predates it and is left alone deliberately:
it is initialised at the overshot point, which is 1.9 sigma from the maximum in
`log fdot` and closer in everything else, and `run_convergence.py` puts it at
R-hat <= 1.02 with ESS >= 900, so the burn-in has demonstrably absorbed the difference.

**An ensemble that starts outside the prior never moves.** If the Laplace width comes
back wider than the prior box — which happens whenever a parameter is barely
constrained; `sigma(log A_g)` reached 1.3 times the full prior range for the faintest
glitches in the P--P set — then the `ball = 0.01` initial cloud straddles the walls and
most walkers start at `log pi = -inf`. A stretch move between two such walkers has
acceptance ratio `-inf - (-inf) = NaN`, which compares false, so it is rejected for
ever and the ensemble locks up. Nothing raises: the chain has the right shape, every
sample is finite, and the only symptom is that the quantiles are exact multiples of
1/16. Six of the 100 P--P runs were dead this way and inflated the tail statistics by
40%. `fd_pipeline.run_chain` now pulls stray walkers back along their own offset from
the maximum, which is a no-op for every well-conditioned ensemble. If you reuse the
sampler, check `len(np.unique(chain[:, i]))` against the walker count before trusting
anything.

**The eleven-parameter posterior is bimodal, exactly.** `run_gb_free_sky.py` now
launches four chains, from the injection and from three candidate degeneracies. The
`psi -> psi + pi/2, phi0 -> phi0 + pi` start reaches the *same* log-posterior as the
injection: it is an exact symmetry of the waveform, and both modes sit inside the
prior. Their medians agree in every other parameter to below 0.4 sigma, so the widths
quoted for the remaining nine are posterior widths, while `psi` and `phi0` are bimodal
by construction and their widths are per mode. The antipodal-sky and inclination-flip
starts are not competitive. Read the `free_*_logpost` entries of `gb_free_sky.npz`
before quoting any width from a single start.

**Storing a chain as float32 destroys `log f0`.** Its width is 2e-7 at a value of
-6.2, below the float32 resolution there, so the tightest parameter in the problem
would be quantised into a handful of levels. Everything here stores float64. The
quantiles in `../pp/pp_summary.npz` were computed before the cast and are unaffected,
but the chains stored under `../pp/ppruns/` carry this limitation.
