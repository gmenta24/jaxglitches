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
| `run_wdm_tdi2.py` | does the time--frequency split depend on the TDI generation, and through which piece of the transfer function? | `wdm_tdi2.npz`, `wdm_tdi2_mechanism.npz` | ~1 h, GPU; `mechanism` ~5 min, CPU |
| `run_wdm_windows.py` | where does the glitch window lose SNR, and what does cutting a segment do to the noise? | `wdm_windows.npz` | ~3 min, CPU |
| `run_wdm_segments.py` | what does a short segment do to the glitch fit, and does padding (or mirroring) the segment fix it? | `wdm_segments_fits.npz`, `wdm_segments_scatter.npz` | ~3 min + ~30 min, CPU |
| `run_gb_only.py` | is the Fisher/MCMC mismatch in psi and fdot the binary's or the glitch's? | `gb_only.npz` | ~1 min, CPU |
| `run_knee_scan.py` | how far from the fiducial configuration does the glitch/binary separation survive? | `knee_scan.npz`, `knee_t0.npz`, `knee_mcmc.npz`, `fig_knee_scan.pdf` | ~40 min, GPU (`--t0scan` ~20 min, CPU) |
| `run_lpf_snr.py` | how loud are LPF-like glitches, seen by LISA on this grid? | `lpf_snr.npz` | ~2 min |
| `run_unmodelled.py` | what breaks if the glitch is *not* modelled, in each representation? | `unmodelled.npz`, `fig_unmodelled.pdf` | ~2 h, GPU |
| `combine_knee_lpf.py` | how often does the population reach its own bias threshold? | — (prints) | seconds |
| `run_convergence.py` | are the chains converged, and has the Newton step? | `convergence.npz`, `tab_convergence.tex`, `tab_posterior.tex` | ~20 s, CPU |
| `make_fig_wdm_tdi.py` | — | `fig_wdm_tdi.pdf` | seconds |
| `make_corner_wdm_vs_fd.py` | — | `fig_wdm_corner.pdf` | seconds |
| `make_fig_fd.py` | — | `fig_corner.pdf`, `fig_residual.pdf` | seconds |
| `make_fig_unmodelled.py` | — | `fig_unmodelled.pdf` | seconds |

`../pp/run_ppplot.py` and `../pp/run_decimation.py` share `fd_pipeline.py` too: the
first calibrates the hybrid binned likelihood over 100 realisations, the second runs
the same realisations through the decimated likelihood of `fd_pipeline.build_decimated`
to show what a stride costs. Both are CPU-parallel and resumable; ~2 h on 20 cores.

`run_unmodelled.py` runs in six phases, each in its own process for the same reason
as `run_wdm_tdi2.py`: `probe` (window geometry, the excision trade-off, the linearised
bias), `ladder` (bias against glitch amplitude), `scatter` and `scattercrit` (24 noise
draws each, at 4 rho_crit and at rho_crit, to tell a bias from a draw), `chains` (the
posteriors themselves) and `seeds` (the same glitch-free posterior sampled from many
seeds, to measure what a chain width is really known to). `merge` collects them, `figure`
draws. Every phase but `probe` takes a list of analyses to re-run, carrying the others
over unchanged; `UNMODELLED_CHAIN_SEED` sets the seed of `chains`. It is the script that
found the blind spot in the split likelihood described below.

`run_wdm_tdi2.py` has two phases outside its default run. `mechanism` takes the
generation transfer function apart -- magnitude, quarter-cycle phase, delay -- applies
each piece to noise draws, and then lengthens the glitch window in time; it writes its
own `wdm_tdi2_mechanism.npz` and is never merged into `wdm_tdi2.npz`, so re-running it
cannot make `fig_wdm_tdi` look stale. `report` only prints: the statistics the paper
quotes from the stored chains and Laplace comparisons. `run_unmodelled.py table` does the
same for the paper's summary table of the six analyses.

`run_wdm_windows.py` has two phases, `wrap` (the glitch's rho^2 by time bin, what the
window keeps, what the notch removes) and `segments` (whitened noise in segments cut from
the year, as cut, with the slow noise removed first, and transformed inside a stretch
three times longer). Neither builds a likelihood.

`run_knee_scan.py` has seven modes, and the ones that are not the scan matter as much
as the scan: `--check` verifies that the degradation and correlation it reports are
independent of both signal amplitudes, `--t0check` verifies that maximising the bias
over a constant arrival phase gives the same answer as walking the arrival time
explicitly, `--mcmc` tests the linearised bias against chains, and `--figure` draws.
Run the scan first; the others read `knee_scan.npz`.

`--t0scan` is the third axis. The scan holds the glitch's arrival **epoch** at the
injected `t0 = 400 s`, which is the worst place for it: `f0` and `fdot` are read from
the phase at the two ends of the year, so a glitch at either end has the most leverage
on them, while one arriving mid-year is largely decorrelated from the binary's
derivatives. Walking the epoch raises `rho_crit` by a factor of a few — at the fiducial
configuration from 532 to 2401 — so `--t0scan` writes the whole `(tau, f0, epoch)` cube
to `knee_t0.npz`, and `combine_knee_lpf.py` reads it to redo the population count with
each event's arrival time **drawn over the record** instead of fixed at its start. It is
cheap because the binary Fisher block does not depend on the epoch: one Hessian per
`(tau, f0)` as in the scan, then two gradients per epoch.

`fd_pipeline.py` is the shared implementation of the hybrid binned likelihood and
the sampler, extracted from `glitch_and_gb.ipynb` so that the scripts and the
P--P study in `../pp/` do not drift apart from it. `run_wdm_tdi2.py` carries its
own copy of the WDM construction, since that one lives in the other notebook.

Chains are stored **walker-major**: a flat `(n_walker * n_iter, dim)` array whose
reshape is `(n_walker, n_iter, dim)`. Reading it the other way round makes walkers
look correlated at 0.8 and shrinks every autocorrelation time by an order of
magnitude; `run_convergence.py` checks the layout before trusting it.

## Nine results worth knowing before reusing this code

**The split likelihood never subtracts the glitch from the binary's window.** Eq. (38)
of the paper models `W_GB` with `h_GB` and `W_gl` with `h_gl`, and that is what makes
the posterior factorise. It also means a *joint* WDM fit carries exactly the same
systematic as a frequency-domain fit that omits the glitch altogether — the glitch is
estimated, but not where the binary is measured. At the fiducial amplitude this is
invisible (the glitch puts SNR 0.67 into `W_GB` against the binary's 283), which is why
nothing in `glitch_and_gb_wdm.ipynb` shows it. `run_unmodelled.py` turns the glitch up
and it appears. Two repairs work and both are in that script: model `W_GB` with
`h_GB + h_gl` (`wdm_sub`, exact, one extra transform per call), or excise the time bins
carrying the onset (`wdm_cut`, needs no glitch model at all, costs a few per cent of the
binary's SNR).

**The glitch window has to reach round the back of the year.** The WDM transform is
periodic and the fiducial onset is 400 s into the year, so the glitch's footprint
straddles the join: on the `Nf = 128` tiling the *last* time bin carries 17% of its
rho^2. A window starting at bin 0, which is what `W_gl` first was, keeps 90.0% of the
SNR; `W_gl` now also keeps bins 1462-1463 and so 99.2% (`run_wdm_windows.py wrap`). The
notebook, `run_wdm_tdi2.py` and `run_unmodelled.py` all use that six-bin window, and the
results of the four-bin one are kept in `superseded_4bin_window/`. Including the wrapped
bins is safe here only because the simulated noise is periodic too. On real data the join
carries a red-noise jump -- the same one that makes a short segment fail -- and the
wrapped bins would pick it up.

**In the WDM domain TDI-1 and TDI-2 differ through the phase, not the magnitude.** The
transfer function between them is `2 sin(4 pi f L) * i * exp(-4 pi i f L)`. Its
magnitude varies across a channel, which looks like the culprit and is not: applied on
its own it leaves the glitch estimate's noise 0.999 correlated between generations. The
quarter-cycle `i` alone drops that to 0.58 in a window of the four bins from the onset
on, because it swaps each wavelet for its quadrature partner, which lives in the
neighbouring *time* bins. With the two wrapped bins of `W_gl` it is 0.97, and with twenty
bins 0.999 (`run_wdm_tdi2.py mechanism`). A windowed time--frequency null test between
generations therefore has a floor set by the window's length in time: over 48 noise draws
the glitch-block difference between generations scatters by 0.2 sigma with `W_gl`, and by
0.64 sigma with the four-bin window (whose results are in `superseded_4bin_window/`).

**A short segment fails by scatter, not by bias, and padding cures it.** Transforming a
segment of the year instead of the whole of it looks free and is not: the segment's two
ends do not meet in red noise, the transform joins them anyway, and the glitch at the
join is fitted against a noise model that does not know about the jump.
`run_wdm_segments.py` rebuilt that likelihood (the original was never kept). On the stored
realisation a 7.9-day segment gives A_g x1.89, 10.4 sigma out, which is where the old
"factor two" came from, but over 24 draws the error has no consistent sign: the log A_g
displacement scatters by 5 posterior widths at 7.9 d and by 21-57 at 23.7-89.9 d, with
half or more of the draws beyond 3 sigma. Transforming a stretch three times longer and
keeping its middle third matches the full year to 0.01 sigma in every draw. Where there
is no data to pad with -- a glitch at the start of a real record -- mirroring the start in
front of it recovers the estimate but not its width: the diagonal noise model counts the
reflected noise twice, the window SNR comes out at 58.5 instead of 42.4, and the Laplace
widths are 28% too narrow.

**A chain can fail to explore a posterior and pass every internal check.** At 4 rho_crit
the glitch-free binary posterior of `run_unmodelled.py` is 17 times wider in fdot than
the joint one and runs into the prior. A 25 000-iteration `wdm_split` chain on it (seed
7, kept as `unmodelled_chains_wdm_split_seed7.npz`) came back 11% narrow in log f0 and 6%
in psi, while its quarters, walker halves, tau_int and R-hat all looked fine. Sixteen
`fd_gbonly` chains of the same length and the same binary posterior from different seeds
(`run_unmodelled.py seeds`, about a minute each on a GPU) put the spread of a width at
1-2% and of a median at 0.1 sigma, which puts that run five standard deviations out; a
re-run from seed 17 (`UNMODELLED_CHAIN_SEED`) lands within 1.4. For a slowly mixing
posterior, compare independent runs before quoting a width.

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

## Regenerating the figures

None of the figures here need the sampler re-run: each `make_fig_*.py` reads the
stored `.npz` and draws in seconds. Those `.npz` files are gitignored, so they exist
only in a working copy — which is what makes recording their hashes worth doing. The
index of which producer makes which figure, what it reads, and how long it costs is
`../../paper/make_figures.py`:

```sh
uv run python paper/make_figures.py --list                  # the whole dependency table
uv run python paper/make_figures.py --run draw              # every redraw here, ~1 min
uv run python paper/make_figures.py --run unmodelled_figure # one of them
uv run python paper/make_figures.py --check                 # is anything stale?
```

`save()` records each figure in `paper/figures/MANIFEST.json` together with the
SHA-256 of every `.npz` it read, so `--check` catches the case that matters here: a
chain re-sampled after the figure drawn from it, which otherwise leaves a stale PDF
in the paper with nothing to show for it. See `paper/validation/README.md` for what
a manifest entry holds.

Two reproducibility details specific to this directory:

- **`run_convergence.py` runs on the CPU** (`JAX_PLATFORMS=cpu`, overridable). GPU
  reductions are not bit-reproducible, and the Newton check was moving `one_vs_conv`
  by 6e-6 between runs — irrelevant to the two decimals the table quotes, but it
  rewrote `convergence.npz` every time and so invalidated the recorded provenance of
  `fig_wdm_tdi`. On the CPU the file is bit-identical run to run, agrees with the GPU
  to five significant figures, and the script is *faster* (20 s, no compilation).

- **The seed bases the P--P runs share** live in `fd_pipeline.py` as `SEED_TRUTH` and
  `SEED_NOISE`. `../pp/run_ppplot.py` and `../pp/run_decimation.py` must analyse the
  same realisations for their comparison to mean anything, and a matched pair of
  literals in two files is not a guarantee of that.
