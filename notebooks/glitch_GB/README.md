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
| `run_gb_free_sky.py` | does separating glitch from binary need the binary localised? | `gb_free_sky.npz` | ~10 min |
| `run_wdm_tdi2.py` | does the time--frequency split depend on the TDI generation? | `wdm_tdi2.npz` | ~1 h, GPU |
| `run_convergence.py` | are the chains converged, and has the Newton step? | `convergence.npz`, `tab_convergence.tex` | ~2 min |
| `make_fig_wdm_tdi.py` | — | `fig_wdm_tdi.pdf` | seconds |
| `make_corner_wdm_vs_fd.py` | — | `fig_wdm_corner.pdf` | seconds |

`fd_pipeline.py` is the shared implementation of the hybrid binned likelihood and
the sampler, extracted from `glitch_and_gb.ipynb` so that the scripts and the
P--P study in `../pp/` do not drift apart from it. `run_wdm_tdi2.py` carries its
own copy of the WDM construction, since that one lives in the other notebook.

Chains are stored **walker-major**: a flat `(n_walker * n_iter, dim)` array whose
reshape is `(n_walker, n_iter, dim)`. Reading it the other way round makes walkers
look correlated at 0.8 and shrinks every autocorrelation time by an order of
magnitude; `run_convergence.py` checks the layout before trusting it.

## Two results worth knowing before reusing this code

**One Newton step is not enough.** The `(log f0, log fdot)` block is ill-conditioned
(cond(H) ≈ 1e17), so an undamped step from the injected values overshoots to a
log-posterior 30 below where it started. `fd_pipeline.laplace` now backtracks and
iterates, converging in five steps to a maximum that agrees with the chain median to
0.2 sigma. Stored outputs produced before that fix were initialised at the overshot
point; that only sets where walkers start, but it has not been regenerated.

**The eleven-parameter run is not converged.** `run_gb_free_sky.py` reaches
R-hat = 1.15 and an effective sample size of 598 in `log f0`. Its glitch block is
fine (R-hat = 1.000); its Galactic-binary block needs a longer run and a multi-start
check.
