"""Duration x SNR comparison for single-glitch inference, over many noise realisations.

Why this exists
---------------
Everything in the paper's joint analysis rests on one glitch at one working point
(tau = 300 s, rho = 42.7) and on one noise realisation. Two separate questions follow
from that, and this module answers both with the same set of runs:

1. *Does duration matter, and in which direction?* The damping time tau sets the
   spectral knee f_knee = 1/(2 pi tau), so it decides how much of the glitch power
   lands inside the 3 mHz analysis band and how sharply the broken power law is
   resolved. "Short" and "long" are therefore statements about the knee, not about
   the amplitude, which is why Deltav is rescaled at every tau to hold the optimal
   SNR fixed. Without that rescaling one measures the trivial fact that a glitch
   with less in-band power is harder to measure.

2. *Are the quoted widths the true scatter?* A single realisation cannot say. Here
   each configuration is repeated over `n_real` independent noise draws, and what is
   recorded per realisation is the standardised offset

       z_i = (posterior median_i - truth_i) / posterior sigma_i ,

   whose spread across realisations is the honest answer: std(z) = 1 means the
   error bars are the scatter, std(z) > 1 means they are optimistic.

Grid
----
tau in {100, 300, 3000} s   -- knee at {1.6, 0.53, 0.053} mHz, i.e. near the top of
                               the band, the fiducial, and near the bottom
rho in {15, 42.7, 150}      -- below, at, and well above the resolvability boundary
                               that Muratore et al. and Pozzoli et al. discuss

The centre cell is the paper's fiducial glitch, so the grid extends the published
working point rather than replacing it.

Parametrisation and priors are the ones of `glitch_GB/fd_pipeline.py`: sampling in
(t0, log A_g, log tau) with A_g = Deltav * tau, flat there, plus the LPF support cut
on Deltav. No Galactic binary is present -- this isolates the glitch-side scaling
from the source-confusion question studied in the joint notebooks.

Usage
-----
    python glitch_comparison.py --n-real 24 --workers 16     # run (CPU, parallel)
    python glitch_comparison.py --collect                    # gather -> npz

Runs are cached one .npz per (configuration, realisation), so the driver is
resumable and the notebook can call `run_all()` without repeating finished work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = os.path.dirname(HERE)                     # noise.py
GLITCH_GB = os.path.join(NOTEBOOKS, "glitch_GB")      # fd_pipeline.py
RUNDIR = os.path.join(HERE, "comparison_runs")
SUMMARY = os.path.join(HERE, "glitch_comparison.npz")

# --- the data set ----------------------------------------------------------
# Same sampling step as the paper (Nyquist at f_max = 3 mHz) but a segment 16 times
# shorter than the year. That is not an approximation here: the optimal SNR and the
# Fisher widths are integrals, rho^2 = 4 int |h|^2/S df, so shortening the segment only
# coarsens the Riemann sum and raises f_min from 3.2e-8 to 5.1e-7 Hz -- a region where
# the steeply rising acceleration PSD leaves the glitch no measurable power. Checked
# directly in the notebook: Deltav(rho), sigma_t0, sigma_Deltav and sigma_tau agree with
# the full-year grid to better than 0.01% at every tau in the grid below, while the
# likelihood costs 16 times less. The one thing that does change is the noise draw,
# which is the point of averaging over realisations.
N_SAMP = 187392 // 16                 # 11712 samples, 5857 positive-frequency bins
DT = 1.0 / (2 * 3e-3)                 # Nyquist at f_max = 3 mHz
T0_TRUE = 5000.0                      # far from both t0 prior walls

TAUS = (100.0, 300.0, 3000.0)
RHOS = (15.0, 42.7, 150.0)

# 99% of the velocity kick is delivered by t0 + X99 * tau, from (1+x) e^-x = 0.01
X99 = 6.638

PARAM_NAMES = ("t0", "log_Ag", "log_tau", "log_Deltav")


def config_id(i_tau: int, i_rho: int) -> str:
    return f"tau{i_tau}_rho{i_rho}"


# ---------------------------------------------------------------------------
# one realisation
# ---------------------------------------------------------------------------

_PINNED = False


def _pin_to_one_core():
    """Hold each worker to a single core.

    Setting OMP/MKL/XLA thread counts is not enough: the JAX CPU backend still opens
    a thread pool sized by nproc, so N workers each spawn N threads and a 22-core box
    ends up at load ~130 doing less work than one process alone. CPU affinity is the
    only cap that holds, and this study is embarrassingly parallel across runs, so
    one core per run costs nothing. Done once per process, before JAX is imported.
    """
    global _PINNED
    if _PINNED:
        return
    _PINNED = True
    try:
        import multiprocessing as mp
        ident = mp.current_process()._identity
        cores = sorted(os.sched_getaffinity(0))
        if ident and cores:
            os.sched_setaffinity(0, {cores[(ident[0] - 1) % len(cores)]})
    except (AttributeError, OSError, IndexError):
        pass


def worker(job) -> dict:
    """One (configuration, realisation) pair, end to end, in its own CPU process."""
    i_tau, i_rho, k = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               "intra_op_parallelism_threads=1")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _pin_to_one_core()
    for p in (NOTEBOOKS, GLITCH_GB):
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr
    import jaxglitches as jg
    import noise as ns
    import fd_pipeline as fp

    out = os.path.join(RUNDIR, config_id(i_tau, i_rho))
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, f"real_{k:03d}.npz")
    if os.path.exists(done):
        return dict(job=job, cached=True)

    tau, rho_target = TAUS[i_tau], RHOS[i_rho]

    freq = jnp.asarray(np.fft.rfftfreq(N_SAMP, DT))
    t_obs = N_SAMP * DT
    psd = ns.psd_tdi1_array(jnp.where(freq > 0, freq, 1.0), t_obs=t_obs)

    def h_of(th):
        """Template from sampling coordinates th = [t0, log A_g, log tau]."""
        tt = jnp.exp(th[2])
        return jg.clean_signal_f(jnp.stack([th[0], jnp.exp(th[1]) / tt, tt]),
                                 freq, T=jg.T_ARM_s, tdi=1).at[0].set(0 + 0j)

    # --- amplitude set by the target SNR: rho is linear in Deltav ----------
    probe = h_of(jnp.array([T0_TRUE, np.log(1e-11 * tau), np.log(tau)]))
    dv = 1e-11 * rho_target / float(jg.snr(probe, psd))
    th_true = jnp.array([T0_TRUE, float(np.log(dv * tau)), float(np.log(tau))])
    h_true = h_of(th_true)
    rho_opt = float(jg.snr(h_true, psd))

    # --- prior: wide box in every direction, LPF support on Deltav --------
    lo = jnp.array([0.0, float(th_true[1]) - 5.0, float(np.log(0.1))])
    hi = jnp.array([1e4, float(th_true[1]) + 5.0, float(np.log(5e4))])
    ldv = (float(np.log(1e-16)), float(np.log(1e-7)))

    @jax.jit
    def log_prior(th):
        ok = jnp.all((th >= lo) & (th <= hi))
        d = th[1] - th[2]
        return jnp.where(ok & (d >= ldv[0]) & (d <= ldv[1]), 0.0, -jnp.inf)

    # --- data: signal + a fresh noise draw --------------------------------
    seed = 700_000 + 1000 * (3 * i_tau + i_rho) + k
    data = h_true + ns.sample_noise_fd(jr.PRNGKey(seed), psd)

    @jax.jit
    def log_lik(th):
        return jg.log_likelihood(data, h_of(th), psd)

    log_post = jax.jit(lambda t: log_lik(t) + log_prior(t))

    fallback = jnp.array([10.0, 0.1, 0.1])
    xmap, sig = fp.laplace(log_post, th_true, fallback)
    if not bool(jnp.all(jnp.isfinite(xmap))):
        xmap = th_true
    # 16 x 2400 post-burn-in samples of a three-parameter posterior: the sampler's
    # per-iteration cost here is Python dispatch, not the likelihood, so chain length
    # rather than grid size sets the wall clock.
    chain = np.asarray(fp.run_chain(log_lik, log_prior, xmap, sig, 3, seed=seed,
                                    nwalkers=16, nburn=600, nsamp=2400))

    # log Deltav = log A_g - log tau, appended as a fourth reported parameter
    chain4 = np.column_stack([chain, chain[:, 1] - chain[:, 2]])
    true4 = np.append(np.asarray(th_true), float(th_true[1] - th_true[2]))

    med = np.median(chain4, axis=0)
    std = chain4.std(axis=0)
    q16, q84 = np.percentile(chain4, [16, 84], axis=0)
    quant = np.array([(chain4[:, i] < true4[i]).mean() for i in range(4)])

    # recovered SNR: the matched filter of the posterior-median template against
    # the data it was fitted to. Its realisation-to-realisation scatter is the
    # quantity Pozzoli et al. report for MBHBs.
    h_med = h_of(jnp.asarray(med[:3]))
    rho_rec = float(jg.inner_product(data, h_med, psd) / jg.snr(h_med, psd))

    np.savez_compressed(
        done, theta_true=true4, median=med, std=std, q16=q16, q84=q84,
        quantile=quant, sigma_laplace=np.asarray(sig), xmap=np.asarray(xmap),
        rho_opt=rho_opt, rho_rec=rho_rec, tau=tau, rho_target=rho_target,
        deltav=dv, seed=seed,
        chain=(chain4[::20].astype(np.float32) if k == 0 else np.zeros((0, 4), np.float32)),
    )
    return dict(job=job, cached=False, rho_opt=rho_opt, rho_rec=rho_rec)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def jobs(n_real: int):
    return [(i, j, k) for i in range(len(TAUS)) for j in range(len(RHOS))
            for k in range(n_real)]


def run_all(n_real: int = 24, workers: int = 16, verbose: bool = True) -> str:
    """Run (or resume) the whole grid, then collect. Returns the summary path."""
    import multiprocessing as mp
    os.makedirs(RUNDIR, exist_ok=True)
    json.dump(dict(N=N_SAMP, DT=DT, t0=T0_TRUE, taus=list(TAUS), rhos=list(RHOS),
                   n_real=n_real),
              open(os.path.join(RUNDIR, "config.json"), "w"), indent=1)
    todo = jobs(n_real)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(worker, todo), 1):
            if verbose:
                i, j, k = r["job"]
                print(f"[{n:4d}/{len(todo)}] tau={TAUS[i]:6.0f} rho={RHOS[j]:6.1f} "
                      f"real {k:3d}{'  (cached)' if r['cached'] else ''}", flush=True)
    if verbose:
        print(f"\n{len(todo)} runs in {(time.time() - t0) / 60:.1f} min")
    return collect(n_real)


def collect(n_real: int | None = None) -> str:
    """Gather the per-realisation files into one npz keyed by configuration."""
    import numpy as np
    out = {}
    chains = {}
    for i in range(len(TAUS)):
        for j in range(len(RHOS)):
            d = os.path.join(RUNDIR, config_id(i, j))
            if not os.path.isdir(d):
                continue
            files = sorted(f for f in os.listdir(d) if f.startswith("real_"))
            if n_real is not None:
                files = files[:n_real]
            rows = [np.load(os.path.join(d, f)) for f in files]
            if not rows:
                continue
            key = config_id(i, j)
            for field in ("theta_true", "median", "std", "q16", "q84", "quantile",
                          "sigma_laplace", "rho_opt", "rho_rec"):
                out[f"{key}/{field}"] = np.array([r[field] for r in rows])
            out[f"{key}/deltav"] = np.array(rows[0]["deltav"])
            ch = rows[0]["chain"]
            if ch.size:
                chains[f"{key}/chain"] = ch
    out.update(chains)
    out["taus"] = np.array(TAUS)
    out["rhos"] = np.array(RHOS)
    out["names"] = np.array(PARAM_NAMES)
    np.savez_compressed(SUMMARY, **out)
    print(f"wrote {SUMMARY}  ({len(out)} arrays)")
    return SUMMARY


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect(a.n_real)
    else:
        run_all(a.n_real, a.workers)
