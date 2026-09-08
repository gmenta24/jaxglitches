"""P--P calibration test for the frequency-domain hybrid-grid analysis.

Everything in the paper rests on one noise realisation. A P--P plot answers the
question that a single realisation cannot: are the quoted credible intervals actually
credible? For each of N realisations we draw parameters from the prior, simulate data,
sample the posterior, and record the quantile at which the injected value falls in each
one-dimensional marginal. If the posterior is calibrated those quantiles are uniform,
so the empirical CDF of each parameter should follow the diagonal.

Chains are written one directory per realisation (`notebooks/pp/ppruns/run_XXXX/`) rather than kept
in memory: 100 chains of 16 x 10000 x 7 float64 would be several GB.

Usage
-----
    python run_ppplot.py --n 100 --workers 20        # run (CPU, parallel)
    python run_ppplot.py --plot                      # aggregate + figure

Workers are CPU-only and single-threaded on purpose. JAX would otherwise start ~22
threads per process and, on the GPU, each process would try to reserve a slice of a
card that only has room for one.
"""
import argparse, os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = os.path.dirname(HERE)                     # noise.py
GLITCH_GB = os.path.join(NOTEBOOKS, "glitch_GB")      # fd_pipeline.py, dataset.npz
REPO_ROOT = os.path.dirname(NOTEBOOKS)

RUNDIR = os.path.join(HERE, "ppruns")

_PINNED = False


def _pin_to_one_core():
    """One core per worker.

    The environment variables below are necessary and not sufficient: XLA still starts
    a thread pool per process, and 20 workers x 22 threads drives the load average past
    the core count and the throughput to nearly zero. Pinning the process to a single
    core is what actually enforces it. See notebooks/glitch_only/glitch_comparison.py.
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


def worker(idx: int) -> dict:
    """One realisation, end to end. Runs in its own single-core CPU process."""
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
    import jax, jax.numpy as jnp, jax.random as jr
    import noise as ns
    import fd_pipeline as fp

    out = os.path.join(RUNDIR, f"run_{idx:04d}")
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, "chain.npz")
    if os.path.exists(done):                       # resumable
        d = np.load(done)
        return dict(idx=idx, quantiles=d["quantiles"].tolist(),
                    snr_gb=float(d["snr_gb"]), snr_gl=float(d["snr_gl"]), cached=True)

    meta = json.load(open(os.path.join(RUNDIR, "config.json")))
    grid = fp.make_grid(meta["N"], meta["DT"])
    model, n_gb = fp.make_gb_model(grid["T_OBS"], meta["N_GB"])
    psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=grid["T_OBS"])
    co = fp.Coords(free_sky=False)
    bounds = {k: tuple(v) for k, v in meta["bounds"].items()}
    lo = np.array([bounds[n][0] for n in co.names])
    hi = np.array([bounds[n][1] for n in co.names])
    log_prior = fp.make_log_prior(co, bounds)

    # ---- draw truth from the prior, rejecting the Deltav-support cut ----
    rng = np.random.default_rng(fp.SEED_TRUTH + idx)
    while True:
        th = jnp.asarray(rng.uniform(lo, hi))
        if np.isfinite(float(log_prior(th))):
            break
    gb8, g3 = co.to_physical(th)

    # ---- data: signals + a fresh noise draw ----
    h_gb = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"])
    h_gl = fp.glitch_fd(g3, grid["freq"]).at[0].set(0 + 0j)
    n_fd = ns.sample_noise_fd(jr.PRNGKey(fp.SEED_NOISE + idx), psd)
    data = h_gb + h_gl + n_fd
    import jaxglitches as jg
    snr_gb, snr_gl = float(jg.snr(h_gb, psd)), float(jg.snr(h_gl, psd))

    k_ref = int(model.get_kmin(gb8[None, 0])[0])
    log_lik, _ = fp.build_hybrid(grid, data, psd, k_ref, model, n_gb, co)
    log_post = jax.jit(lambda t: log_lik(t) + log_prior(t))

    fallback = jnp.asarray([1e-6, 0.3, 0.02, 0.05, 1.0, 0.05, 0.05])
    xmap, sig = fp.laplace(log_post, th, fallback)
    if not bool(jnp.all(jnp.isfinite(xmap))):
        xmap = th
    chain = fp.run_chain(log_lik, log_prior, xmap, sig, co.dim,
                         seed=idx, nwalkers=16, nburn=2000, nsamp=10000)
    ch = np.asarray(chain)
    q = [(ch[:, i] < float(th[i])).mean() for i in range(co.dim)]

    # `q` above is computed in float64, before the cast, and is what the P--P plot
    # uses. The stored chain is float32 to keep 100 of them on disk, and that is lossy
    # for `log_f0`: its width is ~1e-7 at a value of -6.2, below the float32 resolution
    # there, so the marginal comes back quantised into a handful of levels. Re-derive
    # log_f0 statistics from `quantiles`, not from `chain`.
    np.savez_compressed(done, chain=ch.astype(np.float32),
                        theta_true=np.asarray(th), quantiles=np.array(q),
                        snr_gb=snr_gb, snr_gl=snr_gl, names=np.array(co.names))
    return dict(idx=idx, quantiles=q, snr_gb=snr_gb, snr_gl=snr_gl, cached=False)


def setup(n):
    import numpy as np
    os.makedirs(RUNDIR, exist_ok=True)
    DS = np.load(os.path.join(GLITCH_GB, "dataset.npz"))
    df = 1.0 / (int(DS["N"]) * float(DS["DT"]))
    f0 = float(DS["gb_true"][0])
    # Prior for BOTH injection and analysis -- they must be the same for the test to
    # mean anything. Narrow enough that the signals are detectable, wide enough that
    # the posterior never touches a wall (the narrowest is ~13 sigma half-width).
    bounds = {
        "log_f0":   [float(np.log(f0 - 20 * df)), float(np.log(f0 + 20 * df))],
        "log_fdot": [float(np.log(1e-18)), float(np.log(1e-15))],
        "log_A_gb": [float(np.log(5e-22)), float(np.log(5e-21))],
        "psi":      [0.0, float(np.pi)],
        "t0":       [0.0, 1e4],
        "log_Ag":   [float(np.log(1e-9)), float(np.log(1e-8))],
        "log_tau":  [float(np.log(100.0)), float(np.log(1000.0))],
    }
    json.dump(dict(N=int(DS["N"]), DT=float(DS["DT"]), N_GB=int(DS["N_GB"]),
                   n=n, bounds=bounds),
              open(os.path.join(RUNDIR, "config.json"), "w"), indent=1)
    print(f"config written for {n} realisations")


def aggregate():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    sys.path.insert(0, os.path.join(REPO_ROOT, "paper", "validation"))
    from _style import COL_IN, C, save
    sys.path.insert(0, GLITCH_GB)
    import fd_pipeline as fp          # for the seed bases, in the manifest

    runs = sorted(d for d in os.listdir(RUNDIR) if d.startswith("run_"))
    Q, snr = [], []
    for d in runs:
        f = os.path.join(RUNDIR, d, "chain.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        Q.append(z["quantiles"]); snr.append((z["snr_gb"], z["snr_gl"]))
        names = [str(x) for x in z["names"]]
    Q = np.array(Q); snr = np.array(snr)
    n = len(Q)
    print(f"{n} realisations")
    print(f"  GB SNR {snr[:,0].min():.0f}-{snr[:,0].max():.0f}, "
          f"glitch SNR {snr[:,1].min():.0f}-{snr[:,1].max():.0f}")

    disp = [r'$\log f_0$', r'$\log\dot f$', r'$\log\mathcal{A}$', r'$\psi$',
            r'$t_0$', r'$\log A_g$', r'$\log\tau$']
    fig, ax = plt.subplots(figsize=(COL_IN, COL_IN))
    x = np.linspace(0, 1, 200)
    for k in (1, 2, 3):                       # binomial confidence bands
        e = k * np.sqrt(x * (1 - x) / n)
        ax.fill_between(x, x - e, x + e, color=C["grey"], alpha=0.12, lw=0)
    cols = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["cyan"],
            C["yellow"]]
    print(f"\n{'parameter':11s}{'KS p-value':>12s}")
    ps = []
    for i, (nm, col) in enumerate(zip(disp, cols)):
        q = np.sort(Q[:, i])
        ax.plot(q, np.arange(1, n + 1) / n, color=col, lw=1.1, label=nm)
        p = stats.kstest(Q[:, i], "uniform").pvalue
        ps.append(p)
        print(f"{names[i]:11s}{p:12.3f}")
    comb = stats.combine_pvalues(ps).pvalue
    print(f"{'combined':11s}{comb:12.3f}")
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("credible interval"); ax.set_ylabel("fraction of injections")
    ax.set_title(f"$N={n}$, combined $p={comb:.2f}$", fontsize=8, loc="left")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    # written before the figure so that `save` can hash it: `pp_summary.npz` is the
    # single artefact standing in for the whole `ppruns/` directory.
    summary = os.path.join(HERE, "pp_summary.npz")
    np.savez_compressed(summary, quantiles=Q, snr=snr,
                        names=np.array(names), pvalues=np.array(ps))
    save(fig, "fig_pp", inputs=[summary],
         seed={"truth": f"{fp.SEED_TRUTH} + idx", "noise": f"{fp.SEED_NOISE} + idx"},
         note=f"{n} realisations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        aggregate()
    else:
        setup(a.n)
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(worker, range(a.n)), 1):
                print(f"[{k:3d}/{a.n}] run {r['idx']:4d}  "
                      f"SNR GB {r['snr_gb']:7.1f} glitch {r['snr_gl']:6.1f}"
                      f"{'  (cached)' if r['cached'] else ''}", flush=True)
        print(f"\n{a.n} realisations in {(time.time()-t0)/60:.1f} min")
