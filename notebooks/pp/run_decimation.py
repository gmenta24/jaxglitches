"""P--P calibration of the *decimated* likelihood: the test the binning argument needs.

Sec. "Why binning and not decimation" argues that keeping every K-th bin and scaling by
K, Eq. (log L_K), gives the right posterior width but puts the maximum sqrt(K) widths
away from the truth. That is a statement about a distribution, so it can only be checked
over many realisations -- which is exactly what `run_ppplot.py` already does for the
hybrid binned likelihood. This script runs the same realisations through the decimated
likelihood instead.

Controlled comparison
---------------------
Truths and noise draws are generated from the same seeds as `run_ppplot.py`
(`10_000 + idx` and `500_000 + idx`) and the prior box is read from the same
`ppruns/config.json`. Realisation `idx` here is therefore the *same experiment* as
realisation `idx` there, analysed with a different likelihood, so the two P--P curves
differ only in the estimator.

What is measured
----------------
For each realisation and parameter,

    z = (posterior median - truth) / posterior width .

The derivation predicts std(z) = sqrt(K): the width is right, the offset is not. The
quantile of the truth in each marginal gives the P--P curve, which is the same
information in the form a reader recognises as a failure.

Usage
-----
    python run_decimation.py --K 54 --n 100 --workers 18
    python run_decimation.py --plot                      # aggregate + figure
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = os.path.dirname(HERE)
GLITCH_GB = os.path.join(NOTEBOOKS, "glitch_GB")
REPO_ROOT = os.path.dirname(NOTEBOOKS)

PPDIR = os.path.join(HERE, "ppruns")            # config + the hybrid reference runs
RUNDIR = os.path.join(HERE, "decruns")
SUMMARY = os.path.join(HERE, "decimation_summary.npz")

# sqrt(K) = 2.8, 4.0, 7.3, against 1.0 for the exact/binned reference. K = 54 keeps
# 1736 bins, which is the cost of the 1725-block hybrid grid: the like-for-like
# comparison, and the one that matters for the argument.
KS = (54, 16, 8)     # cheapest first, so the lever arm exists early

_PINNED = False


def _pin_to_one_core():
    """One core per worker. See notebooks/glitch_only/glitch_comparison.py for why the
    thread-count environment variables are not enough on their own."""
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
    """One (K, realisation) pair, end to end, in its own single-core CPU process."""
    K, idx = job
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _pin_to_one_core()
    for p in (NOTEBOOKS, GLITCH_GB):
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import noise as ns
    import fd_pipeline as fp

    out = os.path.join(RUNDIR, f"K{K:03d}")
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, f"run_{idx:04d}.npz")
    if os.path.exists(done):
        d = np.load(done)
        return dict(job=job, cached=True, zmax=float(np.abs(d["z"]).max()))

    meta = json.load(open(os.path.join(PPDIR, "config.json")))
    grid = fp.make_grid(meta["N"], meta["DT"])
    model, n_gb = fp.make_gb_model(grid["T_OBS"], meta["N_GB"])
    psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=grid["T_OBS"])
    co = fp.Coords(free_sky=False)
    bounds = {k: tuple(v) for k, v in meta["bounds"].items()}
    lo = np.array([bounds[n][0] for n in co.names])
    hi = np.array([bounds[n][1] for n in co.names])
    log_prior = fp.make_log_prior(co, bounds)

    # ---- same truth as run_ppplot.py realisation `idx` ----
    rng = np.random.default_rng(10_000 + idx)
    while True:
        th = jnp.asarray(rng.uniform(lo, hi))
        if np.isfinite(float(log_prior(th))):
            break
    gb8, g3 = co.to_physical(th)

    # ---- same data as run_ppplot.py realisation `idx` ----
    h_gb = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"])
    h_gl = fp.glitch_fd(g3, grid["freq"]).at[0].set(0 + 0j)
    n_fd = ns.sample_noise_fd(jr.PRNGKey(500_000 + idx), psd)
    data = h_gb + h_gl + n_fd

    log_lik, dmeta = fp.build_decimated(grid, data, psd, K, model, n_gb, co)
    log_post = jax.jit(lambda t: log_lik(t) + log_prior(t))

    fallback = jnp.asarray([1e-6, 0.3, 0.02, 0.05, 1.0, 0.05, 0.05])
    xmap, sig = fp.laplace(log_post, th, fallback)
    if not bool(jnp.all(jnp.isfinite(xmap))):
        xmap = th
    chain = np.asarray(fp.run_chain(log_lik, log_prior, xmap, sig, co.dim,
                                    seed=idx, nwalkers=16, nburn=2000, nsamp=10000))

    # float64 throughout: sigma(log f0) is ~1e-7 at a value of -6.2, below the float32
    # resolution there, so a float32 store would quantise it away.
    med, std = np.median(chain, axis=0), chain.std(axis=0)
    q = np.array([(chain[:, i] < float(th[i])).mean() for i in range(co.dim)])
    z = (med - np.asarray(th)) / std

    np.savez_compressed(done, theta_true=np.asarray(th), median=med, std=std,
                        quantiles=q, z=z, K=K, n_dec=dmeta["n_dec"],
                        names=np.array(co.names))
    return dict(job=job, cached=False, zmax=float(np.abs(z).max()))


def run(ks, n, workers):
    import multiprocessing as mp
    os.makedirs(RUNDIR, exist_ok=True)
    jobs = [(K, i) for K in ks for i in range(n)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for c, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            K, i = r["job"]
            print(f"[{c:4d}/{len(jobs)}] K={K:3d} run {i:4d}  max|z| {r['zmax']:6.2f}"
                  f"{'  (cached)' if r['cached'] else ''}", flush=True)
    print(f"\n{len(jobs)} runs in {(time.time() - t0) / 60:.1f} min")


def aggregate():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(REPO_ROOT, "paper", "validation"))
    from _style import COL_IN, FULL_IN, C, save

    disp = [r'$\log f_0$', r'$\log\dot f$', r'$\log\mathcal{A}$', r'$\psi$',
            r'$t_0$', r'$\log A_g$', r'$\log\tau$']

    # ---- the K = 1 reference: the hybrid binned runs of run_ppplot.py ---------
    hyb = [np.load(os.path.join(PPDIR, d, "chain.npz")) for d in
           sorted(x for x in os.listdir(PPDIR) if x.startswith("run_"))]
    names = [str(x) for x in hyb[0]["names"]]
    snr_gl = np.array([float(r["snr_gl"]) for r in hyb])
    Qh = np.array([r["quantiles"] for r in hyb])
    hmed = np.array([np.median(r["chain"].astype(np.float64), axis=0) for r in hyb])
    hstd = np.array([r["chain"].astype(np.float64).std(axis=0) for r in hyb])
    htru = np.array([r["theta_true"] for r in hyb])
    zh = (hmed - htru) / hstd

    # `log_f0` is not recoverable from those chains: they are stored float32 and its
    # width, 1e-7 at a value of -6.2, is below the float32 resolution there. Its
    # quantile is exact (computed before the cast) and panel (a) uses it; the
    # width-based statistic of panel (b) drops it, for the reference and for the
    # decimated runs alike so that the comparison stays like-for-like.
    keep = [i for i, n in enumerate(names) if n != "log_f0"]

    data = {}
    for K in sorted(KS):
        d = os.path.join(RUNDIR, f"K{K:03d}")
        if not os.path.isdir(d):
            continue
        fs = sorted(x for x in os.listdir(d) if x.startswith("run_"))
        rows = [np.load(os.path.join(d, f)) for f in fs]
        if not rows:
            continue
        data[K] = dict(q=np.array([r["quantiles"] for r in rows]),
                       z=np.array([r["z"] for r in rows]),
                       std=np.array([r["std"] for r in rows]),
                       snr=snr_gl[[int(f[4:8]) for f in fs]],
                       n_dec=int(rows[0]["n_dec"]), n=len(rows))

    # ---- the sample the sqrt(K) law is tested on ----------------------------
    # All hundred realisations, with no signal-to-noise cut. An earlier version of
    # this study restricted to rho_gl > 5 because six low-SNR realisations pushed the
    # K = 1 reference to std(z) = 1.43; that turned out to be a sampler bug rather
    # than a property of those realisations -- their initial ensembles started
    # outside the prior support and never moved -- and it is fixed in
    # fd_pipeline.run_chain. See Sec. "Are the credible intervals credible?".
    #
    # Two scales are reported. std(z) is the one the sqrt(K) law is about. The
    # interquartile scale is reported alongside because at large K a growing fraction
    # of the Galactic-binary marginals become prior-dominated: their width is the
    # prior's, not the likelihood's, so their z is not the linear-regime object the
    # law describes, and it enters std(z) as an outlier either way.
    RHO_MIN = 0.0
    rob = lambda v: float(np.subtract(*np.percentile(v, [75, 25])) / 1.349)

    sets = {1: (zh, snr_gl, len(zh), 93697)}
    for K, d in data.items():
        sets[K] = (d["z"], d["snr"], d["n"], d["n_dec"])

    print(f"\n{'K':>4}{'n':>5}{'bins':>8}{'sqrt(K)':>9}{'std(z)':>9}"
          f"{'/sqrt(K)':>10}{'IQR scale':>11}{'/sqrt(K)':>10}")
    meas, ratio, ratio_rob = {}, {}, {}
    for K, (z, s, n, nb) in sets.items():
        zz = z[:, keep][s > RHO_MIN]
        r = zz.ravel().std(ddof=1)
        meas[K] = zz.std(axis=0, ddof=1)
        ratio[K] = r / np.sqrt(K)
        ratio_rob[K] = rob(zz.ravel()) / np.sqrt(K)
        print(f"{K:>4}{n:5d}{nb:8d}{np.sqrt(K):9.2f}{r:9.2f}{ratio[K]:10.2f}"
              f"{rob(zz.ravel()):11.2f}{ratio_rob[K]:10.2f}")

    print(f"\nper parameter, in units of sqrt(K)")
    print(f"{'param':11s}" + "".join(f"{'K=%d' % K:>9}" for K in sets))
    for j, i in enumerate(keep):
        print(f"{names[i]:11s}"
              + "".join(f"{meas[K][j] / np.sqrt(K):9.2f}" for K in sets))

    # marginals whose +-3 sigma no longer fits the prior box
    meta = json.load(open(os.path.join(PPDIR, "config.json")))
    lo = np.array([meta["bounds"][n][0] for n in names])
    hi = np.array([meta["bounds"][n][1] for n in names])
    igb = [names.index(n) for n in ("log_f0", "log_fdot", "log_A_gb", "psi")]
    pd = {K: float(np.mean(6 * data[K]["std"][:, igb] > (hi - lo)[igb])) for K in data}
    tail = {K: float(np.mean((data[K]["q"] < 0.05) | (data[K]["q"] > 0.95)))
            for K in data}
    tail[1] = float(np.mean((Qh < 0.05) | (Qh > 0.95)))
    print(f"\n{'K':>4}{'prior-dominated GB':>21}{'q outside central 90%':>24}")
    print(f"{1:>4}{'-':>21}{tail[1]:24.3f}")
    for K in data:
        print(f"{K:>4}{pd[K]:21.3f}{tail[K]:24.3f}")

    # ---- figure --------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(FULL_IN, 2.45))
    n = len(Qh)
    ax = axs[0]
    x = np.linspace(0, 1, 200)
    for k in (1, 2, 3):
        e = k * np.sqrt(x * (1 - x) / n)
        ax.fill_between(x, x - e, x + e, color=C["grey"], alpha=0.12, lw=0)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    cols = {8: C["orange"], 16: C["red"], 54: C["purple"]}
    for i in range(Qh.shape[1]):
        ax.plot(np.sort(Qh[:, i]), np.arange(1, n + 1) / n, color=C["blue"], lw=0.9,
                alpha=0.85, label="hybrid (binned)" if i == 0 else None)
    for K, d in data.items():
        m = len(d["q"])
        for i in range(d["q"].shape[1]):
            ax.plot(np.sort(d["q"][:, i]), np.arange(1, m + 1) / m,
                    color=cols.get(K, C["grey"]), lw=0.9, alpha=0.85,
                    label=rf"decimated, $K={K}$" if i == 0 else None)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("credible interval"); ax.set_ylabel("fraction of injections")
    ax.set_title("(a) coverage", loc="left")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(alpha=0.3, lw=0.3)

    ax = axs[1]
    t = np.linspace(0.9, 8.0, 100)
    ax.plot(t, t, color="k", lw=0.9, ls="--", label=r"$\mathrm{std}(z)=\sqrt{K}$")
    pcol = [C["orange"], C["green"], C["red"], C["purple"], C["cyan"], C["yellow"]]
    for j, i in enumerate(keep):
        ax.plot([np.sqrt(K) for K in sets], [meas[K][j] for K in sets], "o", ms=3.2,
                alpha=0.8, color=pcol[j % len(pcol)], label=disp[i])
    ax.plot([np.sqrt(K) for K in sets], [ratio[K] * np.sqrt(K) for K in sets], "k-",
            lw=1.0, alpha=0.6, label="pooled")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([np.sqrt(K) for K in sets])
    ax.set_xticklabels([rf"$\sqrt{{{K}}}$" for K in sets])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"$\sqrt{K}$"); ax.set_ylabel(r"$\mathrm{std}(z)$")
    ax.set_title("(b) offset of the maximum", loc="left")
    ax.legend(fontsize=5.5, loc="lower right", ncol=2, columnspacing=0.8)
    ax.grid(alpha=0.3, which="both", lw=0.3)
    save(fig, "fig_decimation")

    np.savez_compressed(SUMMARY,
                        **{f"K{K}_q": d["q"] for K, d in data.items()},
                        **{f"K{K}_z": d["z"] for K, d in data.items()},
                        **{f"K{K}_ndec": d["n_dec"] for K, d in data.items()},
                        Ks=np.array(sorted(data)), names=np.array(names),
                        hybrid_q=Qh, hybrid_z=zh, snr_gl=snr_gl,
                        std_ratio=np.array([ratio[K] for K in sets]),
                        robust_ratio=np.array([ratio_rob[K] for K in sets]),
                        prior_dominated=np.array([pd[K] for K in data]),
                        tail_fraction=np.array([tail[K] for K in sets]))
    print(f"\nwrote {SUMMARY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="*", default=list(KS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        aggregate()
    else:
        run(a.K, a.n, a.workers)
