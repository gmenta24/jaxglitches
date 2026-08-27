"""Frequency-domain run with the Galactic-binary sky and orientation free.

Sec. "Freeing the sky and the orientation" holds (ra, dec, iota, phi0) fixed elsewhere,
which is defensible for a controlled study of the glitch--binary degeneracy but leaves
open whether the conclusions survive when the binary is not assumed to be already
localised: the sky position sets the annual Doppler modulation, hence the width of the
line, hence how much of the glitch the GB window can absorb.

This script repeats the analysis with all eight GB parameters sampled -- eleven in
total -- on the same stored data stream, using the same hybrid binned likelihood.

Multi-start
-----------
A quasi-monochromatic binary has discrete degeneracies, so a single chain launched from
the injection can sample one mode and report its width as if it were the posterior's.
The free-sky run is therefore launched from four physically motivated starting points,

    truth       the injected parameters
    psi-flip    psi -> psi + pi/2, phi0 -> phi0 + pi
    antipode    ra  -> ra + pi,    dec  -> -dec
    iota-flip   iota -> pi - iota, psi  -> psi + pi/2

each first taken to its local maximum by the damped Newton iteration and then sampled
independently. What the starts are for is the comparison between them: if they agree,
the widths are posterior widths; if they do not, the log-posterior offsets say which
mode carries the probability, and the widths of the others are not comparable.

The chains are also an order of magnitude longer than the earlier version of this
script, which reached an effective sample size of 598 in log f0 and R-hat = 1.15 --
not enough to quote a width to better than a few per cent.

Usage
-----
    python run_gb_free_sky.py                    # full run, ~1 h on one GPU
    python run_gb_free_sky.py --nsamp 5000       # quick check
"""
import argparse
import gc
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # fd_pipeline.py
sys.path.insert(0, os.path.dirname(HERE))      # notebooks/, where noise.py lives

import numpy as np
import jax
import jax.numpy as jnp
import noise as ns
import fd_pipeline as fp

OUT = os.path.join(HERE, "gb_free_sky.npz")

# How far below the best peak a start may sit and still be treated as a mode worth
# sampling. The two that matter here are separated by 0.00 and by 4e4 log-units,
# so nothing depends on the exact value.
LIVE_MARGIN = 20.0

# gb8 = [f0, fdot, A, ra, dec, psi, iota, phi0]
I_RA, I_DEC, I_PSI, I_IOTA, I_PHI0 = 3, 4, 5, 6, 7


def starting_points(gb_true):
    """The injected parameters and the three discrete degeneracies worth probing."""
    g = np.asarray(gb_true, dtype=float)

    psi_flip = g.copy()
    psi_flip[I_PSI] = (g[I_PSI] + np.pi / 2) % np.pi
    psi_flip[I_PHI0] = (g[I_PHI0] + np.pi) % (2 * np.pi)

    antipode = g.copy()
    antipode[I_RA] = (g[I_RA] + np.pi) % (2 * np.pi)
    antipode[I_DEC] = -g[I_DEC]

    iota_flip = g.copy()
    iota_flip[I_IOTA] = np.pi - g[I_IOTA]
    iota_flip[I_PSI] = (g[I_PSI] + np.pi / 2) % np.pi

    return {"truth": g, "psi-flip": psi_flip, "antipode": antipode,
            "iota-flip": iota_flip}


def summarise(chain, names, th_true, n_walker):
    """Median, width, split R-hat and acceptance of one flat walker-major chain.

    The acceptance fraction is measured here, on the *unthinned* chain, because it
    cannot be recovered cleanly afterwards: a Metropolis rejection repeats the
    previous sample, and a stored stride of ten hides all but a saturated one in
    a thousand of them.
    """
    flat = np.asarray(chain)
    per = flat.reshape(n_walker, -1, flat.shape[1]).transpose(1, 0, 2)   # (iter, walker, dim)
    med, sig = np.median(flat, axis=0), flat.std(axis=0)
    rhat = np.array([_split_rhat(per[:, :, i]) for i in range(flat.shape[1])])
    acc = float(np.mean(np.any(np.diff(per, axis=0) != 0.0, axis=-1)))
    return dict(median=med, sigma=sig, rhat=rhat, acceptance=acc,
                z=(med - np.asarray(th_true)) / sig, names=np.array(names))


def _split_rhat(x):
    """Split Gelman--Rubin R-hat of an (n_iter, n_walker) array."""
    n_iter, n_walker = x.shape
    h = n_iter // 2
    parts = np.concatenate([x[:h], x[h:2 * h]], axis=1)          # 2 * n_walker chains
    m, n = parts.shape[1], h
    means, varis = parts.mean(axis=0), parts.var(axis=0, ddof=1)
    B = n * means.var(ddof=1)
    W = varis.mean()
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nburn", type=int, default=20000)
    ap.add_argument("--nsamp", type=int, default=100000)
    ap.add_argument("--thin", type=int, default=10, help="stride for the stored chains")
    a = ap.parse_args()

    DS = np.load(os.path.join(HERE, "dataset.npz"))
    grid = fp.make_grid(int(DS["N"]), float(DS["DT"]))
    model, n_gb = fp.make_gb_model(grid["T_OBS"], int(DS["N_GB"]))
    psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=grid["T_OBS"])
    data = jnp.asarray(DS["data_tdi1"])
    gb_true = jnp.asarray(DS["gb_true"])
    g3_true = jnp.asarray(DS["glitch_true"])
    k_ref = int(DS["k_min"])

    import jaxglitches as jg
    print(f"data: N = {grid['N']:,}, T_obs = {grid['T_OBS']:.4e} s")
    print(f"GB SNR {float(jg.snr(jnp.asarray(DS['h_gb_tdi1']), psd)):.2f}"
          f"   glitch SNR {float(jg.snr(jnp.asarray(DS['h_glitch_tdi1']), psd)):.2f}")
    print(f"sampler: {a.nwalkers} walkers, {a.nburn} burn + {a.nsamp} samples, "
          f"stored thinned by {a.thin}\n")

    base = {
        "log_f0":   (np.log(1e-4), np.log(3e-3)),
        "log_fdot": (np.log(1e-22), np.log(1e-15)),
        "log_A_gb": (np.log(1e-25), np.log(1e-20)),
        "psi":      (0.0, float(np.pi)),
        "t0":       (0.0, float(grid["T_OBS"])),
        "log_Ag":   (float(np.log(1e-17)), float(np.log(5e-3))),
        "log_tau":  (float(np.log(0.1)), float(np.log(5e4))),
    }
    sky_bounds = {"ra": (0.0, 2 * np.pi), "sin_dec": (-1.0, 1.0),
                  "cos_iota": (-1.0, 1.0), "phi0": (0.0, 2 * np.pi)}

    out, results = {}, {}

    # ---------------- fixed sky: the reference the ratios are taken against ----
    co = fp.Coords(free_sky=False)
    th_true = co.to_sampling(gb_true, g3_true)
    log_prior = fp.make_log_prior(co, base)
    log_lik, meta = fp.build_hybrid(grid, data, psd, k_ref, model, n_gb, co)
    log_post = jax.jit(lambda t: log_lik(t) + log_prior(t))
    t0 = time.time()
    xmap, sig = fp.laplace(log_post, th_true, jnp.full(co.dim, 0.05))
    print(f"[fixed] dim = {co.dim}, blocks = {meta['n_blocks']}, "
          f"MAP in {time.time() - t0:.1f} s")
    t0 = time.time()
    chain = fp.run_chain(log_lik, log_prior, xmap, sig, co.dim, seed=11,
                         nwalkers=a.nwalkers, nburn=a.nburn, nsamp=a.nsamp)
    chain = np.asarray(chain)
    print(f"[fixed] chain {tuple(chain.shape)} in {time.time() - t0:.1f} s")
    s = summarise(chain, co.names, th_true, a.nwalkers)
    print(f"   max R-hat {np.nanmax(s['rhat']):.4f}   max |z| {np.abs(s['z']).max():.2f}")
    results["fixed"] = s
    out |= {"fixed_chain": _thin(chain, a.nwalkers, a.thin),
            "fixed_theta_true": np.asarray(th_true), "fixed_sigma": s["sigma"],
            "fixed_median": s["median"], "fixed_rhat": s["rhat"],
            "fixed_names": np.array(co.names),
            "fixed_acceptance": np.array(s["acceptance"])}
    del chain
    gc.collect()
    jax.clear_caches()

    # ---------------- free sky: one chain per starting mode -------------------
    co = fp.Coords(free_sky=True)
    th_true = co.to_sampling(gb_true, g3_true)
    log_prior = fp.make_log_prior(co, base | sky_bounds)
    log_lik, meta = fp.build_hybrid(grid, data, psd, k_ref, model, n_gb, co)
    log_post = jax.jit(lambda t: log_lik(t) + log_prior(t))

    starts = starting_points(np.asarray(gb_true))
    print(f"\n[free] dim = {co.dim}, blocks = {meta['n_blocks']}, "
          f"{len(starts)} starting points")

    # Phase 1 -- take every start to its local maximum. This is seconds each, and it
    # already answers most of the question: a start whose peak is thousands of log-units
    # below the best is not a mode of this posterior and does not need sampling.
    peaks, maxima = {}, {}
    for tag, gb0 in starts.items():
        th0 = co.to_sampling(jnp.asarray(gb0), g3_true)
        t0 = time.time()
        xmap, sig = fp.laplace(log_post, th0, jnp.full(co.dim, 0.05))
        # a start that is not near a maximum can be walked out of the prior box by the
        # Newton step; falling back keeps the reported peak a property of the start
        # rather than of the optimiser
        if float(log_post(xmap)) < float(log_post(th0)):
            xmap, sig = th0, jnp.full(co.dim, 0.05)
        peaks[tag] = float(log_post(xmap))
        maxima[tag] = (xmap, sig)
        print(f"[free/{tag}] MAP in {time.time() - t0:5.1f} s, "
              f"log-posterior {peaks[tag]:.2f}")

    best = max(peaks.values())
    live = [t for t in starts if peaks[t] > best - LIVE_MARGIN]
    print(f"\n=== starting points, log-posterior relative to the best ===")
    for tag, lp in peaks.items():
        print(f"  {tag:10s} {lp - best:+12.2f}"
              f"{'   <- sampled' if tag in live else '   (not a mode; not sampled)'}")

    # Phase 2 -- sample the modes that are actually competitive.
    for si, tag in enumerate(live):
        xmap, sig = maxima[tag]
        t0 = time.time()
        chain = fp.run_chain(log_lik, log_prior, xmap, sig, co.dim,
                             seed=101 + list(starts).index(tag),
                             nwalkers=a.nwalkers, nburn=a.nburn, nsamp=a.nsamp)
        chain = np.asarray(chain)      # to host before the next chain is allocated
        print(f"\n[free/{tag}] chain {tuple(chain.shape)} in {time.time() - t0:.1f} s")
        s = summarise(chain, co.names, th_true, a.nwalkers)
        s["logpost_map"] = peaks[tag]
        print(f"   max R-hat {np.nanmax(s['rhat']):.4f}   "
              f"max |z| {np.abs(s['z']).max():.2f}   acceptance {s['acceptance']:.3f}")
        results[f"free/{tag}"] = s
        key = f"free_{tag.replace('-', '_')}"
        out |= {f"{key}_chain": _thin(chain, a.nwalkers, a.thin),
                f"{key}_sigma": s["sigma"], f"{key}_median": s["median"],
                f"{key}_rhat": s["rhat"], f"{key}_logpost": np.array(peaks[tag]),
                f"{key}_acceptance": np.array(s["acceptance"])}
        if tag == "truth":
            # keys the rest of the repo already reads
            out |= {"free_chain": _thin(chain, a.nwalkers, a.thin),
                    "free_theta_true": np.asarray(th_true), "free_sigma": s["sigma"],
                    "free_median": s["median"], "free_rhat": s["rhat"],
                    "free_names": np.array(co.names),
                    "free_acceptance": np.array(s["acceptance"])}
        del chain
        gc.collect()
        jax.clear_caches()
    out |= {"free_peaks": np.array([peaks[t] for t in starts]),
            "free_start_names": np.array(list(starts))}

    ref = results["free/truth"]
    print(f"\n=== medians of each sampled mode, in units of the 'truth' width ===")
    print(f"{'param':11s}" + "".join(f"{t:>12s}" for t in live))
    for i, nm in enumerate(ref["names"]):
        row = "".join(f"{(results[f'free/{t}']['median'][i] - ref['median'][i]) / ref['sigma'][i]:12.2f}"
                      for t in live)
        print(f"{str(nm):11s}{row}")

    print(f"\n=== widths of each sampled mode, relative to the 'truth' mode ===")
    print(f"{'param':11s}" + "".join(f"{t:>12s}" for t in live))
    for i, nm in enumerate(ref["names"]):
        row = "".join(f"{results[f'free/{t}']['sigma'][i] / ref['sigma'][i]:12.2f}"
                      for t in live)
        print(f"{str(nm):11s}{row}")

    # pooled over every start that is within a few log-units of the best: if the
    # modes are really the same mode, this is the posterior; if not, it is not.
    print(f"\nmodes pooled: {live}")
    pooled = np.concatenate([np.asarray(out[f"free_{t.replace('-', '_')}_chain"])
                             for t in live], axis=0)
    # A parameter is "shared" between the live modes if their medians agree to within a
    # width. For those the pooled spread is the posterior width. For the rest the modes
    # are genuinely separated, the pooled spread measures the separation and not a
    # width, and the number to quote is the within-mode one.
    seps = np.max([np.abs(results[f"free/{t}"]["median"] - ref["median"]) / ref["sigma"]
                   for t in live], axis=0)
    shared = seps < 1.0
    print(f"\n{'param':11s}{'max |median offset| / sigma':>30s}{'shared?':>10s}")
    for i, nm in enumerate(ref["names"]):
        print(f"{str(nm):11s}{seps[i]:30.2f}{'yes' if shared[i] else 'NO':>10s}")
    out |= {"free_pooled_sigma": pooled.std(axis=0),
            "free_pooled_median": np.median(pooled, axis=0),
            "free_mode_separation": seps, "free_shared": shared,
            "free_live_starts": np.array(live)}

    # ---------------- the cost of freeing the sky -----------------------------
    fx, fr = results["fixed"], ref
    names_fx, names_fr = list(fx["names"]), list(fr["names"])
    print(f"\n{'parameter':11s}{'sigma fixed':>14s}{'sigma free':>13s}{'ratio':>8s}"
          f"{'ratio pooled':>14s}")
    ratios = {}
    for nm in names_fx:
        i, j = names_fx.index(nm), names_fr.index(nm)
        r = fr["sigma"][j] / fx["sigma"][i]
        rp = out["free_pooled_sigma"][j] / fx["sigma"][i]
        ratios[nm] = (r, rp)
        flag = "" if out["free_shared"][j] else "   (modes separated)"
        print(f"{str(nm):11s}{fx['sigma'][i]:14.4g}{fr['sigma'][j]:13.4g}"
              f"{r:8.2f}{rp:14.2f}{flag}")
    out |= {"ratio_names": np.array(names_fx),
            "ratio_truth": np.array([ratios[n][0] for n in names_fx]),
            "ratio_pooled": np.array([ratios[n][1] for n in names_fx])}
    out |= {"nwalkers": np.array(a.nwalkers), "thin": np.array(a.thin),
            "nsamp": np.array(a.nsamp), "nburn": np.array(a.nburn)}

    np.savez_compressed(OUT, **out)
    print(f"\nsaved {OUT}")


def _thin(chain, n_walker, stride):
    """Thin a flat walker-major chain, keeping the walker-major layout."""
    flat = np.asarray(chain)
    per = flat.reshape(n_walker, -1, flat.shape[1])[:, ::stride, :]
    # float64, not float32: sigma(log f0) is 2e-7 at a value of -6.2, which is below
    # the float32 resolution there, so a float32 store would quantise the tightest
    # parameter in the problem into a handful of levels.
    return per.reshape(-1, flat.shape[1]).astype(np.float64)


if __name__ == "__main__":
    main()
