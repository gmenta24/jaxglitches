"""Frequency-domain run with the Galactic-binary sky and orientation free.

Sec. 5 of the paper holds (ra, dec, iota, phi0) fixed, which is defensible for a
controlled study of the glitch--binary degeneracy but leaves open whether the
conclusions survive when the binary is not assumed to be already localised: the sky
position sets the annual Doppler modulation, hence the width of the line, hence how
much of the glitch the GB window can absorb.

This script repeats the analysis with all eight GB parameters sampled -- eleven in
total -- on the same stored data stream, using the same hybrid binned likelihood.
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # fd_pipeline.py
sys.path.insert(0, os.path.dirname(HERE))      # notebooks/, where noise.py lives

import numpy as np
import jax, jax.numpy as jnp
import noise as ns
import fd_pipeline as fp

OUT = os.path.join(HERE, "gb_free_sky.npz")

DS = np.load(os.path.join(HERE, "dataset.npz"))
grid = fp.make_grid(int(DS["N"]), float(DS["DT"]))
model, n_gb = fp.make_gb_model(grid["T_OBS"], int(DS["N_GB"]))
psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=grid["T_OBS"])
data = jnp.asarray(DS["data_tdi1"])
gb_true = jnp.asarray(DS["gb_true"])
g3_true = jnp.asarray(DS["glitch_true"])
k_ref = int(DS["k_min"])

print(f"data: N = {grid['N']:,}, T_obs = {grid['T_OBS']:.4e} s")
print(f"GB SNR {float(__import__('jaxglitches').snr(jnp.asarray(DS['h_gb_tdi1']), psd)):.2f}"
      f"   glitch SNR {float(__import__('jaxglitches').snr(jnp.asarray(DS['h_glitch_tdi1']), psd)):.2f}")

results = {}
for free_sky in (False, True):
    tag = "free" if free_sky else "fixed"
    co = fp.Coords(free_sky=free_sky)
    th_true = co.to_sampling(gb_true, g3_true)

    # box prior: same as the paper, plus isotropic sky/orientation when free
    b = {
        "log_f0":   (np.log(1e-4), np.log(3e-3)),
        "log_fdot": (np.log(1e-22), np.log(1e-15)),
        "log_A_gb": (np.log(1e-25), np.log(1e-20)),
        "psi":      (0.0, float(np.pi)),
        "t0":       (0.0, float(grid["T_OBS"])),
        "log_Ag":   (float(np.log(1e-17)), float(np.log(5e-3))),
        "log_tau":  (float(np.log(0.1)), float(np.log(5e4))),
    }
    if free_sky:
        b |= {"ra": (0.0, 2 * np.pi), "sin_dec": (-1.0, 1.0),
              "cos_iota": (-1.0, 1.0), "phi0": (0.0, 2 * np.pi)}
    log_prior = fp.make_log_prior(co, b)
    log_lik, meta = fp.build_hybrid(grid, data, psd, k_ref, model, n_gb, co)

    log_post = jax.jit(lambda t: log_lik(t) + log_prior(t))
    fallback = jnp.where(jnp.arange(co.dim) < 0, 1.0, 0.05)
    t0 = time.time()
    xmap, sig = fp.laplace(log_post, th_true, fallback)
    print(f"\n[{tag}] dim = {co.dim}, blocks = {meta['n_blocks']}, "
          f"MAP in {time.time()-t0:.1f} s")
    for n, s, m, tv in zip(co.names, sig, xmap, th_true):
        print(f"   {n:9s} sigma {float(s):10.4g}   (MAP-truth)/sigma {float((m-tv)/s):+7.2f}")

    t0 = time.time()
    chain = fp.run_chain(log_lik, log_prior, xmap, sig, co.dim, seed=11 + free_sky,
                         nwalkers=32 if free_sky else 16, nburn=4000, nsamp=20000)
    print(f"[{tag}] chain {tuple(chain.shape)} in {time.time()-t0:.1f} s")
    smc = jnp.std(chain, axis=0)
    med = jnp.median(chain, axis=0)
    print(f"   {'param':10s}{'sigma':>12s}{'(med-truth)/sigma':>20s}")
    for n, s, m, tv in zip(co.names, smc, med, th_true):
        print(f"   {n:10s}{float(s):12.4g}{float((m-tv)/s):20.2f}")
    results[tag] = dict(chain=np.asarray(chain), theta_true=np.asarray(th_true),
                        sigma=np.asarray(smc), names=np.array(co.names))

# how much does freeing the sky cost the parameters common to both?
fx, fr = results["fixed"], results["free"]
common = [n for n in fx["names"] if n in set(fr["names"].tolist())]
print(f"\n{'parameter':11s}{'sigma fixed':>13s}{'sigma free':>12s}{'ratio':>8s}")
for n in common:
    i, j = list(fx["names"]).index(n), list(fr["names"]).index(n)
    a, b_ = fx["sigma"][i], fr["sigma"][j]
    print(f"{n:11s}{a:13.4g}{b_:12.4g}{b_/a:8.2f}")

np.savez_compressed(OUT, **{f"{k}_{kk}": vv for k, v in results.items()
                            for kk, vv in v.items()})
print(f"\nsaved {OUT}")
