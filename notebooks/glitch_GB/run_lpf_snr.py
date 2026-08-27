"""SNR distribution of a year of LPF-like glitches, on the grid of `dataset.npz`.

`run_knee_scan.py` returns a critical glitch signal-to-noise ratio at which an
unmodelled glitch would move a Galactic binary by one standard deviation. The number
that makes it meaningful is how often the LPF population actually gets there.

The waveform is linear in Delta_v and the arrival time only sets a phase, so

    rho(Delta_v, tau) = |Delta_v| * R(tau),   R(tau) = sqrt( (h|h) ) at Delta_v = 1,

and one evaluation per tau on a grid, interpolated in log tau, replaces one evaluation
per catalogue event. Twenty year-long draws then cost seconds instead of minutes.

Two resampling modes, and the difference matters
------------------------------------------------
`run_catalog(smooth=True)` draws (Deltav, tau) from a Gaussian KDE fitted to the LPF
catalogue and clips the result to the prior box, which reaches |Deltav| = 1e-7 m/s.
The catalogue itself stops around 2e-8 m/s, so the smoothed population puts events a
factor 5 louder than anything LPF measured, and since rho is linear in Deltav that tail
dominates every percentile above the 99th. `smooth=False` bootstraps the measured pairs
instead and cannot leave the catalogue's support. Both are reported: the bootstrap is
the statement about the data, the KDE is the statement about the prior we sample from,
and quoting only one of them would be misleading in one direction or the other.

Usage
-----
    python run_lpf_snr.py [--catalogues 20]
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # noise.py

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import jaxglitches as jg
import noise as ns
import fd_pipeline as fp

OUT = os.path.join(HERE, "lpf_snr.npz")
N_TAU = 240


def main(n_cat, rho_ref):
    DS = np.load(os.path.join(HERE, "dataset.npz"))
    grid = fp.make_grid(int(DS["N"]), float(DS["DT"]))
    psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=grid["T_OBS"])
    T_OBS = grid["T_OBS"]

    # R(tau): the SNR of a unit-Deltav glitch, on a log grid spanning the catalogue
    tau_grid = np.geomspace(0.05, 1e5, N_TAU)
    R = np.array([float(jg.snr(fp.glitch_fd(jnp.array([0.0, 1.0, t]),
                                            grid["freq"]).at[0].set(0 + 0j), psd))
                  for t in tau_grid])

    store = {}
    for smooth in (False, True):
        rho, taus, dvs, ns_ = [], [], [], []
        for s in range(n_cat):
            cat = jg.run_catalog(float(T_OBS), key=jr.PRNGKey(s),
                                 run_type="ordinary", smooth=smooth)
            t = np.asarray(cat["tau"]); dv = np.abs(np.asarray(cat["Deltav"]))
            r = dv * np.exp(np.interp(np.log(t), np.log(tau_grid), np.log(R)))
            rho.append(r); taus.append(t); dvs.append(dv)
            ns_.append(cat["n_glitches"])
        rho = np.concatenate(rho)
        taus = np.concatenate(taus)
        dvs = np.concatenate(dvs)
        tag = "KDE-smoothed" if smooth else "bootstrap"

        print(f"\n=== {tag} resampling ===")
        print(f"{n_cat} draws of a {T_OBS/86400:.0f}-day window, "
              f"{np.mean(ns_):.0f} events each, {len(rho)} in total; "
              f"max |Deltav| = {dvs.max():.3e} m/s")
        for q in (50, 90, 99, 99.9, 100):
            print(f"  {q:6.1f}th percentile of rho_gl: {np.percentile(rho, q):14.3f}")
        print()
        for r in sorted({42.7, 70.0, float(rho_ref), 1500.0}):
            n_over = int((rho > r).sum())
            print(f"  rho_gl > {r:8.1f}: {n_over:6d} of {len(rho)} "
                  f"({(rho > r).mean()*100:7.4f}%), {n_over/n_cat:9.2f} per year")
        i = int(np.argmax(rho))
        print(f"  loudest: rho = {rho[i]:.4g}, tau = {taus[i]:.2f} s, "
              f"|Deltav| = {dvs[i]:.3e} m/s")
        m = taus > 100.0
        print(f"  tau > 100 s (knee in band): {m.sum()} events "
              f"({m.mean()*100:.2f}%), loudest rho = {rho[m].max():.4g}")
        pre = "kde_" if smooth else "boot_"
        store.update({pre + "rho": rho, pre + "tau": taus, pre + "deltav": dvs})

    np.savez_compressed(OUT, n_catalogues=n_cat, t_obs=float(T_OBS),
                        tau_grid=tau_grid, R=R, **store)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogues", type=int, default=20)
    ap.add_argument("--rho-ref", type=float, default=126.0,
                    help="rho_crit from run_knee_scan.py, for the tail fractions")
    a = ap.parse_args()
    main(a.catalogues, a.rho_ref)
