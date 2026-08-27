"""Put the knee scan and the LPF population on the same axis.

`run_knee_scan.py` gives rho_crit(f0, tau): the glitch SNR at which an unmodelled
glitch of duration tau displaces a Galactic binary at f0 by one standard deviation.
`run_lpf_snr.py` gives the SNR distribution of the LPF population. The question they
answer together is how often a real glitch reaches its own threshold.

For each catalogue event we take the *worst case over f0* -- the most vulnerable binary
frequency for that duration, min_f0 rho_crit(f0, tau), interpolated in log tau -- which
is the strongest statement the scan supports and the weakest assumption about where the
binaries happen to be.

Usage
-----
    python combine_knee_lpf.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(HERE, "knee_scan.npz"))
L = np.load(os.path.join(HERE, "lpf_snr.npz"))

taus, f0s = S["taus"], S["f0s"]
rc_min = S["rho_crit"].min(axis=1)                    # worst f0 for each tau
f_worst = f0s[S["rho_crit"].argmin(axis=1)]

print("worst binary frequency for each glitch duration")
print(f"{'tau [s]':>9}{'f_knee [mHz]':>14}{'f0 worst [mHz]':>16}"
      f"{'x=f0/fknee':>12}{'rho_crit':>10}")
for t, r, f in zip(taus, rc_min, f_worst):
    print(f"{t:9.0f}{1e3/(2*np.pi*t):14.4f}{f*1e3:16.3f}"
          f"{2*np.pi*f*t:12.3f}{r:10.1f}")
i = int(np.argmin(rc_min))
print(f"\nglobal minimum: rho_crit = {rc_min[i]:.1f} at tau = {taus[i]:g} s, "
      f"f0 = {f_worst[i]*1e3:.3f} mHz, x = {2*np.pi*f_worst[i]*taus[i]:.3f}")

# rho_crit at each event's own duration. Below the grid the knee is far above the band
# and the in-band waveform stops depending on tau, so the shortest row is the limit and
# the clip is exact. Above it the clip is conservative: for tau beyond the longest row
# the in-band waveform keeps falling as 1/tau^2, so the true rho_crit is higher than the
# clipped one and the counts below are upper bounds. The catalogue reaches tau = 5e4 s,
# but only a per cent of its events lie past the top of the grid.
lo, hi = taus[0], taus[-1]
for tag in ("boot", "kde"):
    rho, tau = L[f"{tag}_rho"], L[f"{tag}_tau"]
    rcrit = np.exp(np.interp(np.log(np.clip(tau, lo, hi)), np.log(taus),
                             np.log(rc_min)))
    ratio = rho / rcrit
    n_cat = int(L["n_catalogues"])
    over = ratio > 1.0
    name = "bootstrap (catalogue support)" if tag == "boot" else "KDE-smoothed (prior support)"
    print(f"\n=== {name} ===")
    print(f"  {over.sum()} of {len(rho)} events exceed their own rho_crit "
          f"({over.mean()*100:.3f}%), i.e. {over.sum()/n_cat:.1f} per year")
    for q in (99.0, 99.9, 100.0):
        print(f"  {q:5.1f}th percentile of rho/rho_crit: {np.percentile(ratio, q):12.4g}")
    j = int(np.argmax(ratio))
    print(f"  worst event: rho = {rho[j]:.4g}, tau = {tau[j]:.2f} s, "
          f"rho_crit = {rcrit[j]:.1f}, ratio = {ratio[j]:.4g}")
    # the same, with the single loudest catalogue entry removed
    keep = rho < 0.5 * rho.max()
    o2 = ratio[keep] > 1.0
    print(f"  excluding the outlier population above {0.5*rho.max():.3g}: "
          f"{o2.sum()} of {keep.sum()} exceed ({o2.mean()*100:.3f}%), "
          f"{o2.sum()/n_cat:.1f} per year; loudest ratio {ratio[keep].max():.3g}")
