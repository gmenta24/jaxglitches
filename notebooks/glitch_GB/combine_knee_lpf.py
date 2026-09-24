"""Put the knee scan and the LPF population on the same axis.

`run_knee_scan.py` gives rho_crit(f0, tau): the glitch SNR at which an unmodelled
glitch of duration tau displaces a Galactic binary at f0 by one standard deviation.
`run_lpf_snr.py` gives the SNR distribution of the LPF population. The question they
answer together is how often a real glitch reaches its own threshold.

For each catalogue event we take the *worst case over f0* -- the most vulnerable binary
frequency for that duration, min_f0 rho_crit(f0, tau), interpolated in log tau -- which
is the strongest statement the scan supports and the weakest assumption about where the
binaries happen to be.

The arrival epoch is the other axis, and it is not a detail. `run_knee_scan.py` holds it
at the injected t0 = 400 s, at the start of the record, which is where a glitch does the
most damage: f0 and fdot are measured by comparing the phase at the two ends of the year.
If `knee_t0.npz` is present (`run_knee_scan.py --t0scan`) the count is reported twice:
once with every event given the threshold of that worst epoch, and once with each event's
arrival time drawn uniformly over the record, which is what a population actually does.

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

# rho_crit(tau, epoch), worst f0 at each epoch, if the epoch axis has been run.
T0_PATH = os.path.join(HERE, "knee_t0.npz")
T0 = np.load(T0_PATH) if os.path.exists(T0_PATH) else None
if T0 is not None:
    rc_epoch = T0["rho_crit_t0"].min(axis=1)          # (n_tau, n_t0)
    t0s, t_obs = T0["t0s"], float(T0["T_OBS"])
    assert np.allclose(rc_epoch[:, 0], rc_min, rtol=2e-3), (rc_epoch[:, 0], rc_min)
    print(f"\nepoch axis: {len(t0s)} epochs over {t_obs / 86400:.1f} d; at the worst f0 "
          f"of each row the threshold rises by a factor "
          f"{(rc_epoch.max(axis=1) / rc_epoch[:, 0]).min():.1f}-"
          f"{(rc_epoch.max(axis=1) / rc_epoch[:, 0]).max():.1f} "
          f"between the record's start and its middle")
    rng = np.random.default_rng(20260923)

    N_DRAWS = 20          # the count is an average over epoch draws, not one draw

    def rcrit_uniform(tau):
        """rho_crit for each event, with its arrival time drawn over the record.

        Bilinear in (log tau, epoch): np.interp clamps at both ends of both axes, which
        is exact below the tau grid (the in-band waveform stops depending on tau there)
        and conservative above it, as for the fixed-epoch version.
        """
        t0 = rng.uniform(0.0, t_obs, size=len(tau))
        rows = np.stack([np.interp(t0, t0s, np.log(rc_epoch[i]))
                         for i in range(len(taus))])          # (n_tau, n_events)
        x = np.interp(np.log(np.clip(tau, lo, hi)), np.log(taus),
                      np.arange(len(taus)))
        i0 = np.clip(x.astype(int), 0, len(taus) - 2)
        w = x - i0
        return np.exp((1.0 - w) * rows[i0, np.arange(len(tau))]
                      + w * rows[i0 + 1, np.arange(len(tau))])

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
    if T0 is not None:
        all_, kept_, fac = [], [], []
        for _ in range(N_DRAWS):
            rc_u = rcrit_uniform(tau)
            ru = rho / rc_u
            all_.append((ru > 1).sum() / n_cat)
            kept_.append((ru[keep] > 1).sum() / n_cat)
            fac.append(np.median(rc_u / rcrit))
        print(f"  with the arrival epoch drawn uniformly over the record instead of "
              f"fixed at its start ({N_DRAWS} draws, mean +- s.d.):")
        print(f"    {np.mean(all_):.1f} +- {np.std(all_):.1f} per year; "
              f"excluding the outlier population, "
              f"{np.mean(kept_):.1f} +- {np.std(kept_):.1f} per year")
        print(f"    median threshold x{np.mean(fac):.1f} relative to the worst epoch, "
              f"so a factor {np.mean(fac):.1f} in threshold is a factor "
              f"{o2.sum() / n_cat / max(np.mean(kept_), 1e-9):.1f} in rate "
              f"(outlier population excluded, as everywhere else)")
