"""Where the time--frequency split loses signal, and what cutting a segment does to the noise.

Two measurements behind Secs. "What the split costs" and "A trap: short segments and red
noise" of the paper, and behind the red To-do list at its end. Both read the stored data
stream and need neither a likelihood nor a sampler.

wrap
    Where the glitch's rho^2 sits on the N_f = 128 tiling, time bin by time bin, and how
    much of it the glitch window W_gl keeps. The transform is periodic and the onset is
    400 s into the record, so the glitch's footprint straddles the join and the last time
    bin of the year carries a sixth of its rho^2. A window starting at bin 0, as the first
    version of W_gl did, misses that part and keeps 90% of the SNR; W_gl now includes the
    two wrapped bins, 1462-1463, and keeps 99%. Also
    measured: how much of the glitch the three-channel notch removes, and how loud the
    binary is in the notched pixels. Those two decide whether subtracting the binary
    there, instead of notching, could recover glitch SNR. It could not, since the notch
    holds almost none of it.

segments
    Whitened-noise standard deviation, region by region, in segments cut out of the stored
    year. The transform treats a segment as periodic. Red noise drifts over the segment, so
    its two ends do not meet and the join acts as a spurious jump, which the diagonal noise
    model of Eq. (wdmvar) does not describe. The excess sits in the segment's first and last
    time bins and in its lowest channels, and it grows with the segment's length. Each
    segment is measured three ways:

      as cut      the segment transformed on its own;
      detrended   the noise slower than the segment (f < 1/T_seg) removed from the whole
                  year before cutting -- a partial cure;
      padded      a stretch three times longer transformed, and only the pixels of its
                  middle third kept, so the join falls outside the pixels analysed.

    The full simulated year is clean only because the noise was generated periodic by
    construction; real data would show the same jump at the ends of the record.

    python run_wdm_windows.py            # both, a few minutes on a CPU
    python run_wdm_windows.py segments   # one of them

Writes `wdm_windows.npz` next to this script (a phase run on its own updates only its own
entries). Runs on the CPU unless JAX_PLATFORMS says otherwise: nothing here is heavy, and
the CPU keeps the printed numbers identical run to run.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "notebooks"))          # noise.py

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from wdm_transform import TimeSeries, WDM, wdm_noise_variance
from wdm_transform.backends import get_backend

import jaxglitches as jg
import noise as ns

BACKEND = get_backend("jax")
DS = np.load(HERE / "dataset.npz")
DT, N, T_OBS = float(DS["DT"]), int(DS["N"]), float(DS["T_OBS"])
NF_GL = 128                        # the glitch tiling of the paper
NT_WIN, N_WRAP = 4, 2              # W_gl: bins 0-3 and the two wrapped bins before them
F0 = float(DS["gb_true"][0])
freq = np.fft.rfftfreq(N, DT)
f_safe = np.where(freq > 0, freq, 1.0)

# Segment lengths in samples: N_t = L / 128 must be even. 7.9, 23.7, 44.9, 89.9, 180.7 d
# and the full year; the padded version needs 3L <= N, so it stops at 89.9 d.
SEGMENTS = (4096, 12288, 23296, 46592, 93696, N)
N_LOW = 3                          # "lowest channels": 1..3, above the rank-deficient DC


def to_time(H, n=N):
    return jnp.stack([jnp.fft.irfft(jnp.asarray(H)[:, c], n=n) / DT for c in range(3)])


def wdm_of(x, nt):
    return WDM.from_time_series(TimeSeries(jnp.asarray(x), dt=DT, backend=BACKEND), nt=nt)


def pixel_variance(freq_grid, nt, nf):
    """Eq. (wdmvar): the one-sided PSD at each channel centre, shape (3, nt, nf + 1)."""
    fch = np.where(np.asarray(freq_grid) > 0, np.asarray(freq_grid), 1.0)
    S1 = np.stack(ns.psd_tdi1(jnp.asarray(fch)), axis=-1)
    return np.stack([np.asarray(wdm_noise_variance(S1[:, c], nt=nt, nf=nf, dt=DT))
                     for c in range(3)])


def phase_wrap():
    nt = N // NF_GL
    w = wdm_of(to_time(DS["h_glitch_tdi1"]), nt)
    fgrid = np.asarray(w.freq_grid)
    var = pixel_variance(fgrid, nt, NF_GL)
    interior = np.arange(1, NF_GL)                           # DC and Nyquist excluded
    k_notch = int(np.argmin(np.abs(fgrid - F0)))
    notch = np.array([k_notch - 1, k_notch, k_notch + 1])
    kept_ch = np.array([m for m in interior if abs(m - k_notch) > 1])
    P = np.asarray(w.coeffs)[:, :, interior] ** 2 / var[:, :, interior]
    rho2_bin = P.sum(axis=(0, 2))                            # (nt,)
    rho2_wdm = float(P.sum())
    rho_fd = float(jg.snr(jnp.asarray(DS["h_glitch_tdi1"]),
                          ns.psd_tdi1_array(jnp.asarray(f_safe), t_obs=T_OBS)))

    def snr_kept(bins, chans):
        sub = np.asarray(w.coeffs)[:, bins][:, :, chans] ** 2 / var[:, bins][:, :, chans]
        return float(np.sqrt(sub.sum()))

    first, wrapped = np.arange(NT_WIN), np.arange(nt - N_WRAP, nt)
    variants = {"bins 0-3, notch (first W_gl)": (first, kept_ch),
                "bins 0-3, no notch": (first, interior),
                "W_gl: wrapped 1462-1463 + 0-3, notch": (np.r_[wrapped, first], kept_ch),
                "all time bins, notch": (np.arange(nt), kept_ch)}
    kept = {k: snr_kept(*v) for k, v in variants.items()}
    gl_t = np.r_[wrapped, first]
    notch_share = float(P[:, gl_t][:, :, np.searchsorted(interior, notch)].sum() / rho2_wdm)

    wb = wdm_of(to_time(DS["h_gb_tdi1"]), nt)
    Pb = np.asarray(wb.coeffs) ** 2 / var
    gb_in_notch = float(np.sqrt(Pb[:, gl_t][:, :, notch].sum()))
    gb_notch_year = float(np.sqrt(Pb[:, :, notch].sum()))

    order = np.argsort(rho2_bin)[::-1]
    print(f"glitch on the N_f = {NF_GL} tiling: rho = {rho_fd:.2f} (frequency domain), "
          f"{np.sqrt(rho2_wdm):.2f} (WDM, interior channels)")
    print("  largest shares of rho^2 by time bin: "
          + ", ".join(f"bin {b} {100 * rho2_bin[b] / rho2_wdm:.1f}%" for b in order[:5]))
    print("  SNR kept, as a fraction of the frequency-domain SNR:")
    for k, v in kept.items():
        print(f"    {k:40s} {v:6.2f}  ({100 * v / rho_fd:.1f}%)")
    print(f"  share of rho^2 in the notch pixels of W_gl's time bins: {100 * notch_share:.2f}%")
    print(f"  binary SNR in those pixels: {gb_in_notch:.1f}  "
          f"(in the three notch channels over the whole year: {gb_notch_year:.1f})")
    return dict(wrap_rho2_bin=rho2_bin / rho2_wdm, wrap_rho_fd=rho_fd,
                wrap_rho_wdm=np.sqrt(rho2_wdm), wrap_variants=np.array(list(kept)),
                wrap_snr_kept=np.array(list(kept.values())), wrap_notch_share=notch_share,
                wrap_gb_in_notch=gb_in_notch, wrap_gb_notch_year=gb_notch_year)


REGIONS = ("first 4 bins", "last 2 bins", "first 4, lowest ch", "last 2, lowest ch",
           "interior bins", "all pixels")


def _regions(z, b0, b1):
    """Whitened-noise std in each region of the time bins [b0, b1) of z (3, nt, nf+1)."""
    ch, low = slice(1, NF_GL), slice(1, 1 + N_LOW)

    def s(bins, chans=ch):
        return float(np.std(z[:, bins][:, :, chans]))

    return [s(slice(b0, b0 + 4)), s(slice(b1 - 2, b1)), s(slice(b0, b0 + 4), low),
            s(slice(b1 - 2, b1), low), s(slice(b0 + 4, b1 - 2)), s(slice(b0, b1))]


def phase_segments():
    noise_fd = np.asarray(DS["noise_tdi1"])

    def whitened(x):
        nt = x.shape[1] // NF_GL
        w = wdm_of(x, nt)
        return np.asarray(w.coeffs) / np.sqrt(pixel_variance(w.freq_grid, nt, NF_GL))

    table = np.full((len(SEGMENTS), 3, len(REGIONS)), np.nan)   # [as cut, detrended, padded]
    year = np.asarray(to_time(noise_fd))
    for i, L in enumerate(SEGMENTS):
        nt = L // NF_GL
        table[i, 0] = _regions(whitened(year[:, :L]), 0, nt)
        slow = (freq < 1.0 / (L * DT))[:, None]
        table[i, 1] = _regions(whitened(np.asarray(to_time(np.where(slow, 0.0, noise_fd)))[:, :L]),
                               0, nt)
        if 3 * L <= N:
            table[i, 2] = _regions(whitened(year[:, :3 * L]), nt, 2 * nt)

    print(f"whitened-noise standard deviation on the N_f = {NF_GL} tiling (unity if the "
          f"noise model is right)")
    for k, label in enumerate(("as cut", "detrended", "padded")):
        print(f"\n  {label}")
        print(f"  {'segment':>10s}" + "".join(f"{r:>20s}" for r in REGIONS))
        for i, L in enumerate(SEGMENTS):
            row = table[i, k]
            cells = "".join(f"{'--':>20s}" if np.isnan(v) else f"{v:20.2f}" for v in row)
            print(f"  {L * DT / 86400:8.1f} d" + cells)
    return dict(seg_samples=np.array(SEGMENTS), seg_days=np.array(SEGMENTS) * DT / 86400,
                seg_versions=np.array(["as cut", "detrended", "padded"]),
                seg_regions=np.array(REGIONS), seg_std=table)


PHASES = ("wrap", "segments")


def main():
    out_path = HERE / "wdm_windows.npz"
    phases = sys.argv[1:] or list(PHASES)
    bad = [p for p in phases if p not in PHASES]
    if bad:
        raise SystemExit(f"phase must be one of {PHASES}")
    out = dict(np.load(out_path)) if out_path.exists() else {}
    for phase in phases:
        print(f"\n########## {phase} ##########", flush=True)
        out |= globals()[f"phase_{phase}"]()
    np.savez_compressed(out_path, **out)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
