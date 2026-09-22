"""The short-segment trap of Sec. "A trap: short segments and red noise", rebuilt, and its fix.

The glitch occupies six time bins of the N_f = 128 tiling, so it is tempting to transform
a short segment around it instead of the whole year. A first version of the analysis did
exactly that and returned A_g a factor two high. That likelihood was never kept. This
script rebuilds it, measures the bias, and tests the fix proposed for it.

The mechanism is the one `run_wdm_windows.py segments` measures on noise alone. The
transform treats its input as periodic, red noise drifts over a segment so that its two
ends do not meet, and the join acts as a spurious jump that the diagonal noise model of
Eq. (wdmvar) does not describe. The excess sits in the first and last time bins of the
segment and in its lowest channels. The fiducial glitch arrives 400 s into the record, so
its window is exactly those bins: the four after the join and the two before it.

Only the glitch block is fitted: (t0, log A_g, log tau), with the glitch window W_gl of
Eq. (wdmwin), six time bins and the three-channel notch at the binary's frequency. The
binary block of Eq. (wdmsplit) shares no parameter with it and does not need refitting.
The data are the stored stream, binary included, since the binary is part of what a
segment cuts through. Three ways of getting the pixels:

  year     the whole year transformed, as in the paper (the reference);
  cut      a segment of L samples transformed on its own, the join at the glitch;
  padded   a stretch of 3L samples transformed, with the glitch in its middle third, so
           the join is L away from every pixel analysed;
  mirrored the 2L samples from the onset on, preceded by the first L of them in reverse
           order, so that the stretch is continuous where the record starts and its join
           is L away from the glitch. Unlike padding it needs no data before the onset,
           which is the case of a glitch near the start of a real record.

Templates go through exactly the same cut as the data. Each is repeated with the glitch
moved half a year into the record ("mid-year"). For the fiducial glitch, padding reaches
back past the start of the year, which the simulated record allows only because it is
periodic by construction; the mid-year copy needs no such help, so the two together
separate what padding does from what the simulation's periodicity does. Every fit is also
run on noise-free data, where all three must return the injection.

Phases, each resumable and writing its own file:

    python run_wdm_segments.py            # all three, in order
    python run_wdm_segments.py fits       # the stored realisation and noise-free data
    python run_wdm_segments.py scatter    # 24 independent noise draws
    python run_wdm_segments.py report     # print what the paper quotes

Writes `wdm_segments_<phase>.npz` next to this script. Runs on the CPU unless
JAX_PLATFORMS says otherwise; a few minutes for `fits`, about an hour for `scatter`.
"""
import os
import sys
import time
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
import jax.random as jr

from wdm_transform import TimeSeries, WDM, wdm_noise_variance
from wdm_transform.backends import get_backend

import jaxglitches as jg
import noise as ns

BACKEND = get_backend("jax")
DS = np.load(HERE / "dataset.npz")
DT, N, T_OBS = float(DS["DT"]), int(DS["N"]), float(DS["T_OBS"])
NF = 128                                   # the glitch tiling of the paper
NT_WIN, N_WRAP = 4, 2                      # W_gl: the onset bin, three after, two before
# Segment lengths in samples, N_t = L / 128 even: 7.9, 23.7, 44.9 and 89.9 d. The padded
# stretch is 3L, which the year holds up to L = N / 3.
SEGMENTS = (4096, 12288, 23296, 46592)
MID = N // 2                               # onset of the mid-year copy, in samples
SHIFTS = {"fiducial": 0, "mid-year": MID}
N_REALISATIONS = 24
SEED_SCATTER = 6000                        # disjoint from every other script's draws

freq = jnp.asarray(np.fft.rfftfreq(N, DT))
f_safe = jnp.where(freq > 0, freq, 1.0)
PSD_BIN = ns.psd_tdi1_array(f_safe, t_obs=T_OBS)
H_GB = jnp.asarray(DS["h_gb_tdi1"])
NOISE = jnp.asarray(DS["noise_tdi1"])
T0, DELTAV, TAU = (float(v) for v in DS["glitch_true"])
F0_GB = float(DS["gb_true"][0])
THETA_TRUE = np.array([T0, np.log(DELTAV * TAU), np.log(TAU)])
LABELS = ("t0", "log_Ag", "log_tau")


def to_time(H):
    return jnp.stack([jnp.fft.irfft(H[:, c], n=N) / DT for c in range(3)])


def wdm_of(x, nt):
    return WDM.from_time_series(TimeSeries(x, dt=DT, backend=BACKEND), nt=nt)


def glitch_fd(g3):
    return jg.clean_signal_f(g3, freq, tdi=1).at[0].set(0 + 0j)


def glitch_td(th, shift_s):
    """Glitch time series over the year, for sampling coordinates th, onset shifted."""
    tau = jnp.exp(th[2])
    return to_time(glitch_fd(jnp.stack([th[0] + shift_s * DT, jnp.exp(th[1]) / tau, tau])))


_FREQ = np.asarray(wdm_of(jnp.zeros((3, NF * 4)), 4).freq_grid)
_K_NOTCH = int(np.argmin(np.abs(_FREQ - F0_GB)))
GL_CH = np.array([m for m in range(1, NF) if abs(m - _K_NOTCH) > 1])


def pixel_variance(nt):
    """Eq. (wdmvar) at the channel centres, (3, nt, NF + 1)."""
    fch = np.where(_FREQ > 0, _FREQ, 1.0)
    S1 = np.stack(ns.psd_tdi1(jnp.asarray(fch)), axis=-1)
    return np.stack([np.asarray(wdm_noise_variance(S1[:, c], nt=nt, nf=NF, dt=DT))
                     for c in range(3)])


class Stretch:
    """One way of taking samples from the year and transforming them.

    `rel` lists, for each sample of the stretch, which sample of the year it is, counted
    from the time bin the glitch starts in and wrapping round the (periodic) year; `n0`
    is that bin's index inside the stretch. The onset position is an argument of the
    jitted functions rather than a constant, so the fiducial and mid-year copies of a
    variant share one compilation.
    """

    def __init__(self, name, rel, n0):
        self.name, self.rel, self.n0 = name, np.asarray(rel), n0
        self.length = len(rel)
        self.nt = self.length // NF
        assert self.length % NF == 0 and self.nt % 2 == 0
        self.bins = np.arange(n0 - N_WRAP, n0 + NT_WIN) % self.nt
        var = pixel_variance(self.nt)
        self.var = jnp.asarray(var[:, self.bins][:, :, GL_CH])
        nt, bins, ch, rel_j = self.nt, jnp.asarray(self.bins), jnp.asarray(GL_CH), jnp.asarray(rel)

        def window(x_year, start):
            idx = (start + rel_j) % N
            w = wdm_of(jnp.take(x_year, idx, axis=1), nt).coeffs
            return w[:, bins][:, :, ch]

        def loglik(th, d_w, start, shift_s):
            r = d_w - window(glitch_td(th, shift_s), start)
            return -0.5 * jnp.sum(r ** 2 / self.var)

        self.window = jax.jit(window)
        self.f = jax.jit(loglik)
        self.g = jax.jit(jax.grad(loglik))
        self.h = jax.jit(jax.hessian(loglik))

    @staticmethod
    def start(shift_s):
        return shift_s


def variants():
    out = [Stretch("year", np.arange(N), 0)]
    for L in SEGMENTS:
        out.append(Stretch(f"cut {L}", np.arange(L), 0))
        out.append(Stretch(f"padded {L}", np.arange(-L, 2 * L), L // NF))
        out.append(Stretch(f"mirrored {L}", np.r_[np.arange(L)[::-1], np.arange(2 * L)],
                           L // NF))
    return out


def newton(v, d_w, start, shift_s, x0=THETA_TRUE, n_steps=60, tol=1e-6):
    """Backtracked Newton ascent to the maximum, and the Laplace width there.

    Three parameters and a well-conditioned curvature, so a plain Newton step with a
    backtracked length is enough; a shift of the Hessian guards the case where the
    starting point sits outside the concave region, which a factor-two bias can do.
    """
    args = (d_w, start, shift_s)
    x = jnp.asarray(x0)
    for _ in range(n_steps):
        g, Hx = v.g(x, *args), v.h(x, *args)
        H = 0.5 * (Hx + Hx.T)
        ev = jnp.linalg.eigvalsh(H)
        M = H if float(ev[-1]) < 0.0 else H - 2.0 * float(jnp.abs(ev).max()) * jnp.eye(3)
        step = -jnp.linalg.solve(M, g)
        f_now, t = float(v.f(x, *args)), 1.0
        for _ in range(50):
            if float(v.f(x + t * step, *args)) > f_now:
                break
            t *= 0.5
        else:
            break
        x = x + t * step
        sig = jnp.sqrt(jnp.abs(jnp.diag(jnp.linalg.inv(-M))))
        if float(jnp.max(jnp.abs(t * step / sig))) < tol:
            break
    Hx = v.h(x, *args)
    H = 0.5 * (Hx + Hx.T)
    sig = np.sqrt(np.abs(np.diag(np.linalg.inv(-np.asarray(H)))))
    ok = bool(jnp.linalg.eigvalsh(H)[-1] < 0.0)
    return np.asarray(x), sig, ok


def data_year(noise_fd, shift_s, with_noise=True, with_gb=True):
    """Binary + glitch (+ noise) over the year, the glitch onset shifted by shift_s."""
    h = glitch_fd(jnp.array([T0 + shift_s * DT, DELTAV, TAU]))
    h = h + H_GB if with_gb else h
    return to_time(h + noise_fd) if with_noise else to_time(h)


def fit_all(vs, noise_fd, tag, clean=True, whitened=True):
    """Every variant, both glitch positions: maxima, widths, and the noise in the window."""
    nv, ns_ = len(vs), len(SHIFTS)
    res = {k: np.full((ns_, nv, 3), np.nan)
           for k in ("x", "sig", "x_clean", "sig_clean", "x_nogb", "sig_nogb")}
    res |= {"ok": np.zeros((ns_, nv), bool), "ok_clean": np.zeros((ns_, nv), bool),
            "ok_nogb": np.zeros((ns_, nv), bool), "wstd": np.full((ns_, nv, 4), np.nan)}
    for i, (sname, shift) in enumerate(SHIFTS.items()):
        x_data = data_year(noise_fd, shift)
        x_clean = data_year(noise_fd, shift, with_noise=False) if clean else None
        # the same without the binary: what is left on noise-free data is the cut itself
        x_nogb = data_year(noise_fd, shift, with_noise=False, with_gb=False) if clean else None
        x_noise = to_time(noise_fd) if whitened else None
        for j, v in enumerate(vs):
            start = v.start(shift)
            x, s, ok = newton(v, v.window(x_data, start), start, shift)
            res["x"][i, j], res["sig"][i, j], res["ok"][i, j] = x, s, ok
            if clean:
                x, s, ok = newton(v, v.window(x_clean, start), start, shift)
                res["x_clean"][i, j], res["sig_clean"][i, j], res["ok_clean"][i, j] = x, s, ok
                x, s, ok = newton(v, v.window(x_nogb, start), start, shift)
                res["x_nogb"][i, j], res["sig_nogb"][i, j], res["ok_nogb"][i, j] = x, s, ok
            if whitened:
                z = np.asarray(v.window(x_noise, start) / jnp.sqrt(v.var))
                # all six bins; the four from the onset; the two before it; lowest 3 channels
                res["wstd"][i, j] = (z.std(), z[:, N_WRAP:].std(), z[:, :N_WRAP].std(),
                                     z[:, :, :3].std())
        print(f"  [{tag}] {sname:9s} " + "  ".join(
            f"{v.name}: {np.exp(res['x'][i, j, 1] - THETA_TRUE[1]):5.2f}"
            for j, v in enumerate(vs)), flush=True)
    return res


def phase_fits():
    vs = variants()
    print(f"{len(vs)} variants: " + ", ".join(v.name for v in vs))
    print("A_g at the maximum / injected A_g, stored realisation:")
    t = time.time()
    res = fit_all(vs, NOISE, "stored")
    print(f"  ({time.time() - t:.0f} s)")
    z = (res["x"] - THETA_TRUE) / res["sig"]
    zc = (res["x_clean"] - THETA_TRUE) / res["sig_clean"]
    zn = (res["x_nogb"] - THETA_TRUE) / res["sig_nogb"]
    for i, sname in enumerate(SHIFTS):
        print(f"\n== glitch {sname} ==")
        print(f"  {'variant':14s}{'A_g ratio':>10s}  " + "".join(f"{l:>9s}" for l in LABELS)
              + f"{'noise-free max|z|':>19s}{'no binary':>10s}{'std(6 bins)':>12s}{'onset 4':>8s}"
              f"{'before 2':>9s}{'low ch':>8s}")
        for j, v in enumerate(vs):
            print(f"  {v.name:14s}{np.exp(res['x'][i, j, 1] - THETA_TRUE[1]):10.3f}  "
                  + "".join(f"{val:+9.2f}" for val in z[i, j])
                  + f"{np.abs(zc[i, j]).max():19.1e}{np.abs(zn[i, j]).max():10.1e}"
                  + "".join(f"{val:{w}.2f}" for val, w in zip(res["wstd"][i, j],
                                                               (12, 8, 9, 8)))
                  + ("" if res["ok"][i, j] else "  (not a maximum)"))
    return dict(variants=np.array([v.name for v in vs]),
                lengths=np.array([v.length for v in vs]), shifts=np.array(list(SHIFTS)),
                theta_true=THETA_TRUE, labels=np.array(LABELS),
                segments_days=np.array(SEGMENTS) * DT / 86400, **res)


def phase_scatter():
    vs = variants()
    n = N_REALISATIONS
    out = {k: np.full((n, len(SHIFTS), len(vs), 3), np.nan) for k in ("x", "sig")}
    out["ok"] = np.zeros((n, len(SHIFTS), len(vs)), bool)
    out["wstd"] = np.full((n, len(SHIFTS), len(vs), 4), np.nan)
    path = HERE / "wdm_segments_scatter.npz"
    done = 0
    if path.exists():                       # resume
        prev = dict(np.load(path))
        done = int(prev["n_done"])
        for k in ("x", "sig", "ok", "wstd"):
            out[k][:done] = prev[k][:done]
        print(f"resuming after {done} draws")
    for s in range(done, n):
        t = time.time()
        nz = ns.sample_noise_fd(jr.split(jr.PRNGKey(SEED_SCATTER + s))[0], PSD_BIN)
        nz = nz.at[0].set(0 + 0j)
        r = fit_all(vs, nz, f"draw {s:2d}", clean=False)
        for k in ("x", "sig", "ok", "wstd"):
            out[k][s] = r[k]
        np.savez_compressed(path, n_done=s + 1, variants=np.array([v.name for v in vs]),
                            shifts=np.array(list(SHIFTS)), theta_true=THETA_TRUE,
                            segments_days=np.array(SEGMENTS) * DT / 86400, **out)
        print(f"  draw {s} done in {time.time() - t:.0f} s", flush=True)
    return None


def report():
    F = dict(np.load(HERE / "wdm_segments_fits.npz"))
    names = [str(v) for v in F["variants"]]
    print("stored realisation: A_g ratio, and (MAP - truth)/sigma for (t0, log A_g, log tau)")
    for i, sname in enumerate(F["shifts"]):
        print(f"  glitch {sname}")
        for j, name in enumerate(names):
            z = (F["x"][i, j] - F["theta_true"]) / F["sig"][i, j]
            zc = (F["x_clean"][i, j] - F["theta_true"]) / F["sig_clean"][i, j]
            zn = (F["x_nogb"][i, j] - F["theta_true"]) / F["sig_nogb"][i, j]
            print(f"    {name:14s} A_g x {np.exp(F['x'][i, j, 1] - F['theta_true'][1]):.3f}  z "
                  + " ".join(f"{v:+6.2f}" for v in z)
                  + f"   noise-free max|z| {np.abs(zc).max():.1e} (no binary {np.abs(zn).max():.1e})"
                  + f"   whitened std {F['wstd'][i, j, 0]:.2f} (low ch {F['wstd'][i, j, 3]:.2f})")
    path = HERE / "wdm_segments_scatter.npz"
    if not path.exists():
        print("\n(no scatter file yet)")
        return
    S = dict(np.load(path))
    n = int(S["n_done"])
    names = [str(v) for v in S["variants"]]
    print(f"\n{n} noise draws: mean +- s.e. (and spread) of (MAP - truth)/sigma, in each "
          f"variant's own Laplace width")
    for i, sname in enumerate(S["shifts"]):
        print(f"  glitch {sname}")
        for j, name in enumerate(names):
            z = (S["x"][:n, i, j] - S["theta_true"]) / S["sig"][:n, i, j]
            ratio = np.exp(S["x"][:n, i, j, 1] - S["theta_true"][1])
            m, se, sd = z.mean(0), z.std(0) / np.sqrt(n), z.std(0)
            print(f"    {name:14s} " + "  ".join(f"{a:+6.2f}+-{b:.2f} (sd {c:4.2f})"
                                                for a, b, c in zip(m, se, sd))
                  + f"   A_g x {np.median(ratio):.2f} [{ratio.min():.2f}, {ratio.max():.2f}]"
                  + f"   whitened std {S['wstd'][:n, i, j, 0].mean():.2f}"
                  + f" (low ch {S['wstd'][:n, i, j, 3].mean():.2f})"
                  + f"   draws with max|z| > 3: {int((np.abs(z) > 3).any(axis=1).sum())}")
    j0 = names.index("year")
    print(f"\ndraw by draw against the full year, in the full year's widths: max over draws "
          f"of |MAP - MAP_year| / sigma_year, and the median width ratio sigma / sigma_year")
    for i, sname in enumerate(S["shifts"]):
        print(f"  glitch {sname}")
        for j, name in enumerate(names):
            d = np.abs(S["x"][:n, i, j] - S["x"][:n, i, j0]) / S["sig"][:n, i, j0]
            w = S["sig"][:n, i, j] / S["sig"][:n, i, j0]
            print(f"    {name:14s} max " + " ".join(f"{v:6.3f}" for v in d.max(0))
                  + "   rms " + " ".join(f"{v:6.3f}" for v in np.sqrt((d ** 2).mean(0)))
                  + "   width ratio " + " ".join(f"{v:5.3f}" for v in np.median(w, axis=0)))


def main():
    phases = sys.argv[1:] or ["fits", "scatter", "report"]
    bad = [ph for ph in phases if ph not in ("fits", "scatter", "report")]
    if bad:
        raise SystemExit("phase must be one of fits, scatter, report")
    print("devices:", jax.devices())
    for phase in phases:
        if phase == "fits":
            out = phase_fits()
            np.savez_compressed(HERE / "wdm_segments_fits.npz", **out)
            print(f"saved {HERE / 'wdm_segments_fits.npz'}")
        elif phase == "scatter":
            phase_scatter()
        else:
            report()


if __name__ == "__main__":
    main()
