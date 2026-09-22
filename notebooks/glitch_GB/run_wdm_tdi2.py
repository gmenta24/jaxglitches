"""TDI-1 against TDI-2 in the WDM domain: the null test of Sec. "TDI-1 against TDI-2: a
null test", repeated in the time--frequency analysis (Sec. "TDI-1 against TDI-2, in the
time--frequency domain").

In the frequency domain the two generations are related by an exact identity ---
h^(2) = -TF_X^(1) h^(1) and S^(2) = |TF_X^(1)|^2 S^(1), so the transfer function
cancels bin by bin and the two likelihoods are equal to machine precision. Any
disagreement there is a bug.

The time--frequency analysis is different, and that is what makes the test worth
repeating rather than copying. Between the generations every Fourier component is
multiplied by -TF_X1(f) = 2 sin(4 pi f L) * i * exp(-4 pi i f L): a magnitude, a
quarter-cycle phase and a delay of 2L. A WDM pixel is a weighted sum over its channel,
and the quarter-cycle swaps each wavelet for its quadrature partner, which the basis
(one phase per pixel) carries mostly in the neighbouring time bins. The binary's window
keeps every time bin of its channels and cannot tell; the glitch's window keeps six,
so content crosses its edges and each generation keeps a slightly different part of the
same noise realisation. With a four-bin window, which misses the part of the glitch's
footprint that wraps round the end of the year, that difference is several times
larger. The obvious suspect -- the magnitude varying across a channel while
the pixel variance of Eq. (wdmvar) takes it at the channel centre -- turns out to be
negligible. The comparison therefore measures the size of an effect instead of testing
an identity, and this script separates the things that can produce a difference:

  * the machinery -- templates, transform, windows, split -- which is checked on
    noise-free data, where the two generations must agree to round-off (`checks`);
  * the noise, which is checked by repeating the comparison over independent
    realisations (`scatter`) and by running a full TDI-2 chain against the stored
    TDI-1 one (`chain`);
  * the mechanism, which applies the magnitude, the phase, the quarter-cycle and the
    delay one at a time to noise draws, and then lengthens the glitch window in time
    (`mechanism`). It is linear and noise-only: no likelihood, no sampler, a few
    minutes on a CPU.

Writes `wdm_tdi2.npz` next to this script; `make_fig_wdm_tdi.py` turns that into
`paper/figures/fig_wdm_tdi.pdf`. Runtime is dominated by the chain, about half an hour
on a GPU, with the realisation scan adding roughly as much again. `mechanism` writes
its own `wdm_tdi2_mechanism.npz` and is not merged, so re-running it leaves the
figure's recorded inputs untouched.

The phases run as separate processes:

    python run_wdm_tdi2.py              # checks, scatter, chain, then merge
    python run_wdm_tdi2.py chain        # one phase only
    python run_wdm_tdi2.py mechanism    # which piece of the transfer function matters
    python run_wdm_tdi2.py report       # print the numbers the paper quotes, from the
                                        # stored files

That is not cosmetic. Each realisation builds its own jitted likelihood closure holding
about 20 MB of device arrays, and with a few dozen of them alive the XLA compilation of
the sampler's scan slows to a crawl. Running the phases in fresh processes keeps the
chain compiling against an empty cache.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "notebooks"))          # noise.py

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import lisaorbits
from jaxgb import jaxgb as jgb_mod
from wdm_transform import TimeSeries, WDM, wdm_noise_variance
from wdm_transform.backends import get_backend

import jaxglitches as jg
import noise as ns
from jaxglitches.priors import _DELTAV_MIN, _DELTAV_MAX, _TAU_MIN, _TAU_MAX

BACKEND = get_backend("jax")
N_REALISATIONS = 48
SEED_CHAIN = 1            # the seed the stored TDI-1 chain used: common random numbers
SEED_MECHANISM = 5000     # noise draws of `mechanism`, disjoint from those of `scatter`
N_WALKERS, N_BURN, N_SAMP, DIM = 16, 2_000, 10_000, 7
LABELS = ["log_f0", "log_fdot", "log_A_gb", "psi", "t0", "log_Ag", "log_tau"]

# ---------------------------------------------------------------------------
# grid, tilings, injections -- identical to glitch_and_gb_wdm.ipynb
# ---------------------------------------------------------------------------
DS = np.load(HERE / "dataset.npz")
DT, N = float(DS["DT"]), int(DS["N"])
T_OBS, T_ARM = float(DS["T_OBS"]), float(DS["T_ARM"])
NF_GB, NF_GL = 1536, 128
NT_GB, NT_GL = N // NF_GB, N // NF_GL

freq = jnp.asarray(np.fft.rfftfreq(N, DT))
f_safe = jnp.where(freq > 0, freq, 1.0)
n_fine = len(freq)

# TDI-1 -> TDI-2 acts on data and templates alike, so both generations describe one
# physical realisation rather than two independent draws.
TFX1 = -1.0 + jnp.exp(-4j * T_ARM * 2.0 * jnp.pi * f_safe)


def to_tdi2(H):
    return ((-TFX1)[:, None] * H).at[0].set(0 + 0j)


IDENTITY = lambda H: H                                            # noqa: E731
XFORM = {1: IDENTITY, 2: to_tdi2}

gb_true = jnp.asarray(DS["gb_true"])
g3_true = jnp.asarray(DS["glitch_true"])
F0_TRUE = float(gb_true[0])
RA, DEC, IOTA, PHI0 = (float(gb_true[i]) for i in (3, 4, 6, 7))
N_GB = int(DS["N_GB"])
JGB = jgb_mod.JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=T_OBS, t0=0.0, n=N_GB)


def to_time(H):
    return jnp.stack([jnp.fft.irfft(H[:, c], n=N) / DT for c in range(3)])


def wdm_of(x, nt):
    return WDM.from_time_series(TimeSeries(x, dt=DT, backend=BACKEND), nt=nt)


def pixel_variance(freq_grid, nt, nf, gen):
    fch = jnp.where(jnp.asarray(freq_grid) > 0, jnp.asarray(freq_grid), 1.0)
    psd = ns.psd_tdi1 if gen == 1 else ns.psd_tdi2
    S1 = jnp.stack(psd(fch), axis=-1)
    return jnp.stack([jnp.asarray(wdm_noise_variance(np.asarray(S1[:, c]),
                                                     nt=nt, nf=nf, dt=DT))
                      for c in range(3)])


def gb_fd(gb8):
    p = gb8[None, :]
    segs = JGB.get_tdi(p, tdi_generation=1.5, tdi_combination="AET")
    seg = jnp.stack(segs, axis=0).astype(jnp.complex128)[:, 0, :].T
    k = JGB.get_kmin(p[:, 0])[0].astype(jnp.int32)
    return jax.lax.dynamic_update_slice(jnp.zeros((n_fine, 3), jnp.complex128), seg,
                                        (k, jnp.zeros((), jnp.int32)))


def glitch_fd(g3):
    return jg.clean_signal_f(g3, freq, tdi=1).at[0].set(0 + 0j)


def to_physical(th):
    gb8 = (jnp.array([0., 0., 0., RA, DEC, 0., IOTA, PHI0])
           .at[0].set(jnp.exp(th[0])).at[1].set(jnp.exp(th[1]))
           .at[2].set(jnp.exp(th[2])).at[5].set(th[3]))
    tau = jnp.exp(th[6])
    return gb8, jnp.stack([th[4], jnp.exp(th[5]) / tau, tau])


THETA_TRUE = jnp.array([jnp.log(gb_true[0]), jnp.log(gb_true[1]), jnp.log(gb_true[2]),
                        gb_true[5], g3_true[0], jnp.log(g3_true[1] * g3_true[2]),
                        jnp.log(g3_true[2])])

_w = wdm_of(to_time(jnp.asarray(DS["h_gb_tdi1"])), NT_GB)
FREQ_GB, TIME_GB = np.asarray(_w.freq_grid), np.asarray(_w.time_grid)
_w = wdm_of(to_time(jnp.asarray(DS["h_glitch_tdi1"])), NT_GL)
FREQ_GL, TIME_GL = np.asarray(_w.freq_grid), np.asarray(_w.time_grid)
VAR = {(g, "gb"): pixel_variance(FREQ_GB, NT_GB, NF_GB, g) for g in (1, 2)}
VAR |= {(g, "gl"): pixel_variance(FREQ_GL, NT_GL, NF_GL, g) for g in (1, 2)}

K_GB = int(np.argmin(np.abs(FREQ_GB - F0_TRUE)))
GB_CH = jnp.arange(max(1, K_GB - 2), min(NF_GB, K_GB + 3))
# W_gl: the four time bins from the onset on, and the two before it. The transform is
# periodic and the onset is 400 s into the record, so those two are the last two bins
# of the year; bin 1463 alone carries 17% of the glitch's rho^2
# (`run_wdm_windows.py wrap`). Keeping bins 0-3 only, as a first version did, keeps
# 90.0% of the SNR; this keeps 99.2%.
NT_WIN, N_WRAP = 4, 2
GL_T = jnp.asarray(np.r_[np.arange(NT_GL - N_WRAP, NT_GL), np.arange(NT_WIN)])
K_NOTCH = int(np.argmin(np.abs(FREQ_GL - F0_TRUE)))
GL_CH = jnp.asarray([m for m in range(1, NF_GL) if abs(m - K_NOTCH) > 1])

T0_MAX = float(TIME_GL[NT_WIN - 1])


@jax.jit
def log_prior(th):
    ok = ((th[0] >= jnp.log(1e-4)) & (th[0] <= jnp.log(3e-3))
          & (th[1] >= jnp.log(1e-22)) & (th[1] <= jnp.log(1e-15))
          & (th[2] >= jnp.log(1e-25)) & (th[2] <= jnp.log(1e-20))
          & (th[3] >= 0.0) & (th[3] <= float(jnp.pi))
          & (th[4] >= 0.0) & (th[4] <= T0_MAX)
          & (th[5] >= float(np.log(_DELTAV_MIN * _TAU_MIN)))
          & (th[5] <= float(np.log(_DELTAV_MAX * _TAU_MAX)))
          & (th[5] - th[6] >= float(np.log(_DELTAV_MIN)))
          & (th[5] - th[6] <= float(np.log(_DELTAV_MAX)))
          & (th[6] >= float(np.log(_TAU_MIN))) & (th[6] <= float(np.log(_TAU_MAX))))
    return jnp.where(ok, 0.0, -jnp.inf)


def build(gen, data_fd):
    """Split WDM log-likelihood for one TDI generation on one data stream."""
    xf = XFORM[gen]
    d = xf(data_fd)
    d_gb = wdm_of(to_time(d), NT_GB).coeffs[:, :, GB_CH]
    d_gl = wdm_of(to_time(d), NT_GL).coeffs[:, GL_T][:, :, GL_CH]
    v_gb = VAR[(gen, "gb")][:, :, GB_CH]
    v_gl = VAR[(gen, "gl")][:, GL_T][:, :, GL_CH]

    @jax.jit
    def log_lik(th):
        gb8, g3 = to_physical(th)
        w_gb = wdm_of(to_time(xf(gb_fd(gb8))), NT_GB).coeffs[:, :, GB_CH]
        w_gl = wdm_of(to_time(xf(glitch_fd(g3))), NT_GL).coeffs[:, GL_T][:, :, GL_CH]
        return (-0.5 * jnp.sum((d_gb - w_gb) ** 2 / v_gb)
                - 0.5 * jnp.sum((d_gl - w_gl) ** 2 / v_gl))

    return log_lik


def laplace(log_lik, x0=THETA_TRUE):
    lp = jax.jit(lambda th: log_lik(th) + log_prior(th))
    grad, hess = jax.grad(lp)(x0), jax.hessian(lp)(x0)
    x_map = x0 - jnp.linalg.solve(hess, grad)
    sigma = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(hess))))
    return np.asarray(x_map), np.asarray(sigma)


def run_chain(log_lik, x0, cov, seed):
    from jexplore.sampler import JaxSampler, Steps
    from jexplore.sampling import EpochMH, SamplingMH
    from jexplore.steps import Stretch
    from jexplore.backends import DefaultBackend

    sampling = SamplingMH(nwalker=N_WALKERS, temps=jnp.array([1.0]),
                          loglik=log_lik, logprior=log_prior, dim=DIM)
    steps = Steps([{Stretch(permute=True).builder: 1.0}])
    p0 = jnp.asarray(x0) + jr.multivariate_normal(
        jr.PRNGKey(seed), jnp.zeros(DIM), jnp.asarray(cov) * 0.01, shape=(N_WALKERS,))
    backend = DefaultBackend(burn=N_BURN, inmem_epochs=1)
    JaxSampler(sampling, steps, backend).run(EpochMH({"p": p0}),
                                             niters=N_BURN + N_SAMP, nepoch=1, seed=seed)
    s = backend.get_samples()["p"]
    return np.asarray(jnp.array(s.transpose(0, 2, 1).reshape(-1, DIM)))


PHASES = ("checks", "scatter", "chain")
# Run on request only, and never merged into wdm_tdi2.npz, which the figure reads.
EXTRA_PHASES = ("mechanism",)


def _out(phase):
    return HERE / f"wdm_tdi2_{phase}.npz"


def phase_checks():
    """Noise-free agreement, then the realisation the paper analyses."""
    data_fd = jnp.asarray(DS["data_tdi1"])
    clean = jnp.asarray(DS["h_gb_tdi1"]) + jnp.asarray(DS["h_glitch_tdi1"])
    out = {}

    print("== noise-free data: is the machinery generation-independent? ==")
    x1, s1 = laplace(build(1, clean))
    x2, s2 = laplace(build(2, clean))
    out["clean_shift"], out["clean_ratio"] = (x2 - x1) / s1, s2 / s1
    print(f"{'param':10s}{'(MAP2-MAP1)/sigma':>20}{'sigma ratio':>14}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:10s}{out['clean_shift'][i]:20.2e}{out['clean_ratio'][i]:14.5f}")

    print("\n== the stored realisation ==")
    x1, s1 = laplace(build(1, data_fd))
    x2, s2 = laplace(build(2, data_fd))
    out.update(laplace_shift=(x2 - x1) / s1, laplace_ratio=s2 / s1,
               sigma1=s1, sigma2=s2, map1=x1, map2=x2)
    print(f"{'param':10s}{'(MAP2-MAP1)/sigma':>20}{'sigma ratio':>14}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:10s}{out['laplace_shift'][i]:20.3f}{out['laplace_ratio'][i]:14.4f}")
    return out


def phase_scatter():
    """Is the shift on one realisation a bias, or a draw?"""
    clean = jnp.asarray(DS["h_gb_tdi1"]) + jnp.asarray(DS["h_glitch_tdi1"])
    psd_bin = ns.psd_tdi1_array(f_safe, t_obs=T_OBS)
    print(f"== {N_REALISATIONS} independent noise realisations ==")
    shifts = []
    for seed in range(N_REALISATIONS):
        nz = ns.sample_noise_fd(jr.split(jr.PRNGKey(1000 + seed))[0], psd_bin)
        nz = nz.at[0].set(0 + 0j)
        a, sa = laplace(build(1, clean + nz))
        b, _ = laplace(build(2, clean + nz))
        shifts.append((b - a) / sa)
        print(f"  seed {seed:2d}: " + " ".join(f"{v:+6.2f}" for v in shifts[-1]),
              flush=True)
        jax.clear_caches()          # each closure holds ~20 MB of device arrays
    shifts = np.array(shifts)
    n = shifts.shape[0]
    print("  " + "-" * 60)
    print("  mean:    " + " ".join(f"{v:+6.2f}" for v in shifts.mean(0)))
    print("  std:     " + " ".join(f"{v:+6.2f}" for v in shifts.std(0)))
    print("  mean/se: " + " ".join(f"{v:+6.2f}"
                                   for v in shifts.mean(0) / (shifts.std(0) / np.sqrt(n))))
    return {"scatter": shifts}


def phase_chain():
    """The TDI-2 chain, against the stored TDI-1 one."""
    print("devices:", jax.devices())
    data_fd = jnp.asarray(DS["data_tdi1"])
    log_lik = build(2, data_fd)
    x2, s2 = laplace(log_lik)
    log_lik(jnp.asarray(x2)).block_until_ready()
    t0 = time.time()
    for _ in range(50):
        log_lik(jnp.asarray(x2)).block_until_ready()
    per_call = (time.time() - t0) / 50
    print(f"likelihood: {1e3 * per_call:.2f} ms/call  ->  chain should take "
          f"{per_call * N_WALKERS * (N_BURN + N_SAMP) / 60:.0f} min")

    print(f"== TDI-2 chain ({N_WALKERS} walkers x {N_BURN + N_SAMP} iters, seed "
          f"{SEED_CHAIN} -- the seed the stored TDI-1 chain used) ==", flush=True)
    t0 = time.time()
    chain2 = run_chain(log_lik, x2, np.diag(s2 ** 2), seed=SEED_CHAIN)
    print(f"  done in {time.time() - t0:.0f} s, {chain2.shape[0]:,} samples")

    chain1 = np.load(HERE / "wdm_chain.npz")["chain"]
    med1, med2 = np.median(chain1, axis=0), np.median(chain2, axis=0)
    sig1, sig2 = np.std(chain1, axis=0), np.std(chain2, axis=0)
    # Yardstick: the same statistic on two disjoint halves of the TDI-1 walkers.
    # The stored chains are walker-major, so splitting the flat array splits the
    # ensemble, and the difference of the two medians is roughly what the sampler
    # can resolve. `run_convergence.py` does this properly, from the integrated
    # autocorrelation time; this is the cheap version.
    half = chain1.reshape(2, -1, DIM)
    mc = np.abs(np.median(half[0], axis=0) - np.median(half[1], axis=0)) / sig1
    out = dict(chain2=chain2, med_shift=(med2 - med1) / sig1, sig_ratio=sig2 / sig1,
               mc_error=mc, sig1_chain=sig1, sig2_chain=sig2, med1=med1, med2=med2)
    print(f"\n{'param':10s}{'(med2-med1)/sigma':>20}{'sigma ratio':>14}{'MC error':>12}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:10s}{out['med_shift'][i]:20.3f}{out['sig_ratio'][i]:14.4f}"
              f"{mc[i]:12.3f}")
    return out


def _pieces():
    """The generation transfer function taken apart.

    -TF_X1(f) = 1 - exp(-8 pi i f L) = 2 sin(4 pi f L) * i * exp(-4 pi i f L). Each entry
    maps to (the factor applied to every Fourier component, the generation whose
    channel-centre PSD sets the pixel variance): a piece that changes the magnitude
    changes the variance with it, a pure phase does not.
    """
    F = np.asarray(-TFX1)
    return {"full":          (F, 2),
            "magnitude":     (np.abs(F), 2),
            "phase":         (F / np.abs(F), 1),
            "quarter_cycle": (np.full(F.shape, 1j), 1),
            "delay":         (np.exp(-4j * np.pi * np.asarray(f_safe) * T_ARM), 1)}


# Glitch windows of growing length, as (label, bins before bin 0, bins after bin 3).
# `None` keeps every time bin of the year.
MECH_WINDOWS = (("bins 0-3", 0, 0),
                ("+ wrapped bins 1462-1463 (W_gl)", 2, 0),
                ("2 more either side", 2, 2),
                ("4 more either side", 4, 4),
                ("8 more either side", 8, 8),
                ("all time bins", None, None))
# The windows the transfer function is taken apart in: bins 0-3, the window of the first
# version of this analysis, where the effect is large enough to dissect, and W_gl.
PIECE_WINDOWS = (("bins 0-3", np.arange(NT_WIN)),
                 ("W_gl", np.asarray(GL_T)))


def phase_mechanism():
    """Which part of the transfer function makes the glitch window generation-dependent?

    The yardstick is the quantity that drives the glitch amplitude estimate: the noise's
    matched-filter projection onto the glitch template inside a glitch window, (n|h)/rho.
    For TDI-1 and for each piece of the transfer function it is computed on the same noise
    draws, and the two are compared by their correlation and by the spread of their
    difference. A piece that does nothing gives correlation 1. The pixel-by-pixel
    correlation of the whitened noise is recorded as well, in the glitch window and in
    W_GB. All of this is done in the two windows of PIECE_WINDOWS.

    The second half keeps the full transfer function and lengthens the glitch window in
    time, bins added on either side of the onset and wrapping round the end of the year.
    If the difference comes from content crossing the window's time edges, it must shrink
    as the window grows, and it does.
    """
    psd_bin = ns.psd_tdi1_array(f_safe, t_obs=T_OBS)
    h1 = np.asarray(DS["h_glitch_tdi1"])
    rho_fd = float(jg.snr(jnp.asarray(h1), psd_bin))
    pieces = _pieces()
    names = list(pieces)
    gl_ch, gb_ch = np.asarray(GL_CH), np.asarray(GB_CH)

    def coeffs(H, nt):
        return np.asarray(wdm_of(to_time(jnp.asarray(H)), nt).coeffs)

    def sel(A, b):
        return A[:, b][:, :, gl_ch]

    def win_gb(A):
        return A[:, :, gb_ch]

    var = {(g, k): np.asarray(VAR[(g, k)]) for g in (1, 2) for k in ("gl", "gb")}
    full_t = {1: coeffs(h1, NT_GL), 2: coeffs(pieces["full"][0][:, None] * h1, NT_GL)}
    # glitch templates over the whole glitch tiling, with the generation that sets the
    # pixel variance, for TDI-1 and for each piece
    plane = {"tdi1": (full_t[1], 1)}
    for name, (fac, gen) in pieces.items():
        plane[name] = (coeffs(fac[:, None] * h1, NT_GL), gen)
    nw = len(PIECE_WINDOWS)
    tmpl = [{k: sel(A, b) for k, (A, _) in plane.items()} for _, b in PIECE_WINDOWS]
    vgl = [{k: sel(var[(g, "gl")], b) for k, (_, g) in plane.items()}
           for _, b in PIECE_WINDOWS]
    vgb = {k: win_gb(var[(g, "gb")]) for k, (_, g) in plane.items()}
    rho = [{k: float(np.sqrt(np.sum(tmpl[w][k] ** 2 / vgl[w][k]))) for k in plane}
           for w in range(nw)]

    bins = []
    for _, before, after in MECH_WINDOWS:
        bins.append(np.arange(NT_GL) if before is None else
                    np.r_[np.arange(NT_GL - before, NT_GL), np.arange(0, NT_WIN + after)]
                    .astype(int))

    wrho = np.array([[np.sqrt(np.sum(sel(full_t[g], b) ** 2 / sel(var[(g, "gl")], b)))
                      for g in (1, 2)] for b in bins])

    n = N_REALISATIONS
    proj1 = np.zeros((nw, n))
    proj = np.zeros((nw, len(names), n))
    pix = np.zeros((nw, len(names), n))             # pixel correlation, glitch window
    pix_gb = np.zeros((len(names), n))              # ... and in W_GB
    wproj = np.zeros((len(bins), n, 2))             # [TDI-1, full factor]
    print(f"== {n} noise draws, each through TDI-1 and {len(names)} pieces of the "
          f"transfer function ==", flush=True)
    for s in range(n):
        nz = np.array(ns.sample_noise_fd(jr.split(jr.PRNGKey(SEED_MECHANISM + s))[0],
                                         psd_bin))
        nz[0] = 0.0
        c1_gl, c1_gb = coeffs(nz, NT_GL), coeffs(nz, NT_GB)
        z1_gb = (win_gb(c1_gb) / np.sqrt(vgb["tdi1"])).ravel()
        z1_gl = [(sel(c1_gl, b) / np.sqrt(vgl[w]["tdi1"])).ravel()
                 for w, (_, b) in enumerate(PIECE_WINDOWS)]
        for w, (_, b) in enumerate(PIECE_WINDOWS):
            proj1[w, s] = (np.sum(sel(c1_gl, b) * tmpl[w]["tdi1"] / vgl[w]["tdi1"])
                           / rho[w]["tdi1"])
        for i, (name, (fac, gen)) in enumerate(pieces.items()):
            nv = fac[:, None] * nz
            cv_gl, cv_gb = coeffs(nv, NT_GL), coeffs(nv, NT_GB)
            for w, (_, b) in enumerate(PIECE_WINDOWS):
                proj[w, i, s] = (np.sum(sel(cv_gl, b) * tmpl[w][name] / vgl[w][name])
                                 / rho[w][name])
                pix[w, i, s] = np.corrcoef(
                    z1_gl[w], (sel(cv_gl, b) / np.sqrt(vgl[w][name])).ravel())[0, 1]
            pix_gb[i, s] = np.corrcoef(
                z1_gb, (win_gb(cv_gb) / np.sqrt(vgb[name])).ravel())[0, 1]
            if name == "full":
                for j, b in enumerate(bins):
                    for g, c in ((1, c1_gl), (2, cv_gl)):
                        wproj[j, s, g - 1] = (np.sum(sel(c, b) * sel(full_t[g], b)
                                                     / sel(var[(g, "gl")], b)) / wrho[j, g - 1])
        if (s + 1) % 12 == 0:
            print(f"  {s + 1}/{n}", flush=True)

    corr = np.array([[np.corrcoef(proj1[w], p)[0, 1] for p in proj[w]] for w in range(nw)])
    spread = np.array([[np.std(p - proj1[w]) for p in proj[w]] for w in range(nw)])
    for w, (wlabel, _) in enumerate(PIECE_WINDOWS):
        print(f"\nnoise projection onto the glitch template in {wlabel}, TDI-1 against "
              f"each piece")
        print(f"  (the window keeps SNR {rho[w]['tdi1']:.2f} of {rho_fd:.2f} in TDI-1)")
        print(f"  {'piece':15s}{'window SNR':>11s}{'corr':>8s}{'std diff':>10s}"
              f"{'pixel corr':>12s}{'W_GB':>8s}")
        for i, name in enumerate(names):
            print(f"  {name:15s}{rho[w][name]:11.2f}{corr[w, i]:8.3f}{spread[w, i]:10.3f}"
                  f"{pix[w, i].mean():12.3f}{pix_gb[i].mean():8.3f}")
    wcorr = np.array([np.corrcoef(w[:, 0], w[:, 1])[0, 1] for w in wproj])
    wspread = np.array([np.std(w[:, 1] - w[:, 0]) for w in wproj])
    print(f"\nthe same, full transfer function, glitch window lengthened in time")
    print(f"  {'window':32s}{'SNR kept TDI-1':>15s}{'TDI-2':>7s}{'corr':>8s}{'std diff':>10s}")
    for j, (label, _, _) in enumerate(MECH_WINDOWS):
        print(f"  {label:32s}{wrho[j, 0] / rho_fd:15.3f}{wrho[j, 1] / rho_fd:7.3f}"
              f"{wcorr[j]:8.3f}{wspread[j]:10.3f}")
    return dict(pieces=np.array(names),
                piece_windows=np.array([w[0] for w in PIECE_WINDOWS]), rho_fd=rho_fd,
                rho_tdi1=np.array([r["tdi1"] for r in rho]),
                rho_pieces=np.array([[r[k] for k in names] for r in rho]),
                proj_tdi1=proj1, proj=proj, pixel_corr=pix, pixel_corr_gb=pix_gb,
                corr=corr, spread=spread,
                windows=np.array([w[0] for w in MECH_WINDOWS]), window_rho=wrho,
                window_proj=wproj, window_corr=wcorr, window_spread=wspread)


def report():
    """Print the numbers the paper quotes, read off the stored files.

    Nothing is computed that the stored chains and Laplace comparisons do not already
    contain; this only gathers the statistics Sec. "TDI-1 against TDI-2, in the
    time--frequency domain" quotes, so that each has a place it comes from.
    """
    W = np.load(HERE / "wdm_chain.npz")
    T = np.load(HERE / "wdm_tdi2.npz")
    truth = W["theta_true"]
    print("posterior medians against the injection, in units of each chain's own width")
    print(f"  {'':12s}" + "".join(f"{lab:>10s}" for lab in LABELS))
    for name, ch in (("exact FD", W["chain_fd"]), ("WDM TDI-1", W["chain"]),
                     ("WDM TDI-2", T["chain2"])):
        z = (np.median(ch, axis=0) - truth) / ch.std(axis=0)
        print(f"  {name:12s}" + "".join(f"{v:+10.2f}" for v in z))

    print("\nnoise-free data:")
    print(f"  max |MAP2 - MAP1| / sigma     = {np.abs(T['clean_shift']).max():.2e}")
    print(f"  max |sigma ratio - 1|         = {np.abs(T['clean_ratio'] - 1).max():.2e}")
    print("stored realisation, (TDI-2 - TDI-1) / sigma:")
    print(f"  {'':12s}" + "".join(f"{lab:>10s}" for lab in LABELS))
    print(f"  {'Laplace':12s}" + "".join(f"{v:+10.3f}" for v in T["laplace_shift"]))
    print(f"  {'chains':12s}" + "".join(f"{v:+10.3f}" for v in T["med_shift"]))
    print(f"  Laplace and chains agree to {np.abs(T['laplace_shift'] - T['med_shift'])[4:].max():.3f}"
          f" sigma in the glitch block")
    print(f"  width ratio: Laplace {T['laplace_ratio'].min():.3f}-{T['laplace_ratio'].max():.3f},"
          f" chains {T['sig_ratio'].min():.3f}-{T['sig_ratio'].max():.3f}")

    S = T["scatter"]
    n = S.shape[0]
    sd = S.std(axis=0)
    print(f"\n{n} noise draws, Laplace (TDI-2 - TDI-1) / sigma:")
    print(f"  binary block: spread {sd[:4].max():.3f}, worst |shift| {np.abs(S[:, :4]).max():.3f}")
    print(f"  glitch block: spread " + " ".join(f"{v:.3f}" for v in sd[4:])
          + ", mean " + " ".join(f"{v:+.3f}" for v in S[:, 4:].mean(axis=0))
          + ", mean/s.e. " + " ".join(f"{v:+.2f}" for v in S[:, 4:].mean(axis=0) / (sd[4:] / np.sqrt(n))))
    print(f"  draws exceeding 1 sigma in at least one glitch parameter: "
          f"{int((np.abs(S[:, 4:]) > 1.0).any(axis=1).sum())} of {n}")
    print(f"  the stored realisation, in units of that spread: "
          + " ".join(f"{v:+.2f}" for v in T["laplace_shift"][4:] / sd[4:]))

    path = _out("mechanism")
    if not path.exists():
        print(f"\n({path.name} not found: run `{Path(__file__).name} mechanism` for the rest)")
        return
    M = np.load(path)
    for w, wlabel in enumerate(M["piece_windows"]):
        print(f"\nmechanism ({M['proj'].shape[-1]} draws), {wlabel}: TDI-1 against each "
              f"piece, noise projection onto the glitch template")
        for i, name in enumerate(M["pieces"]):
            print(f"  {str(name):15s} corr {M['corr'][w, i]:.3f}   pixel corr "
                  f"{M['pixel_corr'][w, i].mean():.3f}, W_GB {M['pixel_corr_gb'][i].mean():.3f}")
    print()
    for j, label in enumerate(M["windows"]):
        print(f"  {str(label):32s} SNR kept {M['window_rho'][j, 0] / M['rho_fd']:.3f}   "
              f"corr {M['window_corr'][j]:.3f}")


def merge():
    out = {"theta_true": np.asarray(THETA_TRUE), "labels": np.array(LABELS)}
    for phase in PHASES:
        path = _out(phase)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run `{sys.argv[0]} {phase}`")
        out |= dict(np.load(path))
    np.savez_compressed(HERE / "wdm_tdi2.npz", **out)
    print(f"saved {HERE / 'wdm_tdi2.npz'}")


def main():
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        if phase == "merge":
            merge()
            return
        if phase == "report":
            report()
            return
        if phase not in PHASES + EXTRA_PHASES:
            raise SystemExit(f"phase must be one of {PHASES + EXTRA_PHASES + ('merge', 'report')}")
        np.savez_compressed(_out(phase), **globals()[f"phase_{phase}"]())
        print(f"saved {_out(phase)}")
        return
    # Default: each phase in its own process, so the chain compiles against an
    # empty XLA cache rather than behind a few dozen live likelihood closures.
    for phase in PHASES:
        print(f"\n########## {phase} ##########", flush=True)
        subprocess.run([sys.executable, "-u", __file__, phase], check=True)
    merge()


if __name__ == "__main__":
    main()
