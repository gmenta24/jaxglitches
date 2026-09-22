"""What happens if the glitch is not modelled -- in both representations.

Every posterior in the paper so far has the glitch in the model. That shows the joint
fit working; it does not show that the joint fit was needed. Sec. "How far does the
separation extend?" fills half the gap analytically, with the linearised displacement of
Cutler & Vallisneri and the critical amplitude rho_crit at which it reaches one standard
deviation, checked against noiseless maxima. What is missing is the thing a reader
actually wants to see: noisy data, a glitch above threshold, and the biased posterior
next to the joint one.

This script runs that experiment in six analyses of the same data stream.

    fd_joint     exact frequency-domain likelihood, glitch modelled       7 parameters
    fd_gbonly    the same, glitch absent from the model                   4
    wdm_split    the split likelihood exactly as Eq. (wdmsplit) writes it 7
    wdm_sub      the same, but with h_gl also subtracted in W_GB          7
    wdm_gbonly   the binary window alone, glitch absent from the model    4
    wdm_cut      the same, with the time bins carrying the onset excised  4

Why running it in two representations is not redundant
------------------------------------------------------
The WDM transform is orthogonal, so a likelihood that keeps *all* pixels cannot know
which representation it is written in, and the bias must come out the same. It does:
`fd_gbonly` and `wdm_gbonly` agree to 1e-4 sigma out of 4, which is the check that both
implementations are right rather than a result.

The result is what happens to the two analyses in between. `wdm_split` is the paper's
own likelihood, and it is *not* protected: Eq. (wdmsplit) models the binary's window
with h_GB and the glitch's window with h_gl, so the glitch is estimated but never
subtracted from the pixels the binary is measured in. A joint WDM fit therefore carries
the full systematic of a fit that omits the glitch altogether. At the fiducial amplitude
this is invisible -- the glitch deposits SNR 0.67 in W_GB against the binary's 283 --
which is why the notebook never showed it.

Two things fix it, and they are different in kind. `wdm_sub` restores the missing term,
which is exact and costs one extra transform per call, but gives up the factorisation
that made the split cheap to reason about. `wdm_cut` instead drops the time bins that
carry the onset: it needs no glitch model at all, and it is available only in the
time--frequency plane, because excising an interval is a time-domain statement about a
narrow band of frequencies and notching the glitch out of a frequency-domain likelihood
would mean removing the band the binary lives in.

Phases
------
    python run_unmodelled.py                # all of them, each in its own process
    python run_unmodelled.py probe          # geometry, excision trade-off, linear bias
    python run_unmodelled.py ladder         # MAP bias vs glitch amplitude
    python run_unmodelled.py scatter        # is the shift a bias or a draw? 24 draws
    python run_unmodelled.py chains         # the posteriors themselves, at 4 rho_crit
    python run_unmodelled.py seeds          # seed-to-seed scatter of those widths
    python run_unmodelled.py scatter wdm_split wdm_sub
                                            # any phase but probe: re-run only the named
                                            # analyses, carry the rest over
    python run_unmodelled.py figure         # paper/figures/fig_unmodelled.pdf
    python run_unmodelled.py table          # print the numbers of the paper's summary
                                            # table (tab:unmodelled) from unmodelled.npz

Writes `unmodelled.npz` next to this script. About two hours on one GPU, the chains
dominating.

Separate processes for the same reason as `run_wdm_tdi2.py`: every rung of the amplitude
ladder builds its own jitted WDM closure holding tens of MB of device arrays, and with a
few dozen alive the XLA compilation of the sampler's scan slows to a crawl.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))                        # fd_pipeline.py, run_knee_scan.py
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

# The amplitude ladder, in units of the critical glitch signal-to-noise ratio of
# Eq. (rho_crit) at this configuration and this arrival time. Zero is the control:
# the same analyses on data that contains no glitch at all, which is what separates a
# displacement caused by the glitch from one caused by the noise draw.
RUNGS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
HEADLINE = 4.0                     # the rung the chains and the scatter test use
N_REALISATIONS = 24
EXCISE_HALF = 3                    # time bins dropped either side of the onset

N_WALKERS = 16
# Overridable so that a chain whose diagnostics come back marginal can be re-run longer
# without disturbing the ones that were fine. The glitch-free posteriors of this section
# mix worse than anything else in the paper -- their curvature at the injection is not
# negative definite -- and needed it.
N_BURN = int(os.environ.get("UNMODELLED_NBURN", 2_000))
N_SAMP = int(os.environ.get("UNMODELLED_NSAMP", 10_000))
# The seed of `chains`. Overridable because a chain can fail to explore this posterior
# without its own diagnostics noticing: the first wdm_split run with the six-bin glitch
# window came back 11% narrow in log f0, five standard deviations below sixteen
# fd_gbonly runs of the same binary posterior (`seeds`), and was re-run from seed 17.
CHAIN_SEED = int(os.environ.get("UNMODELLED_CHAIN_SEED", 7))
GB_LABELS = ["log_f0", "log_fdot", "log_A_gb", "psi"]
ALL_LABELS = GB_LABELS + ["t0", "log_Ag", "log_tau"]
ANALYSES = ("fd_joint", "fd_gbonly", "wdm_split", "wdm_sub", "wdm_gbonly", "wdm_cut")

# ---------------------------------------------------------------------------
# grid, injections, tilings -- identical to glitch_and_gb_wdm.ipynb
# ---------------------------------------------------------------------------
DS = np.load(HERE / "dataset.npz")
DT, N = float(DS["DT"]), int(DS["N"])
T_OBS, T_ARM = float(DS["T_OBS"]), float(DS["T_ARM"])
NF_GB, NF_GL = 1536, 128
NT_GB, NT_GL = N // NF_GB, N // NF_GL

freq = jnp.asarray(np.fft.rfftfreq(N, DT))
f_safe = jnp.where(freq > 0, freq, 1.0)
n_fine = len(freq)
PSD = ns.psd_tdi1_array(f_safe, t_obs=T_OBS)

GB_TRUE = jnp.asarray(DS["gb_true"])
G3_TRUE = jnp.asarray(DS["glitch_true"])
F0_TRUE = float(GB_TRUE[0])
RA, DEC, IOTA, PHI0 = (float(GB_TRUE[i]) for i in (3, 4, 6, 7))
T0_TRUE, DELTAV_TRUE, TAU_TRUE = (float(v) for v in G3_TRUE)
N_GB = int(DS["N_GB"])
JGB = jgb_mod.JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=T_OBS, t0=0.0, n=N_GB)

H_GB = jnp.asarray(DS["h_gb_tdi1"])
H_GL1 = jnp.asarray(DS["h_glitch_tdi1"])           # at Deltav = DELTAV_TRUE
NOISE = jnp.asarray(DS["noise_tdi1"])
RHO_GB = float(jg.snr(H_GB, PSD))
RHO_GL1 = float(jg.snr(H_GL1, PSD))                # SNR per unit of the fiducial Deltav


def to_time(H):
    return jnp.stack([jnp.fft.irfft(H[:, c], n=N) / DT for c in range(3)])


def wdm_of(x, nt):
    return WDM.from_time_series(TimeSeries(x, dt=DT, backend=BACKEND), nt=nt)


def pixel_variance(freq_grid, nt, nf):
    fch = jnp.where(jnp.asarray(freq_grid) > 0, jnp.asarray(freq_grid), 1.0)
    S1 = jnp.stack(ns.psd_tdi1(fch), axis=-1)
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


_w = wdm_of(to_time(H_GB), NT_GB)
FREQ_GB, TIME_GB = np.asarray(_w.freq_grid), np.asarray(_w.time_grid)
_w = wdm_of(to_time(H_GL1), NT_GL)
FREQ_GL, TIME_GL = np.asarray(_w.freq_grid), np.asarray(_w.time_grid)
VAR_GB = pixel_variance(FREQ_GB, NT_GB, NF_GB)
VAR_GL = pixel_variance(FREQ_GL, NT_GL, NF_GL)

K_GB = int(np.argmin(np.abs(FREQ_GB - F0_TRUE)))
GB_CH = jnp.arange(max(1, K_GB - 2), min(NF_GB, K_GB + 3))
# W_gl: the four time bins from the onset on, and the two before it, which the periodic
# transform puts at the end of the year. Same window as glitch_and_gb_wdm.ipynb and
# run_wdm_tdi2.py; only wdm_split and wdm_sub read it.
NT_WIN, N_WRAP = 4, 2
GL_T = jnp.asarray(np.r_[np.arange(NT_GL - N_WRAP, NT_GL), np.arange(NT_WIN)])
K_NOTCH = int(np.argmin(np.abs(FREQ_GL - F0_TRUE)))
GL_CH = jnp.asarray([m for m in range(1, NF_GL) if abs(m - K_NOTCH) > 1])
T0_MAX = float(TIME_GL[NT_WIN - 1])


# ---------------------------------------------------------------------------
# the excision window: which time bins of the GB tiling carry the glitch
# ---------------------------------------------------------------------------

def _windowed_power(coeffs):
    """Per-time-bin rho^2 inside the GB window, (NT_GB,)."""
    r = np.nan_to_num(np.asarray(coeffs)[:, :, GB_CH] ** 2
                      / np.asarray(VAR_GB)[:, :, GB_CH])
    return r.sum(axis=(0, 2))


_P_GL_T = _windowed_power(wdm_of(to_time(H_GL1), NT_GB).coeffs)
_P_GB_T = _windowed_power(wdm_of(to_time(H_GB), NT_GB).coeffs)
_ORDER = np.argsort(_P_GL_T)[::-1]


def excision(n_half=EXCISE_HALF):
    """Time bins of the binary tiling to drop: the onset bin plus `n_half` either side.

    Contiguous and centred on the onset, wrapping at the ends of the record -- the
    glitch here arrives 400 s into the year, so its wavelet footprint straddles the
    join and bin 121 carries as much of it as bin 1. The window is chosen from the
    glitch *template*, not from the data: an analysis reaches this stage with a trigger
    that localises the onset to far better than the three-day time bin, so the
    footprint is known in advance. Nothing here uses the noise.

    A contiguous window rather than the `n_half` bins with the most glitch power in
    them, which is what one writes first. Ranking by power picks up the wavelet's
    side lobes before the bins between them -- bins 3 and 119 before bins 2 and 120 --
    and the side lobes alternate in sign, so a ranked cut can leave a residual that a
    wider ranked cut removes and a narrower one cancels by accident. The bias then
    moves non-monotonically with the number of bins dropped, for a reason that has
    nothing to do with the glitch.
    """
    i0 = int(np.argmin(np.abs(TIME_GB - T0_TRUE)))
    cut = np.unique([(i0 + k) % NT_GB for k in range(-n_half, n_half + 1)])
    keep = np.setdiff1d(np.arange(NT_GB), cut)
    return cut, keep, dict(n_cut=len(cut), i0=i0,
                           gl_removed=float(1.0 - _P_GL_T[keep].sum() / _P_GL_T.sum()),
                           gb_snr_kept=float(np.sqrt(_P_GB_T[keep].sum()
                                                     / _P_GB_T.sum())))


CUT_T, KEEP_T, CUT_INFO = excision()


# ---------------------------------------------------------------------------
# coordinates and priors
# ---------------------------------------------------------------------------

def to_physical(th):
    """7-vector -> (gb8, g3)."""
    gb8 = (jnp.array([0., 0., 0., RA, DEC, 0., IOTA, PHI0])
           .at[0].set(jnp.exp(th[0])).at[1].set(jnp.exp(th[1]))
           .at[2].set(jnp.exp(th[2])).at[5].set(th[3]))
    tau = jnp.exp(th[6])
    return gb8, jnp.stack([th[4], jnp.exp(th[5]) / tau, tau])


def gb8_of(th4):
    return (jnp.array([0., 0., 0., RA, DEC, 0., IOTA, PHI0])
            .at[0].set(jnp.exp(th4[0])).at[1].set(jnp.exp(th4[1]))
            .at[2].set(jnp.exp(th4[2])).at[5].set(th4[3]))


def theta_true(deltav):
    return jnp.array([jnp.log(GB_TRUE[0]), jnp.log(GB_TRUE[1]), jnp.log(GB_TRUE[2]),
                      GB_TRUE[5], T0_TRUE, jnp.log(deltav * TAU_TRUE),
                      jnp.log(TAU_TRUE)])


_GB_BOX = ((float(np.log(1e-4)), float(np.log(3e-3))),
           (float(np.log(1e-22)), float(np.log(1e-15))),
           (float(np.log(1e-25)), float(np.log(1e-20))),
           (0.0, float(np.pi)))


@jax.jit
def log_prior4(th):
    ok = jnp.all(jnp.array([(th[i] >= lo) & (th[i] <= hi)
                            for i, (lo, hi) in enumerate(_GB_BOX)]))
    return jnp.where(ok, 0.0, -jnp.inf)


@jax.jit
def log_prior7(th):
    ok = (jnp.isfinite(log_prior4(th[:4]))
          & (th[4] >= 0.0) & (th[4] <= T0_MAX)
          & (th[5] >= float(np.log(_DELTAV_MIN * _TAU_MIN)))
          & (th[5] <= float(np.log(_DELTAV_MAX * _TAU_MAX)))
          & (th[5] - th[6] >= float(np.log(_DELTAV_MIN)))
          & (th[5] - th[6] <= float(np.log(_DELTAV_MAX)))
          & (th[6] >= float(np.log(_TAU_MIN))) & (th[6] <= float(np.log(_TAU_MAX))))
    return jnp.where(ok, 0.0, -jnp.inf)


PRIOR = {4: log_prior4, 7: log_prior7}


# ---------------------------------------------------------------------------
# the five likelihoods
# ---------------------------------------------------------------------------

def build_fd(data_fd, with_glitch):
    """Exact fine-grid frequency-domain likelihood, DC excluded.

    Exact rather than the hybrid binned likelihood of `fd_pipeline.build_hybrid`:
    the question here is what omitting the glitch costs, and binning is a second
    approximation that would have to be disentangled from it afterwards. At 1 ms a
    call there is no reason to introduce it.
    """
    d = data_fd[1:]

    @jax.jit
    def log_lik(th):
        if with_glitch:
            gb8, g3 = to_physical(th)
            h = gb_fd(gb8) + glitch_fd(g3)
        else:
            h = gb_fd(gb8_of(th))
        r = d - h[1:]
        return -jnp.sum((r.real ** 2 + r.imag ** 2) / PSD[1:])

    return log_lik


def build_wdm(data_fd, with_glitch, t_keep=None, subtract_in_gb=False):
    """Split WDM likelihood, or its Galactic-binary piece alone.

    `with_glitch` keeps both windows and the three glitch parameters. Otherwise only the
    binary window survives -- the glitch window holds no binary signal by construction,
    so dropping the glitch from the model leaves nothing in it -- and `t_keep` selects
    which of its time bins are summed over.

    `subtract_in_gb` is the repair of Sec. "Two tilings of the same data". As written,
    Eq. (wdmsplit) puts h_GB in the binary's window and h_gl in the glitch's, and
    nothing else: the glitch is fitted, but it is never subtracted from the pixels the
    binary is fitted in. That is exactly what makes the posterior factorise, and at the
    fiducial amplitude it costs nothing, because the glitch deposits SNR 0.67 there
    against the binary's 283. It stops being free as soon as the glitch is loud. With
    this flag the binary window is modelled by h_GB + h_gl instead, which couples the
    two blocks again and costs one extra transform per likelihood call.
    """
    tsel = jnp.arange(NT_GB) if t_keep is None else jnp.asarray(t_keep)
    x = to_time(data_fd)
    d_gb = wdm_of(x, NT_GB).coeffs[:, tsel][:, :, GB_CH]
    v_gb = VAR_GB[:, tsel][:, :, GB_CH]
    if with_glitch:
        d_gl = wdm_of(x, NT_GL).coeffs[:, GL_T][:, :, GL_CH]
        v_gl = VAR_GL[:, GL_T][:, :, GL_CH]

    @jax.jit
    def log_lik(th):
        if with_glitch:
            gb8, g3 = to_physical(th)
        else:
            gb8 = gb8_of(th)
        h_gb = gb_fd(gb8)
        if subtract_in_gb:
            h_gb = h_gb + glitch_fd(g3)
        w_gb = wdm_of(to_time(h_gb), NT_GB).coeffs[:, tsel][:, :, GB_CH]
        out = -0.5 * jnp.sum((d_gb - w_gb) ** 2 / v_gb)
        if with_glitch:
            w_gl = wdm_of(to_time(glitch_fd(g3)), NT_GL).coeffs[:, GL_T][:, :, GL_CH]
            out = out - 0.5 * jnp.sum((d_gl - w_gl) ** 2 / v_gl)
        return out

    return log_lik


BUILDERS = {
    "fd_joint":   lambda d: (build_fd(d, True), 7),
    "fd_gbonly":  lambda d: (build_fd(d, False), 4),
    "wdm_split":  lambda d: (build_wdm(d, True), 7),
    "wdm_sub":    lambda d: (build_wdm(d, True, subtract_in_gb=True), 7),
    "wdm_gbonly": lambda d: (build_wdm(d, False), 4),
    "wdm_cut":    lambda d: (build_wdm(d, False, KEEP_T), 4),
}


def data_at(scale, noise=NOISE):
    """h_GB + scale * h_gl + noise. The waveform is linear in Deltav, so `scale` is
    exactly a rescaling of the kick and of the glitch's signal-to-noise ratio."""
    return H_GB + scale * H_GL1 + noise


# ---------------------------------------------------------------------------
# maximisation
# ---------------------------------------------------------------------------

def newton(log_post, x0, n_steps=100, tol=1e-5):
    """Damped Levenberg--Marquardt ascent to the maximum, and the Laplace width there.

    Plain Newton is not enough, for two separate reasons, and both are met here.

    The first is the one `fd_pipeline.laplace` already handles: the
    (log f0, log fdot) block is ill-conditioned, so an undamped step overshoots and
    the step length has to be backtracked.

    The second only appears once the glitch is left out of the model. The curvature of
    a Gaussian log-likelihood is -(dh|dh) + (r|d2h), and the second term is not
    negative: with an unmodelled glitch in the residual r it can flip the sign of the
    weakest direction. It does exactly that along log fdot -- whose second derivative
    is large, since the phase is quadratic in time -- at some of the amplitudes and
    excisions scanned here. The injected point is then a *saddle*, -H^{-1} g points
    downhill, every backtracked step is rejected, and a plain Newton iteration returns
    its own starting point while reporting nothing. That failure is silent and reads
    as an absence of bias, which is the answer one is hoping for; it must not be
    trusted without a check.

    So the Hessian is shifted, H -> H - lambda I, with lambda zero wherever H is
    already negative definite -- which must be the common case, or the ill-conditioned
    (log f0, log fdot) block would be damped into uselessness -- and otherwise just
    past the offending eigenvalue, escalating whenever the line search runs out. The
    step is an ascent direction by construction and degenerates to gradient ascent as
    lambda grows.

    `ok` reports whether the endpoint is a genuine maximum: negative definite
    curvature, and a residual Newton step below `tol` in units of the Laplace width.
    """
    grad = jax.jit(jax.grad(log_post))
    hess = jax.jit(jax.hessian(log_post))
    eye = jnp.eye(x0.shape[0])
    x, lam = x0, 0.0

    def _step(H, g, lam):
        M = H - lam * eye
        s = -jnp.linalg.solve(M, g)
        return s if bool(jnp.all(jnp.isfinite(s))) else None

    for _ in range(n_steps):
        g = grad(x)
        H = 0.5 * (hess(x) + hess(x).T)
        ev_max = float(jnp.linalg.eigvalsh(H)[-1])
        lam = max(lam, 0.0 if ev_max < 0.0 else 2.0 * ev_max)
        f_now, moved = float(log_post(x)), False
        for _ in range(12):                       # escalate the damping if need be
            step = _step(H, g, lam)
            if step is not None:
                t = 1.0
                for _ in range(60):               # backtrack the length
                    if float(log_post(x + t * step)) > f_now:
                        moved = True
                        break
                    t *= 0.5
            if moved:
                break
            lam = 10.0 * lam if lam > 0.0 else max(abs(ev_max) * 1e-8, 1e-30)
        if not moved:
            break
        sig = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(H))))
        x = x + t * step
        lam = 0.1 * lam
        if float(jnp.max(jnp.abs(t * step / sig))) < tol:
            break

    H = 0.5 * (hess(x) + hess(x).T)
    sig = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(H))))
    negdef = bool(jnp.linalg.eigvalsh(H)[-1] < 0.0)
    resid = float(jnp.max(jnp.abs(jnp.linalg.solve(H, grad(x)) / sig)))
    return np.asarray(x), np.asarray(sig), bool(negdef and resid < 1e-2)


def cv_bias(builder, scale, th4):
    """Linearised systematic displacement of the binary, Eq. (cvbias), in units of sigma.

    Built from two evaluations at the injected point: the curvature of the *glitch-free*
    problem, which is exactly -(dh|dh) because the residual vanishes there and is
    therefore negative definite whatever the glitch does, and the gradient of the same
    likelihood on data that contains the glitch, which is (d_j h | h_gl).

    This is the estimator to use when the full Hessian is not negative definite -- see
    `newton` -- because it never involves the residual second-derivative term that
    causes the trouble. It is also the object Sec. "How far does the separation extend?"
    scans, so the two are directly comparable. What it cannot do is follow the maximum
    once the displacement is large enough for first order to fail; that is what the
    chains are for.
    """
    H0 = jax.hessian(builder(H_GB))(jnp.asarray(th4))
    g = jax.grad(builder(H_GB + scale * H_GL1))(jnp.asarray(th4))
    H0 = 0.5 * np.asarray(H0 + H0.T)
    sig = np.sqrt(np.abs(np.diag(-np.linalg.inv(H0))))
    return -np.linalg.solve(H0, np.asarray(g)) / sig


def fit(name, data_fd, deltav):
    """MAP and Laplace width of one analysis on one data stream, GB block only."""
    log_lik, dim = BUILDERS[name](data_fd)
    lp = PRIOR[dim]
    log_post = jax.jit(lambda th: log_lik(th) + lp(th))
    th_t = theta_true(deltav)[:dim]
    x, s, ok = newton(log_post, th_t)
    return x[:4], s[:4], np.asarray(th_t)[:4], ok


# ---------------------------------------------------------------------------
# phase: probe
# ---------------------------------------------------------------------------

def phase_probe():
    """Window geometry, the excision trade-off, and the linearised bias in both domains."""
    import run_knee_scan as ks

    grid, model, n_gb, psd = ks._setup(N, DT, N_GB)
    r = ks.point(F0_TRUE, TAU_TRUE, grid, model, n_gb, psd,
                 deltav=DELTAV_TRUE, a_gb=float(GB_TRUE[2]), t0=T0_TRUE)
    rho_crit_worst = float(r["rho_crit"])
    rho_crit_fid = 1.0 / float(r["beta_fid"])
    half = ks.prior_half(F0_TRUE, grid["df"])
    print("== the configuration ==")
    print(f"  f0 = {F0_TRUE * 1e3:.3f} mHz   tau = {TAU_TRUE:.0f} s   "
          f"x = 2 pi f0 tau = {r['x']:.3f}   t0 = {T0_TRUE:.0f} s")
    print(f"  rho_GB = {RHO_GB:.1f}   rho_gl(fiducial Deltav) = {RHO_GL1:.2f}")
    print(f"  rho_crit  = {rho_crit_fid:.1f} at the injected arrival time, "
          f"{rho_crit_worst:.1f} at the worst one")
    print(f"  likelihood width / prior half-width, binary block: "
          + "  ".join(f"{lab}={s / h:.2e}"
                      for lab, s, h in zip(GB_LABELS, r["sig_only"], half)))

    print(f"\n== the binary window on the Nf = {NF_GB} tiling ==")
    print(f"  {len(GB_CH)} channels, {NT_GB} time bins of "
          f"{(TIME_GB[1] - TIME_GB[0]) / 86400:.3f} d")
    print(f"  glitch power per time bin, ranked:")
    for k in _ORDER[:6]:
        print(f"    bin {k:4d}  t = {TIME_GB[k] / 86400:7.2f} d   "
              f"glitch {_P_GL_T[k] / _P_GL_T.sum():9.5f}   "
              f"binary {_P_GB_T[k] / _P_GB_T.sum():9.5f}")

    # What excising costs and what it buys, on noiseless data at the headline rung:
    # every bin removed takes some of the glitch and some of the binary with it.
    print(f"\n== the excision trade-off, noiseless data at rho_gl = "
          f"{HEADLINE:g} rho_crit ==")
    scale_head = HEADLINE * rho_crit_fid / RHO_GL1
    clean_head = H_GB + scale_head * H_GL1
    th4 = np.asarray(theta_true(DELTAV_TRUE)[:4])
    halves = np.array([-1, 0, 1, 2, 3, 4, 6, 8])
    trade, bias_cut, bias_map = [], [], []
    for c in halves:
        keep = None if c < 0 else excision(int(c))[1]
        info = (dict(n_cut=0, gl_removed=0.0, gb_snr_kept=1.0) if c < 0
                else excision(int(c))[2])
        bias_cut.append(float(np.abs(cv_bias(
            lambda d: build_wdm(d, False, keep), scale_head, th4)).max()))
        x, s_, ok = newton(jax.jit(lambda th: build_wdm(clean_head, False, keep)(th)
                                   + log_prior4(th)), jnp.asarray(th4))
        bias_map.append((float(np.abs((x - th4) / s_).max()), ok))
        trade.append((info["n_cut"], info["gl_removed"], info["gb_snr_kept"]))
        print(f"  cut {info['n_cut']:3d} bins ({info['gl_removed'] * 100:7.3f}% of the "
              f"glitch): binary SNR x {info['gb_snr_kept']:.5f}, "
              f"linearised {bias_cut[-1]:8.4f} sigma, maximum {bias_map[-1][0]:8.4f} "
              f"{'' if ok else '(not a maximum)'}")
        jax.clear_caches()

    print(f"\n== linearised bias at rho_gl = {HEADLINE:g} rho_crit, noiseless data ==")
    zs, cvs = {}, {}
    for name in ANALYSES:
        x, s, t, ok = fit(name, H_GB + scale_head * H_GL1, scale_head * DELTAV_TRUE)
        zs[name] = (x - t) / s
        cv = cv_bias((lambda n: (lambda d: BUILDERS[n](d)[0]))(name), scale_head, th4) \
            if name in ("fd_gbonly", "wdm_gbonly", "wdm_cut") else np.zeros(4)
        cvs[name] = cv
        print(f"  {name:11s} maximum   " + "  ".join(
            f"{lab}={v:+8.4f}" for lab, v in zip(GB_LABELS, zs[name]))
            + ("" if ok else "   (not a maximum)"))
        if cv.any():
            print(f"  {'':11s} linearised " + " ".join(
                f"{lab}={v:+8.4f}" for lab, v in zip(GB_LABELS, cv)))
    d = np.abs(zs["fd_gbonly"] - zs["wdm_gbonly"]).max()
    print(f"  frequency vs time--frequency, glitch unmodelled, all pixels kept: "
          f"max difference {d:.4f} sigma "
          f"({d / max(np.abs(zs['fd_gbonly']).max(), 1e-30) * 100:.2f}% of the bias)")

    return dict(rho_crit_fid=rho_crit_fid, rho_crit_worst=rho_crit_worst,
                rho_gb=RHO_GB, rho_gl_unit=RHO_GL1, x_sep=float(r["x"]),
                sig_only=np.asarray(r["sig_only"]), prior_half=np.asarray(half),
                p_gl_t=_P_GL_T, p_gb_t=_P_GB_T, time_gb=TIME_GB,
                cut_bins=CUT_T, cut_info=np.array([CUT_INFO["n_cut"],
                                                   CUT_INFO["gl_removed"],
                                                   CUT_INFO["gb_snr_kept"]]),
                trade_half=halves, trade=np.array(trade),
                trade_bias=np.array(bias_cut),
                trade_map=np.array([b for b, _ in bias_map]),
                trade_map_ok=np.array([o for _, o in bias_map]),
                probe_z=np.array([zs[a] for a in ANALYSES]),
                probe_cv=np.array([cvs[a] for a in ANALYSES]))


# ---------------------------------------------------------------------------
# phase: ladder
# ---------------------------------------------------------------------------

def _subset(only):
    """Indices into ANALYSES to (re)compute: all of them, or the named ones."""
    if not only:
        return list(range(len(ANALYSES)))
    bad = [a for a in only if a not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analyses {bad}; choose from {ANALYSES}")
    return [ANALYSES.index(a) for a in only]


def _previous(phase, only):
    """The stored result of `phase`, which a subset run updates in place."""
    if not only:
        return None
    prev = _out(phase)
    if not prev.exists():
        raise FileNotFoundError(f"{prev} missing -- run the full phase first")
    print(f"re-running only {list(only)}; the rest are carried over unchanged")
    return dict(np.load(prev, allow_pickle=True))


def phase_ladder(only=None):
    """Bias of the four binary parameters against glitch amplitude, five analyses,
    on noiseless data and on the stored noise draw.

    `only` re-computes the named analyses and carries the others over from the stored
    file, as `phase_chains` does -- used when a change touches some analyses and not
    others, such as the glitch window, which only wdm_split and wdm_sub read.
    """
    rho_crit = float(np.load(_out("probe"))["rho_crit_fid"])
    print(f"rho_crit = {rho_crit:.1f} (injected arrival time)")
    prev = _previous("ladder", only)
    todo = _subset(only)
    if prev is None:
        z = {k: np.zeros((len(RUNGS), len(ANALYSES), 4)) for k in ("clean", "noisy")}
        sig = {k: np.zeros((len(RUNGS), len(ANALYSES), 4)) for k in ("clean", "noisy")}
        conv = {k: np.zeros((len(RUNGS), len(ANALYSES)), bool) for k in ("clean", "noisy")}
    else:
        z = {k: prev[f"z_{k}"].copy() for k in ("clean", "noisy")}
        sig = {k: prev[f"sig_{k}"].copy() for k in ("clean", "noisy")}
        conv = {k: prev[f"conv_{k}"].copy() for k in ("clean", "noisy")}
    for i, rung in enumerate(RUNGS):
        scale = rung * rho_crit / RHO_GL1
        dv = max(scale, 1e-12) * DELTAV_TRUE
        print(f"\n-- rho_gl = {rung:g} rho_crit = {rung * rho_crit:8.1f}, "
              f"Deltav = {dv:.3e} m/s", flush=True)
        for tag, nz in (("clean", jnp.zeros_like(NOISE)), ("noisy", NOISE)):
            for j in todo:
                name = ANALYSES[j]
                x, s, t, ok = fit(name, data_at(scale, nz), dv)
                z[tag][i, j], sig[tag][i, j] = (x - t) / s, s
                conv[tag][i, j] = ok
            print(f"   {tag:6s} " + "  ".join(
                f"{n.split('_')[-1][:6]:>6s}:{np.abs(z[tag][i, k]).max():7.3f}"
                for k, n in enumerate(ANALYSES)), flush=True)
        jax.clear_caches()
    return dict(rungs=np.array(RUNGS), rho_crit=rho_crit,
                z_clean=z["clean"], z_noisy=z["noisy"],
                sig_clean=sig["clean"], sig_noisy=sig["noisy"],
                conv_clean=conv["clean"], conv_noisy=conv["noisy"],
                analyses=np.array(ANALYSES), gb_labels=np.array(GB_LABELS))


# ---------------------------------------------------------------------------
# phase: scatter
# ---------------------------------------------------------------------------

def _scatter(rung, tag, only=None):
    """A displacement on one noise draw is not a bias. Repeat over independent draws.

    The draws are keyed on fixed seeds, so `only` can re-compute some analyses on the
    same realisations and carry the others over from the stored file."""
    rho_crit = float(np.load(_out("probe"))["rho_crit_fid"])
    scale = rung * rho_crit / RHO_GL1
    dv = scale * DELTAV_TRUE
    print(f"== {N_REALISATIONS} noise realisations at rho_gl = {rung * rho_crit:.0f} "
          f"= {rung:g} rho_crit ==", flush=True)
    prev = _previous(tag, only)
    todo = _subset(only)
    if prev is None:
        z = np.zeros((N_REALISATIONS, len(ANALYSES), 4))
        conv = np.zeros((N_REALISATIONS, len(ANALYSES)), bool)
    else:
        z, conv = prev[f"{tag}_z"].copy(), prev[f"{tag}_conv"].copy()
    for s_ in range(N_REALISATIONS):
        nz = ns.sample_noise_fd(jr.split(jr.PRNGKey(2000 + s_))[0], PSD).at[0].set(0 + 0j)
        for j in todo:
            name = ANALYSES[j]
            x, sg, t, ok = fit(name, data_at(scale, nz), dv)
            z[s_, j] = (x - t) / sg
            conv[s_, j] = ok
        print(f"  seed {s_:2d}: " + "  ".join(
            f"{np.abs(z[s_, k]).max():6.2f}" for k in range(len(ANALYSES))), flush=True)
        jax.clear_caches()
    print(f"\n  {'analysis':12s}{'max_i |z_i|':>26s}   most displaced parameter")
    for j, name in enumerate(ANALYSES):
        mx = np.abs(z[:, j]).max(axis=1)
        k = int(np.abs(z[:, j].mean(0)).argmax())
        print(f"  {name:12s}{mx.mean():8.3f} +- {mx.std() / np.sqrt(N_REALISATIONS):5.3f}"
              f"  (scatter {mx.std():6.3f})   {GB_LABELS[k]:9s} "
              f"{z[:, j, k].mean():+7.3f} +- {z[:, j, k].std() / np.sqrt(N_REALISATIONS):.3f}")
    return {f"{tag}_z": z, f"{tag}_conv": conv, f"{tag}_rho": rung * rho_crit,
            f"{tag}_rung": rung}


def phase_scatter(only=None):
    return _scatter(HEADLINE, "scatter", only)


def phase_scattercrit(only=None):
    """The same at exactly rho_crit, where the one-sigma definition is testable and the
    displacement is small enough for the maximum and the median to still agree."""
    return _scatter(1.0, "scattercrit", only)


# ---------------------------------------------------------------------------
# phase: chains
# ---------------------------------------------------------------------------

def run_chain(log_lik, x0, sigma, dim, seed):
    from jexplore.sampler import JaxSampler, Steps
    from jexplore.sampling import EpochMH, SamplingMH
    from jexplore.steps import Stretch
    from jexplore.backends import DefaultBackend

    lp = PRIOR[dim]
    sampling = SamplingMH(nwalker=N_WALKERS, temps=jnp.array([1.0]),
                          loglik=log_lik, logprior=lp, dim=dim)
    steps = Steps([{Stretch(permute=True).builder: 1.0}])
    p0 = jnp.asarray(x0) + jr.multivariate_normal(
        jr.PRNGKey(seed), jnp.zeros(dim), jnp.diag(jnp.asarray(sigma) ** 2) * 0.01,
        shape=(N_WALKERS,))
    backend = DefaultBackend(burn=N_BURN, inmem_epochs=1)
    JaxSampler(sampling, steps, backend).run(EpochMH({"p": p0}),
                                             niters=N_BURN + N_SAMP, nepoch=1, seed=seed)
    s = backend.get_samples()["p"]
    return np.asarray(jnp.array(s.transpose(0, 2, 1).reshape(-1, dim)))


def phase_chains(only=None):
    """The posteriors themselves, at the headline amplitude, on the stored noise draw.

    `only` re-runs a subset and merges the result into the stored file, leaving the
    other chains bit-identical. Used to lengthen the two whose diagnostics came back
    marginal without paying for the three that did not.
    """
    rho_crit = float(np.load(_out("probe"))["rho_crit_fid"])
    scale = HEADLINE * rho_crit / RHO_GL1
    dv = scale * DELTAV_TRUE
    data = data_at(scale)
    print(f"rho_gl = {HEADLINE * rho_crit:.0f} = {HEADLINE:g} rho_crit, "
          f"Deltav = {dv:.3e} m/s, rho_GB = {RHO_GB:.1f}")
    print(f"sampler: {N_WALKERS} walkers, {N_BURN} burn + {N_SAMP} samples")
    out = {"chain_rho_gl": HEADLINE * rho_crit, "chain_deltav": dv}
    # wdm_gbonly is omitted: `probe` shows it reproduces fd_gbonly to 1e-4 sigma, so a
    # second 40-minute chain of the same posterior would buy nothing.
    names = [a for a in ANALYSES if a != "wdm_gbonly"]
    if only:
        prev = _out("chains")
        if not prev.exists():
            raise FileNotFoundError(f"{prev} missing -- run the full phase first")
        out = dict(np.load(prev, allow_pickle=True))
        names = list(only)
        print(f"re-running only {names}; the rest are carried over unchanged")
    for name in names:
        log_lik, dim = BUILDERS[name](data)
        lp = PRIOR[dim]
        log_post = jax.jit(lambda th: log_lik(th) + lp(th))
        x, s, ok = newton(log_post, theta_true(dv)[:dim])
        if not ok:
            print(f"   [{name}] warning: the maximum is not a clean one; the chain "
                  f"is the statement, the start merely a start")
        log_lik(jnp.asarray(x)).block_until_ready()
        t0 = time.time()
        for _ in range(20):
            log_lik(jnp.asarray(x)).block_until_ready()
        per = (time.time() - t0) / 20
        print(f"\n[{name}] dim {dim}, {1e3 * per:.2f} ms/call -> "
              f"{per * N_WALKERS * (N_BURN + N_SAMP) / 60:.1f} min", flush=True)
        t0 = time.time()
        ch = run_chain(log_lik, x, s, dim, seed=CHAIN_SEED)
        print(f"[{name}] {ch.shape} in {time.time() - t0:.0f} s", flush=True)
        t = np.asarray(theta_true(dv)[:dim])
        med, sg = np.median(ch, axis=0), ch.std(axis=0)
        out[f"chain_{name}"] = ch
        out[f"seed_chain_{name}"] = CHAIN_SEED
        out[f"map_{name}"] = x
        out[f"true_{name}"] = t
        print(f"   {'param':10s}{'median-truth':>14s}{'in sigma':>10s}{'sigma':>12s}")
        for i in range(dim):
            print(f"   {ALL_LABELS[i]:10s}{med[i] - t[i]:14.5g}"
                  f"{(med[i] - t[i]) / sg[i]:10.2f}{sg[i]:12.4g}")
        del ch
        jax.clear_caches()
    return out


def phase_seeds(only=None):
    """Seed-to-seed scatter of sampled widths and medians, at the headline rung.

    The glitch-free posteriors of this section are 17 times wider in log fdot than the
    joint one and reach the prior wall, and their chains mix slowly. Their
    autocorrelation diagnostics look acceptable and their halves agree -- yet two runs of
    wdm_split differing only in the glitch window, which does not enter its binary block,
    returned log f0 widths 11% apart. The spread between independent runs is the honest
    Monte Carlo error on such a width, and this measures it on fd_gbonly, the cheapest
    analysis with the same binary posterior: a few chains of the same length as the
    wdm_split one, from different seeds, on the stored noise draw.

    Seeds come from UNMODELLED_SEEDS (default 8,9,10,11; the stored chains use 7).
    """
    rho_crit = float(np.load(_out("probe"))["rho_crit_fid"])
    scale = HEADLINE * rho_crit / RHO_GL1
    dv = scale * DELTAV_TRUE
    data = data_at(scale)
    seeds = [int(v) for v in os.environ.get("UNMODELLED_SEEDS", "8,9,10,11").split(",")]
    names = list(only) if only else ["fd_gbonly"]
    print(f"sampler: {N_WALKERS} walkers, {N_BURN} burn + {N_SAMP} samples; seeds {seeds}")
    out = {}
    for name in names:
        log_lik, dim = BUILDERS[name](data)
        lp = PRIOR[dim]
        log_post = jax.jit(lambda th: log_lik(th) + lp(th))
        x, sg, ok = newton(log_post, theta_true(dv)[:dim])
        t = np.asarray(theta_true(dv)[:dim])
        med, wid = [], []
        for seed in seeds:
            t0 = time.time()
            ch = run_chain(log_lik, x, sg, dim, seed=seed)
            med.append(np.median(ch, axis=0))
            wid.append(ch.std(axis=0))
            print(f"[{name}] seed {seed}: {time.time() - t0:.0f} s; widths "
                  + " ".join(f"{v:.4g}" for v in wid[-1][:4]) + "; median-truth / width "
                  + " ".join(f"{v:+.2f}" for v in ((med[-1] - t) / wid[-1])[:4]), flush=True)
            del ch
        out[f"seed_list_{name}"] = np.array(seeds)
        out[f"seed_median_{name}"] = np.array(med)
        out[f"seed_width_{name}"] = np.array(wid)
        out[f"seed_true_{name}"] = t
        jax.clear_caches()
    return out


# ---------------------------------------------------------------------------
# merge and drive
# ---------------------------------------------------------------------------
PHASES = ("probe", "ladder", "scatter", "scattercrit", "chains", "seeds")


def _out(phase):
    return HERE / f"unmodelled_{phase}.npz"


# Phases the figure and the paper's quoted numbers actually need. The others are
# supporting tests, so a merge without them is incomplete but usable -- which lets the
# figure be drawn as soon as the chains land, while a slow phase is still running.
REQUIRED = ("probe", "ladder", "chains")


def merge():
    out, missing = {}, []
    for phase in PHASES:
        p = _out(phase)
        if not p.exists():
            missing.append(phase)
            continue
        out |= dict(np.load(p, allow_pickle=True))
    if set(missing) & set(REQUIRED):
        raise FileNotFoundError(
            f"required phase(s) {sorted(set(missing) & set(REQUIRED))} missing -- "
            f"run `{sys.argv[0]} <phase>`")
    if missing:
        print(f"WARNING: merging without {missing}; re-run merge once they finish")
    np.savez_compressed(HERE / "unmodelled.npz", **out)
    print(f"saved {HERE / 'unmodelled.npz'}  ({len(PHASES) - len(missing)}"
          f"/{len(PHASES)} phases)")


def table():
    """Print the numbers of the paper's Table `tab:unmodelled`, read off `unmodelled.npz`.

    Nothing is recomputed. Displacements are of the four binary parameters, in units of
    each analysis's own width: Laplace widths at the maximum for the ladder and the draws,
    sampled widths for the chains. The two rows without a glitch in the data are rung 0
    of the ladder, where only the four-parameter analyses were run.
    """
    U = dict(np.load(HERE / "unmodelled.npz", allow_pickle=True))
    names = [str(a) for a in U["analyses"]]
    j = {a: names.index(a) for a in names}
    rungs = [float(r) for r in U["rungs"]]
    r0, rh = rungs.index(0.0), rungs.index(HEADLINE)
    rho_crit = float(U["rho_crit"])
    head = "".join(f"{lab:>11s}" for lab in GB_LABELS)

    def row(label, v, fmt="{:+11.3f}"):
        print(f"  {label:18s}" + "".join(fmt.format(x) for x in v))

    print(f"rho_crit = {rho_crit:.1f}; headline rung {HEADLINE:g} rho_crit = "
          f"{HEADLINE * rho_crit:.0f}")

    print("\n-- no glitch in the data (rung 0): maxima --")
    print(f"  {'':18s}{head}")
    for a in ("fd_gbonly", "wdm_gbonly"):
        row(a + " clean", U["z_clean"][r0, j[a]])
        row(a + " noisy", U["z_noisy"][r0, j[a]])
    print(f"  max |fd_gbonly - wdm_gbonly|, stored noise draw: "
          f"{np.abs(U['z_noisy'][r0, j['fd_gbonly']] - U['z_noisy'][r0, j['wdm_gbonly']]).max():.1e}")
    print(f"  max |fd_joint (glitch in data, rung {HEADLINE:g}) - fd_gbonly (rung 0)|, "
          f"stored draw: {np.abs(U['z_noisy'][rh, j['fd_joint']] - U['z_noisy'][r0, j['fd_gbonly']]).max():.4f}")

    print(f"\n-- glitch in the data, {HEADLINE:g} rho_crit: noiseless maxima --")
    print(f"  {'':18s}{head}")
    for a in names:
        row(a, U["z_clean"][rh, j[a]])
    print(f"  max |wdm_gbonly - fd_gbonly| = "
          f"{np.abs(U['z_clean'][rh, j['wdm_gbonly']] - U['z_clean'][rh, j['fd_gbonly']]).max():.1e}, "
          f"max |wdm_split - fd_gbonly| = "
          f"{np.abs(U['z_clean'][rh, j['wdm_split']] - U['z_clean'][rh, j['fd_gbonly']]).max():.1e}")

    for tag in ("scattercrit", "scatter"):
        if f"{tag}_z" not in U:
            print(f"\n({tag} not merged into unmodelled.npz)")
            continue
        z = U[f"{tag}_z"]
        n = z.shape[0]
        print(f"\n-- {n} noise draws at rho_gl = {float(U[tag + '_rho']):.0f}: "
              f"mean displacement +- standard error --")
        print(f"  {'':18s}{head}")
        for a in names:
            m, se = z[:, j[a]].mean(axis=0), z[:, j[a]].std(axis=0) / np.sqrt(n)
            print(f"  {a:12s}" + "".join(f"{x:+6.2f}+-{e:.2f}" for x, e in zip(m, se)))
        print(f"  draw by draw: max |wdm_sub - fd_joint| = "
              f"{np.abs(z[:, j['wdm_sub']] - z[:, j['fd_joint']]).max():.4f}, "
              f"max |wdm_split - fd_gbonly| = "
              f"{np.abs(z[:, j['wdm_split']] - z[:, j['fd_gbonly']]).max():.4f}, "
              f"max |wdm_gbonly - fd_gbonly| = "
              f"{np.abs(z[:, j['wdm_gbonly']] - z[:, j['fd_gbonly']]).max():.4f}")

    print(f"\n-- chains at {HEADLINE:g} rho_crit on the stored draw --")
    ref = U["chain_fd_joint"][:, :4].std(axis=0)
    print(f"  {'':18s}{head}")
    for a in ("fd_joint", "fd_gbonly", "wdm_split", "wdm_sub", "wdm_cut"):
        ch = U[f"chain_{a}"]
        row(a + " width", ch[:, :4].std(axis=0) / ref, "{:11.3f}")
    for a in ("fd_joint", "fd_gbonly", "wdm_split", "wdm_sub", "wdm_cut"):
        ch, t = U[f"chain_{a}"], U[f"true_{a}"]
        z = (np.median(ch, axis=0) - t) / ch.std(axis=0)
        row(a + " median", z[:4], "{:+11.2f}")
        if ch.shape[1] == 7:
            print(f"  {'':12s}glitch block (t0, log A_g, log tau): "
                  + " ".join(f"{x:+.2f}" for x in z[4:]))


def main():
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        if phase == "merge":
            return merge()
        if phase == "figure":
            import make_fig_unmodelled
            return make_fig_unmodelled.main()
        if phase == "table":
            return table()
        if phase not in PHASES:
            raise SystemExit(f"phase must be one of {PHASES + ('merge', 'figure', 'table')}")
        extra = sys.argv[2:]
        if extra and phase == "probe":
            raise SystemExit("`probe` does not take a list of analyses")
        res = globals()[f"phase_{phase}"](extra) if extra \
            else globals()[f"phase_{phase}"]()
        np.savez_compressed(_out(phase), **res)
        print(f"saved {_out(phase)}")
        return
    for phase in PHASES:
        print(f"\n########## {phase} ##########", flush=True)
        subprocess.run([sys.executable, "-u", __file__, phase], check=True)
    merge()
    subprocess.run([sys.executable, "-u", __file__, "figure"], check=True)


if __name__ == "__main__":
    main()
