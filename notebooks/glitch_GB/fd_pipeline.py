"""Reusable frequency-domain pipeline: hybrid binned likelihood + sampler.

Extracted from `glitch_and_gb.ipynb` so that the extra studies -- the run with the
Galactic-binary sky and orientation free, and the P--P calibration test -- share one
implementation with the notebook rather than three copies that can drift apart.

Sampling coordinates
--------------------
Galactic binary, either 4 parameters (sky and orientation held fixed) or 8:

    log f0, log fdot, log A            [, ra, sin dec, cos iota, phi0]
    psi

Uniform priors in these coordinates give the usual isotropic sky and orientation
priors. The glitch always contributes three:

    t0, log A_g, log tau      with A_g = Deltav * tau

`A_g` rather than `Deltav` because the two are nearly perfectly anti-correlated at
fixed A_g once the spectral knee approaches the band edge.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import jaxglitches as jg
from jaxglitches.waveform import tdi1_1exp_f_glitch, AET

# noise.py sits one level up, in notebooks/. This module is imported both from
# glitch_GB/ and from pp/, so it puts that directory on the path itself rather
# than relying on whoever imports it.
_NOTEBOOKS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)
import noise as ns  # noqa: E402

T_ARM = jg.T_ARM_s

# ---------------------------------------------------------------------------
# grid and models
# ---------------------------------------------------------------------------

def make_grid(N: int, DT: float):
    freq = jnp.asarray(np.fft.rfftfreq(N, DT))
    return dict(N=N, DT=DT, T_OBS=N * DT, freq=freq,
                f_safe=jnp.where(freq > 0, freq, 1.0), n_fine=len(freq),
                df=float(freq[1]))


def make_gb_model(T_OBS: float, n_gb: int = 256):
    import lisaorbits
    from jaxgb import jaxgb as jgb_mod
    return jgb_mod.JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=float(T_OBS),
                         t0=0.0, n=n_gb), n_gb


def glitch_fd(g3, freqs):
    """Glitch TDI-1 (A,E,T) at arbitrary frequencies. g3 = [t0, Deltav, tau]."""
    fs = jnp.where(freqs > 0, freqs, 1.0)
    return jnp.stack(AET(*tdi1_1exp_f_glitch(fs, g3[0], g3[1], g3[2], T_ARM)), axis=-1)


def gb_segment(model, n_gb, gb8):
    """GB on its own n_gb bins, plus the starting bin index."""
    p = gb8[None, :]
    segs = model.get_tdi(p, tdi_generation=1.5, tdi_combination="AET")
    seg = jnp.stack(segs, axis=0).astype(jnp.complex128)[:, 0, :].T
    return seg, model.get_kmin(p[:, 0])[0].astype(jnp.int32)


def gb_fd_full(model, n_gb, gb8, n_fine):
    """GB on the full fine grid -- for building data, not for the likelihood."""
    seg, k = gb_segment(model, n_gb, gb8)
    out = jnp.zeros((n_fine, 3), dtype=jnp.complex128)
    return jax.lax.dynamic_update_slice(out, seg, (k, jnp.zeros((), jnp.int32)))


# ---------------------------------------------------------------------------
# parameterisation
# ---------------------------------------------------------------------------

class Coords:
    """Map sampling vector <-> (gb8, g3), with the sky either fixed or free."""

    def __init__(self, free_sky: bool, fixed=(1.0, -0.5, 1.0, 0.0)):
        self.free_sky = free_sky
        self.ra0, self.dec0, self.iota0, self.phi00 = fixed
        self.names = (["log_f0", "log_fdot", "log_A_gb", "psi"]
                      + (["ra", "sin_dec", "cos_iota", "phi0"] if free_sky else [])
                      + ["t0", "log_Ag", "log_tau"])
        self.dim = len(self.names)

    def to_sampling(self, gb8, g3):
        v = [jnp.log(gb8[0]), jnp.log(gb8[1]), jnp.log(gb8[2]), gb8[5]]
        if self.free_sky:
            v += [gb8[3], jnp.sin(gb8[4]), jnp.cos(gb8[6]), gb8[7]]
        v += [g3[0], jnp.log(g3[1] * g3[2]), jnp.log(g3[2])]
        return jnp.array(v)

    def to_physical(self, th):
        if self.free_sky:
            ra, dec = th[4], jnp.arcsin(th[5])
            iota, phi0 = jnp.arccos(th[6]), th[7]
            j = 8
        else:
            ra, dec, iota, phi0 = self.ra0, self.dec0, self.iota0, self.phi00
            j = 4
        gb8 = jnp.stack([jnp.exp(th[0]), jnp.exp(th[1]), jnp.exp(th[2]),
                         ra, dec, th[3], iota, phi0])
        tau = jnp.exp(th[j + 2])
        g3 = jnp.stack([th[j], jnp.exp(th[j + 1]) / tau, tau])
        return gb8, g3


def make_log_prior(coords: Coords, bounds: dict):
    """Flat prior inside a box given as {name: (lo, hi)}; -inf outside."""
    lo = jnp.array([bounds[n][0] for n in coords.names])
    hi = jnp.array([bounds[n][1] for n in coords.names])
    i_ag = coords.names.index("log_Ag")
    i_tau = coords.names.index("log_tau")
    ldv = (float(np.log(1e-16)), float(np.log(1e-7)))   # LPF support on Deltav

    @jax.jit
    def log_prior(th):
        ok = jnp.all((th >= lo) & (th <= hi))
        dv = th[i_ag] - th[i_tau]
        ok = ok & (dv >= ldv[0]) & (dv <= ldv[1])
        return jnp.where(ok, 0.0, -jnp.inf)

    return log_prior


# ---------------------------------------------------------------------------
# hybrid binned likelihood
# ---------------------------------------------------------------------------

def build_hybrid(grid, data_fd, psd, k_ref, model, n_gb, coords,
                 buf=64, relw=0.005, df_max=5e-6):
    """Hybrid likelihood: binned coarse grid (glitch only) + fine GB window."""
    freq, n_fine, df = grid["freq"], grid["n_fine"], grid["df"]

    k_lo = int(max(1, k_ref - buf))
    n_win = n_gb + 2 * buf
    k_hi = k_lo + n_win
    freq_win = freq[k_lo:k_hi]
    data_win, psd_win = data_fd[k_lo:k_hi], psd[k_lo:k_hi]

    w_max = max(1, int(round(df_max / df)))
    starts, k = [], 1
    while k < n_fine:
        starts.append(k)
        k += min(max(int(relw * k), 1), w_max)
    starts = jnp.asarray(starts)
    nb_full = len(starts)
    seg = jnp.searchsorted(starts, jnp.arange(n_fine), side="right") - 1
    seg = jnp.where(seg < 0, nb_full, seg)

    kk = jnp.arange(n_fine)
    in_win = (kk >= k_lo) & (kk < k_hi)
    inv_S = jnp.where(in_win[:, None], 0.0, 1.0 / psd)
    ssum = lambda v: jax.ops.segment_sum(v, seg, num_segments=nb_full)
    W, D = ssum(inv_S), ssum(data_fd * inv_S)
    X = ssum(jnp.abs(data_fd) ** 2 * inv_S)
    wf = jnp.sum(inv_S, axis=1)
    wsum = ssum(wf)
    fbar = ssum(wf * freq) / jnp.where(wsum > 0, wsum, 1.0)
    keep = wsum > 0
    W, D, X, fbar = W[keep], D[keep], X[keep], fbar[keep]
    Xsum = jnp.sum(X)
    zero32 = jnp.zeros((), jnp.int32)

    @jax.jit
    def log_lik(th):
        gb8, g3 = coords.to_physical(th)
        hb = glitch_fd(g3, fbar)
        L_c = -(Xsum - 2.0 * jnp.sum(jnp.real(jnp.conj(hb) * D))
                + jnp.sum(jnp.abs(hb) ** 2 * W))
        seg_gb, k_gb = gb_segment(model, n_gb, gb8)
        off = (k_gb - k_lo).astype(jnp.int32)
        hw = jax.lax.dynamic_update_slice(
            jnp.zeros((n_win, 3), jnp.complex128), seg_gb, (off, zero32))
        r = data_win - hw - glitch_fd(g3, freq_win)
        return L_c - jnp.sum((r.real ** 2 + r.imag ** 2) / psd_win)

    return log_lik, dict(k_lo=k_lo, n_win=n_win, n_blocks=int(keep.sum()))


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

def run_chain(log_lik, log_prior, x0, sigma, dim, seed,
              nwalkers=16, nburn=2000, nsamp=10000, ball=0.01):
    from jexplore.sampler import JaxSampler, Steps
    from jexplore.sampling import EpochMH, SamplingMH
    from jexplore.steps import Stretch
    from jexplore.backends import DefaultBackend

    sampling = SamplingMH(nwalker=nwalkers, temps=jnp.array([1.0]),
                          loglik=log_lik, logprior=log_prior, dim=dim)
    steps = Steps([{Stretch(permute=True).builder: 1.0}])
    p0 = x0 + jr.multivariate_normal(jr.PRNGKey(seed), jnp.zeros(dim),
                                     jnp.diag(sigma ** 2) * ball, shape=(nwalkers,))
    backend = DefaultBackend(burn=nburn, inmem_epochs=1)
    JaxSampler(sampling, steps, backend).run(EpochMH({"p": p0}),
                                             niters=nburn + nsamp, nepoch=1, seed=seed)
    s = backend.get_samples()["p"]
    return jnp.array(s.transpose(0, 2, 1).reshape(-1, dim))


def laplace(log_post, x0, fallback):
    """One Newton step plus the Laplace width, with a guard for stiff directions."""
    g = jax.grad(log_post)(x0)
    H = jax.hessian(log_post)(x0)
    xmap = x0 - jnp.linalg.solve(H, g)
    sig = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(H))))
    ok = jnp.isfinite(sig) & (sig > 1e-12) & (sig < 1e3)
    return jnp.where(jnp.isfinite(xmap), xmap, x0), jnp.where(ok, sig, fallback)
