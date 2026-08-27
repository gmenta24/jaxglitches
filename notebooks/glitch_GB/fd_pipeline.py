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

import math
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

def block_df_max(dt0, tau_max, eps_phi=0.05):
    """Widest block for which the glitch phase drifts by less than `eps_phi` rad.

    The glitch phase winds at  d arg h / df = -2 pi (t0 + 2 tau), so binning is
    limited by how late the glitch arrives -- not by anything physical, since
    |h(f)|, the SNR and the exact fine-grid likelihood are all independent of t0.
    Heterodyning the data statistic at `t_ref` (see `build_hybrid`) replaces t0 by
    t0 - t_ref, and the bound becomes

        df_max = eps_phi / (2 pi (|t0 - t_ref| + 2 tau_max)) .

    Pass `dt0 = t0` and leave `t_ref = 0` to recover the un-heterodyned bound.

    `tau_max` is the largest decay time the sampler will visit, not the one at the
    maximum: the heterodyne removes only the part of the phase that is linear in f,
    so 2 tau survives it and sets a floor on how coarsely one may bin.
    """
    return float(eps_phi) / (2.0 * math.pi * (abs(float(dt0)) + 2.0 * float(tau_max)))


def gen2(freqs):
    """TDI-1 -> TDI-2 transfer, h^(2) = (1 - D^4) h^(1), for a signal at `freqs`."""
    return 1.0 - jnp.exp(-4j * T_ARM * 2.0 * jnp.pi * freqs)


def build_hybrid(grid, data_fd, psd, k_ref, model, n_gb, coords,
                 buf=64, relw=0.005, df_max=5e-6, t_ref=0.0, tdi=1):
    """Hybrid likelihood: binned coarse grid (glitch only) + fine GB window.

    `tdi` selects the generation of the *templates*; `data_fd` and `psd` must be
    supplied in the same generation. TDI-2 multiplies both signals by `gen2`, which
    cancels against the matching factor in the PSD, so the two generations give the
    same likelihood up to the binning -- that is the null test of Sec. "TDI-1 against
    TDI-2".

    `t_ref` heterodynes the block statistic, D_b -> sum_k d_k e^{+2 pi i f_k t_ref}/S_k,
    and multiplies the model by the same factor, so that what has to be smooth across
    a block is h(f) e^{+2 pi i f t_ref} rather than h(f) itself. It is algebraically a
    no-op -- and numerically exact at t_ref = 0 -- but it lets `df_max` be set by how
    well the onset is already localised instead of by how late it is. Use
    `block_df_max` for the matching width. `t_ref` is baked into the precomputed
    statistic, so it has to be fixed before sampling: take it from the MAP.
    """
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
    het = jnp.exp(2j * jnp.pi * freq * t_ref)[:, None] if t_ref else 1.0
    W, D = ssum(inv_S), ssum(data_fd * het * inv_S)
    X = ssum(jnp.abs(data_fd) ** 2 * inv_S)
    wf = jnp.sum(inv_S, axis=1)
    wsum = ssum(wf)
    fbar = ssum(wf * freq) / jnp.where(wsum > 0, wsum, 1.0)
    keep = wsum > 0
    W, D, X, fbar = W[keep], D[keep], X[keep], fbar[keep]
    Xsum = jnp.sum(X)
    zero32 = jnp.zeros((), jnp.int32)
    if tdi not in (1, 2):
        raise ValueError(f"tdi must be 1 or 2, got {tdi!r}")
    g2_coarse = gen2(fbar)[:, None] if tdi == 2 else 1.0
    g2_win = gen2(freq_win)[:, None] if tdi == 2 else 1.0

    @jax.jit
    def log_lik(th):
        gb8, g3 = coords.to_physical(th)
        hb = glitch_fd(g3, fbar) * g2_coarse
        if t_ref:
            hb = hb * jnp.exp(2j * jnp.pi * fbar * t_ref)[:, None]
        L_c = -(Xsum - 2.0 * jnp.sum(jnp.real(jnp.conj(hb) * D))
                + jnp.sum(jnp.abs(hb) ** 2 * W))
        seg_gb, k_gb = gb_segment(model, n_gb, gb8)
        off = (k_gb - k_lo).astype(jnp.int32)
        if tdi == 2:
            # the binary lives at a dynamic slice of the window, so the transfer has
            # to follow it rather than being applied to the fixed window grid
            seg_gb = seg_gb * gen2(jax.lax.dynamic_slice(freq_win, (off,), (n_gb,)))[:, None]
        hw = jax.lax.dynamic_update_slice(
            jnp.zeros((n_win, 3), jnp.complex128), seg_gb, (off, zero32))
        r = data_win - hw - glitch_fd(g3, freq_win) * g2_win
        return L_c - jnp.sum((r.real ** 2 + r.imag ** 2) / psd_win)

    return log_lik, dict(k_lo=k_lo, n_win=n_win, n_blocks=int(keep.sum()),
                         df_max=df_max, t_ref=float(t_ref), tdi=tdi)


def build_decimated(grid, data_fd, psd, K, model, n_gb, coords, tdi=1, k_first=1):
    """The tempting cheap alternative to binning: keep every K-th bin, scale by K.

        log L_K(theta) = -K sum_{k in S_K} |d_k - h_k|^2 / S_k

    This exists so that the argument of Sec. "Why binning and not decimation" can be
    measured rather than asserted. Both give the same Fisher information -- the
    prefactor K compensates the 1/K retained terms -- so the posterior comes out the
    right width. What decimation breaks is the noise: the score picks up K twice in its
    variance and once in the information, so the maximum scatters about the truth by
    sqrt(K) posterior widths instead of one. Run through the P--P machinery it produces
    credible intervals that look healthy and fail to cover.

    The retained set is k = k_first, k_first + K, ... The DC bin is excluded, as
    everywhere else.

    The Galactic binary is the only awkward part. Its model returns `n_gb` *contiguous*
    bins starting at a k_min that moves with f0, so the retained bins that land on it
    are a contiguous run in the decimated index whose offset is dynamic. Rather than
    scattering into a full-length array (which would cost exactly the O(N) this is
    supposed to avoid), the segment is gathered at the local offsets
    `k_first + K*m - k_min` and dynamically placed, which is O(n_gb / K).
    """
    freq, n_fine = grid["freq"], grid["n_fine"]
    idx = jnp.arange(k_first, n_fine, K)
    n_dec = int(idx.shape[0])
    freq_d, data_d, psd_d = freq[idx], data_fd[idx], psd[idx]
    zero32 = jnp.zeros((), jnp.int32)

    # widest run of retained bins the GB segment can cover, plus one for the offset
    n_win = n_gb // K + 2
    j = jnp.arange(n_win)

    if tdi not in (1, 2):
        raise ValueError(f"tdi must be 1 or 2, got {tdi!r}")
    g2_d = gen2(freq_d)[:, None] if tdi == 2 else 1.0

    @jax.jit
    def log_lik(th):
        gb8, g3 = coords.to_physical(th)
        seg, k_gb = gb_segment(model, n_gb, gb8)                  # (n_gb, 3), scalar
        # first retained index at or beyond k_gb, and the local offsets it implies
        m0 = jnp.ceil((k_gb - k_first) / K).astype(jnp.int32)
        local = k_first + K * (m0 + j) - k_gb                     # (n_win,)
        ok = (local >= 0) & (local < n_gb)
        vals = jnp.take(seg, jnp.clip(local, 0, n_gb - 1), axis=0) * ok[:, None]
        h = jax.lax.dynamic_update_slice(
            jnp.zeros((n_dec, 3), jnp.complex128), vals,
            (jnp.clip(m0, 0, n_dec - n_win), zero32))
        h = h * g2_d + glitch_fd(g3, freq_d) * g2_d
        r = data_d - h
        return -K * jnp.sum((r.real ** 2 + r.imag ** 2) / psd_d)

    return log_lik, dict(K=K, n_dec=n_dec, n_fine=n_fine, tdi=tdi, k_first=k_first)


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

def _pull_inside(p0, x0, log_prior, n_halve=40):
    """Guarantee every walker starts inside the prior support.

    A stretch move between two walkers that both sit outside the support has
    acceptance ratio -inf - (-inf) = NaN, which compares false and is rejected for
    ever. An ensemble that starts with most of its walkers outside can therefore
    freeze completely, and it does so silently: the chain has the right shape and
    every sample is the walker's starting point. It happened for a handful of the
    P--P realisations of Sec. "Are the credible intervals credible?", the ones where
    the glitch is too faint to curve the posterior, so that the Laplace width comes
    back wider than the prior box and a ball of `ball` times it straddles the walls.

    Offending walkers are pulled back along their own offset from `x0` rather than
    clipped to the wall, which would pile them up on a face of the box and destroy
    the ensemble's spread in that direction. Walkers that are already valid are left
    bit-identical, so this is a no-op for every run that was not broken.
    """
    lp = np.asarray(jax.vmap(log_prior)(p0))
    bad = ~np.isfinite(lp)
    if not bad.any():
        return p0
    if not np.isfinite(float(log_prior(x0))):
        raise ValueError("starting point is outside the prior support")
    p = np.array(p0, dtype=float)
    x = np.asarray(x0, dtype=float)
    for w in np.flatnonzero(bad):
        d = p[w] - x
        for _ in range(n_halve):
            d = 0.5 * d
            if np.isfinite(float(log_prior(jnp.asarray(x + d)))):
                break
        else:
            d = np.zeros_like(d)
        p[w] = x + d
    print(f"    run_chain: pulled {int(bad.sum())}/{len(p)} walkers back inside the "
          f"prior support")
    return jnp.asarray(p)


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
    p0 = _pull_inside(p0, x0, log_prior)
    backend = DefaultBackend(burn=nburn, inmem_epochs=1)
    JaxSampler(sampling, steps, backend).run(EpochMH({"p": p0}),
                                             niters=nburn + nsamp, nepoch=1, seed=seed)
    s = backend.get_samples()["p"]
    return jnp.array(s.transpose(0, 2, 1).reshape(-1, dim))


def laplace(log_post, x0, fallback, n_steps=8, tol=1e-3, return_steps=False):
    """Newton to the maximum, plus the Laplace width there.

    One step is *not* enough on this posterior, which is why this iterates. The
    (log f0, log fdot) block is ill-conditioned -- cond(H) ~ 1e17 at the fiducial
    injection -- so an undamped step overshoots: from the injected values it lands
    at a log-posterior some 30 below where it started, and only recovers over the
    following steps. Each step is therefore backtracked until it does not decrease
    the posterior, and the iteration stops when the accepted step is below `tol`
    times the Laplace width. See `run_convergence.py`, which measures all of this.

    Note that the covariance returned is the curvature at the *maximum*, which is
    not the same thing as the Fisher matrix at the injected truth: the paper quotes
    the latter when it compares a forecast against a chain, and the two differ by
    more than they sound like they should along the curved degeneracy.
    """
    grad = jax.jit(jax.grad(log_post))
    hess = jax.jit(jax.hessian(log_post))
    x, used = x0, 0
    for k in range(n_steps):
        g, H = grad(x), hess(x)
        step = -jnp.linalg.solve(H, g)
        if not bool(jnp.all(jnp.isfinite(step))):
            break
        sig = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(H))))
        f0, t = float(log_post(x)), 1.0
        for _ in range(30):
            if float(log_post(x + t * step)) >= f0:
                break
            t *= 0.5
        x, used = x + t * step, k + 1
        if float(jnp.max(jnp.abs(t * step / sig))) < tol:
            break
    H = hess(x)
    sig = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(H))))
    ok = jnp.isfinite(sig) & (sig > 1e-12) & (sig < 1e3)
    out = jnp.where(jnp.isfinite(x), x, x0), jnp.where(ok, sig, fallback)
    return out + (used,) if return_steps else out
