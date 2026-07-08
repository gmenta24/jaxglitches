"""
Frequency-domain likelihood for a single single-exponential glitch in stationary noise.

Noise model
-----------
    n_fd[k, c]  ~  CN(0, S_c(f_k))        for each positive-frequency bin k
    E[|n_fd[k,c]|^2] = S_c(f_k)

where S_c is the noise PSD supplied by the caller (not computed here).

Log-likelihood
--------------
    log p(d | h) = -sum_{k>0, c}  |d[k,c] - h[k,c]|^2 / S_c(f_k)  + const

Matched-filter SNR
------------------
    SNR  = sqrt( (h|h) )
    (a|b) = 2 * Re[ sum_{k>0, c} conj(a[k,c]) b[k,c] / S_c[k] ]

Public API
----------
inner_product(a_fd, b_fd, psd_fd)      -> scalar
snr(h_fd, psd_fd)                      -> scalar
log_likelihood(data_fd, h_fd, psd_fd)  -> scalar
make_log_likelihood(data_fd, psd_fd, freq, T, tdi) -> callable(params)
make_log_likelihood_unequal(data_fd, psd_fd, freq, ltt, tdi) -> callable(params)
fisher_matrix(params, freq, psd_fd, T, tdi)          -> (n, n) array
fisher_matrix_unequal(params, freq, psd_fd, ltt, tdi) -> (n, n) array
"""
import jax
import jax.numpy as jnp
from functools import partial

from .waveform import T_ARM_s
from .data import clean_signal_f, clean_signal_f_unequal

jax.config.update("jax_enable_x64", True)


def inner_product(a_fd, b_fd, psd_fd):
    """
    Noise-weighted inner product  (a|b).

    (a|b) = 2 * Re[ sum_{k>0, c} conj(a[k,c]) b[k,c] / S_c[k] ]

    The DC bin (index 0) is always excluded.

    Parameters
    ----------
    a_fd, b_fd : complex (F, 3) or (F,) arrays.
    psd_fd     : real array of the same shape — S_c(f_k) per bin.

    Returns
    -------
    Scalar float.
    """
    return 2.0 * jnp.real(jnp.sum(jnp.conj(a_fd[1:]) * b_fd[1:] / psd_fd[1:]))


def snr(h_fd, psd_fd):
    """Optimal matched-filter SNR = sqrt( (h|h) )."""
    return jnp.sqrt(inner_product(h_fd, h_fd, psd_fd))


@jax.jit
def log_likelihood(data_fd, h_fd, psd_fd):
    """
    Log-likelihood  log p(data | h).

    Parameters
    ----------
    data_fd : complex (F, 3) observed data.
    h_fd    : complex (F, 3) template.
    psd_fd  : real   (F, 3) noise PSD for the TDI channels.

    Returns
    -------
    Scalar float (up to an additive constant independent of h).
    """
    # real**2 + imag**2 rather than jnp.abs(.)**2: the complex abs primitive
    # has an undefined derivative at 0, which makes jax.hessian silently
    # wrong (not NaN) wherever the residual vanishes — e.g. noiseless
    # injections evaluated at the true parameters.
    residual = (data_fd - h_fd)[1:]
    return -jnp.sum((residual.real ** 2 + residual.imag ** 2) / psd_fd[1:])


def make_log_likelihood(data_fd, psd_fd, freq, T: float = T_ARM_s, tdi: int = 1):
    """
    Build a JIT-compiled  log_likelihood(params) -> scalar  closure.

    Parameters
    ----------
    data_fd : complex (F, 3) fixed data array.
    psd_fd  : real   (F, 3) noise PSD — use the correct PSD for this TDI generation.
    freq    : (F,) frequency array (Hz).
    T       : LISA arm light travel time (s).
    tdi     : TDI generation (1 or 2) — static arg for clean_signal_f.

    Returns
    -------
    log_L : callable, log_L(params) -> scalar.
    """
    @partial(jax.jit, static_argnames=())
    def log_L(params):
        h_fd = clean_signal_f(params, freq, T=T, tdi=tdi)
        return log_likelihood(data_fd, h_fd, psd_fd)

    return log_L


def make_log_likelihood_unequal(data_fd, psd_fd, freq, ltt, tdi: int = 1):
    """
    Build a JIT-compiled  log_likelihood(params) -> scalar  closure for the
    unequal-arm (numerical-orbits) template.

    The six light travel times are frozen at the glitch epoch — pass
    ltt = orbits.link_ltt(t_glitch).  Since the arms drift by ~3e-8 s per
    second, one evaluation per glitch (e.g. at the detection time) is enough
    for any t0 within the analysis window.

    Parameters
    ----------
    data_fd : complex (F, 3) fixed data array.
    psd_fd  : real   (F, 3) noise PSD — use the correct PSD for this TDI generation.
    freq    : (F,) frequency array (Hz).
    ltt     : (6,) light travel times (s), lisaorbits order
              [L_12, L_23, L_31, L_13, L_32, L_21].
    tdi     : TDI generation (1 or 2) — static arg for clean_signal_f_unequal.

    Returns
    -------
    log_L : callable, log_L(params) -> scalar.
    """
    ltt = jnp.asarray(ltt)

    @partial(jax.jit, static_argnames=())
    def log_L(params):
        h_fd = clean_signal_f_unequal(params, freq, ltt, tdi=tdi)
        return log_likelihood(data_fd, h_fd, psd_fd)

    return log_L


def fisher_matrix(params, freq, psd_fd, T: float = T_ARM_s, tdi: int = 1):
    """
    Fisher information matrix in physical parameters [t0, Deltav, tau].

    Gamma_ij = (dh/dtheta_i | dh/dtheta_j)
             = 2 Re[ sum_{k>0,c} conj(dh[k,c,i]) dh[k,c,j] / S_c[k] ]

    Parameters
    ----------
    params : (n,) array [t0, Deltav, tau].
    freq   : (F,) frequency grid (Hz).
    psd_fd : (F, 3) noise PSD (must match the tdi generation).
    T      : one-way arm travel time (s).
    tdi    : TDI generation (1 or 2).

    Returns
    -------
    Gamma : (n, n) real symmetric positive-definite Fisher matrix.
    """
    F_len = freq.shape[0]
    n_p   = params.shape[0]

    def h_split(p):
        h = clean_signal_f(p, freq, T=T, tdi=tdi)  # (F, 3) complex
        return jnp.concatenate([h.real.ravel(), h.imag.ravel()])  # (2*F*3,) real

    J  = jax.jacobian(h_split)(params)    # (2*F*3, n_p) real
    dh = (J[:F_len * 3].reshape(F_len, 3, n_p)
          + 1j * J[F_len * 3:].reshape(F_len, 3, n_p))  # (F, 3, n_p) complex

    dh_pos = dh[1:]   # skip DC bin  (F-1, 3, n_p)
    return 2.0 * jnp.real(
        jnp.einsum('kci,kcj->ij', jnp.conj(dh_pos), dh_pos / psd_fd[1:, :, None])
    )


def fisher_matrix_unequal(params, freq, psd_fd, ltt, tdi: int = 1):
    """
    Fisher information matrix for the unequal-arm (numerical-orbits) template.

    Same as fisher_matrix, with the single arm time T replaced by the six
    per-link light travel times frozen at the glitch epoch.

    Parameters
    ----------
    params : (n,) array [t0, Deltav, tau].
    freq   : (F,) frequency grid (Hz).
    psd_fd : (F, 3) noise PSD (must match the tdi generation).
    ltt    : (6,) light travel times (s), lisaorbits order.
    tdi    : TDI generation (1 or 2).

    Returns
    -------
    Gamma : (n, n) real symmetric positive-definite Fisher matrix.
    """
    ltt   = jnp.asarray(ltt)
    F_len = freq.shape[0]
    n_p   = params.shape[0]

    def h_split(p):
        h = clean_signal_f_unequal(p, freq, ltt, tdi=tdi)  # (F, 3) complex
        return jnp.concatenate([h.real.ravel(), h.imag.ravel()])  # (2*F*3,) real

    J  = jax.jacobian(h_split)(params)    # (2*F*3, n_p) real
    dh = (J[:F_len * 3].reshape(F_len, 3, n_p)
          + 1j * J[F_len * 3:].reshape(F_len, 3, n_p))  # (F, 3, n_p) complex

    dh_pos = dh[1:]   # skip DC bin  (F-1, 3, n_p)
    return 2.0 * jnp.real(
        jnp.einsum('kci,kcj->ij', jnp.conj(dh_pos), dh_pos / psd_fd[1:, :, None])
    )


def log_posterior(data_fd, h_fd, psd_fd, params, t_obs: float):
    """
    Un-normalised log posterior = log_likelihood + log_prior.

    Parameters
    ----------
    data_fd : complex (F, 3).
    h_fd    : complex (F, 3) — template at params.
    psd_fd  : real   (F, 3).
    params  : (3,) array [t0, Deltav, tau].
    t_obs   : observation window (s) — needed by the prior.
    """
    from .priors import log_prior
    return log_likelihood(data_fd, h_fd, psd_fd) + log_prior(params, t_obs)
