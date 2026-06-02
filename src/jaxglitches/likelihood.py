"""
Frequency-domain likelihood for a single shapelet glitch in stationary noise.

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
"""
import jax
import jax.numpy as jnp
from functools import partial

from .constants import T_ARM_s
from .data import clean_signal

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
    residual = data_fd - h_fd
    return -jnp.sum(jnp.abs(residual[1:]) ** 2 / psd_fd[1:])


def make_log_likelihood(data_fd, psd_fd, freq, T: float = T_ARM_s, tdi: int = 1):
    """
    Build a JIT-compiled  log_likelihood(params) -> scalar  closure.

    Parameters
    ----------
    data_fd : complex (F, 3) fixed data array.
    psd_fd  : real   (F, 3) noise PSD — use the correct PSD for this TDI generation.
    freq    : (F,) frequency array (Hz).
    T       : LISA arm light travel time (s).
    tdi     : TDI generation (1 or 2) — static arg for clean_signal.

    Returns
    -------
    log_L : callable, log_L(params) -> scalar.
    """
    @partial(jax.jit, static_argnames=())
    def log_L(params):
        h_fd = clean_signal(params, freq, T=T, tdi=tdi)
        return log_likelihood(data_fd, h_fd, psd_fd)

    return log_L


def log_posterior(data_fd, h_fd, psd_fd, params, t_obs: float):
    """
    Un-normalised log posterior = log_likelihood + log_prior.

    Parameters
    ----------
    data_fd : complex (F, 3).
    h_fd    : complex (F, 3) — template at params.
    psd_fd  : real   (F, 3).
    params  : (3,) array [tau, Deltav, beta].
    t_obs   : observation window (s) — needed by the prior.
    """
    from .priors import log_prior
    return log_likelihood(data_fd, h_fd, psd_fd) + log_prior(params, t_obs)
