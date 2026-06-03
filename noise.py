"""
LISA noise PSDs and frequency-domain noise sampling — outside jaxglitches package.

TDI-1 vs TDI-2 noise
---------------------
The TDI-1 and TDI-2 X channels carry the same physical signal but with
different transfer functions (see waveform.py):

    h_X^{TDI1}(f) ~ TF1(f) * Δν(f),   TF1 = exp(-4iTf·2π) - 1
    h_X^{TDI2}(f) ~ TF1(f)² * Δν(f)

The noise in TDI-2 is consequently amplified by |TF1|²:

    S_TDI2(f) = |TF1(f)|² · S_TDI1(f)
              = 4 sin²(4πfT) · S_TDI1(f)

where T = L/c is the one-way arm travel time.  With this correction the
optimal SNR is identical in both TDI channels (as it should be physically).

Convention
----------
All PSD functions return the *bin variance* S_c(f_k) defined so that a
frequency-domain noise draw n_fd satisfies E[|n_fd[k,c]|²] = S_c(f_k).
This matches the convention in jaxglitches/likelihood.py.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Default LISA noise parameters (LISA red-book / SciRD values)
_ARM_m  = 2.5e9      # arm length (m)
_C_SI   = 299792458.0
_A_DEF  = 3.0        # test-mass acceleration noise (pm/s²/√Hz)
_P_DEF  = 15.0       # OMS optical path noise (pm/√Hz)


# ---------------------------------------------------------------------------
# TDI-1 base PSD (A, E, T channels)
# ---------------------------------------------------------------------------

def psd_tdi1(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """
    TDI-1 noise PSD for A, E, T channels.

    The formulas follow the standard equal-armlength LISA sensitivity model
    (e.g. Larson et al. 2000, Barack & Cutler 2004) for the AET combination.

    Parameters
    ----------
    f  : array of positive frequencies (Hz).
    A  : test-mass acceleration noise (pm/s²/√Hz).
    P  : OMS noise (pm/√Hz).
    L  : arm length (m).

    Returns
    -------
    S_A, S_E, S_T : arrays of shape (len(f),).
    """
    T = L / _C_SI
    x = 2.0 * jnp.pi * f * T        # dimensionless phase per one-way trip

    S_acc = (
        (A / L) ** 2 * 1e-30
        * (1.0 + (4e-4 / f) ** 2)
        * (1.0 + (f / 8e-3) ** 4)
        / (2.0 * jnp.pi * f) ** 4
    )
    S_oms = (P / L) ** 2 * 1e-24 * (1.0 + (2e-3 / f) ** 4)

    cx = jnp.cos(x)
    S_A = 0.5 * (2.0 + cx) * S_oms + 2.0 * (1.0 + cx + cx ** 2) * S_acc
    S_E = S_A
    S_T = (1.0 - cx) * S_oms + 2.0 * (1.0 - cx) ** 2 * S_acc
    return S_A, S_E, S_T


def psd_tdi1_array(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """Return TDI-1 PSD as an (F, 3) array — columns [S_A, S_E, S_T]."""
    S_A, S_E, S_T = psd_tdi1(f, A=A, P=P, L=L)
    return jnp.stack([S_A, S_E, S_T], axis=-1)


# ---------------------------------------------------------------------------
# TDI-2 PSD  (= TDI-1 × |TF1|²)
# ---------------------------------------------------------------------------

def psd_tdi2(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """
    TDI-2 noise PSD for A, E, T channels.

    S_TDI2(f) = 4 sin²(4πfT) · S_TDI1(f)

    The factor 4 sin²(4πfT) = |TF1(f)|² is the squared magnitude of the
    TDI-1 transfer function.  With this PSD the optimal SNR is the same as
    in TDI-1.

    Returns
    -------
    S_A, S_E, S_T : arrays of shape (len(f),).
    """
    T = L / _C_SI
    S_A1, S_E1, S_T1 = psd_tdi1(f, A=A, P=P, L=L)
    # |TF1|^2 = |exp(-4i*T*2πf) - 1|^2 = 4 sin^2(4πfT)
    factor = 4.0 * jnp.sin(4.0 * jnp.pi * f * T) ** 2
    return factor * S_A1, factor * S_E1, factor * S_T1


def psd_tdi2_array(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """Return TDI-2 PSD as an (F, 3) array — columns [S_A, S_E, S_T]."""
    S_A, S_E, S_T = psd_tdi2(f, A=A, P=P, L=L)
    return jnp.stack([S_A, S_E, S_T], axis=-1)


# ---------------------------------------------------------------------------
# Frequency-domain noise sampling
# ---------------------------------------------------------------------------

@jax.jit
def sample_noise_fd(key, psd_fd):
    """
    Draw frequency-domain Gaussian noise matching the supplied PSD.

    Each positive-frequency bin k satisfies  E[|n[k,c]|²] = psd_fd[k,c].
    The DC bin (index 0) is set to zero.

    Parameters
    ----------
    key    : JAX PRNG key.
    psd_fd : real (F, C) array — noise PSD, already evaluated at the
             desired frequency grid.  Typically psd_tdi1_array(f_safe) or
             psd_tdi2_array(f_safe) where f_safe = jnp.where(f>0, f, 1.).

    Returns
    -------
    n_fd : complex (F, C) array.
    """
    shape = psd_fd.shape
    k_r, k_i = jax.random.split(key)
    z_r = jax.random.normal(k_r, shape)
    z_i = jax.random.normal(k_i, shape)
    n_fd = jnp.sqrt(psd_fd / 2.0) * (z_r + 1j * z_i)
    # zero DC bin to avoid issues at f=0
    return n_fd.at[0].set(0.0 + 0.0j)
