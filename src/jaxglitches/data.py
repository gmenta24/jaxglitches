"""
Clean (noise-free) single-exponential glitch signal in the frequency and time domains.

Convention
----------
    h_fd[k, c]  =  H_c(f_k)        (frequency domain)
    h_td[n, c]  =  h_c(t_n)        (time domain)

where H_c(f) is the analytical Fourier transform of TDI channel c evaluated
at discrete frequency f_k (the DC bin is set to zero), and h_c(t) is the
corresponding time-domain TDI channel evaluated at time t_n.

Public API
----------
freq_grid(t_obs, dt)                                 -> (F,) real array
clean_signal_f(params, freq, T, tdi)                 -> (F, 3) complex array
clean_signal_t(params, t, T, tdi)                    -> (N, 3) real array
"""
import jax
import jax.numpy as jnp
from functools import partial

from .waveform import (
    tdi1_1exp_f_glitch, tdi2_1exp_f_glitch,
    tdi1_1exp_glitch, tdi2_1exp_glitch,
    AET, T_ARM_s,
)

DT_s = 0.25          # time step → Nyquist at 2 Hz
T_OBS_s = 3600.0     # 1 hour observation window

jax.config.update("jax_enable_x64", True)


def freq_grid(t_obs: float = T_OBS_s, dt: float = DT_s):
    """Return the one-sided rfft frequency grid for an observation of length t_obs."""
    n = int(t_obs / dt)
    return jnp.fft.rfftfreq(n, dt)


@partial(jax.jit, static_argnames=("tdi",))
def clean_signal_f(params, freq, T: float = T_ARM_s, tdi: int = 1):
    """
    Frequency-domain TDI A/E/T signal for a single-exponential glitch.

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
             t0     : glitch onset time (s).
             Deltav : velocity-kick amplitude (m/s).
             tau    : exponential decay timescale (s).
    freq   : (F,) frequency array (Hz), e.g. from freq_grid().
    T      : LISA one-way light travel time along one arm (s).
    tdi    : TDI generation — 1 or 2.

    Returns
    -------
    h_fd : complex (F, 3), columns [A, E, T_ch].
           DC bin is zero.
    """
    t0     = params[0]
    Deltav = params[1]
    tau    = params[2]

    # Replace f=0 to avoid 1/f singularity; DC bin is zeroed afterwards.
    f_safe = jnp.where(freq > 0, freq, 1.0)

    if tdi == 1:
        X, Y, Z = tdi1_1exp_f_glitch(f_safe, t0, Deltav, tau, T)
    else:
        X, Y, Z = tdi2_1exp_f_glitch(f_safe, t0, Deltav, tau, T)

    A_ch, E_ch, T_ch = AET(X, Y, Z)
    h_fd = jnp.stack([A_ch, E_ch, T_ch], axis=-1)   # (F, 3)
    return h_fd.at[0].set(0.0 + 0.0j)


@partial(jax.jit, static_argnames=("tdi",))
def clean_signal_t(params, t, T: float = T_ARM_s, tdi: int = 1):
    """
    Time-domain TDI A/E/T signal for a single-exponential glitch.

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
             t0     : glitch onset time (s).
             Deltav : velocity-kick amplitude (m/s).
             tau    : exponential decay timescale (s).
    t      : (N,) time array (s).
    T      : LISA one-way light travel time along one arm (s).
    tdi    : TDI generation — 1 or 2.

    Returns
    -------
    h_td : real (N, 3), columns [A, E, T_ch].
    """
    t0     = params[0]
    Deltav = params[1]
    tau    = params[2]

    if tdi == 1:
        X, Y, Z = tdi1_1exp_glitch(t, t0, Deltav, tau, T)
    else:
        X, Y, Z = tdi2_1exp_glitch(t, t0, Deltav, tau, T)

    A_ch, E_ch, T_ch = AET(X, Y, Z)
    return jnp.stack([A_ch, E_ch, T_ch], axis=-1)   # (N, 3)
