"""
Clean (noise-free) single-exponential glitch signal in the frequency domain.

Convention
----------
    h_fd[k, c]  =  H_c(f_k)

where H_c(f) is the analytical Fourier transform of TDI channel c evaluated
at discrete frequency f_k.  The DC bin is set to zero.

Public API
----------
freq_grid(t_obs, dt)                                 -> (F,) real array
clean_signal(params, freq, T, tdi)                   -> (F, 3) complex array
"""
import jax
import jax.numpy as jnp
from functools import partial

from .constants import T_ARM_s, DT_s, T_OBS_s
from .waveform import tdi1_1exp_f_glitch, tdi2_1exp_f_glitch, AET

jax.config.update("jax_enable_x64", True)


def freq_grid(t_obs: float = T_OBS_s, dt: float = DT_s):
    """Return the one-sided rfft frequency grid for an observation of length t_obs."""
    n = int(t_obs / dt)
    return jnp.fft.rfftfreq(n, dt)


@partial(jax.jit, static_argnames=("tdi",))
def clean_signal(params, freq, T: float = T_ARM_s, tdi: int = 1):
    """
    Frequency-domain TDI A/E/T signal for a single-exponential glitch.

    Parameters
    ----------
    params : (3,) array  [tau (s), Deltav (m/s), beta (s)].
             tau    : glitch onset time (s).
             Deltav : velocity-kick amplitude (m/s).
             beta   : exponential decay timescale (s).
    freq   : (F,) frequency array (Hz), e.g. from freq_grid().
    T      : LISA one-way light travel time along one arm (s).
    tdi    : TDI generation — 1 or 2.

    Returns
    -------
    h_fd : complex (F, 3), columns [A, E, T_ch].
           DC bin is zero.
    """
    tau    = params[0]
    Deltav = params[1]
    beta   = params[2]

    # Replace f=0 to avoid 1/f singularity; DC bin is zeroed afterwards.
    f_safe = jnp.where(freq > 0, freq, 1.0)

    # The single-exponential waveform signature is (freq, t0, Deltav, tau_decay, T);
    # here tau plays the onset t0 and beta plays the decay time tau_decay.
    if tdi == 1:
        X, Y, Z = tdi1_1exp_f_glitch(f_safe, tau, Deltav, beta, T)
    else:
        X, Y, Z = tdi2_1exp_f_glitch(f_safe, tau, Deltav, beta, T)

    A_ch, E_ch, T_ch = AET(X, Y, Z)
    h_fd = jnp.stack([A_ch, E_ch, T_ch], axis=-1)   # (F, 3)
    return h_fd.at[0].set(0.0 + 0.0j)
