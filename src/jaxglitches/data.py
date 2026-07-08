"""
Clean (noise-free) single-exponential glitch signal in the frequency and time domains.

Convention
----------
    h_fd[k, c]  =  H_c(f_k)        (frequency domain)
    h_td[n, c]  =  h_c(t_n)        (time domain)

where H_c(f) is the analytical Fourier transform of TDI channel c evaluated
at discrete frequency f_k (the DC bin is set to zero), and h_c(t) is the
corresponding time-domain TDI channel evaluated at time t_n.  The channel
columns c are [A, E, T] when basis="AET" (default) or [X, Y, Z] when
basis="XYZ".

Raw (pre-TDI) signal
--------------------
The "raw" glitch is the single-link fractional-frequency signal injected on
link/MOSA 12, *before* the TDI Michelson combination is applied:

    raw_t(t) = (Deltav / c) * [ -1 + e^{-(t-t0)/tau} (1 + (t-t0)/tau) ]   (t >= t0)

Its analytic Fourier transform is

    raw_f(f) = 1 / (i w c) * e^{-i w t0} * Deltav / (-i + tau w)^2 ,  w = 2*pi*f .

`compute_TDI` applies the TDI transfer functions to a raw frequency-domain
signal so that, by construction,

    compute_TDI(raw_glitch_f(params, freq), freq, ...) == clean_signal_f(params, freq, ...)

to machine precision.  The transfer functions are the frequency-domain delay
operators D = e^{-i w T} acting on link 12:

    TDI-1:  X = (1 - D^4) raw,                 Y = 2 (D^3 - D) raw,           Z = 0
    TDI-2:  X = (1 - D^4)^2 raw,               Y = 2 (-D + D^3 + D^5 - D^7) raw, Z = 0

Unequal arms (numerical orbits)
-------------------------------
The `*_unequal` variants replace the single arm time T by the six per-link
light travel times L_ij (frozen at the glitch epoch — see orbits.py), in the
lisaorbits ordering  ltt = [L_12, L_23, L_31, L_13, L_32, L_21].
With D_ij = e^{-i w L_ij} the link-12 test-mass glitch response is

    TDI-1:  X = (1 - D_13 D_31)(1 + D_12 D_21) raw
            Y = 2 D_21 (D_23 D_32 - 1) raw
            Z = 0
    TDI-2:  X = (1 - D_12 D_21 D_13 D_31) * X_TDI1
            Y = (1 - D_23 D_32 D_12 D_21) * Y_TDI1
            Z = 0

which reduce exactly to the equal-arm transfer functions when all L_ij = T.

Public API
----------
freq_grid(t_obs, dt)                                 -> (F,) real array
clean_signal_f(params, freq, T, tdi, basis)          -> (F, 3) complex array
clean_signal_t(params, t, T, tdi, basis)             -> (N, 3) real array
raw_glitch_f(params, freq, T)                        -> (F,) complex array
raw_glitch_t(params, t, T)                           -> (N,) real array
compute_TDI(raw_f, freq, T, tdi, basis)              -> (F, 3) complex array
clean_signal_f_unequal(params, freq, ltt, tdi, basis) -> (F, 3) complex array
clean_signal_t_unequal(params, t, ltt, tdi, basis)    -> (N, 3) real array
compute_TDI_unequal(raw_f, freq, ltt, tdi, basis)     -> (F, 3) complex array
"""
import jax
import jax.numpy as jnp
from functools import partial

from .waveform import (
    tdi1_1exp_f_glitch, tdi2_1exp_f_glitch,
    tdi1_1exp_glitch, tdi2_1exp_glitch,
    AET, T_ARM_s, C_SI,
)

DT_s = 0.25          # time step → Nyquist at 2 Hz
T_OBS_s = 3600.0     # 1 hour observation window

jax.config.update("jax_enable_x64", True)


def freq_grid(t_obs: float = T_OBS_s, dt: float = DT_s):
    """Return the one-sided rfft frequency grid for an observation of length t_obs."""
    n = round(t_obs / dt)   # int() would truncate, e.g. int(0.9999...e4) -> 9999
    return jnp.fft.rfftfreq(n, dt)


@partial(jax.jit, static_argnames=("tdi", "basis"))
def clean_signal_f(params, freq, T: float = T_ARM_s, tdi: int = 1, basis: str = "AET"):
    """
    Frequency-domain TDI signal for a single-exponential glitch.

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
             t0     : glitch onset time (s).
             Deltav : velocity-kick amplitude (m/s).
             tau    : exponential decay timescale (s).
    freq   : (F,) frequency array (Hz), e.g. from freq_grid().
    T      : LISA one-way light travel time along one arm (s).
    tdi    : TDI generation — 1 or 2.
    basis  : output channel basis — "AET" (default) or "XYZ".

    Returns
    -------
    h_fd : complex (F, 3), columns [A, E, T] if basis="AET" else [X, Y, Z].
           DC bin is zero.
    """
    t0     = params[0]
    Deltav = params[1]
    tau    = params[2]

    # Replace f=0 to avoid 1/f singularity; DC bin is zeroed afterwards.
    f_safe = jnp.where(freq > 0, freq, 1.0)

    if tdi == 1:
        X, Y, Z = tdi1_1exp_f_glitch(f_safe, t0, Deltav, tau, T)
    elif tdi == 2:
        X, Y, Z = tdi2_1exp_f_glitch(f_safe, t0, Deltav, tau, T)
    else:
        raise ValueError(f"tdi must be 1 or 2, got {tdi!r}")

    if basis == "AET":
        c0, c1, c2 = AET(X, Y, Z)
    elif basis == "XYZ":
        c0, c1, c2 = X, Y, Z
    else:
        raise ValueError(f"basis must be 'AET' or 'XYZ', got {basis!r}")

    h_fd = jnp.stack([c0, c1, c2], axis=-1)   # (F, 3)
    # zero the DC bin wherever it sits (don't assume index 0 is f = 0)
    return jnp.where(freq[:, None] > 0, h_fd, 0.0 + 0.0j)


@partial(jax.jit, static_argnames=("tdi", "basis"))
def clean_signal_t(params, t, T: float = T_ARM_s, tdi: int = 1, basis: str = "AET"):
    """
    Time-domain TDI signal for a single-exponential glitch.

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
             t0     : glitch onset time (s).
             Deltav : velocity-kick amplitude (m/s).
             tau    : exponential decay timescale (s).
    t      : (N,) time array (s).
    T      : LISA one-way light travel time along one arm (s).
    tdi    : TDI generation — 1 or 2.
    basis  : output channel basis — "AET" (default) or "XYZ".

    Returns
    -------
    h_td : real (N, 3), columns [A, E, T] if basis="AET" else [X, Y, Z].
    """
    t0     = params[0]
    Deltav = params[1]
    tau    = params[2]

    if tdi == 1:
        X, Y, Z = tdi1_1exp_glitch(t, t0, Deltav, tau, T)
    elif tdi == 2:
        X, Y, Z = tdi2_1exp_glitch(t, t0, Deltav, tau, T)
    else:
        raise ValueError(f"tdi must be 1 or 2, got {tdi!r}")

    if basis == "AET":
        c0, c1, c2 = AET(X, Y, Z)
    elif basis == "XYZ":
        c0, c1, c2 = X, Y, Z
    else:
        raise ValueError(f"basis must be 'AET' or 'XYZ', got {basis!r}")

    return jnp.stack([c0, c1, c2], axis=-1)   # (N, 3)


# ── Raw (pre-TDI) single-link signal ──────────────────────────────────────────

@jax.jit
def raw_glitch_t(params, t, T: float = T_ARM_s):
    """
    Time-domain raw glitch on link 12, *before* the TDI combination.

    This is the single-link fractional-frequency signal (a smooth velocity step
    in units of c).  It is the integrated n=1 shapelet whose time derivative is
    the acceleration shapelet (delta_t/tau^2) e^{-delta_t/tau}.

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
    t      : (N,) time array (s).
    T      : kept for signature symmetry with the TDI functions (unused here:
             the raw single-link signal carries no arm delay).

    Returns
    -------
    raw_td : real (N,) fractional-frequency single-link signal.
    """
    del T  # the raw single-link signal has no light-travel delay
    t0     = params[0]
    Deltav = params[1]
    tau    = params[2]

    x    = t - t0
    mask = t >= t0
    xs   = jnp.where(mask, x, 0.0)                       # guard exp for x < 0
    g    = jnp.where(mask, -1.0 + jnp.exp(-xs / tau) * (1.0 + xs / tau), 0.0)
    return Deltav / C_SI * g                             # (N,)


@jax.jit
def raw_glitch_f(params, freq, T: float = T_ARM_s):
    """
    Frequency-domain raw glitch on link 12, *before* the TDI combination.

    Analytic Fourier transform of `raw_glitch_t`:

        raw_f(f) = 1 / (i w c) * e^{-i w t0} * Deltav / (-i + tau w)^2 ,  w = 2*pi*f.

    The DC bin is set to zero (the step has an ill-defined zero-frequency value;
    every TDI transfer function vanishes there anyway).

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
    freq   : (F,) frequency array (Hz), e.g. from freq_grid().
    T      : kept for signature symmetry (unused).

    Returns
    -------
    raw_fd : complex (F,) single-link signal; DC bin is zero.
    """
    del T
    t0     = params[0]
    Deltav = params[1]
    tau    = params[2]

    f_safe = jnp.where(freq > 0, freq, 1.0)             # avoid 1/f singularity
    w      = 2.0 * jnp.pi * f_safe
    raw    = (1.0 / (1j * C_SI * w)) * jnp.exp(-1j * t0 * w) * Deltav / (-1j + tau * w) ** 2
    return jnp.where(freq > 0, raw, 0.0 + 0.0j)         # (F,)


@partial(jax.jit, static_argnames=("tdi", "basis"))
def compute_TDI(raw_f, freq, T: float = T_ARM_s, tdi: int = 1, basis: str = "AET"):
    """
    Numerically apply the link-12 TDI combination to a raw frequency-domain glitch.

    By construction this reproduces `clean_signal_f`:

        compute_TDI(raw_glitch_f(params, freq), freq, T, tdi, basis)
            == clean_signal_f(params, freq, T, tdi, basis)

    The TDI Michelson combination is implemented as frequency-domain delay
    operators D = e^{-i w T} acting on the single-link signal raw_f.

    Parameters
    ----------
    raw_f : complex (F,) single-link signal, e.g. from raw_glitch_f().
    freq  : (F,) frequency array (Hz) — same grid raw_f was evaluated on.
    T     : LISA one-way light travel time along one arm (s).
    tdi   : TDI generation — 1 or 2.
    basis : output channel basis — "AET" (default) or "XYZ".

    Returns
    -------
    h_fd : complex (F, 3), columns [A, E, T] if basis="AET" else [X, Y, Z].
           DC bin is zero.
    """
    w = 2.0 * jnp.pi * freq
    D = jnp.exp(-1j * w * T)                             # delay by one arm T

    if tdi == 1:
        TFX = 1.0 - D ** 4
        TFY = 2.0 * (D ** 3 - D)
    elif tdi == 2:
        TFX = (1.0 - D ** 4) ** 2
        TFY = 2.0 * (-D + D ** 3 + D ** 5 - D ** 7)
    else:
        raise ValueError(f"tdi must be 1 or 2, got {tdi!r}")

    X = TFX * raw_f
    Y = TFY * raw_f
    Z = jnp.zeros_like(X)

    if basis == "AET":
        c0, c1, c2 = AET(X, Y, Z)
    elif basis == "XYZ":
        c0, c1, c2 = X, Y, Z
    else:
        raise ValueError(f"basis must be 'AET' or 'XYZ', got {basis!r}")

    h_fd = jnp.stack([c0, c1, c2], axis=-1)             # (F, 3)
    # zero the DC bin wherever it sits (don't assume index 0 is f = 0)
    return jnp.where(freq[:, None] > 0, h_fd, 0.0 + 0.0j)


# ── Unequal-arm TDI (numerical orbits, frozen-arm approximation) ──────────────
#
# ltt is the (6,) array of one-way light travel times in the lisaorbits
# ordering [L_12, L_23, L_31, L_13, L_32, L_21], e.g. from orbits.link_ltt(t0).

def _unequal_transfer(freq, ltt, tdi: int):
    """Frequency-domain TDI transfer functions (TFX, TFY) for a link-12 glitch
    with six constant, unequal delays.  TFZ is identically zero."""
    w = 2.0 * jnp.pi * freq
    D12, D23, D31, D13, D32, D21 = (jnp.exp(-1j * w * ltt[k]) for k in range(6))

    A = D12 * D21     # round trip along the 1-2 arm
    B = D13 * D31     # round trip along the 1-3 arm
    C = D23 * D32     # round trip along the 2-3 arm

    TFX = (1.0 - B) * (1.0 + A)
    TFY = 2.0 * D21 * (C - 1.0)
    if tdi == 2:
        TFX = (1.0 - A * B) * TFX
        TFY = (1.0 - C * A) * TFY
    elif tdi != 1:
        raise ValueError(f"tdi must be 1 or 2, got {tdi!r}")
    return TFX, TFY


def _psi(t, ts, tau):
    """Integrated one-exponential shapelet with onset ts:
    psi(t) = Theta(t - ts) * [-1 + e^{-(t-ts)/tau} (1 + (t-ts)/tau)]."""
    x    = t - ts
    mask = x >= 0.0
    xs   = jnp.where(mask, x, 0.0)                      # guard exp for x < 0
    return jnp.where(mask, -1.0 + jnp.exp(-xs / tau) * (1.0 + xs / tau), 0.0)


def _unequal_pulses(ltt, tdi: int):
    """Time-domain pulse decomposition of the unequal-arm transfer functions.

    Returns (delays_X, signs_X, delays_Y, signs_Y): each TDI channel is
    sum_k sign_k * psi(t; t0 + delay_k) times Deltav/c (2*Deltav/c for Y).
    Obtained by expanding the delay-operator products in _unequal_transfer.
    """
    L12, L23, L31, L13, L32, L21 = (ltt[k] for k in range(6))
    a = L12 + L21     # round trip along the 1-2 arm
    b = L13 + L31     # round trip along the 1-3 arm
    c = L23 + L32     # round trip along the 2-3 arm

    if tdi == 1:
        # (1 - B)(1 + A) = 1 + A - B - AB
        dX = (0.0, a, b, a + b)
        sX = (1.0, 1.0, -1.0, -1.0)
        # D21 (C - 1)
        dY = (L21 + c, L21)
        sY = (1.0, -1.0)
    elif tdi == 2:
        # (1 - AB)(1 + A - B - AB) = 1 + A - B - 2AB - A^2B + AB^2 + A^2B^2
        dX = (0.0, a, b, a + b, 2 * a + b, a + 2 * b, 2 * (a + b))
        sX = (1.0, 1.0, -1.0, -2.0, -1.0, 1.0, 1.0)
        # (1 - CA) D21 (C - 1) = D21 (C - 1 + CA - C^2 A)
        dY = (L21 + c, L21, L21 + c + a, L21 + 2 * c + a)
        sY = (1.0, -1.0, 1.0, -1.0)
    else:
        raise ValueError(f"tdi must be 1 or 2, got {tdi!r}")
    return dX, sX, dY, sY


@partial(jax.jit, static_argnames=("tdi", "basis"))
def clean_signal_f_unequal(params, freq, ltt, tdi: int = 1, basis: str = "AET"):
    """
    Frequency-domain TDI signal for a link-12 glitch with unequal arms.

    Same conventions as clean_signal_f, but the single arm time T is replaced
    by the six per-link light travel times frozen at the glitch epoch.

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
    freq   : (F,) frequency array (Hz), e.g. from freq_grid().
    ltt    : (6,) light travel times (s), lisaorbits order
             [L_12, L_23, L_31, L_13, L_32, L_21] — see orbits.link_ltt.
    tdi    : TDI generation — 1 or 2.
    basis  : "AET" (default) or "XYZ".

    Returns
    -------
    h_fd : complex (F, 3); DC bin is zero.
    """
    f_safe = jnp.where(freq > 0, freq, 1.0)
    raw    = raw_glitch_f(params, freq)                 # (F,) — DC already zero

    TFX, TFY = _unequal_transfer(f_safe, ltt, tdi)
    X = TFX * raw
    Y = TFY * raw
    Z = jnp.zeros_like(X)

    if basis == "AET":
        c0, c1, c2 = AET(X, Y, Z)
    elif basis == "XYZ":
        c0, c1, c2 = X, Y, Z
    else:
        raise ValueError(f"basis must be 'AET' or 'XYZ', got {basis!r}")

    h_fd = jnp.stack([c0, c1, c2], axis=-1)             # (F, 3)
    # zero the DC bin wherever it sits (don't assume index 0 is f = 0)
    return jnp.where(freq[:, None] > 0, h_fd, 0.0 + 0.0j)


@partial(jax.jit, static_argnames=("tdi", "basis"))
def clean_signal_t_unequal(params, t, ltt, tdi: int = 1, basis: str = "AET"):
    """
    Time-domain TDI signal for a link-12 glitch with unequal arms.

    Exact time-domain counterpart of clean_signal_f_unequal: each TDI channel
    is a signed sum of shifted copies of the integrated shapelet, with the
    shifts built from the six frozen light travel times.

    Parameters
    ----------
    params : (3,) array  [t0 (s), Deltav (m/s), tau (s)].
    t      : (N,) time array (s).
    ltt    : (6,) light travel times (s), lisaorbits order
             [L_12, L_23, L_31, L_13, L_32, L_21] — see orbits.link_ltt.
    tdi    : TDI generation — 1 or 2.
    basis  : "AET" (default) or "XYZ".

    Returns
    -------
    h_td : real (N, 3).
    """
    t0     = params[0]
    Deltav = params[1]
    tau    = params[2]

    dX, sX, dY, sY = _unequal_pulses(ltt, tdi)

    X = Deltav / C_SI * sum(s * _psi(t, t0 + d, tau) for d, s in zip(dX, sX))
    Y = 2.0 * Deltav / C_SI * sum(s * _psi(t, t0 + d, tau) for d, s in zip(dY, sY))
    Z = jnp.zeros_like(X)

    if basis == "AET":
        c0, c1, c2 = AET(X, Y, Z)
    elif basis == "XYZ":
        c0, c1, c2 = X, Y, Z
    else:
        raise ValueError(f"basis must be 'AET' or 'XYZ', got {basis!r}")

    return jnp.stack([c0, c1, c2], axis=-1)             # (N, 3)


@partial(jax.jit, static_argnames=("tdi", "basis"))
def compute_TDI_unequal(raw_f, freq, ltt, tdi: int = 1, basis: str = "AET"):
    """
    Apply the unequal-arm link-12 TDI combination to a raw frequency-domain glitch.

    By construction:
        compute_TDI_unequal(raw_glitch_f(params, freq), freq, ltt, tdi, basis)
            == clean_signal_f_unequal(params, freq, ltt, tdi, basis)

    Parameters
    ----------
    raw_f : complex (F,) single-link signal, e.g. from raw_glitch_f().
    freq  : (F,) frequency array (Hz) — same grid raw_f was evaluated on.
    ltt   : (6,) light travel times (s), lisaorbits order.
    tdi   : TDI generation — 1 or 2.
    basis : "AET" (default) or "XYZ".

    Returns
    -------
    h_fd : complex (F, 3); DC bin is zero.
    """
    TFX, TFY = _unequal_transfer(freq, ltt, tdi)
    X = TFX * raw_f
    Y = TFY * raw_f
    Z = jnp.zeros_like(X)

    if basis == "AET":
        c0, c1, c2 = AET(X, Y, Z)
    elif basis == "XYZ":
        c0, c1, c2 = X, Y, Z
    else:
        raise ValueError(f"basis must be 'AET' or 'XYZ', got {basis!r}")

    h_fd = jnp.stack([c0, c1, c2], axis=-1)             # (F, 3)
    # zero the DC bin wherever it sits (don't assume index 0 is f = 0)
    return jnp.where(freq[:, None] > 0, h_fd, 0.0 + 0.0j)
