"""
Numerical LISA orbits (lisaorbits) -> per-link light travel times for unequal-arm TDI.

Frozen-arm approximation
------------------------
Glitches last seconds to minutes, while the LISA arm lengths breathe on
month timescales (dL/dt ~ 10 m/s, i.e. a light-travel-time drift of
~3e-8 s per second).  Over the duration of one glitch the six one-way
light travel times L_ij are therefore constant to excellent accuracy,
but they are *not* equal to each other: the Keplerian constellation shows
inter-link differences of ~30 ms.  We freeze the six delays at the glitch
epoch and build the TDI response with those constant, unequal delays.

Link ordering
-------------
All (6,) light-travel-time arrays in this package follow the
lisaorbits / lisaconstants convention:

    LINKS = (12, 23, 31, 13, 32, 21)

so that ltt[0] = L_12, ltt[1] = L_23, ltt[2] = L_31,
        ltt[3] = L_13, ltt[4] = L_32, ltt[5] = L_21.
Link ij is the beam *received* by spacecraft i, *emitted* by spacecraft j.

Public API
----------
default_orbits(**kwargs)      -> lisaorbits.KeplerianOrbits instance
link_ltt(t, orbits)           -> (6,) [scalar t] or (N, 6) light travel times (s)
equal_arm_ltt(T)              -> (6,) constant array (for testing/reduction checks)
"""
import numpy as np

from .waveform import T_ARM_s

# lisaorbits / lisaconstants link ordering (receiver, emitter)
LINKS = (12, 23, 31, 13, 32, 21)


def default_orbits(**kwargs):
    """Return a lisaorbits.KeplerianOrbits instance (imported lazily).

    Keyword arguments are forwarded to KeplerianOrbits, e.g.
    L=2.5e9 (mean arm length, m), lambda1, m_init1 (initial constellation
    phase angles, rad).
    """
    from lisaorbits import KeplerianOrbits
    return KeplerianOrbits(**kwargs)


def link_ltt(t, orbits=None):
    """
    Light travel times of the six LISA links at epoch(s) t.

    Parameters
    ----------
    t      : float or (N,) array — epoch(s) since the orbit reference time (s).
             For a glitch, pass the glitch onset epoch (the frozen-arm
             approximation evaluates the orbits once per glitch).
    orbits : lisaorbits.Orbits instance or None — defaults to KeplerianOrbits().

    Returns
    -------
    ltt : (6,) array if t is scalar, else (N, 6) — L_ij in seconds,
          ordered as LINKS = (12, 23, 31, 13, 32, 21).
    """
    if orbits is None:
        orbits = default_orbits()
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    ltt = np.asarray(orbits.compute_ltt(t_arr))    # (N, 6)
    return ltt[0] if np.ndim(t) == 0 else ltt


def equal_arm_ltt(T: float = T_ARM_s):
    """(6,) array with all links set to the same travel time T (s).

    Feeding this to the unequal-arm functions must reproduce the
    equal-arm functions exactly — useful as a reduction check.
    """
    return np.full(6, float(T))
