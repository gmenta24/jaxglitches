"""
Realistic glitch-population catalogue generator.

``run_catalog(t_obs)`` returns a realistic population of LISA glitches over an
observation window of ``t_obs`` seconds, reproducing the pipeline of the LISA
Glitch tool (https://gitlab.in2p3.fr/lisa-simulation/glitch, ``sampler/``) and
the LISA Pathfinder statistics of Baghi et al. 2022 (arXiv:2112.07490):

* **Arrival times** are a homogeneous Poisson process. The rate is estimated the
  same way LISA Glitch does it — ``lambda = 1 / mean(interarrivals)`` from the
  LPF interval catalogue — giving ``lambda ~ 4.97e-5 /s`` (~4.3 events/day) for
  ordinary runs, with cold runs about ten times more active. ``lambda`` is the
  *total* event rate; each event is then assigned one of the six test-mass
  injection points uniformly (as in ``sampler/sample_lpf.py``).

* **Parameters** ``(Delta_v, tau)`` are drawn from the *empirical* LPF glitch
  catalogue (the ``effective_glitch_parameters`` files that the tool's
  normalizing flow was trained on), by Gaussian-kernel resampling in
  ``(log tau, log|Delta_v|)`` space. This reproduces the true, highly non-uniform
  LPF population — damping times clustered below a minute with a long tail to a
  few hours, and velocity kicks around ~3e-13 m/s — instead of a flat prior.
  Glitch polarity (the sign of ``Delta_v``) is randomised, as in the tool.

Mapping to the LISA Glitch model
--------------------------------
Glitches are single-component (``n = 1``) integrated exponential shapelets
(``IntegratedShapeletGlitch``): ``s(t) = level * [1 - (1 + dt/beta) e^{-dt/beta}]``,
which asymptotes to ``level``. Hence the tool's ``level`` is exactly this
package's velocity kick ``Delta_v`` and its ``beta`` is the decay time ``tau``
— the same shape as :func:`jaxglitches.waveform.tdi1_1exp_glitch`.

The per-event parameters use the bounds and coordinate transforms of
:mod:`priors`; only the population *shape* comes from the LPF catalogue.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from . import priors

jax.config.update("jax_enable_x64", True)

# ── Bundled LPF catalogue data (copied from lisa-simulation/glitch) ─────────────
_DATA_DIR = Path(__file__).parent / "lpf_data"
_PARAM_FILES = {
    "ordinary": _DATA_DIR / "2021-09-17-effective_glitch_parameters_ordinary.txt",
    "cold":     _DATA_DIR / "2021-09-17-effective_glitch_parameters_cold.txt",
}
_INTERVAL_FILE = _DATA_DIR / "2021-09-17-intervals_ordinary.txt"

# ── Event rates (Baghi et al. 2022) ────────────────────────────────────────────
# Total glitch rate [s^-1]. RATE_ORDINARY is estimated from the LPF interval
# catalogue (1/mean interarrival ~ 4.3 events/day). No cold-run interval file is
# published; LPF cold runs were ~10x more active, so we scale accordingly.
RATE_ORDINARY = 1.0 / float(np.mean(np.loadtxt(_INTERVAL_FILE, comments="#", usecols=0)))
RATE_COLD     = 10.0 * RATE_ORDINARY

_RATES = {"ordinary": RATE_ORDINARY, "cold": RATE_COLD}

# ── Injection points ───────────────────────────────────────────────────────────
# The six test-mass interferometers (LISA Glitch naming, ``sampler/sample_lpf.py``).
INJ_POINTS = ("tm_12", "tm_23", "tm_31", "tm_13", "tm_32", "tm_21")


def _load_lpf_catalog(run_type: str):
    """Return (beta, level) arrays (s, m/s) of the empirical LPF glitch catalogue.

    Columns of the data file are ``Beta [s]``, ``Amplitude``, ``SNR``. The tool's
    sampler feeds the amplitude column straight in as ``IntegratedShapeletGlitch``'s
    ``level`` (asymptotic velocity kick), so we treat it as ``Delta_v`` [m/s]
    regardless of the (legacy) unit label in the header.
    """
    try:
        path = _PARAM_FILES[run_type]
    except KeyError:
        raise ValueError(f"run_type must be one of {tuple(_PARAM_FILES)}; got {run_type!r}")
    arr = np.loadtxt(path, comments="#", usecols=(0, 1))
    return arr[:, 0], arr[:, 1]


def _resolve_rate(run_type: str, rate):
    if rate is not None:
        return float(rate)
    try:
        return _RATES[run_type]
    except KeyError:
        raise ValueError(
            f"run_type must be one of {tuple(_RATES)} or pass an explicit rate; got {run_type!r}"
        )


def expected_count(t_obs: float, run_type: str = "ordinary", rate=None) -> float:
    """Expected number of glitches over ``t_obs`` seconds: ``lambda * t_obs``."""
    return _resolve_rate(run_type, rate) * t_obs


def sample_parameters(key, n: int, run_type: str = "ordinary",
                      smooth: bool = True, signed: bool = True):
    """
    Draw ``n`` glitch parameter pairs from the empirical LPF population.

    Parameters
    ----------
    key      : JAX PRNG key.
    n        : number of parameter pairs.
    run_type : "ordinary" or "cold" — selects the LPF catalogue.
    smooth   : if True (default), Gaussian-kernel-density resampling in
               ``(log tau, log|Delta_v|)`` space (Scott's-rule bandwidth); if
               False, plain bootstrap of the catalogue's discrete values.
    signed   : if True (default), randomise the sign of ``Delta_v`` (glitch
               polarity), matching ``sampler/sample_lpf.py``.

    Returns
    -------
    (Deltav, tau) : two arrays of shape ``(n,)`` — velocity kick (m/s, signed if
    requested) and damping time (s), clipped to the :mod:`priors` bounds.
    """
    beta, level = _load_lpf_catalog(run_type)
    log_data = jnp.stack([jnp.log(jnp.asarray(beta)), jnp.log(jnp.abs(jnp.asarray(level)))])  # (2, M)
    m = log_data.shape[1]

    k_idx, k_noise, k_sign = jr.split(key, 3)
    idx = jr.randint(k_idx, (n,), 0, m)
    means = log_data[:, idx].T                      # (n, 2)

    if smooth:
        # scipy.stats.gaussian_kde-style resampling: pick a data point, add
        # Gaussian noise with covariance (Scott factor)^2 * data covariance.
        cov = jnp.cov(log_data)                     # (2, 2)
        factor2 = m ** (-2.0 / (2 + 4))             # Scott's rule, d=2
        noise = jr.multivariate_normal(k_noise, jnp.zeros(2), factor2 * cov, (n,))
        samples = means + noise
    else:
        samples = means

    tau    = jnp.exp(samples[:, 0])
    deltav = jnp.exp(samples[:, 1])

    lower, upper = priors.prior_bounds()
    tau    = jnp.clip(tau,    lower[2], upper[2])
    deltav = jnp.clip(deltav, lower[1], upper[1])

    if signed:
        signs = jnp.where(jr.bernoulli(k_sign, 0.5, (n,)), 1.0, -1.0)
        deltav = deltav * signs

    return deltav, tau


def run_catalog(t_obs: float, key=None, run_type: str = "ordinary", rate=None,
                inj_points=INJ_POINTS, smooth: bool = True, signed: bool = True,
                sort: bool = True):
    """
    Generate a realistic glitch catalogue over an observation window.

    Parameters
    ----------
    t_obs      : observation window length (s).
    key        : JAX PRNG key. If ``None``, a fixed key (``jr.PRNGKey(0)``) is
                 used for reproducibility; pass an explicit key for a new draw.
    run_type   : "ordinary" or "cold" — selects the LPF rate and population.
    rate       : optional total event rate (s^-1); overrides ``run_type``'s rate.
    inj_points : injection-point names to draw from (default: the six ``tm_ij``).
    smooth     : KDE-smoothed (True) vs bootstrap (False) parameter resampling.
    signed     : randomise glitch polarity (sign of ``Delta_v``).
    sort       : order the catalogue by arrival time t0 (default True).

    Returns
    -------
    catalog : dict with entries
        "n_glitches" : int, number of events drawn (~ Poisson(rate * t_obs)).
        "t0"         : (N,) arrival times (s), uniform on [0, t_obs].
        "Deltav"     : (N,) velocity kicks (m/s), signed if ``signed``.
        "tau"        : (N,) damping times (s).
        "params"     : (N, 3) physical parameters [t0, Deltav, tau].
        "xi"         : (N, 3) sampling coordinates [t0, log|Deltav|, log tau].
        "inj_idx"    : (N,) integer index into ``inj_points``.
        "inj_point"  : list[str] of length N, injection-point name per event.
        "t_obs"      : the observation window (s).
        "rate"       : the total event rate used (s^-1).
    """
    if key is None:
        key = jr.PRNGKey(0)
    lam = _resolve_rate(run_type, rate)

    k_n, k_t0, k_par, k_inj = jr.split(key, 4)

    # Number of events over the window: homogeneous Poisson process.
    n = int(jr.poisson(k_n, lam * t_obs))

    # Arrival times: uniform on [0, t_obs] (Poisson-process arrival law).
    t0 = jr.uniform(k_t0, (n,), minval=0.0, maxval=t_obs)

    # Parameters from the empirical LPF population.
    deltav, tau = sample_parameters(k_par, n, run_type, smooth=smooth, signed=signed)

    # Uniformly assign an injection point.
    n_inj = len(inj_points)
    inj_idx = jr.randint(k_inj, (n,), 0, n_inj)

    if sort and n > 0:
        order = jnp.argsort(t0)
        t0, deltav, tau, inj_idx = t0[order], deltav[order], tau[order], inj_idx[order]

    params = jnp.stack([t0, deltav, tau], axis=-1)
    # Sampling coords use log|Delta_v| (the sign is the glitch polarity, tracked
    # separately in ``Deltav``/``params``).
    xi = jnp.stack([t0, jnp.log(jnp.abs(deltav)), jnp.log(tau)], axis=-1)

    inj_idx_py = [int(i) for i in inj_idx]
    inj_names = [inj_points[i] for i in inj_idx_py]

    return {
        "n_glitches": n,
        "t0": t0,
        "Deltav": deltav,
        "tau": tau,
        "params": params,
        "xi": xi,
        "inj_idx": inj_idx,
        "inj_point": inj_names,
        "t_obs": float(t_obs),
        "rate": lam,
    }
