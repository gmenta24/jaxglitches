"""
Prior distributions for a single single-exponential glitch.

Parameter vector (3 components, last axis):
    [t0, Deltav, tau]

    t0    : glitch onset time (s)   — uniform over the analysis window
    Deltav : velocity-kick amplitude (m/s) — log-uniform
    tau   : exponential decay timescale (s)  — log-uniform

The prior is expressed through its inverse CDF so it can be composed
with any sampler that expects unit-hypercube inputs (flowMC, numpyro, etc.).
Log-prior functions are also provided for samplers that work directly in
parameter space.
"""
import jax
import jax.numpy as jnp
from .constants import T_OBS_s

jax.config.update("jax_enable_x64", True)

# Prior bounds
# Rule of thumb: the boundary must be > 5 marginal-sigma from the MAP so that
# the prior does not asymmetrically truncate the posterior.  For a typical
# glitch SNR~50 the marginal sigma_log(Dv) ~ 0.12, so the wall must be at
# least 0.6 log-units (~factor 2) from the MAP.  The old Dv_MIN=1e-13 was
# only 1.5 sigma from the MAP and was visibly skewing the posteriors.
_T0_MIN   = 0.0      # onset time lower bound (s)
_DELTAV_MIN = 1e-16   # minimum velocity kick (m/s)  — 57 sigma from a typical MAP
_DELTAV_MAX = 1e-8    # maximum velocity kick (m/s)
_TAU_MIN   = 0.01    # minimum decay timescale (s)
_TAU_MAX   = 1000.0  # maximum decay timescale (s)


def prior_inverse_cdf(u, t_obs: float = T_OBS_s):
    """
    Map unit-hypercube samples to glitch parameters.

    Parameters
    ----------
    u     : array of shape (..., 3) with values in [0, 1].
    t_obs : observation window length (s); upper bound for t0.

    Returns
    -------
    params : array of shape (..., 3) = [t0, Deltav, tau].
    """
    u_t0, u_Dv, u_tau = u[..., 0], u[..., 1], u[..., 2]

    t0   = u_t0 * t_obs
    Deltav = jnp.exp(jnp.log(_DELTAV_MIN) + u_Dv * jnp.log(_DELTAV_MAX / _DELTAV_MIN))
    tau  = jnp.exp(jnp.log(_TAU_MIN)   + u_tau * jnp.log(_TAU_MAX  / _TAU_MIN))

    return jnp.stack([t0, Deltav, tau], axis=-1)


def log_prior(params, t_obs: float = T_OBS_s):
    """
    Log prior density p(t0, Deltav, tau).

    Parameters
    ----------
    params : array of shape (..., 3) = [t0, Deltav, tau].
    t_obs  : observation window (s).

    Returns
    -------
    Scalar log-prior value (sum over last axis).
    """
    t0, Deltav, tau = params[..., 0], params[..., 1], params[..., 2]

    # uniform in t0
    log_p_t0 = -jnp.log(t_obs)

    # log-uniform in Deltav and tau
    log_p_Dv   = -jnp.log(Deltav)  - jnp.log(jnp.log(_DELTAV_MAX / _DELTAV_MIN))
    log_p_tau = -jnp.log(tau)    - jnp.log(jnp.log(_TAU_MAX   / _TAU_MIN))

    in_bounds = (
        (t0   >= 0.0)         & (t0   <= t_obs)
        & (Deltav >= _DELTAV_MIN) & (Deltav <= _DELTAV_MAX)
        & (tau  >= _TAU_MIN)  & (tau  <= _TAU_MAX)
    )

    lp = log_p_t0 + log_p_Dv + log_p_tau
    return jnp.where(in_bounds, lp, -jnp.inf)


def prior_bounds(t_obs: float = T_OBS_s):
    """Return (lower, upper) bound arrays of shape (3,) for box-constrained samplers."""
    lower = jnp.array([0.0,        _DELTAV_MIN, _TAU_MIN])
    upper = jnp.array([t_obs,      _DELTAV_MAX, _TAU_MAX])
    return lower, upper
