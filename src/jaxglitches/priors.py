"""
Prior distributions for a single single-exponential glitch.

Parameter vector (3 components, last axis):
    [t0, Deltav, tau]

    t0    : glitch onset time (s)   — uniform over the analysis window
    Deltav : velocity-kick amplitude (m/s) — log-uniform
    tau   : exponential decay timescale (s)  — log-uniform

Physical model and bounds
-------------------------
These parameters follow the LISA Pathfinder glitch model implemented in the
LISA Glitch tool (https://gitlab.in2p3.fr/lisa-simulation/glitch). A glitch is a
single-component (n = 1) shapelet in differential acceleration; integrated in
time it produces a velocity kick

    Delta_v = "equivalent transferred impulse"   (the amplitude here, m/s)
    tau     = "equivalent damping time" beta      (s)
    t0      = arrival time                        (s)

The bounds below are chosen from the observed LPF population characterised in
Baghi et al. 2022 (PRD 105, 042002; arXiv:2112.07490): transferred impulses are
mostly in 1e-15 - 1e-13 m/s with a detection floor near 2e-16 m/s, and damping
times range from below the sampling time up to a few hours, most under a minute.
`catalog_generator.py` draws from exactly these priors and adds the LPF event
rate to synthesise realistic glitch catalogues.

The prior is expressed through its inverse CDF so it can be composed
with any sampler that expects unit-hypercube inputs (flowMC, numpyro, etc.).
Log-prior functions are also provided for samplers that work directly in
parameter space.

Samplers usually explore the *sampling* coordinates

    xi = [t0, log Deltav, log tau]

in which the log-uniform priors on Deltav and tau become flat (their Jacobian
cancels). `to_sampling`/`to_physical` convert between the two, and
`log_prior_sampling` / `sample_prior` give the prior and prior draws directly in
these coordinates.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
T_OBS_s = 3600.0     # 1 hour observation window

jax.config.update("jax_enable_x64", True)

# Prior bounds
# Anchored on the LISA Pathfinder glitch population of the LISA Glitch tool
# (lisa-simulation/glitch; Baghi et al. 2022, arXiv:2112.07490). The tau bounds
# are the exact damping-time support the tool renormalises its sampler onto
# (sample_lpf.py: exp(-2.3026)=0.1 s to exp(10.8198)=5e4 s). The Delta_v bounds
# span the amplitude support of the fitted catalogue (|level| up to ~2.15e-8 m/s,
# median ~3e-13 m/s) with a practical lower detection floor; the box is wide
# enough not to asymmetrically truncate posteriors (rule of thumb: the wall
# should sit > 5 marginal-sigma from the MAP; SNR~50 gives sigma_log(Dv) ~ 0.12).
# The *realistic* (non-uniform) population lives in catalog_generator.py, which
# resamples the empirical LPF catalogue rather than this log-uniform box.
_T0_MIN   = 0.0      # onset time lower bound (s)
_DELTAV_MIN = 1e-16   # minimum velocity kick (m/s)  — practical detection floor
_DELTAV_MAX = 1e-7    # maximum velocity kick (m/s)  — above brightest LPF event (~2e-8)
_TAU_MIN   = 0.1     # minimum decay timescale (s)  — LPF sampler support (~sample rate)
_TAU_MAX   = 5e4     # maximum decay timescale (s)  — LPF sampler support (longest events)


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


# ── Sampling coordinates xi = [t0, log Deltav, log tau] ────────────────────────

def to_sampling(params):
    """
    Physical -> sampling coordinates, on the last axis:

        [t0, Deltav, tau]  ->  [t0, log Deltav, log tau]

    Accepts a single (3,) vector or a batch (..., 3).
    """
    return jnp.stack(
        [params[..., 0], jnp.log(params[..., 1]), jnp.log(params[..., 2])], axis=-1
    )


def to_physical(xi):
    """
    Sampling -> physical coordinates (inverse of `to_sampling`), on the last axis:

        [t0, log Deltav, log tau]  ->  [t0, Deltav, tau]

    Accepts a single (3,) vector or a batch (..., 3).
    """
    return jnp.stack(
        [xi[..., 0], jnp.exp(xi[..., 1]), jnp.exp(xi[..., 2])], axis=-1
    )


def sampling_bounds(t_obs: float = T_OBS_s):
    """Return (lower, upper) bound arrays of shape (3,) in sampling coordinates."""
    lower, upper = prior_bounds(t_obs)
    return to_sampling(lower), to_sampling(upper)


def log_prior_sampling(xi, t_obs: float = T_OBS_s):
    """
    Log prior in sampling coordinates [t0, log Deltav, log tau].

    The physical prior is uniform in t0 and log-uniform in Deltav and tau; in these
    log coordinates its Jacobian cancels, so the prior is *flat* inside the box.
    Returns 0.0 inside the prior support and -inf outside. Accepts a single (3,)
    vector or a batch (..., 3) (then the result has shape (...,)).
    """
    lower, upper = sampling_bounds(t_obs)
    in_bounds = jnp.all((xi >= lower) & (xi <= upper), axis=-1)
    return jnp.where(in_bounds, 0.0, -jnp.inf)


def sample_prior(key, n: int, t_obs: float = T_OBS_s):
    """
    Draw `n` samples from the prior, returned in sampling coordinates
    [t0, log Deltav, log tau] with shape (n, 3).

    Uses `prior_inverse_cdf` on uniform-hypercube draws, so the samples follow the
    exact prior (uniform t0, log-uniform Deltav and tau).
    """
    u = jr.uniform(key, (n, 3))
    return to_sampling(prior_inverse_cdf(u, t_obs))
