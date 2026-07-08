"""Tests for jaxglitches.priors: inverse CDF, densities, coordinate
transforms, and prior sampling."""
import jax.numpy as jnp
import jax.random as jr
import pytest

import jaxglitches as jg
from jaxglitches.priors import (
    _DELTAV_MIN, _DELTAV_MAX, _TAU_MIN, _TAU_MAX, T_OBS_s,
)

T_OBS = 3600.0


class TestPriorInverseCdf:
    def test_endpoints(self):
        lower, upper = jg.prior_bounds(T_OBS)
        assert jnp.allclose(jg.prior_inverse_cdf(jnp.zeros(3), T_OBS), lower)
        assert jnp.allclose(jg.prior_inverse_cdf(jnp.ones(3), T_OBS), upper, rtol=1e-12)

    def test_midpoint_is_geometric_mean_for_loguniform(self):
        p = jg.prior_inverse_cdf(jnp.full(3, 0.5), T_OBS)
        assert jnp.isclose(p[0], T_OBS / 2)
        assert jnp.isclose(p[1], jnp.sqrt(_DELTAV_MIN * _DELTAV_MAX), rtol=1e-12)
        assert jnp.isclose(p[2], jnp.sqrt(_TAU_MIN * _TAU_MAX), rtol=1e-12)

    def test_batch_shape(self):
        u = jr.uniform(jr.PRNGKey(0), (7, 5, 3))
        assert jg.prior_inverse_cdf(u, T_OBS).shape == (7, 5, 3)


class TestLogPrior:
    def test_analytic_value_inside_support(self, params):
        t0, dv, tau = params
        expected = (
            -jnp.log(T_OBS)
            - jnp.log(dv) - jnp.log(jnp.log(_DELTAV_MAX / _DELTAV_MIN))
            - jnp.log(tau) - jnp.log(jnp.log(_TAU_MAX / _TAU_MIN))
        )
        assert jnp.isclose(jg.log_prior(params, T_OBS), expected, rtol=1e-14)

    @pytest.mark.parametrize("bad", [
        [-1.0, 1e-13, 1.0],          # t0 < 0
        [4000.0, 1e-13, 1.0],        # t0 > t_obs
        [100.0, 1e-17, 1.0],         # Deltav below floor
        [100.0, 1e-6, 1.0],          # Deltav above ceiling
        [100.0, -1e-13, 1.0],        # negative Deltav (polarity not modelled)
        [100.0, 1e-13, 0.01],        # tau too small
        [100.0, 1e-13, 1e5],         # tau too large
    ])
    def test_out_of_support_is_minus_inf(self, bad):
        assert jg.log_prior(jnp.array(bad), T_OBS) == -jnp.inf

    def test_batch(self, params):
        batch = jnp.stack([params, jnp.array([-1.0, 1e-13, 1.0])])
        lp = jg.log_prior(batch, T_OBS)
        assert lp.shape == (2,)
        assert jnp.isfinite(lp[0]) and lp[1] == -jnp.inf


class TestSamplingCoordinates:
    def test_round_trip(self, params):
        assert jnp.allclose(jg.to_physical(jg.to_sampling(params)), params, rtol=1e-14)

    def test_round_trip_batch(self):
        xi = jg.sample_prior(jr.PRNGKey(1), 100)
        assert jnp.allclose(jg.to_sampling(jg.to_physical(xi)), xi, rtol=1e-12)

    def test_sampling_bounds_are_log_of_physical(self):
        lo_p, up_p = jg.prior_bounds(T_OBS)
        lo_s, up_s = jg.sampling_bounds(T_OBS)
        assert jnp.allclose(lo_s, jg.to_sampling(lo_p))
        assert jnp.allclose(up_s, jg.to_sampling(up_p))

    def test_log_prior_sampling_flat_inside(self, params):
        xi = jg.to_sampling(params)
        assert jg.log_prior_sampling(xi, T_OBS) == 0.0
        outside = xi.at[2].set(jnp.log(_TAU_MAX) + 1.0)
        assert jg.log_prior_sampling(outside, T_OBS) == -jnp.inf


N_DRAWS = 20_000


@pytest.fixture(scope="module")
def draws():
    return jg.sample_prior(jr.PRNGKey(42), N_DRAWS, T_OBS)


class TestSamplePrior:
    N = N_DRAWS

    def test_shape_and_support(self, draws):
        assert draws.shape == (self.N, 3)
        lo, up = jg.sampling_bounds(T_OBS)
        assert jnp.all(draws >= lo) and jnp.all(draws <= up)

    def test_marginal_means(self, draws):
        """Each sampling coordinate is uniform on its box; the sample mean
        must land within ~5 sigma of the box centre."""
        lo, up = jg.sampling_bounds(T_OBS)
        centre = (lo + up) / 2
        sigma = (up - lo) / jnp.sqrt(12.0) / jnp.sqrt(self.N)
        assert jnp.all(jnp.abs(jnp.mean(draws, axis=0) - centre) < 5.0 * sigma)

    def test_consistent_with_inverse_cdf(self):
        u = jr.uniform(jr.PRNGKey(7), (50, 3))
        xi = jg.to_sampling(jg.prior_inverse_cdf(u, T_OBS))
        assert jnp.all(jnp.isfinite(xi))
        assert jnp.all(jg.log_prior_sampling(xi, T_OBS) == 0.0)


def test_default_t_obs_duplicated_constant():
    """priors.T_OBS_s duplicates data.T_OBS_s — keep them in sync (see
    review: consider a single shared constant)."""
    from jaxglitches.data import T_OBS_s as data_t_obs
    assert T_OBS_s == data_t_obs
