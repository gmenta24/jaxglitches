"""Gradient health checks — jaxglitches exists to be differentiated, so NaN
gradients are first-class bugs."""
import jax
import jax.numpy as jnp
import pytest

import jaxglitches as jg


def test_grad_clean_signal_f_finite(params, freq):
    for tdi in (1, 2):
        g = jax.grad(
            lambda p: jnp.sum(jnp.abs(jg.clean_signal_f(p, freq, tdi=tdi)) ** 2)
        )(params)
        assert bool(jnp.all(jnp.isfinite(g)))


def test_grad_unequal_loglike_finite(params, freq, psd1, unequal_ltt):
    data = jg.clean_signal_f_unequal(params, freq, unequal_ltt, tdi=1)
    log_L = jg.make_log_likelihood_unequal(1.3 * data, psd1, freq, unequal_ltt, tdi=1)
    g = jax.grad(log_L)(params)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_grad_log_prior_finite_inside_support(params):
    g = jax.grad(jg.log_prior)(params)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_grad_raw_glitch_t_finite_extreme_onset(times):
    """raw_glitch_t clamps its exponent before exp, so gradients stay finite
    even when (t0 - t)/tau is huge."""
    late = jnp.array([3000.0, 1.2e-13, 0.5])  # (t0 - 0)/tau = 6000 >> 709
    g = jax.grad(lambda p: jnp.sum(jg.raw_glitch_t(p, times) ** 2))(late)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_grad_clean_signal_t_unequal_finite_extreme_onset(times):
    """The unequal-arm time-domain path uses the guarded _psi and is safe."""
    ltt = jnp.asarray(jg.equal_arm_ltt())
    late = jnp.array([3000.0, 1.2e-13, 0.5])
    g = jax.grad(
        lambda p: jnp.sum(jg.clean_signal_t_unequal(p, times, ltt, tdi=1) ** 2)
    )(late)
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize("tdi", [1, 2])
def test_grad_clean_signal_t_finite_extreme_onset(times, tdi):
    """The equal-arm TD waveforms clamp their exponent before exp, so
    gradients stay finite even when (t0 - t)/tau is far past the float64
    overflow threshold (~709)."""
    late = jnp.array([3000.0, 1.2e-13, 0.5])  # (t0 - 0)/tau = 6000
    g = jax.grad(lambda p: jnp.sum(jg.clean_signal_t(p, times, tdi=tdi) ** 2))(late)
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize("fn_name", ["tdi1_2exp_glitch", "tdi2_2exp_glitch"])
def test_grad_2exp_td_finite_extreme_onset_and_equal_taus(times, fn_name):
    """Two-exponential TD templates: gradients must stay finite both far
    before onset (clamped exp) and at the tau1 == tau2 singular point
    (continuous extension with a NaN-free unselected branch)."""
    from jaxglitches import waveform
    fn = getattr(waveform, fn_name)

    def loss(theta):
        X, Y, _ = fn(times, theta[0], theta[1], theta[2], theta[3])
        return jnp.sum(X ** 2 + Y ** 2)

    for theta in (jnp.array([3000.0, 1.2e-13, 0.5, 0.7]),   # extreme onset
                  jnp.array([400.0, 1.2e-13, 5.0, 5.0])):   # equal taus
        val = loss(theta)
        g = jax.grad(loss)(theta)
        assert bool(jnp.isfinite(val))
        assert bool(jnp.all(jnp.isfinite(g)))
