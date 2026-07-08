"""Tests for jaxglitches.likelihood: inner product algebra, likelihood
factories, and Fisher matrices."""
import jax
import jax.numpy as jnp
import pytest

import jaxglitches as jg
from jaxglitches.waveform import T_ARM_s


@pytest.fixture(scope="module")
def h1(params, freq):
    return jg.clean_signal_f(params, freq, tdi=1)


def test_inner_product_definition(h1, psd1):
    expected = 2.0 * jnp.real(jnp.sum(jnp.conj(h1[1:]) * h1[1:] / psd1[1:]))
    assert jnp.isclose(jg.inner_product(h1, h1, psd1), expected, rtol=1e-14)


def test_inner_product_excludes_dc(h1, psd1):
    """Whatever sits in the DC bin must not contribute."""
    spiked = h1.at[0].set(1e6 + 0.0j)
    assert jnp.isclose(
        jg.inner_product(spiked, spiked, psd1),
        jg.inner_product(h1, h1, psd1),
        rtol=1e-14,
    )


def test_snr_is_sqrt_of_self_inner_product(h1, psd1):
    assert jnp.isclose(jg.snr(h1, psd1) ** 2,
                       jg.inner_product(h1, h1, psd1), rtol=1e-13)


def test_log_likelihood_is_minus_half_residual_norm(h1, psd1):
    """log L = -sum |r|^2 / S = -(1/2) (r|r) — the two public functions must
    agree on the noise convention."""
    data = 1.1 * h1  # nonzero residual
    r = data - h1
    assert jnp.isclose(
        jg.log_likelihood(data, h1, psd1),
        -0.5 * jg.inner_product(r, r, psd1),
        rtol=1e-13,
    )


def test_noiseless_loglike_peaks_at_truth(params, freq, psd1, h1):
    log_L = jg.make_log_likelihood(h1, psd1, freq, tdi=1)
    assert jnp.isclose(log_L(params), 0.0, atol=1e-20)
    for i, rel in [(0, 1e-3), (1, 1e-3), (2, 1e-3)]:
        perturbed = params.at[i].multiply(1.0 + rel)
        assert log_L(perturbed) < log_L(params)


def test_make_log_likelihood_matches_direct(params, freq, psd1, h1):
    data = h1 * (1.0 + 0.05j)
    log_L = jg.make_log_likelihood(data, psd1, freq, T=T_ARM_s, tdi=1)
    direct = jg.log_likelihood(data, jg.clean_signal_f(params, freq, tdi=1), psd1)
    assert jnp.isclose(log_L(params), direct, rtol=1e-14)


def test_make_log_likelihood_unequal_equal_arm_reduction(params, freq, psd1, h1):
    ltt = jnp.asarray(jg.equal_arm_ltt(T_ARM_s))
    data = 0.9 * h1
    a = jg.make_log_likelihood_unequal(data, psd1, freq, ltt, tdi=1)(params)
    b = jg.make_log_likelihood(data, psd1, freq, T=T_ARM_s, tdi=1)(params)
    assert jnp.isclose(a, b, rtol=1e-12)


def test_log_posterior_is_loglike_plus_logprior(params, freq, psd1, h1):
    data = 1.05 * h1
    lp = jg.log_posterior(data, h1, psd1, params, t_obs=3600.0)
    expected = jg.log_likelihood(data, h1, psd1) + jg.log_prior(params, 3600.0)
    assert jnp.isclose(lp, expected, rtol=1e-14)


def _fisher_close(a, b, tol):
    """Compare Fisher matrices entrywise against the natural per-entry scale
    sqrt(Gamma_ii * Gamma_jj). The (t0, Deltav) entry is analytically zero
    (it is 2 Re[i * positive]) and carries catastrophic cancellation, so a
    plain per-element rtol is meaningless there."""
    scale = jnp.sqrt(jnp.outer(jnp.diag(b), jnp.diag(b)))
    return bool(jnp.all(jnp.abs(a - b) <= tol * scale))


class TestFisher:
    def test_symmetric_positive_definite(self, params, freq, psd1):
        gamma = jg.fisher_matrix(params, freq, psd1, tdi=1)
        assert gamma.shape == (3, 3)
        assert _fisher_close(gamma, gamma.T, 1e-12)
        # scale to O(1) before checking eigenvalues (parameters span ~26
        # orders of magnitude, so raw eigenvalues are meaningless)
        d = jnp.sqrt(jnp.diag(gamma))
        corr = gamma / jnp.outer(d, d)
        assert jnp.all(jnp.linalg.eigvalsh((corr + corr.T) / 2) > 0)

    def test_equals_minus_hessian_near_truth(self, params, freq, psd1):
        """With a small nonzero residual, -hessian(log L) approaches the
        Fisher matrix (the residual curvature term is O(|r|/|h|))."""
        h_true = jg.clean_signal_f(params, freq, tdi=1)
        log_L = jg.make_log_likelihood(h_true * (1.0 + 1e-6), psd1, freq, tdi=1)
        H = jax.hessian(log_L)(params)
        gamma = jg.fisher_matrix(params, freq, psd1, tdi=1)
        assert _fisher_close(-H, gamma, 1e-4)

    def test_equals_minus_hessian_at_zero_residual(self, params, freq, psd1):
        """For a Gaussian likelihood at exactly zero residual, the Hessian of
        log L must equal minus the Fisher matrix. Requires log_likelihood to
        avoid jnp.abs(r)**2, whose derivative is undefined at r = 0."""
        h_true = jg.clean_signal_f(params, freq, tdi=1)
        log_L = jg.make_log_likelihood(h_true, psd1, freq, tdi=1)
        H = jax.hessian(log_L)(params)
        gamma = jg.fisher_matrix(params, freq, psd1, tdi=1)
        assert _fisher_close(-H, gamma, 1e-8)

    def test_unequal_equal_arm_reduction(self, params, freq, psd1):
        ltt = jnp.asarray(jg.equal_arm_ltt(T_ARM_s))
        a = jg.fisher_matrix_unequal(params, freq, psd1, ltt, tdi=1)
        b = jg.fisher_matrix(params, freq, psd1, tdi=1)
        assert _fisher_close(a, b, 1e-10)

    def test_scales_with_deltav_squared(self, params, freq, psd1):
        """h is linear in Deltav, so the t0/tau block of the Fisher matrix
        scales as Deltav^2."""
        g1 = jg.fisher_matrix(params, freq, psd1, tdi=1)
        g2 = jg.fisher_matrix(params.at[1].multiply(2.0), freq, psd1, tdi=1)
        assert jnp.isclose(g2[0, 0], 4.0 * g1[0, 0], rtol=1e-10)
        assert jnp.isclose(g2[2, 2], 4.0 * g1[2, 2], rtol=1e-10)
        # the (Deltav, Deltav) entry is independent of Deltav
        assert jnp.isclose(g2[1, 1], g1[1, 1], rtol=1e-10)


def test_grad_of_loglike_is_finite(params, freq, psd1, h1):
    data = 1.2 * h1
    log_L = jg.make_log_likelihood(data, psd1, freq, tdi=1)
    g = jax.grad(log_L)(params)
    assert bool(jnp.all(jnp.isfinite(g)))
