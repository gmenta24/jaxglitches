"""Tests for jaxglitches.waveform: analytic FD vs TD consistency, template
family consistency, and agreement with the lisaglitch reference model."""
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglitches.waveform import (
    AET, C_SI, T_ARM_s,
    tdi1_1exp_glitch, tdi2_1exp_glitch,
    tdi1_2exp_glitch, tdi2_2exp_glitch,
    tdi1_1exp_f_glitch, tdi2_1exp_f_glitch,
    tdi1_2exp_f_glitch, tdi2_2exp_f_glitch,
)

from conftest import DT, T0_TRUE, DELTAV_TRUE, TAU_TRUE


class TestAET:
    def test_matrix(self):
        X, Y, Z = jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)
        A, E, T = AET(X, Y, Z)
        assert jnp.allclose(A, (Z - X) / jnp.sqrt(2.0))
        assert jnp.allclose(E, (X - 2 * Y + Z) / jnp.sqrt(6.0))
        assert jnp.allclose(T, (X + Y + Z) / jnp.sqrt(3.0))

    def test_orthonormal(self):
        """The AET map is orthonormal: it preserves the channel-space norm."""
        rng = np.random.default_rng(0)
        X, Y, Z = (jnp.asarray(rng.normal(size=64) + 1j * rng.normal(size=64))
                   for _ in range(3))
        A, E, T = AET(X, Y, Z)
        norm_xyz = jnp.sum(jnp.abs(X) ** 2 + jnp.abs(Y) ** 2 + jnp.abs(Z) ** 2)
        norm_aet = jnp.sum(jnp.abs(A) ** 2 + jnp.abs(E) ** 2 + jnp.abs(T) ** 2)
        assert jnp.allclose(norm_xyz, norm_aet, rtol=1e-12)


@pytest.mark.parametrize("tdi,f_td,f_fd", [
    (1, tdi1_1exp_glitch, tdi1_1exp_f_glitch),
    (2, tdi2_1exp_glitch, tdi2_1exp_f_glitch),
])
def test_fd_matches_fft_of_td_1exp(tdi, f_td, f_fd, times, freq, f_safe):
    """The analytic Fourier transform must match dt * rfft of the time-domain
    waveform (up to discretization/leakage errors)."""
    X_t, Y_t, _ = f_td(times, T0_TRUE, DELTAV_TRUE, TAU_TRUE)
    X_f, Y_f, _ = f_fd(f_safe, T0_TRUE, DELTAV_TRUE, TAU_TRUE)
    for td, fd in [(X_t, X_f), (Y_t, Y_f)]:
        num = jnp.fft.rfft(td) * DT
        err = jnp.max(jnp.abs(num[1:] - fd[1:])) / jnp.max(jnp.abs(fd[1:]))
        assert err < 2e-3  # discretization/leakage floor at dt = 0.5 s


@pytest.mark.parametrize("tdi,f_td,f_fd", [
    (1, tdi1_2exp_glitch, tdi1_2exp_f_glitch),
    (2, tdi2_2exp_glitch, tdi2_2exp_f_glitch),
])
def test_fd_matches_fft_of_td_2exp(tdi, f_td, f_fd, times, freq, f_safe):
    """Internal consistency of each two-exponential pair (TD vs FD)."""
    tau1, tau2 = 5.0, 8.0
    X_t, Y_t, _ = f_td(times, T0_TRUE, DELTAV_TRUE, tau1, tau2)
    X_f, Y_f, _ = f_fd(f_safe, T0_TRUE, DELTAV_TRUE, tau1, tau2)
    for td, fd in [(X_t, X_f), (Y_t, Y_f)]:
        num = jnp.fft.rfft(td) * DT
        err = jnp.max(jnp.abs(num[1:] - fd[1:])) / jnp.max(jnp.abs(fd[1:]))
        assert err < 2e-3  # discretization/leakage floor at dt = 0.5 s


class TestTwoExpReducesToOneExp:
    """In the limit tau2 -> tau1 the two-exponential glitch must reduce to
    the one-exponential glitch with tau = tau1 (same polarity!)."""

    TAU1 = TAU_TRUE
    TAU2 = TAU_TRUE * (1.0 + 1e-7)

    def _check(self, one, two):
        for a, b in zip(one[:2], two[:2]):  # X and Y channels
            scale = jnp.max(jnp.abs(a))
            assert jnp.max(jnp.abs(a - b)) / scale < 1e-5

    def test_tdi1_time_domain(self, times):
        one = tdi1_1exp_glitch(times, T0_TRUE, DELTAV_TRUE, self.TAU1)
        two = tdi1_2exp_glitch(times, T0_TRUE, DELTAV_TRUE, self.TAU1, self.TAU2)
        self._check(one, two)

    def test_tdi1_freq_domain(self, f_safe):
        one = tdi1_1exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, self.TAU1)
        two = tdi1_2exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, self.TAU1, self.TAU2)
        self._check(one, two)

    def test_tdi2_time_domain(self, times):
        one = tdi2_1exp_glitch(times, T0_TRUE, DELTAV_TRUE, self.TAU1)
        two = tdi2_2exp_glitch(times, T0_TRUE, DELTAV_TRUE, self.TAU1, self.TAU2)
        self._check(one, two)

    def test_tdi2_freq_domain(self, f_safe):
        one = tdi2_1exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, self.TAU1)
        two = tdi2_2exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, self.TAU1, self.TAU2)
        self._check(one, two)


class TestTdi2IsOneMinusD4TimesTdi1:
    """For any input signal, TDI-2 = (1 - D^4) * TDI-1 in the frequency
    domain. This pins the relative sign between generations per family."""

    def _factor(self, f):
        return 1.0 - jnp.exp(-1j * 2.0 * jnp.pi * f * 4.0 * T_ARM_s)

    def test_1exp_family(self, f_safe):
        X1, Y1, _ = tdi1_1exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, TAU_TRUE)
        X2, Y2, _ = tdi2_1exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, TAU_TRUE)
        fac = self._factor(f_safe)
        assert jnp.max(jnp.abs(X2 - fac * X1)) / jnp.max(jnp.abs(X2)) < 1e-12
        assert jnp.max(jnp.abs(Y2 - fac * Y1)) / jnp.max(jnp.abs(Y2)) < 1e-12

    def test_2exp_family(self, f_safe):
        X1, Y1, _ = tdi1_2exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, 5.0, 8.0)
        X2, Y2, _ = tdi2_2exp_f_glitch(f_safe, T0_TRUE, DELTAV_TRUE, 5.0, 8.0)
        fac = self._factor(f_safe)
        assert jnp.max(jnp.abs(X2 - fac * X1)) / jnp.max(jnp.abs(X2)) < 1e-12


@pytest.mark.parametrize("f_td", [tdi1_2exp_glitch, tdi2_2exp_glitch])
def test_2exp_equal_taus_continuous_extension(f_td, times):
    """tau1 == tau2 uses the continuous extension (as in lisaglitch): the
    result must be finite and continuous with nearly-equal taus."""
    X_eq, Y_eq, _ = f_td(times, T0_TRUE, DELTAV_TRUE, 5.0, 5.0)
    assert bool(jnp.all(jnp.isfinite(X_eq))) and bool(jnp.all(jnp.isfinite(Y_eq)))
    X_near, Y_near, _ = f_td(times, T0_TRUE, DELTAV_TRUE, 5.0, 5.0 * (1 + 1e-6))
    for eq, near in [(X_eq, X_near), (Y_eq, Y_near)]:
        scale = jnp.max(jnp.abs(eq))
        assert jnp.max(jnp.abs(eq - near)) < 1e-4 * scale


def test_raw_glitch_matches_lisaglitch_up_to_polarity(times, params):
    """raw_glitch_t (fractional frequency) times c equals MINUS the
    lisaglitch IntegratedShapeletGlitch signal (m/s) — the package uses the
    opposite polarity convention to lisaglitch. If this test starts failing
    the convention changed; update the catalog/docs accordingly."""
    lisaglitch = pytest.importorskip("lisaglitch")
    import jaxglitches as jg

    gl = lisaglitch.IntegratedShapeletGlitch(
        level=DELTAV_TRUE, beta=TAU_TRUE, inj_point="tm_12", t_inj=T0_TRUE,
    )
    ref = gl.compute_signal(np.asarray(times))
    ours = np.asarray(jg.raw_glitch_t(params, times)) * C_SI
    np.testing.assert_allclose(ours, -ref, atol=1e-25 * DELTAV_TRUE / 1e-13)
