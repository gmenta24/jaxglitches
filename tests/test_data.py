"""Tests for jaxglitches.data: signal builders, TDI application, and the
equal-arm reduction of the unequal-arm code paths."""
import jax.numpy as jnp
import pytest

import jaxglitches as jg
from jaxglitches.data import DT_s, T_OBS_s
from jaxglitches.waveform import AET, T_ARM_s

from conftest import DT, T_OBS


def test_freq_grid_matches_rfftfreq():
    n = int(T_OBS / DT)
    assert jnp.array_equal(jg.freq_grid(T_OBS, DT), jnp.fft.rfftfreq(n, DT))
    assert jg.freq_grid(T_OBS, DT).shape == (n // 2 + 1,)


def test_module_constants():
    assert DT_s == 0.25
    assert T_OBS_s == 3600.0
    assert jnp.isclose(T_ARM_s, 2.5e9 / 299792458.0)


@pytest.mark.parametrize("tdi", [1, 2])
class TestCleanSignal:
    def test_shapes_and_dtypes(self, params, freq, times, tdi):
        h_fd = jg.clean_signal_f(params, freq, tdi=tdi)
        h_td = jg.clean_signal_t(params, times, tdi=tdi)
        assert h_fd.shape == (freq.shape[0], 3)
        assert h_td.shape == (times.shape[0], 3)
        assert h_fd.dtype == jnp.complex128  # x64 must be enabled
        assert h_td.dtype == jnp.float64

    def test_dc_bin_is_zero(self, params, freq, tdi):
        h_fd = jg.clean_signal_f(params, freq, tdi=tdi)
        assert jnp.all(h_fd[0] == 0.0)

    def test_z_channel_zero_in_xyz(self, params, freq, times, tdi):
        h_fd = jg.clean_signal_f(params, freq, tdi=tdi, basis="XYZ")
        h_td = jg.clean_signal_t(params, times, tdi=tdi, basis="XYZ")
        assert jnp.all(h_fd[:, 2] == 0.0)
        assert jnp.all(h_td[:, 2] == 0.0)

    def test_aet_basis_is_aet_of_xyz(self, params, freq, tdi):
        xyz = jg.clean_signal_f(params, freq, tdi=tdi, basis="XYZ")
        aet = jg.clean_signal_f(params, freq, tdi=tdi, basis="AET")
        A, E, T = AET(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        expected = jnp.stack([A, E, T], axis=-1).at[0].set(0.0)
        assert jnp.allclose(aet, expected, atol=0.0, rtol=1e-14)

    def test_compute_tdi_reproduces_clean_signal_f(self, params, freq, tdi):
        """Documented contract: compute_TDI(raw_glitch_f(...)) equals
        clean_signal_f(...) to machine precision."""
        raw = jg.raw_glitch_f(params, freq)
        a = jg.compute_TDI(raw, freq, T_ARM_s, tdi=tdi)
        b = jg.clean_signal_f(params, freq, T=T_ARM_s, tdi=tdi)
        scale = jnp.max(jnp.abs(b))
        assert jnp.max(jnp.abs(a - b)) < 1e-13 * scale

    def test_signal_linear_in_deltav(self, params, freq, tdi):
        """The template is exactly linear in Deltav (sign flip included).
        Compared against the overall signal scale: the AET projection has
        cancelling bins whose per-element relative error is unbounded."""
        h = jg.clean_signal_f(params, freq, tdi=tdi)
        p2 = params.at[1].multiply(-3.0)
        h2 = jg.clean_signal_f(p2, freq, tdi=tdi)
        scale = jnp.max(jnp.abs(h))
        assert jnp.max(jnp.abs(h2 + 3.0 * h)) < 1e-12 * scale


def test_raw_glitch_f_dc_is_zero(params, freq):
    raw = jg.raw_glitch_f(params, freq)
    assert raw[0] == 0.0
    assert raw.shape == freq.shape


def test_raw_glitch_t_step_shape(times, params):
    """Raw glitch: zero before onset, asymptotes to -Deltav/c after."""
    raw = jg.raw_glitch_t(params, times)
    t0, dv = params[0], params[1]
    assert jnp.all(raw[times < t0] == 0.0)
    c = 299792458.0
    assert jnp.isclose(raw[-1], -dv / c, rtol=1e-6)


def test_invalid_tdi_and_basis_raise(params, freq):
    with pytest.raises(ValueError):
        jg.clean_signal_f(params, freq, tdi=3)
    with pytest.raises(ValueError):
        jg.clean_signal_f(params, freq, basis="XYW")
    with pytest.raises(ValueError):
        jg.compute_TDI(jg.raw_glitch_f(params, freq), freq, tdi=0)


# ── Unequal-arm code paths ─────────────────────────────────────────────────────

@pytest.mark.parametrize("tdi", [1, 2])
class TestUnequalArm:
    def test_equal_arm_reduction_fd(self, params, freq, tdi):
        """With all six delays equal to T, the unequal-arm frequency-domain
        signal must reproduce the equal-arm one exactly."""
        ltt = jnp.asarray(jg.equal_arm_ltt(T_ARM_s))
        a = jg.clean_signal_f_unequal(params, freq, ltt, tdi=tdi)
        b = jg.clean_signal_f(params, freq, T=T_ARM_s, tdi=tdi)
        scale = jnp.max(jnp.abs(b))
        assert jnp.max(jnp.abs(a - b)) < 1e-13 * scale

    def test_equal_arm_reduction_td(self, params, times, tdi):
        ltt = jnp.asarray(jg.equal_arm_ltt(T_ARM_s))
        a = jg.clean_signal_t_unequal(params, times, ltt, tdi=tdi)
        b = jg.clean_signal_t(params, times, T=T_ARM_s, tdi=tdi)
        scale = jnp.max(jnp.abs(b))
        assert jnp.max(jnp.abs(a - b)) < 1e-13 * scale

    def test_compute_tdi_unequal_contract(self, params, freq, unequal_ltt, tdi):
        raw = jg.raw_glitch_f(params, freq)
        a = jg.compute_TDI_unequal(raw, freq, unequal_ltt, tdi=tdi)
        b = jg.clean_signal_f_unequal(params, freq, unequal_ltt, tdi=tdi)
        scale = jnp.max(jnp.abs(b))
        assert jnp.max(jnp.abs(a - b)) < 1e-13 * scale

    def test_fd_matches_fft_of_td(self, params, freq, times, unequal_ltt, tdi):
        """The pulse-sum time-domain construction must be the inverse FT of
        the frequency-domain transfer-function construction."""
        h_fd = jg.clean_signal_f_unequal(params, freq, unequal_ltt, tdi=tdi)
        h_td = jg.clean_signal_t_unequal(params, times, unequal_ltt, tdi=tdi)
        num = jnp.fft.rfft(h_td, axis=0) * DT
        scale = jnp.max(jnp.abs(h_fd[1:]))
        assert jnp.max(jnp.abs(num[1:] - h_fd[1:])) / scale < 2e-3

    def test_dc_bin_is_zero(self, params, freq, unequal_ltt, tdi):
        h_fd = jg.clean_signal_f_unequal(params, freq, unequal_ltt, tdi=tdi)
        assert jnp.all(h_fd[0] == 0.0)

    def test_differs_from_equal_arm(self, params, freq, unequal_ltt, tdi):
        """Sanity: realistic unequal arms produce a measurably different
        signal from the equal-arm approximation."""
        a = jg.clean_signal_f_unequal(params, freq, unequal_ltt, tdi=tdi)
        b = jg.clean_signal_f(params, freq, T=T_ARM_s, tdi=tdi)
        scale = jnp.max(jnp.abs(b))
        assert jnp.max(jnp.abs(a - b)) > 1e-3 * scale
