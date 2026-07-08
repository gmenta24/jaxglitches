"""Tests for the root-level noise.py: PSD relations, SNR invariance across
TDI generations, and frequency-domain noise sampling.

Note: noise.py sits at the repository root (added to sys.path by conftest),
not inside the jaxglitches package.
"""
import jax.numpy as jnp
import jax.random as jr
import pytest

import jaxglitches as jg
from jaxglitches.waveform import T_ARM_s, C_SI

import noise as ns


def test_psd_positive_and_finite(f_safe, psd1, psd2):
    for psd in (psd1, psd2):
        assert bool(jnp.all(jnp.isfinite(psd)))
        assert bool(jnp.all(psd[1:] > 0))


def test_array_matches_tuple_form(f_safe):
    S_A, S_E, S_T = ns.psd_tdi1(f_safe)
    arr = ns.psd_tdi1_array(f_safe)
    assert jnp.array_equal(arr, jnp.stack([S_A, S_E, S_T], axis=-1))
    assert jnp.array_equal(S_A, S_E)


def test_tdi2_is_transfer_squared_times_tdi1(f_safe, psd1, psd2):
    """S_TDI2 = |1 - D^4|^2 S_TDI1 = 4 sin^2(4 pi f T) S_TDI1."""
    T = 2.5e9 / C_SI
    factor = 4.0 * jnp.sin(4.0 * jnp.pi * f_safe * T) ** 2
    assert jnp.allclose(psd2, factor[:, None] * psd1, rtol=1e-12)


def test_snr_invariant_across_tdi_generations(params, freq, psd1, psd2):
    """The physical content of TDI-1 and TDI-2 is the same, so the optimal
    SNR must agree when each uses its own PSD."""
    h1 = jg.clean_signal_f(params, freq, tdi=1)
    h2 = jg.clean_signal_f(params, freq, tdi=2)
    snr1 = jg.snr(h1, psd1)
    snr2 = jg.snr(h2, psd2)
    assert jnp.isclose(snr1, snr2, rtol=1e-9)


def test_psd_shape_characterization(f_safe, psd1):
    """CHARACTERIZATION (see review): the A-channel PSD equals the standard
    fractional-frequency TDI-1 AET PSD (Cornish & Robson conventions,
    S_A = 8 sin^2 x [4(1+cx+cx^2) S_pm + (2+cx) S_op]) DIVIDED by
    16 x^2 sin^2 x, with x = 2 pi f T. Equivalently it is a strain-like,
    transfer-function-divided PSD, NOT the raw fractional-frequency TDI PSD
    that the clean_signal_* templates live in. Internally self-consistent
    (noise is sampled from the same PSD), but absolute SNRs are not physical
    LISA SNRs until this is reconciled. If this test fails, the PSD
    convention changed — update likelihood/SNR docs accordingly."""
    x = 2 * jnp.pi * f_safe * T_ARM_s
    A_amp, P_amp = 3e-15, 15e-12  # 3 fm/s^2, 15 pm (SciRD values)
    M_acc = (1 + (4e-4 / f_safe) ** 2) * (1 + (f_safe / 8e-3) ** 4)
    M_oms = 1 + (2e-3 / f_safe) ** 4
    S_pm_ff = A_amp ** 2 * M_acc / (2 * jnp.pi * f_safe * C_SI) ** 2
    S_op_ff = P_amp ** 2 * M_oms * (2 * jnp.pi * f_safe / C_SI) ** 2
    cx = jnp.cos(x)
    S_A_std = 8 * jnp.sin(x) ** 2 * (4 * (1 + cx + cx ** 2) * S_pm_ff
                                     + (2 + cx) * S_op_ff)
    sl = slice(10, None)
    ratio = psd1[sl, 0] * (16 * x[sl] ** 2 * jnp.sin(x[sl]) ** 2) / S_A_std[sl]
    finite = jnp.isfinite(ratio)  # sin x = 0 bins produce 0/0; skip them
    assert bool(jnp.all(jnp.abs(ratio[finite] - 1) < 1e-10))


class TestSampleNoise:
    def test_dc_bin_zero_and_shape(self, psd1):
        n = ns.sample_noise_fd(jr.PRNGKey(0), psd1)
        assert n.shape == psd1.shape
        assert jnp.all(n[0] == 0.0)
        assert n.dtype == jnp.complex128

    def test_variance_matches_psd(self, psd1):
        """E|n_k|^2 = S_k: whitened bin powers must average to 1."""
        n = ns.sample_noise_fd(jr.PRNGKey(3), psd1)
        white = jnp.abs(n[1:]) ** 2 / psd1[1:]
        m = white.size
        # mean of m unit-mean exponentials: std = 1/sqrt(m); allow 6 sigma
        assert abs(float(jnp.mean(white)) - 1.0) < 6.0 / m ** 0.5

    def test_noise_realizations_differ(self, psd1):
        a = ns.sample_noise_fd(jr.PRNGKey(0), psd1)
        b = ns.sample_noise_fd(jr.PRNGKey(1), psd1)
        # compare whitened draws: raw values are ~1e-21, so any absolute
        # tolerance would call different arrays "equal"
        diff = jnp.abs(a[1:] - b[1:]) / jnp.sqrt(psd1[1:])
        assert float(jnp.max(diff)) > 0.1
