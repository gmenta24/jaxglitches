"""Tests for jaxglitches.orbits: link ordering, equal-arm helper, and the
lisaorbits-backed light travel times."""
import numpy as np
import pytest

import jaxglitches as jg
from jaxglitches.waveform import T_ARM_s


def test_links_ordering():
    assert jg.LINKS == (12, 23, 31, 13, 32, 21)


def test_equal_arm_ltt():
    ltt = jg.equal_arm_ltt(T_ARM_s)
    assert ltt.shape == (6,)
    assert np.all(ltt == T_ARM_s)
    assert np.all(jg.equal_arm_ltt(5.0) == 5.0)


@pytest.fixture(scope="module")
def orbits():
    lisaorbits = pytest.importorskip("lisaorbits")
    return lisaorbits.KeplerianOrbits()


class TestLinkLtt:
    """Requires lisaorbits; skipped if not installed."""

    def test_scalar_epoch(self, orbits):
        ltt = jg.link_ltt(0.0, orbits)
        assert ltt.shape == (6,)
        # one-way travel times close to the nominal arm, unequal across links
        assert np.all((ltt > 8.0) & (ltt < 8.7))
        assert 1e-3 < np.ptp(ltt) < 0.1  # inter-link spread ~30 ms

    def test_array_epochs(self, orbits):
        t = np.array([0.0, 86400.0, 30 * 86400.0])
        ltt = jg.link_ltt(t, orbits)
        assert ltt.shape == (3, 6)
        assert np.all((ltt > 8.0) & (ltt < 8.7))

    def test_frozen_arm_drift_is_slow(self, orbits):
        """The frozen-arm approximation rests on |dL/dt| ~ 3e-8 s/s; a glitch
        lasting 1000 s must see the arms move by less than a microsecond."""
        ltt = jg.link_ltt(np.array([0.0, 1000.0]), orbits)
        assert np.max(np.abs(ltt[1] - ltt[0])) < 1e-4
