"""Tests for jaxglitches.catalog_generator: rates, population resampling,
and catalogue structure."""
import re
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import jaxglitches as jg
from jaxglitches import catalog_generator as cg
from jaxglitches import priors

DATA_DIR = Path(cg.__file__).parent / "lpf_data"


def test_rate_ordinary_matches_lisaglitch_estimator():
    """RATE_ORDINARY must equal 1 / mean(interarrival), the estimator used by
    lisaglitch's estimate_poisson_beta."""
    intervals = np.loadtxt(DATA_DIR / "2021-09-17-intervals_ordinary.txt",
                           comments="#", usecols=0)
    assert np.isclose(cg.RATE_ORDINARY, 1.0 / np.mean(intervals), rtol=1e-12)
    # ~4.3 events/day (Baghi et al. 2022 ordinary-run rate)
    assert 3.0 < cg.RATE_ORDINARY * 86400 < 6.0
    assert np.isclose(cg.RATE_COLD, 10.0 * cg.RATE_ORDINARY)


def test_expected_count():
    assert np.isclose(jg.expected_count(1e5), cg.RATE_ORDINARY * 1e5)
    assert np.isclose(jg.expected_count(1e5, rate=1e-3), 100.0)
    with pytest.raises(ValueError):
        jg.expected_count(1e5, run_type="bogus")


def test_inj_points_follow_lisaorbits_link_order():
    assert cg.INJ_POINTS == ("tm_12", "tm_23", "tm_31", "tm_13", "tm_32", "tm_21")
    assert tuple(int(p.split("_")[1]) for p in cg.INJ_POINTS) == jg.LINKS


class TestSampleParameters:
    def test_within_prior_bounds(self):
        dv, tau = cg.sample_parameters(jr.PRNGKey(0), 500)
        lower, upper = priors.prior_bounds()
        assert jnp.all(tau >= lower[2]) and jnp.all(tau <= upper[2])
        assert jnp.all(jnp.abs(dv) >= lower[1]) and jnp.all(jnp.abs(dv) <= upper[1])

    def test_signed_produces_both_polarities(self):
        dv, _ = cg.sample_parameters(jr.PRNGKey(1), 200, signed=True)
        assert jnp.any(dv > 0) and jnp.any(dv < 0)

    def test_unsigned_all_positive(self):
        dv, _ = cg.sample_parameters(jr.PRNGKey(2), 200, signed=False)
        assert jnp.all(dv > 0)

    def test_bootstrap_draws_catalog_values(self):
        """With smooth=False every sample must be an (appropriately clipped)
        row of the empirical LPF catalogue."""
        dv, tau = cg.sample_parameters(jr.PRNGKey(3), 300, smooth=False, signed=False)
        beta, level = np.loadtxt(
            DATA_DIR / "2021-09-17-effective_glitch_parameters_ordinary.txt",
            comments="#", usecols=(0, 1), unpack=True)
        lower, upper = priors.prior_bounds()
        cat_tau = np.clip(beta, lower[2], upper[2])
        cat_dv = np.clip(np.abs(level), lower[1], upper[1])
        cat = np.stack([np.log(cat_tau), np.log(cat_dv)], axis=1)
        pts = np.stack([np.log(np.asarray(tau)), np.log(np.asarray(dv))], axis=1)
        d = np.min(np.linalg.norm(pts[:, None, :] - cat[None, :, :], axis=-1), axis=1)
        assert np.max(d) < 1e-9

    def test_smooth_differs_from_bootstrap(self):
        dv_s, tau_s = cg.sample_parameters(jr.PRNGKey(4), 100, smooth=True, signed=False)
        beta, _ = np.loadtxt(
            DATA_DIR / "2021-09-17-effective_glitch_parameters_ordinary.txt",
            comments="#", usecols=(0, 1), unpack=True)
        # KDE-smoothed taus are (generically) not exact catalogue values
        assert not np.all(np.isin(np.asarray(tau_s), beta))

    def test_bad_run_type_raises(self):
        with pytest.raises(ValueError):
            cg.sample_parameters(jr.PRNGKey(0), 10, run_type="bogus")


CAT_T_OBS = 1e5
CAT_RATE = 1e-3  # ~100 expected events; keeps the test fast and populated


@pytest.fixture(scope="module")
def cat():
    return jg.run_catalog(CAT_T_OBS, key=jr.PRNGKey(0), rate=CAT_RATE)


class TestRunCatalog:
    T_OBS = CAT_T_OBS
    RATE = CAT_RATE

    def test_reproducible(self, cat):
        again = jg.run_catalog(self.T_OBS, key=jr.PRNGKey(0), rate=self.RATE)
        assert again["n_glitches"] == cat["n_glitches"]
        assert jnp.array_equal(again["params"], cat["params"])

    def test_default_key_is_fixed(self):
        a = jg.run_catalog(self.T_OBS, rate=self.RATE)
        b = jg.run_catalog(self.T_OBS, rate=self.RATE)
        assert jnp.array_equal(a["params"], b["params"])

    def test_structure_and_shapes(self, cat):
        n = cat["n_glitches"]
        assert n > 0
        for k in ("t0", "Deltav", "tau", "inj_idx"):
            assert cat[k].shape == (n,)
        assert cat["params"].shape == (n, 3)
        assert cat["xi"].shape == (n, 3)
        assert len(cat["inj_point"]) == n
        assert cat["rate"] == self.RATE
        assert cat["t_obs"] == self.T_OBS

    def test_count_is_plausible(self, cat):
        """n ~ Poisson(100): a fixed-seed draw must lie within 6 sigma."""
        mu = self.RATE * self.T_OBS
        assert abs(cat["n_glitches"] - mu) < 6 * np.sqrt(mu)

    def test_t0_sorted_and_in_window(self, cat):
        t0 = cat["t0"]
        assert jnp.all(jnp.diff(t0) >= 0)
        assert jnp.all((t0 >= 0) & (t0 <= self.T_OBS))

    def test_params_and_xi_consistent(self, cat):
        assert jnp.array_equal(cat["params"][:, 0], cat["t0"])
        assert jnp.array_equal(cat["params"][:, 1], cat["Deltav"])
        assert jnp.array_equal(cat["params"][:, 2], cat["tau"])
        assert jnp.allclose(cat["xi"][:, 1], jnp.log(jnp.abs(cat["Deltav"])))
        assert jnp.allclose(cat["xi"][:, 2], jnp.log(cat["tau"]))

    def test_injection_points(self, cat):
        assert jnp.all((cat["inj_idx"] >= 0) & (cat["inj_idx"] < len(cg.INJ_POINTS)))
        assert all(re.fullmatch(r"tm_\d\d", p) for p in cat["inj_point"])
        assert all(cg.INJ_POINTS[i] == p
                   for i, p in zip(cat["inj_idx"], cat["inj_point"]))

    def test_both_polarities_present(self, cat):
        assert jnp.any(cat["Deltav"] > 0) and jnp.any(cat["Deltav"] < 0)

    def test_empty_catalog(self):
        cat = jg.run_catalog(10.0, key=jr.PRNGKey(0), rate=1e-9)
        assert cat["n_glitches"] == 0
        assert cat["params"].shape == (0, 3)
        assert cat["inj_point"] == []
