"""Shared fixtures for the jaxglitches test suite.

The suite uses a shorter grid than the package defaults (T_OBS = 1800 s,
dt = 0.5 s) to keep runtimes low; every consistency check here is
grid-independent.
"""
import sys
from pathlib import Path

import pytest

# noise.py lives at the repository root, outside the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp  # noqa: E402  (jaxglitches enables x64 on import)
import jaxglitches as jg  # noqa: E402

T_OBS = 1800.0
DT = 0.5

# Truth used across tests: knee 1/(2*pi*tau) ~ 0.2 Hz is inside the band and
# t0 leaves room for the glitch to settle well before the window ends.
T0_TRUE = 400.0
DELTAV_TRUE = 1.2e-13
TAU_TRUE = 0.79


@pytest.fixture(scope="session")
def freq():
    return jg.freq_grid(T_OBS, DT)


@pytest.fixture(scope="session")
def f_safe(freq):
    return jnp.where(freq > 0, freq, 1.0)


@pytest.fixture(scope="session")
def times():
    return jnp.arange(0.0, T_OBS, DT)


@pytest.fixture(scope="session")
def params():
    return jnp.array([T0_TRUE, DELTAV_TRUE, TAU_TRUE])


@pytest.fixture(scope="session")
def unequal_ltt():
    """Realistic frozen light travel times (values from lisaorbits
    KeplerianOrbits at t=0), hardcoded so the tests don't need lisaorbits."""
    return jnp.array([8.3324, 8.3028, 8.3324, 8.3316, 8.3045, 8.3316])


@pytest.fixture(scope="session")
def psd1(f_safe):
    import noise as ns
    return ns.psd_tdi1_array(f_safe)


@pytest.fixture(scope="session")
def psd2(f_safe):
    import noise as ns
    return ns.psd_tdi2_array(f_safe)
