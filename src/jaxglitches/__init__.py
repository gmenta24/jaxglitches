"""jaxglitches — JAX-based LISA glitch waveforms and parameter-estimation utilities."""

from .priors import (
    prior_inverse_cdf, log_prior, prior_bounds,
    to_sampling, to_physical, sampling_bounds, log_prior_sampling, sample_prior,
)
from .data import freq_grid, clean_signal_f, clean_signal_t, DT_s, T_OBS_s, T_ARM_s
from .likelihood import inner_product, snr, log_likelihood, make_log_likelihood, log_posterior, fisher_matrix
from .catalog_generator import (
    run_catalog, expected_count, RATE_ORDINARY, RATE_COLD, INJ_POINTS,
)
