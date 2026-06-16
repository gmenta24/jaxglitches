"""jaxglitches — JAX-based LISA glitch waveforms and parameter-estimation utilities."""

from .priors import prior_inverse_cdf, log_prior, prior_bounds
from .data import freq_grid, clean_signal_f, clean_signal_t, DT_s, T_OBS_s, T_ARM_s
from .likelihood import inner_product, snr, log_likelihood, make_log_likelihood, log_posterior, fisher_matrix
