"""jaxglitches — JAX-based LISA glitch waveforms and parameter-estimation utilities."""

from .constants import C_SI, T_ARM_s, DT_s, T_OBS_s, ARM_LENGTH_m, YRSID_SI
from .priors import prior_inverse_cdf, log_prior, prior_bounds
from .data import freq_grid, clean_signal
from .likelihood import inner_product, snr, log_likelihood, make_log_likelihood, log_posterior
