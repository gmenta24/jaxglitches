"""Generate the one data stream that both analyses consume.

Frequency-domain (`glitch_and_gb.ipynb`) and time--frequency
(`glitch_and_gb_wdm.ipynb`) inference must see *identical* data -- same grid, same
injections, same noise draw -- otherwise their posteriors sit at different
noise-shifted maxima and only their widths can be compared, not their centres.

Run this once; both notebooks then load `dataset.npz`, which is written next to
this script (`notebooks/glitch_GB/dataset.npz`).

The sample count is chosen divisible by 3072 so that the two WDM tilings used in the
time--frequency analysis (Nf = 128 and Nf = 1536) divide it exactly. Truncating to fit
a tiling would break periodicity and leak the Galactic binary's line across the band.
"""
import os
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import lisaorbits
from jaxgb import jaxgb as jgb_mod

import jaxglitches as jg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # notebooks/, where noise.py lives
import noise as ns  # noqa: E402

# -- grid --------------------------------------------------------------------
DT     = 1.0 / (2.0 * 3e-3)        # Nyquist at 3 mHz -> 166.667 s
N      = 187392                    # = 61 x 3072; divides both WDM tilings exactly
T_OBS  = N * DT
T_ARM  = jg.T_ARM_s

freq   = jnp.asarray(np.fft.rfftfreq(N, DT))
f_safe = jnp.where(freq > 0, freq, 1.0)
n_fine = len(freq)

psd1 = ns.psd_tdi1_array(f_safe, t_obs=T_OBS)     # per-bin variance
psd2 = ns.psd_tdi2_array(f_safe, t_obs=T_OBS)

# -- injections --------------------------------------------------------------
F0, FDOT, A_GB, PSI = 2.0e-3, 5.0e-17, 1.0e-21, float(jnp.pi / 4)
RA, DEC, IOTA, PHI0 = 1.0, -0.5, 1.0, 0.0
T0, TAU, DELTAV     = 400.0, 300.0, 1.0e-11

gb_true     = jnp.array([[F0, FDOT, A_GB, RA, DEC, PSI, IOTA, PHI0]])
glitch_true = jnp.array([T0, DELTAV, TAU])
N_GB        = 256

model  = jgb_mod.JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=float(T_OBS),
                       t0=0.0, n=N_GB)
k_min  = int(model.get_kmin(gb_true[:, 0])[0])
segs   = model.get_tdi(gb_true, tdi_generation=1.5, tdi_combination='AET')
seg    = jnp.stack(segs, axis=0).astype(jnp.complex128)[:, 0, :].T
h_gb   = jnp.zeros((n_fine, 3), dtype=jnp.complex128).at[k_min + jnp.arange(N_GB)].set(seg)
h_gl   = jg.clean_signal_f(glitch_true, freq, tdi=1).at[0].set(0 + 0j)

# -- noise, drawn once -------------------------------------------------------
SEED = 0
n_fd = ns.sample_noise_fd(jr.split(jr.PRNGKey(SEED))[0], psd1)
data = h_gb + h_gl + n_fd

print(f'N = {N:,}   T_obs = {T_OBS:.6e} s = {T_OBS/86400:.3f} d   dt = {DT:.6f} s')
print(f'n_fine = {n_fine:,}   df = {float(freq[1]):.6e} Hz   k_min = {k_min:,}')
print(f'SNR  GB {float(jg.snr(h_gb, psd1)):8.3f}   glitch {float(jg.snr(h_gl, psd1)):8.3f}'
      f'   both {float(jg.snr(h_gb + h_gl, psd1)):8.3f}')

OUT = os.path.join(HERE, 'dataset.npz')
np.savez_compressed(
    OUT,
    data_tdi1=np.asarray(data), noise_tdi1=np.asarray(n_fd),
    h_gb_tdi1=np.asarray(h_gb), h_glitch_tdi1=np.asarray(h_gl),
    N=N, DT=DT, T_OBS=T_OBS, T_ARM=float(T_ARM), seed=SEED,
    k_min=k_min, N_GB=N_GB,
    gb_true=np.asarray(gb_true[0]), glitch_true=np.asarray(glitch_true),
)
print(f'\nsaved {OUT}')
