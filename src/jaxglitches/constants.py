import jax
jax.config.update("jax_enable_x64", True)

C_SI = 299792458.0           # speed of light, m/s
YRSID_SI = 31558149.763545603  # sidereal year, s
ARM_LENGTH_m = 2.5e9           # LISA arm length, m
T_ARM_s = ARM_LENGTH_m / C_SI  # light travel time along one arm, ~8.336 s

# Default glitch-analysis window: short enough for fast computation,
# long enough to have good frequency resolution (df = 1/T_OBS ~ 3e-4 Hz).
DT_s = 0.25          # time step → Nyquist at 2 Hz
T_OBS_s = 3600.0     # 1 hour observation window
