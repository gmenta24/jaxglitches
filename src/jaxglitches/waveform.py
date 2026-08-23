import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

C_SI = 299792458.
YRSID_SI = 31558149.763545603
ARM_LENGTH_m = 2.5e9           # LISA arm length, m
T_ARM_s = ARM_LENGTH_m / C_SI  # light travel time along one arm, ~8.336 s

def AET(X, Y, Z):
        return (
            (Z - X) / jnp.sqrt(2.0),
            (X - 2.0 * Y + Z) / jnp.sqrt(6.0),
            (X + Y + Z) / jnp.sqrt(3.0))

#########  Time domain TDI  ######

def _psi1(t, ts, tau):
    """Integrated one-exponential shapelet with onset ts:

        psi(t) = Theta(t - ts) * [-1 + e^{-(t-ts)/tau} (1 + (t-ts)/tau)]

    The exponent is clamped to the masked region before exp so that
    gradients stay finite even when (ts - t)/tau is large (an unclamped
    exp overflows in the unselected where-branch and poisons the grad)."""
    x = t - ts
    mask = x >= 0.0
    xs = jnp.where(mask, x, 0.0)
    return jnp.where(mask, -1.0 + jnp.exp(-xs / tau) * (1.0 + xs / tau), 0.0)

#TDI 1 glitch in time domain, with one exponential template
def tdi1_1exp_glitch(t, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau=0.79357148 ,T=T_ARM_s):
    # X channel: psi(t0) - psi(t0 + 4T)      <-> transfer (1 - D^4)
    tdiX1link12 = Deltav / C_SI * (_psi1(t, t0, tau) - _psi1(t, t0 + 4.0 * T, tau))
    # Y channel: 2 [psi(t0 + 3T) - psi(t0 + T)]  <-> transfer 2 (D^3 - D)
    tdiY1link12 = 2.0 * Deltav / C_SI * (_psi1(t, t0 + 3.0 * T, tau) - _psi1(t, t0 + T, tau))
    tdiZ1link12 = jnp.zeros_like(tdiX1link12)

    return tdiX1link12, tdiY1link12, tdiZ1link12

# TDI 2 glitch in time domain, with one exponential template
def tdi2_1exp_glitch(t, t0=1.9394221536001746, Deltav=2.22616837e-11, tau=0.79357148, T=T_ARM_s):
    # X channel: psi coefficients [1, -2, 1] at delays {0, 4T, 8T}  <-> (1 - D^4)^2
    tdiX1link12 = Deltav / C_SI * (
        _psi1(t, t0, tau)
        - 2.0 * _psi1(t, t0 + 4.0 * T, tau)
        + _psi1(t, t0 + 8.0 * T, tau)
    )
    # Y channel: coefficients 2*[-1, +1, +1, -1] at delays {T, 3T, 5T, 7T}
    #            <-> 2 (-D + D^3 + D^5 - D^7)
    tdiY1link12 = 2.0 * Deltav / C_SI * (
        -_psi1(t, t0 + T, tau)
        + _psi1(t, t0 + 3.0 * T, tau)
        + _psi1(t, t0 + 5.0 * T, tau)
        - _psi1(t, t0 + 7.0 * T, tau)
    )
    tdiZ1link12 = jnp.zeros_like(tdiX1link12)
    return tdiX1link12, tdiY1link12, tdiZ1link12

####  Frequency domain for TDI  ###

#TDI 1 glitch in frequency domain, with one exponential template
def tdi1_1exp_f_glitch( f, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau=0.79357148 ,T=T_ARM_s ):

    Deltanuh = (- 1/(1j*C_SI*2*jnp.pi* f) * jnp.exp(-1j * t0*2*jnp.pi* f  ) 
                * Deltav /(-1j + tau* 2*jnp.pi*f  )**2)

    TFX_single_glich_tm12 = (-1 + jnp.exp(-4 * 1j* T* 2*jnp.pi*f  ))
    TFY_single_glich_tm12 = 4 * 1j * jnp.exp(-2 * 1j* T *  2*jnp.pi*f  )*jnp.sin(T*2*jnp.pi*f  )
    TFZ_single_glich_tm12 = jnp.zeros_like(TFY_single_glich_tm12)

    return TFX_single_glich_tm12*Deltanuh , TFY_single_glich_tm12*Deltanuh, TFZ_single_glich_tm12*Deltanuh

#TDI 2 glitch in frequency domain, with one exponential template
def tdi2_1exp_f_glitch( f, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau=0.79357148 ,T=T_ARM_s ):

    Deltanuh = (1/(1j*C_SI*2*jnp.pi* f) * jnp.exp(-1j * t0*2*jnp.pi* f  ) 
                * Deltav /(-1j + tau* 2*jnp.pi*f  )**2)

    TFX_single_glich_tm12 = jnp.exp(-8 *1j * T *2*jnp.pi*f  )* (-1 + jnp.exp(4 * 1j* T* 2*jnp.pi*f  ))**2
    TFY_single_glich_tm12 = 8 * jnp.exp(-4 * 1j*T *2*jnp.pi*f  ) * jnp.sin(2 * T *  2*jnp.pi*f  )*jnp.sin(T*2*jnp.pi*f  )
    TFZ_single_glich_tm12 = jnp.zeros_like(TFY_single_glich_tm12)

    return TFX_single_glich_tm12*Deltanuh , TFY_single_glich_tm12*Deltanuh, TFZ_single_glich_tm12*Deltanuh

