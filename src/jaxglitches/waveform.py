import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

C_SI = 299792458.
YRSID_SI = 31558149.763545603


def AET(X, Y, Z):
        return (
            (Z - X) / jnp.sqrt(2.0),
            (X - 2.0 * Y + Z) / jnp.sqrt(6.0),
            (X + Y + Z) / jnp.sqrt(3.0))

#########  Time domain TDI  ###### 

#TDI 1 glitch in time domain, with one exponential template
def tdi1_1exp_glitch(t, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau=0.79357148 ,T=8.3):    
    mask1 =  t  >= t0
   
    expression1 = (jnp.where(mask1, jnp.heaviside(t  - t0 ,0),0) * (-1 + jnp.where(mask1,jnp.exp( (- t  + t0  ) / tau ),0) + 
                    t  * jnp.where(mask1,jnp.exp( (- t  + t0  ) / tau ),0) /  tau  - jnp.where(mask1,jnp.exp( (- t  + t0  ) / tau ),0)*   t0  /  tau ))

    mask2 = t  >= t0  + 4 * T

    expression2 =( jnp.where(mask2, jnp.heaviside(t  -t0  - 4 * T ,0),0) * (1 - jnp.where(mask2,jnp.exp( (- t  + t0  + 4*T ) / tau ),0) 
                    - t  * jnp.where(mask2,jnp.exp( (-t+t0  + 4*T ) / tau ),0) / tau + 4 * T * jnp.where(mask2,jnp.exp( (- t  + t0  + 4*T ) / tau ),0)/tau  
                     +  t0  * jnp.where(mask2,jnp.exp( (- t  + t0  + 4*T ) / tau ),0) /  tau   ))

    mask3 =  t  >= t0  + 3 *T

    expression3 =(jnp.where(mask3,  jnp.heaviside(t  - t0  - 3 *T,0),0) 
                 *  (-1 + jnp.where(mask3,jnp.exp( (- t  + t0  + 3 *T) / tau ),0) *(1 +  t  /tau  - 3 * T /tau  - t0  /tau ) ))
             

    mask4 = t  >= t0  + T

    expression4 = (jnp.where(mask4,  jnp.heaviside(t  - t0  - T,0) ,0) * 
                  (1 + jnp.where(mask4,jnp.exp( (- t  + t0  + T) / tau ),0) *(-1 -  t  /tau  + T /tau  + t0  /tau )  ))
    

    tdiX1link12 = Deltav /C_SI*  (expression1 +  expression2)
    tdiY1link12 =  2* Deltav /C_SI* (expression3  + expression4)
    tdiZ1link12 = jnp.zeros_like(tdiX1link12)
 
    return tdiX1link12, tdiY1link12, tdiZ1link12

#TDI 1 glitch in time domain, with two exponentials template
def tdi1_2exp_glitch(t, t0=950,  Deltav=2.22616837*10**(-11), tau1=10, tau2=11 ,T=8.3):
    # First-generation TDI: the X channel combines two delays {t0, t0+4T} and
    # the Y channel two delays {t0+T, t0+3T} (cf. tdi1_1exp_glitch). The previous
    # implementation used the second-generation delay pattern {t0, t0+4T, t0+8T} /
    # {t0+T..t0+7T}, which is the TDI-2 waveform (see tdi2_2exp_glitch) and did not
    # match the first-gen frequency-domain formula tdi1_2exp_f_glitch.
    def phi(ts):
        x = t - ts
        mask = t >= ts
        return (
            Deltav / (C_SI * (tau1 - tau2))
            * (
                (tau1 - jnp.where(mask, jnp.exp(-x / tau1), 0.0) * tau1
                 + (-1.0 + jnp.where(mask, jnp.exp(-x / tau2), 0.0)) * tau2)
                * jnp.where(mask, jnp.heaviside(x, 0), 0.0)
            )
        )

    # X channel: -phi(t0) + phi(t0 + 4T)   -> transfer factor (-1 + exp(-4i T w))
    tdiX1link12 = -phi(t0) + phi(t0 + 4.0 * T)

    # Y channel: 2 phi(t0 + T) - 2 phi(t0 + 3T) -> 4i exp(-2i T w) sin(T w)
    tdiY1link12 = 2.0 * phi(t0 + T) - 2.0 * phi(t0 + 3.0 * T)

    tdiZ1link12 = jnp.zeros_like(tdiX1link12)

    return tdiX1link12, tdiY1link12, tdiZ1link12


# TDI 2 glitch in time domain, with one exponential template
def tdi2_1exp_glitch(t, t0=1.9394221536001746, Deltav=2.22616837e-11, tau=0.79357148, T=8.3):
    # Primitive used in your TDI1 implementation, kept in the same convention
    def phi(ts):
        x = t - ts
        mask = t >= ts
        return (
            jnp.where(mask, jnp.heaviside(x, 0), 0.0)
            * (-1.0 + jnp.where(mask, jnp.exp(-x / tau), 0.0)
               + jnp.where(mask, t * jnp.exp(-x / tau) / tau, 0.0)
               - jnp.where(mask, ts * jnp.exp(-x / tau) / tau, 0.0))
        )

    # X channel: coefficients [1, -2, 1] with your TDI1 outer convention
    expr1 = phi(t0)
    expr2 = phi(t0 + 4.0 * T)
    expr3 = phi(t0 + 8.0 * T)
    tdiX1link12 = Deltav / C_SI * (expr1 - 2.0 * expr2 + expr3)

    # Y channel: coefficients [-1, +1, +1, -1] in the same style as TDI1
    expr4 = phi(t0 + T)
    expr5 = phi(t0 + 3.0 * T)
    expr6 = phi(t0 + 5.0 * T)
    expr7 = phi(t0 + 7.0 * T)
    tdiY1link12 = Deltav / C_SI * (-2.0 * expr4 + 2.0 * expr5 + 2.0 * expr6 - 2.0 * expr7)

    tdiZ1link12 = jnp.zeros_like(tdiX1link12)
    return tdiX1link12, tdiY1link12, tdiZ1link12

# TDI 2 glitch in time domain, with two exponentials template
def tdi2_2exp_glitch(t, t0=950, Deltav=2.22616837e-11, tau1=10, tau2=11, T=8.3):
    def phi(ts):
        x = t - ts
        mask = t >= ts
        return (
            Deltav / (C_SI * (tau1 - tau2))
            * (
                (tau1 - jnp.where(mask, jnp.exp(-x / tau1), 0.0) * tau1
                 + (-1.0 + jnp.where(mask, jnp.exp(-x / tau2), 0.0)) * tau2)
                * jnp.where(mask, jnp.heaviside(x, 0), 0.0)
            )
        )

    # X channel
    expr1 = phi(t0)
    expr2 = phi(t0 + 4.0 * T)
    expr3 = phi(t0 + 8.0 * T)
    tdiX1link12 = expr1 - 2.0 * expr2 + expr3

    # Y channel
    expr4 = phi(t0 + T)
    expr5 = phi(t0 + 3.0 * T)
    expr6 = phi(t0 + 5.0 * T)
    expr7 = phi(t0 + 7.0 * T)
    tdiY1link12 = -2.0 * expr4 + 2.0 * expr5 + 2.0 * expr6 - 2.0 * expr7

    tdiZ1link12 = jnp.zeros_like(tdiX1link12)
    return tdiX1link12, tdiY1link12, tdiZ1link12


####  Frequency domain for TDI  ###

#TDI 1 glitch in frequency domain, with one exponential template
def tdi1_1exp_f_glitch( f, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau=0.79357148 ,T=8.3 ):

    Deltanuh = (- 1/(1j*C_SI*2*jnp.pi* f) * jnp.exp(-1j * t0*2*jnp.pi* f  ) 
                * Deltav /(-1j + tau* 2*jnp.pi*f  )**2)

    TFX_single_glich_tm12 = (-1 + jnp.exp(-4 * 1j* T* 2*jnp.pi*f  ))
    TFY_single_glich_tm12 = 4 * 1j * jnp.exp(-2 * 1j* T *  2*jnp.pi*f  )*jnp.sin(T*2*jnp.pi*f  )
    TFZ_single_glich_tm12 = jnp.zeros_like(TFY_single_glich_tm12)

    return TFX_single_glich_tm12*Deltanuh , TFY_single_glich_tm12*Deltanuh, TFZ_single_glich_tm12*Deltanuh

#TDI 1 glitch in frequency domain, with two exponentials template
def tdi1_2exp_f_glitch(freq, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau1=0.79357148,tau2=0.79357148 ,T=8.3 ):    
   
    Deltanuh = (1/(1j*C_SI*2*jnp.pi*freq  ) * jnp.exp(-1j * t0   *2*jnp.pi*freq  ) * Deltav 
                /((1+1j * tau1  * 2*jnp.pi*freq  )*(1 +1j * tau2  * 2*jnp.pi*freq  )))

    TFX_single_glich_tm12 = (-1 + jnp.exp(-4 * 1j* T* 2*jnp.pi*freq  ))
    TFY_single_glich_tm12 = 4 * 1j * jnp.exp(-2 * 1j* T *  2*jnp.pi*freq  )*jnp.sin(T*2*jnp.pi*freq  )
    TFZ_single_glich_tm12 = jnp.zeros_like(TFY_single_glich_tm12)

    return TFX_single_glich_tm12*Deltanuh , TFY_single_glich_tm12*Deltanuh, TFZ_single_glich_tm12*Deltanuh


#TDI 2 glitch in frequency domain, with one exponential template
def tdi2_1exp_f_glitch( f, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau=0.79357148 ,T=8.3 ):

    Deltanuh = (1/(1j*C_SI*2*jnp.pi* f) * jnp.exp(-1j * t0*2*jnp.pi* f  ) 
                * Deltav /(-1j + tau* 2*jnp.pi*f  )**2)

    TFX_single_glich_tm12 = jnp.exp(-8 *1j * T *2*jnp.pi*f  )* (-1 + jnp.exp(4 * 1j* T* 2*jnp.pi*f  ))**2
    TFY_single_glich_tm12 = 8 * jnp.exp(-4 * 1j*T *2*jnp.pi*f  ) * jnp.sin(2 * T *  2*jnp.pi*f  )*jnp.sin(T*2*jnp.pi*f  )
    TFZ_single_glich_tm12 = jnp.zeros_like(TFY_single_glich_tm12)

    return TFX_single_glich_tm12*Deltanuh , TFY_single_glich_tm12*Deltanuh, TFZ_single_glich_tm12*Deltanuh
    
#TDI 2 glitch in frequency domain, with two exponentials template
def tdi2_2exp_f_glitch(freq, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau1=0.79357148,tau2=0.79357148 ,T=8.3 ):    
   
    Deltanuh = (1/(1j*C_SI*2*jnp.pi*freq  ) * jnp.exp(-1j * t0   *2*jnp.pi*freq  ) * Deltav
                 /((1+1j * tau1  * 2*jnp.pi*freq  )*(1 +1j * tau2  * 2*jnp.pi*freq  )))

    TFX_single_glich_tm12 = jnp.exp(-8 *1j * T *2*jnp.pi*freq  )* (-1 + jnp.exp(4 * 1j* T* 2*jnp.pi*freq  ))**2
    TFY_single_glich_tm12 = 8 * jnp.exp(-4 * 1j*T *2*jnp.pi*freq  ) * jnp.sin(2 * T *  2*jnp.pi*freq  )*jnp.sin(T*2*jnp.pi*freq  )
    TFZ_single_glich_tm12 = jnp.zeros_like(TFY_single_glich_tm12)

    return TFX_single_glich_tm12*Deltanuh , TFY_single_glich_tm12*Deltanuh, TFZ_single_glich_tm12*Deltanuh

