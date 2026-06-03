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
 
    mask1 =  t >= t0
    mask2 =  t >= t0 + 8*T
    mask3 =  t >= t0 + 4*T
   
    tdiX1link12 =( Deltav/(C_SI*(tau1 - tau2)) *
     ((tau1  - jnp.where(mask1,jnp.exp((-t + t0)/tau1 ),0)*tau1  + 
            (-1 +jnp.where(mask1,jnp.exp((-t + t0)/tau2 ),0))*tau2 )* jnp.where(mask1, jnp.heaviside(t - t0,0),0) + 
           (tau1  - jnp.where(mask2,jnp.exp((-t + 8*T + t0)/tau1 ),0)*tau1  + 
           (-1 +jnp.where(mask2,jnp.exp((-t + 8*T + t0)/tau2 ),0))*tau2 )* jnp.where(mask2, jnp.heaviside(t - 8*T - t0,0),0) 
            - 2*(tau1  - jnp.where(mask3,jnp.exp((-t + 4*T + t0)/tau1 ),0)*tau1   
         +  (-1 +jnp.where(mask3,jnp.exp((-t + 4*T + t0)/tau2 ),0))*tau2 )* jnp.where(mask3, jnp.heaviside(t - 4*T - t0,0),0)))

    mask4 =  t >= t0  + 7*T
    mask5 =  t >= t0  + 5*T
    mask6 =  t >= t0  + 3*T
    mask7 =  t >= t0  + T


    tdiY1link12 = ( - 2* Deltav/(C_SI*(tau1  - tau2 )) * ( ((tau1  -jnp.where(mask7,jnp.exp((-t + 7*T + t0)/tau1 ),0)*tau1   
      +  (-1 + jnp.where(mask7,jnp.exp((-t + 7*T + t0)/tau2 ),0))*tau2 )* jnp.where(mask4, jnp.heaviside(t - 7*T - t0,0),0))  
        + ((-1 + jnp.where(mask5,jnp.exp((-t + 5*T + t0)/tau1 ),0))*tau1 + tau2   
        -   jnp.where(mask5,jnp.exp((-t + 5*T + t0)/tau2 ),0)*tau2 )* jnp.where(mask5, jnp.heaviside(t - 5*T - t0,0),0)
     +  ((-1 + jnp.where(mask6,jnp.exp((-t + 3*T + t0)/tau1 ),0))*tau1  + tau2   
     -     jnp.where(mask6,jnp.exp((-t + 3*T + t0)/tau2 ),0)*tau2 )*  jnp.where(mask6, jnp.heaviside(t - 3*T - t0,0) ,0) 
      +  (tau1  -  jnp.where(mask7,jnp.exp((-t + T + t0)/tau1 ),0)*tau1 
      +     (-1 + jnp.where(mask7,jnp.exp((-t + T + t0)/tau2),0))*tau2)*jnp.where(mask7, jnp.heaviside(t - T - t0,0),0)))



    tdiZ1link12 = jnp.zeros_like(tdiX1link12)
 
    return tdiX1link12, tdiY1link12, tdiZ1link12

#TDI 1 glitch in time domain, with shaplet template
def tdi1_shap_glitch(t, tau=1.9394221536001746,  Deltav=2.22616837*10**(-11), beta=0.79357148 ,T=8.3):

    tdiX1link12 =  (-2*(Deltav*(beta   - 
              jnp.exp((-t   + tau  )/beta  )*(t   + beta   -tau  ))*
             jnp.heaviside(t   - tau  ,0) - 
           Deltav*(beta   + 
              jnp.exp((-t   + 4*T + tau  )/beta  )*
              (-t   + 4*T -beta   +tau  ))*
             jnp.heaviside(t   - 4*T - tau  ,0)))/C_SI
 
    tdiY1link12 = (-4*(Deltav*(beta   + 
              jnp.exp((-t + 3*T + tau  )/beta  )*
               (-t + 3*T - beta   + tau  ))*
             jnp.heaviside(t - 3*T -tau  ,0) - 
           Deltav*(beta   + 
              jnp.exp((-t + T + tau  )/beta  )*
               (-t + T - beta   + tau  ))*
             jnp.heaviside(t - T -tau  ,0)))/C_SI
  

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

# TDI 2 glitch in time domain, with shaplet template
def tdi2_shap_glitch(t, tau=1.9394221536001746, Deltav=2.22616837e-11, beta=0.79357148, T=8.3):
    def phi(ts):
        x = t - ts
        mask = t >= ts
        return (
            jnp.where(mask, jnp.heaviside(x, 0), 0.0)
            * (Deltav * (beta - jnp.where(mask, jnp.exp(-x / beta), 0.0)* (x + beta)))
        )

    # X channel: 1, -2, 1
    expr1 = phi(tau)
    expr2 = phi(tau + 4.0 * T)
    expr3 = phi(tau + 8.0 * T)
    tdiX1link12 = (-2.0 * (expr1 - 2.0 * expr2 + expr3)) / C_SI

    # Y channel: alternating pattern
    expr4 = phi(tau + T)
    expr5 = phi(tau + 3.0 * T)
    expr6 = phi(tau + 5.0 * T)
    expr7 = phi(tau + 7.0 * T)
    tdiY1link12 = (-4.0 * (-expr4 + expr5 + expr6 - expr7)) / C_SI

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

#TDI 1 glitch in frequency domain, with shaplet template
def tdi1_shap_f_glitch(freq, tau=0.79357148,  Deltav=2.22616837*10**(-11), beta=1.9394221536001746 ,T=8.3):    
   
    Deltanus = (- 2*Deltav* beta  /(1j*C_SI*2*jnp.pi*freq  ) 
                * jnp.exp(-1j * tau   *2*jnp.pi*freq  )/(-1j + beta  * 2*jnp.pi*freq  )**2)


    TFX1_single_glich_tm12 = (-1 + jnp.exp(-4 * 1j* T* 2*jnp.pi*freq  ))
    TFY1_single_glich_tm12 = 4 * 1j * jnp.exp(-2 * 1j* T *  2*jnp.pi*freq  )*jnp.sin(T*2*jnp.pi*freq  )
    TFZ1_single_glich_tm12 = jnp.zeros_like(TFY1_single_glich_tm12)


    return TFX1_single_glich_tm12*Deltanus, TFY1_single_glich_tm12*Deltanus, TFZ1_single_glich_tm12*Deltanus
    
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

#TDI 2 glitch in frequency domain, with shaplet template
def tdi2_shap_f_glitch(freq, tau=0.79357148,  Deltav=2.22616837*10**(-11), beta=1.9394221536001746  ,T=8.3):    
   
    Deltanus = (2*Deltav* beta  /(1j*C_SI*2*jnp.pi*freq  ) 
                * jnp.exp(-1j * tau   *2*jnp.pi*freq  )/(-1j + beta  * 2*jnp.pi*freq  )**2)
    
    TFX_single_glich_tm12 = jnp.exp(-8 *1j * T *2*jnp.pi*freq  )* (-1 + jnp.exp(4 * 1j* T* 2*jnp.pi*freq  ))**2
    TFY_single_glich_tm12 = 8 * jnp.exp(-4 * 1j*T *2*jnp.pi*freq  ) * jnp.sin(2 * T *  2*jnp.pi*freq  )*jnp.sin(T*2*jnp.pi*freq  )
    TFZ_single_glich_tm12 = jnp.zeros_like(TFY_single_glich_tm12)

    return TFX_single_glich_tm12*Deltanus, TFY_single_glich_tm12*Deltanus, TFZ_single_glich_tm12*Deltanus


if __name__ == "__main__":
  
    ## defining constant to evaluate the glitches/shapelet

    dt = 0.25
    Tobs = 1/12. *YRSID_SI
    N = int(Tobs / dt)
    Tobs = dt * N
    t_in = jnp.arange(N) * dt

    freqs = jnp.fft.rfftfreq(N, dt)  # fs =1/dt
    #freqs[freqs == 0] = 1e-50

    fmin = 2.5e-5
    fmax = 1e-1
    frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL: we MUST find a way to include them
    freqs_cut = jnp.array(freqs[frequencymask])

