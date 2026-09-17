"""What is still measurable when a glitch is shorter than the analysis can resolve?

The paper's glitch has tau = 300 s, and its analysis band stops at f_max = 3 mHz, i.e. the
data are sampled every 167 s. The decay time tau is resolvable only if the glitch's knee,
f_knee = 1/(2 pi tau), lies inside the band, which needs tau > 1/(2 pi f_max) = 53 s. One
might expect a shorter glitch to be unobservable. It is not, and this script measures why.

Below its knee the frequency-domain glitch is a step of height Deltav in the velocity,
placed at the pulse's centroid t0 + 2 tau; tau enters the waveform only at order
(2 pi f tau)^2. So a short glitch is *more* detectable per unit kick than a long one, its
kick Deltav and its centroid are measured as well as ever, and only tau -- and with it
t0 separately from the centroid -- is lost.

snr       rho against tau at fixed Deltav = 1e-11 m/s.
fisher    Fisher widths against tau at fixed rho = 42.7, the paper's SNR: t0, the centroid
          t0 + 2 tau, Deltav, tau and corr(t0, log tau).
chain     the posterior at tau = 1 s and rho = 42.7, with the paper's priors. The Fisher
          matrix gives sigma(Deltav)/Deltav = 3.8% at every tau, which for tau below
          ~10 s is a linearisation artefact: it marginalises a weak (2 pi f tau)^2
          correlation over an unbounded log tau. The chain gives the ~1/rho that the
          step picture predicts, and only an upper limit on tau.
centroid  the SNR of the difference between a tau = 1 s template and templates with a
          longer tau at the same centroid, or a slightly different Deltav at the same
          tau: how distinguishable each change is.

The grid is glitch-only and 1/16 as long as the paper's (11 712 samples, 22.6 d, band to
3 mHz), so everything runs in minutes on a CPU; a transient's SNR does not depend on the
record length once the record contains it. Sampling coordinates and priors for `chain`
are those of `../glitch_GB/fd_pipeline.py`: (t0, log A_g, log tau) with A_g = Deltav tau,
flat, plus the LPF support cut on Deltav.

    python short_glitch.py            # all four
    python short_glitch.py chain      # one of them

Writes `short_glitch.npz` next to this script (a phase run on its own updates only its
own entries). CPU by default.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent
sys.path.insert(0, str(NOTEBOOKS))                   # noise.py
sys.path.insert(0, str(NOTEBOOKS / "glitch_GB"))     # fd_pipeline.py

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import jaxglitches as jg
import noise as ns

F_MAX = 3e-3
N_SAMP = 187_392 // 16
DT = 1.0 / (2 * F_MAX)
FREQ = jnp.asarray(np.fft.rfftfreq(N_SAMP, DT))
T_OBS = N_SAMP * DT
PSD = ns.psd_tdi1_array(jnp.where(FREQ > 0, FREQ, 1.0), t_obs=T_OBS)
T0 = 5000.0                        # onset, well inside the record
DV_REF = 1e-11                     # the paper's kick
RHO = 42.7                         # the paper's glitch SNR


def template(t0, deltav, tau):
    return jg.clean_signal_f(jnp.stack([jnp.asarray(t0, float), jnp.asarray(deltav, float),
                                        jnp.asarray(tau, float)]),
                             FREQ, T=jg.T_ARM_s, tdi=1).at[0].set(0 + 0j)


def inner(a, b):
    return 2.0 * float(jnp.real(jnp.sum(jnp.conj(a[1:]) * b[1:] / PSD[1:])))


def deltav_at_rho(tau, rho=RHO):
    """The kick that gives signal-to-noise ratio `rho` at duration `tau`."""
    return DV_REF * rho / float(jg.snr(template(T0, DV_REF, tau), PSD))


def phase_snr():
    taus = np.array([0.01, 0.1, 1.0, 10.0, 30.0, 53.0, 100.0, 300.0, 1000.0, 3000.0])
    rho = np.array([float(jg.snr(template(T0, DV_REF, t), PSD)) for t in taus])
    print(f"SNR at fixed Deltav = {DV_REF:g} m/s, band to {1e3 * F_MAX:g} mHz")
    for t, r in zip(taus, rho):
        print(f"  tau = {t:8.2f} s   rho = {r:8.2f}")
    return dict(snr_tau=taus, snr_rho=rho)


def phase_fisher():
    taus = np.array([0.1, 1.0, 10.0, 30.0, 53.0, 100.0, 300.0])
    rows = []
    print(f"Fisher at fixed rho = {RHO}, in (t0, log Deltav, log tau)")
    print(f"  {'tau [s]':>8s}{'Deltav':>10s}{'s(t0) [s]':>11s}{'s(t0+2tau) [s]':>16s}"
          f"{'s(Dv)/Dv':>10s}{'s(tau)/tau':>12s}{'corr(t0,log tau)':>18s}")
    for tau in taus:
        dv = deltav_at_rho(tau)
        th = jnp.array([T0, np.log(dv), np.log(tau)])

        def h_real(t):
            h = template(t[0], jnp.exp(t[1]), jnp.exp(t[2]))
            return jnp.concatenate([h.real.ravel(), h.imag.ravel()])

        J = jax.jacfwd(h_real)(th)
        w = jnp.concatenate([1 / PSD.ravel(), 1 / PSD.ravel()])
        w = jnp.where(jnp.isfinite(w), w, 0.0)
        F = 2.0 * np.asarray((J.T * w) @ J)
        C = np.linalg.pinv(F)
        s = np.sqrt(np.diag(C))
        g = np.array([1.0, 0.0, 2.0 * tau])              # d(t0 + 2 tau)/d(t0, log Dv, log tau)
        s_centroid = float(np.sqrt(g @ C @ g))
        corr = C[0, 2] / (s[0] * s[2])
        rows.append([tau, dv, s[0], s_centroid, s[1], s[2], corr])
        print(f"  {tau:8.1f}{dv:10.2e}{s[0]:11.2f}{s_centroid:16.2f}{s[1]:10.3f}"
              f"{s[2]:12.3f}{corr:18.3f}")
    return dict(fisher_columns=np.array(["tau", "deltav", "sigma_t0", "sigma_centroid",
                                         "sigma_log_deltav", "sigma_log_tau",
                                         "corr_t0_log_tau"]),
                fisher=np.array(rows))


def phase_chain():
    import fd_pipeline as fp

    tau = 1.0
    dv = deltav_at_rho(tau)

    def h_sampling(th):                      # (t0, log A_g, log tau), A_g = Deltav tau
        tt = jnp.exp(th[2])
        return template(th[0], jnp.exp(th[1]) / tt, tt)

    th = jnp.array([T0, float(np.log(dv * tau)), float(np.log(tau))])
    h = h_sampling(th)
    print(f"tau = {tau} s, Deltav = {dv:.3e} m/s, rho = {float(jg.snr(h, PSD)):.2f}")

    lo = jnp.array([0.0, float(th[1]) - 8.0, float(np.log(0.1))])
    hi = jnp.array([1e4, float(th[1]) + 12.0, float(np.log(5e4))])
    ldv = (float(np.log(1e-16)), float(np.log(1e-7)))      # LPF support on Deltav

    @jax.jit
    def log_prior(t):
        ok = jnp.all((t >= lo) & (t <= hi))
        d = t[1] - t[2]
        return jnp.where(ok & (d >= ldv[0]) & (d <= ldv[1]), 0.0, -jnp.inf)

    data = h + ns.sample_noise_fd(jr.PRNGKey(700_123), PSD)

    @jax.jit
    def log_lik(t):
        return jg.log_likelihood(data, h_sampling(t), PSD)

    # Deliberately broad start along the flat direction; run_chain scales it by `ball`.
    sig0 = jnp.array([20.0, 0.5, 0.5])
    t_start = time.time()
    chain = np.asarray(fp.run_chain(log_lik, log_prior, th, sig0 / 0.1, 3, seed=5,
                                    nwalkers=32, nburn=3000, nsamp=12000, ball=0.01))
    print(f"chain {chain.shape} in {time.time() - t_start:.0f} s")
    t0s, taus = chain[:, 0], np.exp(chain[:, 2])
    log_dv = chain[:, 1] - chain[:, 2]
    pct = np.percentile(taus, [5, 50, 95])
    print(f"  log Deltav: median offset {np.median(log_dv) - np.log(dv):+.4f}, "
          f"std {log_dv.std():.4f}   (1/rho = {1 / RHO:.4f})")
    print(f"  tau: 5/50/95% = {pct[0]:.2f} / {pct[1]:.2f} / {pct[2]:.2f} s   "
          f"(prior floor 0.1 s)")
    print(f"  t0: std {t0s.std():.2f} s;  t0 + 2 tau: std {(t0s + 2 * taus).std():.2f} s, "
          f"median offset {np.median(t0s + 2 * taus) - (T0 + 2 * tau):+.2f} s")
    print(f"  corr(log Deltav, log tau) = {np.corrcoef(log_dv, chain[:, 2])[0, 1]:+.3f}")
    return dict(chain_tau=tau, chain_deltav=dv, chain=chain, chain_theta_true=np.asarray(th))


def phase_centroid():
    tau_ref = 1.0
    dv = deltav_at_rho(tau_ref)
    ref = template(T0, dv, tau_ref)
    print(f"reference: tau = {tau_ref} s, rho = {np.sqrt(inner(ref, ref)):.1f}")
    print("SNR of the difference to the reference")
    taus = np.array([3.0, 10.0, 13.0, 20.0, 30.0, 53.0])
    d_tau = []
    for tau in taus:                          # same kick, same centroid t0 + 2 tau
        diff = template(T0 + 2 * (tau_ref - tau), dv, tau) - ref
        d_tau.append(np.sqrt(inner(diff, diff)))
        print(f"  tau = {tau:5.1f} s, same Deltav and centroid: {d_tau[-1]:6.2f}")
    fracs = np.array([0.01, 0.025, 0.05])
    d_dv = []
    for frac in fracs:
        diff = template(T0, dv * (1 + frac), tau_ref) - ref
        d_dv.append(np.sqrt(inner(diff, diff)))
        print(f"  Deltav {100 * frac:+4.1f}%, same tau:            {d_dv[-1]:6.2f}")
    return dict(centroid_tau=taus, centroid_snr_tau=np.array(d_tau),
                centroid_dv_frac=fracs, centroid_snr_dv=np.array(d_dv))


PHASES = ("snr", "fisher", "chain", "centroid")


def main():
    out_path = HERE / "short_glitch.npz"
    phases = sys.argv[1:] or list(PHASES)
    if any(p not in PHASES for p in phases):
        raise SystemExit(f"phase must be one of {PHASES}")
    out = dict(np.load(out_path)) if out_path.exists() else {}
    for phase in phases:
        print(f"\n########## {phase} ##########", flush=True)
        out |= globals()[f"phase_{phase}"]()
    np.savez_compressed(out_path, **out)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
