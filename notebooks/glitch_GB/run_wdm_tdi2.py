"""TDI-1 against TDI-2 in the WDM domain: the null test of Sec. 5.5, repeated
in the time--frequency analysis.

In the frequency domain the two generations are related by an exact identity ---
h^(2) = -TF_X^(1) h^(1) and S^(2) = |TF_X^(1)|^2 S^(1), so the transfer function
cancels bin by bin and the two likelihoods are equal to machine precision. Any
disagreement there is a bug.

The time--frequency analysis is different, and that is what makes the test worth
repeating rather than copying. The split likelihood keeps 1106 pixels out of the
187394 real numbers of the fine grid, and the per-pixel variance of Eq. (35) is
evaluated at the channel centre. The generation transfer function varies across a
channel, so the two generations do not weight the plane identically and do not
discard the same part of the same noise realisation. The comparison therefore
measures the size of that effect instead of testing an identity, and this script
separates the two things that can produce a difference:

  * the machinery -- templates, transform, windows, split -- which is checked on
    noise-free data, where the two generations must agree to round-off;
  * the noise, which is checked by repeating the comparison over independent
    realisations and by running a full TDI-2 chain against the stored TDI-1 one.

Writes `wdm_tdi2.npz` next to this script; `make_fig_wdm_tdi.py` turns that into
`paper/figures/fig_wdm_tdi.pdf`. Runtime is dominated by the chain, about half an hour
on a GPU, with the realisation scan adding roughly as much again.

The three phases run as separate processes:

    python run_wdm_tdi2.py            # checks, scatter, chain, then merge
    python run_wdm_tdi2.py chain      # one phase only

That is not cosmetic. Each realisation builds its own jitted likelihood closure holding
about 20 MB of device arrays, and with a few dozen of them alive the XLA compilation of
the sampler's scan slows to a crawl. Running the phases in fresh processes keeps the
chain compiling against an empty cache.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "notebooks"))          # noise.py

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import lisaorbits
from jaxgb import jaxgb as jgb_mod
from wdm_transform import TimeSeries, WDM, wdm_noise_variance
from wdm_transform.backends import get_backend

import jaxglitches as jg
import noise as ns
from jaxglitches.priors import _DELTAV_MIN, _DELTAV_MAX, _TAU_MIN, _TAU_MAX

BACKEND = get_backend("jax")
N_REALISATIONS = 48
SEED_CHAIN = 1            # the seed the stored TDI-1 chain used: common random numbers
N_WALKERS, N_BURN, N_SAMP, DIM = 16, 2_000, 10_000, 7
LABELS = ["log_f0", "log_fdot", "log_A_gb", "psi", "t0", "log_Ag", "log_tau"]

# ---------------------------------------------------------------------------
# grid, tilings, injections -- identical to glitch_and_gb_wdm.ipynb
# ---------------------------------------------------------------------------
DS = np.load(HERE / "dataset.npz")
DT, N = float(DS["DT"]), int(DS["N"])
T_OBS, T_ARM = float(DS["T_OBS"]), float(DS["T_ARM"])
NF_GB, NF_GL = 1536, 128
NT_GB, NT_GL = N // NF_GB, N // NF_GL

freq = jnp.asarray(np.fft.rfftfreq(N, DT))
f_safe = jnp.where(freq > 0, freq, 1.0)
n_fine = len(freq)

# TDI-1 -> TDI-2 acts on data and templates alike, so both generations describe one
# physical realisation rather than two independent draws.
TFX1 = -1.0 + jnp.exp(-4j * T_ARM * 2.0 * jnp.pi * f_safe)


def to_tdi2(H):
    return ((-TFX1)[:, None] * H).at[0].set(0 + 0j)


IDENTITY = lambda H: H                                            # noqa: E731
XFORM = {1: IDENTITY, 2: to_tdi2}

gb_true = jnp.asarray(DS["gb_true"])
g3_true = jnp.asarray(DS["glitch_true"])
F0_TRUE = float(gb_true[0])
RA, DEC, IOTA, PHI0 = (float(gb_true[i]) for i in (3, 4, 6, 7))
N_GB = int(DS["N_GB"])
JGB = jgb_mod.JaxGB(lisaorbits.EqualArmlengthOrbits(), t_obs=T_OBS, t0=0.0, n=N_GB)


def to_time(H):
    return jnp.stack([jnp.fft.irfft(H[:, c], n=N) / DT for c in range(3)])


def wdm_of(x, nt):
    return WDM.from_time_series(TimeSeries(x, dt=DT, backend=BACKEND), nt=nt)


def pixel_variance(freq_grid, nt, nf, gen):
    fch = jnp.where(jnp.asarray(freq_grid) > 0, jnp.asarray(freq_grid), 1.0)
    psd = ns.psd_tdi1 if gen == 1 else ns.psd_tdi2
    S1 = jnp.stack(psd(fch), axis=-1)
    return jnp.stack([jnp.asarray(wdm_noise_variance(np.asarray(S1[:, c]),
                                                     nt=nt, nf=nf, dt=DT))
                      for c in range(3)])


def gb_fd(gb8):
    p = gb8[None, :]
    segs = JGB.get_tdi(p, tdi_generation=1.5, tdi_combination="AET")
    seg = jnp.stack(segs, axis=0).astype(jnp.complex128)[:, 0, :].T
    k = JGB.get_kmin(p[:, 0])[0].astype(jnp.int32)
    return jax.lax.dynamic_update_slice(jnp.zeros((n_fine, 3), jnp.complex128), seg,
                                        (k, jnp.zeros((), jnp.int32)))


def glitch_fd(g3):
    return jg.clean_signal_f(g3, freq, tdi=1).at[0].set(0 + 0j)


def to_physical(th):
    gb8 = (jnp.array([0., 0., 0., RA, DEC, 0., IOTA, PHI0])
           .at[0].set(jnp.exp(th[0])).at[1].set(jnp.exp(th[1]))
           .at[2].set(jnp.exp(th[2])).at[5].set(th[3]))
    tau = jnp.exp(th[6])
    return gb8, jnp.stack([th[4], jnp.exp(th[5]) / tau, tau])


THETA_TRUE = jnp.array([jnp.log(gb_true[0]), jnp.log(gb_true[1]), jnp.log(gb_true[2]),
                        gb_true[5], g3_true[0], jnp.log(g3_true[1] * g3_true[2]),
                        jnp.log(g3_true[2])])

_w = wdm_of(to_time(jnp.asarray(DS["h_gb_tdi1"])), NT_GB)
FREQ_GB, TIME_GB = np.asarray(_w.freq_grid), np.asarray(_w.time_grid)
_w = wdm_of(to_time(jnp.asarray(DS["h_glitch_tdi1"])), NT_GL)
FREQ_GL, TIME_GL = np.asarray(_w.freq_grid), np.asarray(_w.time_grid)
VAR = {(g, "gb"): pixel_variance(FREQ_GB, NT_GB, NF_GB, g) for g in (1, 2)}
VAR |= {(g, "gl"): pixel_variance(FREQ_GL, NT_GL, NF_GL, g) for g in (1, 2)}

K_GB = int(np.argmin(np.abs(FREQ_GB - F0_TRUE)))
GB_CH = jnp.arange(max(1, K_GB - 2), min(NF_GB, K_GB + 3))
NT_WIN = 4
GL_T = jnp.arange(NT_WIN)
K_NOTCH = int(np.argmin(np.abs(FREQ_GL - F0_TRUE)))
GL_CH = jnp.asarray([m for m in range(1, NF_GL) if abs(m - K_NOTCH) > 1])

T0_MAX = float(TIME_GL[NT_WIN - 1])


@jax.jit
def log_prior(th):
    ok = ((th[0] >= jnp.log(1e-4)) & (th[0] <= jnp.log(3e-3))
          & (th[1] >= jnp.log(1e-22)) & (th[1] <= jnp.log(1e-15))
          & (th[2] >= jnp.log(1e-25)) & (th[2] <= jnp.log(1e-20))
          & (th[3] >= 0.0) & (th[3] <= float(jnp.pi))
          & (th[4] >= 0.0) & (th[4] <= T0_MAX)
          & (th[5] >= float(np.log(_DELTAV_MIN * _TAU_MIN)))
          & (th[5] <= float(np.log(_DELTAV_MAX * _TAU_MAX)))
          & (th[5] - th[6] >= float(np.log(_DELTAV_MIN)))
          & (th[5] - th[6] <= float(np.log(_DELTAV_MAX)))
          & (th[6] >= float(np.log(_TAU_MIN))) & (th[6] <= float(np.log(_TAU_MAX))))
    return jnp.where(ok, 0.0, -jnp.inf)


def build(gen, data_fd):
    """Split WDM log-likelihood for one TDI generation on one data stream."""
    xf = XFORM[gen]
    d = xf(data_fd)
    d_gb = wdm_of(to_time(d), NT_GB).coeffs[:, :, GB_CH]
    d_gl = wdm_of(to_time(d), NT_GL).coeffs[:, GL_T][:, :, GL_CH]
    v_gb = VAR[(gen, "gb")][:, :, GB_CH]
    v_gl = VAR[(gen, "gl")][:, GL_T][:, :, GL_CH]

    @jax.jit
    def log_lik(th):
        gb8, g3 = to_physical(th)
        w_gb = wdm_of(to_time(xf(gb_fd(gb8))), NT_GB).coeffs[:, :, GB_CH]
        w_gl = wdm_of(to_time(xf(glitch_fd(g3))), NT_GL).coeffs[:, GL_T][:, :, GL_CH]
        return (-0.5 * jnp.sum((d_gb - w_gb) ** 2 / v_gb)
                - 0.5 * jnp.sum((d_gl - w_gl) ** 2 / v_gl))

    return log_lik


def laplace(log_lik, x0=THETA_TRUE):
    lp = jax.jit(lambda th: log_lik(th) + log_prior(th))
    grad, hess = jax.grad(lp)(x0), jax.hessian(lp)(x0)
    x_map = x0 - jnp.linalg.solve(hess, grad)
    sigma = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(hess))))
    return np.asarray(x_map), np.asarray(sigma)


def run_chain(log_lik, x0, cov, seed):
    from jexplore.sampler import JaxSampler, Steps
    from jexplore.sampling import EpochMH, SamplingMH
    from jexplore.steps import Stretch
    from jexplore.backends import DefaultBackend

    sampling = SamplingMH(nwalker=N_WALKERS, temps=jnp.array([1.0]),
                          loglik=log_lik, logprior=log_prior, dim=DIM)
    steps = Steps([{Stretch(permute=True).builder: 1.0}])
    p0 = jnp.asarray(x0) + jr.multivariate_normal(
        jr.PRNGKey(seed), jnp.zeros(DIM), jnp.asarray(cov) * 0.01, shape=(N_WALKERS,))
    backend = DefaultBackend(burn=N_BURN, inmem_epochs=1)
    JaxSampler(sampling, steps, backend).run(EpochMH({"p": p0}),
                                             niters=N_BURN + N_SAMP, nepoch=1, seed=seed)
    s = backend.get_samples()["p"]
    return np.asarray(jnp.array(s.transpose(0, 2, 1).reshape(-1, DIM)))


PHASES = ("checks", "scatter", "chain")


def _out(phase):
    return HERE / f"wdm_tdi2_{phase}.npz"


def phase_checks():
    """Noise-free agreement, then the realisation the paper analyses."""
    data_fd = jnp.asarray(DS["data_tdi1"])
    clean = jnp.asarray(DS["h_gb_tdi1"]) + jnp.asarray(DS["h_glitch_tdi1"])
    out = {}

    print("== noise-free data: is the machinery generation-independent? ==")
    x1, s1 = laplace(build(1, clean))
    x2, s2 = laplace(build(2, clean))
    out["clean_shift"], out["clean_ratio"] = (x2 - x1) / s1, s2 / s1
    print(f"{'param':10s}{'(MAP2-MAP1)/sigma':>20}{'sigma ratio':>14}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:10s}{out['clean_shift'][i]:20.2e}{out['clean_ratio'][i]:14.5f}")

    print("\n== the stored realisation ==")
    x1, s1 = laplace(build(1, data_fd))
    x2, s2 = laplace(build(2, data_fd))
    out.update(laplace_shift=(x2 - x1) / s1, laplace_ratio=s2 / s1,
               sigma1=s1, sigma2=s2, map1=x1, map2=x2)
    print(f"{'param':10s}{'(MAP2-MAP1)/sigma':>20}{'sigma ratio':>14}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:10s}{out['laplace_shift'][i]:20.3f}{out['laplace_ratio'][i]:14.4f}")
    return out


def phase_scatter():
    """Is the shift on one realisation a bias, or a draw?"""
    clean = jnp.asarray(DS["h_gb_tdi1"]) + jnp.asarray(DS["h_glitch_tdi1"])
    psd_bin = ns.psd_tdi1_array(f_safe, t_obs=T_OBS)
    print(f"== {N_REALISATIONS} independent noise realisations ==")
    shifts = []
    for seed in range(N_REALISATIONS):
        nz = ns.sample_noise_fd(jr.split(jr.PRNGKey(1000 + seed))[0], psd_bin)
        nz = nz.at[0].set(0 + 0j)
        a, sa = laplace(build(1, clean + nz))
        b, _ = laplace(build(2, clean + nz))
        shifts.append((b - a) / sa)
        print(f"  seed {seed:2d}: " + " ".join(f"{v:+6.2f}" for v in shifts[-1]),
              flush=True)
        jax.clear_caches()          # each closure holds ~20 MB of device arrays
    shifts = np.array(shifts)
    n = shifts.shape[0]
    print("  " + "-" * 60)
    print("  mean:    " + " ".join(f"{v:+6.2f}" for v in shifts.mean(0)))
    print("  std:     " + " ".join(f"{v:+6.2f}" for v in shifts.std(0)))
    print("  mean/se: " + " ".join(f"{v:+6.2f}"
                                   for v in shifts.mean(0) / (shifts.std(0) / np.sqrt(n))))
    return {"scatter": shifts}


def phase_chain():
    """The TDI-2 chain, against the stored TDI-1 one."""
    print("devices:", jax.devices())
    data_fd = jnp.asarray(DS["data_tdi1"])
    log_lik = build(2, data_fd)
    x2, s2 = laplace(log_lik)
    log_lik(jnp.asarray(x2)).block_until_ready()
    t0 = time.time()
    for _ in range(50):
        log_lik(jnp.asarray(x2)).block_until_ready()
    per_call = (time.time() - t0) / 50
    print(f"likelihood: {1e3 * per_call:.2f} ms/call  ->  chain should take "
          f"{per_call * N_WALKERS * (N_BURN + N_SAMP) / 60:.0f} min")

    print(f"== TDI-2 chain ({N_WALKERS} walkers x {N_BURN + N_SAMP} iters, seed "
          f"{SEED_CHAIN} -- the seed the stored TDI-1 chain used) ==", flush=True)
    t0 = time.time()
    chain2 = run_chain(log_lik, x2, np.diag(s2 ** 2), seed=SEED_CHAIN)
    print(f"  done in {time.time() - t0:.0f} s, {chain2.shape[0]:,} samples")

    chain1 = np.load(HERE / "wdm_chain.npz")["chain"]
    med1, med2 = np.median(chain1, axis=0), np.median(chain2, axis=0)
    sig1, sig2 = np.std(chain1, axis=0), np.std(chain2, axis=0)
    # Yardstick: the same statistic on two disjoint halves of the TDI-1 walkers.
    # The stored chains are walker-major, so splitting the flat array splits the
    # ensemble, and the difference of the two medians is roughly what the sampler
    # can resolve. `run_convergence.py` does this properly, from the integrated
    # autocorrelation time; this is the cheap version.
    half = chain1.reshape(2, -1, DIM)
    mc = np.abs(np.median(half[0], axis=0) - np.median(half[1], axis=0)) / sig1
    out = dict(chain2=chain2, med_shift=(med2 - med1) / sig1, sig_ratio=sig2 / sig1,
               mc_error=mc, sig1_chain=sig1, sig2_chain=sig2, med1=med1, med2=med2)
    print(f"\n{'param':10s}{'(med2-med1)/sigma':>20}{'sigma ratio':>14}{'MC error':>12}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:10s}{out['med_shift'][i]:20.3f}{out['sig_ratio'][i]:14.4f}"
              f"{mc[i]:12.3f}")
    return out


def merge():
    out = {"theta_true": np.asarray(THETA_TRUE), "labels": np.array(LABELS)}
    for phase in PHASES:
        path = _out(phase)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run `{sys.argv[0]} {phase}`")
        out |= dict(np.load(path))
    np.savez_compressed(HERE / "wdm_tdi2.npz", **out)
    print(f"saved {HERE / 'wdm_tdi2.npz'}")


def main():
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        if phase == "merge":
            merge()
            return
        if phase not in PHASES:
            raise SystemExit(f"phase must be one of {PHASES + ('merge',)}")
        np.savez_compressed(_out(phase), **globals()[f"phase_{phase}"]())
        print(f"saved {_out(phase)}")
        return
    # Default: each phase in its own process, so the chain compiles against an
    # empty XLA cache rather than behind a few dozen live likelihood closures.
    for phase in PHASES:
        print(f"\n########## {phase} ##########", flush=True)
        subprocess.run([sys.executable, "-u", __file__, phase], check=True)
    merge()


if __name__ == "__main__":
    main()
