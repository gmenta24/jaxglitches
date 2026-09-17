"""Is the Fisher/MCMC mismatch in psi and fdot a property of the binary, or of the glitch?

Sec. "Is the Fisher forecast trustworthy?" finds the Fisher width at the injection wider
than the sampled one by 17% for psi and 64% for fdot, while the three glitch parameters
and the amplitude agree to 2%, and it attributes the two exceptions to the curved
(log f0, log fdot) degeneracy of the binary, which psi inherits through its correlations
with both. The glitch is not involved in that argument, and this script checks that it
is not involved in the numbers either. It removes the glitch from the data AND from the
model -- same stored noise, same binary, four parameters instead of seven -- and repeats
the comparison:

  1. the Fisher matrix at the injection, with the glitch marginalised (7 parameters,
     joint data) against the glitch absent (4 parameters, glitch-free data): if the
     glitch mattered to the binary's widths the two would differ;
  2. a chain of the glitch-free posterior, and sigma_Fisher / sigma_MCMC for it: if the
     mismatch belonged to the glitch it would go away.

It also prints the correlations of psi in the joint chain of `fd_chains.npz`, which the
same section quotes.

The glitch-free likelihood is the hybrid binned one of `fd_pipeline.build_hybrid`,
evaluated with the glitch amplitude at 1e-30 m, where the template is zero to machine
precision; the data are the stored ones with the stored glitch signal subtracted.

    python run_gb_only.py         # a few minutes on a CPU

Writes `gb_only.npz` next to this script.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))                        # fd_pipeline.py
sys.path.insert(0, str(REPO / "notebooks"))          # noise.py

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import fd_pipeline as fp
import noise as ns

GB_NAMES = ["log_f0", "log_fdot", "log_A_gb", "psi"]
N_WALKERS, N_BURN, N_SAMP, SEED = 16, 2_000, 20_000, 1


def main():
    DS = np.load(HERE / "dataset.npz")
    grid = fp.make_grid(int(DS["N"]), float(DS["DT"]))
    T = grid["T_OBS"]
    psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=T)
    data_joint = jnp.asarray(DS["data_tdi1"])
    data_gb = data_joint - jnp.asarray(DS["h_glitch_tdi1"])     # glitch removed from the data
    model, n_gb = fp.make_gb_model(T, int(DS["N_GB"]))
    co = fp.Coords(free_sky=False)
    th7 = co.to_sampling(jnp.asarray(DS["gb_true"]), jnp.asarray(DS["glitch_true"]))
    bounds = {"log_f0": (np.log(1e-4), np.log(3e-3)),
              "log_fdot": (np.log(1e-22), np.log(1e-15)),
              "log_A_gb": (np.log(1e-25), np.log(1e-20)), "psi": (0.0, float(np.pi)),
              "t0": (0.0, float(T)), "log_Ag": (float(np.log(1e-17)), float(np.log(5e-3))),
              "log_tau": (float(np.log(0.1)), float(np.log(5e4)))}

    # --- 1. Fisher at the injection, glitch marginalised against glitch absent ----------
    hyb7, _ = fp.build_hybrid(grid, data_joint, psd, int(DS["k_min"]), model, n_gb, co)
    H7 = np.asarray(jax.hessian(hyb7)(th7))
    C7 = np.linalg.inv(-H7)
    hyb_gb, _ = fp.build_hybrid(grid, data_gb, psd, int(DS["k_min"]), model, n_gb, co)
    glitch_off = jnp.array([float(th7[4]), float(np.log(1e-30)), float(th7[6])])
    log_lik4 = jax.jit(lambda t4: hyb_gb(jnp.concatenate([t4, glitch_off])))
    th4 = th7[:4]
    H4 = np.asarray(jax.hessian(log_lik4)(th4))
    C4 = np.linalg.inv(-H4)
    s7, s4 = np.sqrt(np.diag(C7))[:4], np.sqrt(np.diag(C4))
    print("Fisher sigma at the injection")
    print(f"  {'':10s}{'glitch marginalised':>21s}{'glitch absent':>15s}{'ratio':>11s}")
    for i, n in enumerate(GB_NAMES):
        print(f"  {n:10s}{s7[i]:21.5g}{s4[i]:15.5g}{s7[i] / s4[i]:11.6f}")

    # --- 2. the glitch-free posterior, sampled ---------------------------------------
    lo = jnp.array([bounds[n][0] for n in GB_NAMES])
    hi = jnp.array([bounds[n][1] for n in GB_NAMES])
    log_prior4 = jax.jit(lambda t4: jnp.where(jnp.all((t4 >= lo) & (t4 <= hi)), 0.0, -jnp.inf))
    log_post4 = jax.jit(lambda t4: log_lik4(t4) + log_prior4(t4))
    x_map, sig = fp.laplace(log_post4, th4, jnp.full(4, 0.05))
    t_start = time.time()
    chain = np.asarray(fp.run_chain(log_lik4, log_prior4, x_map, sig, 4, seed=SEED,
                                    nwalkers=N_WALKERS, nburn=N_BURN, nsamp=N_SAMP))
    print(f"\nglitch-free chain: {N_WALKERS} walkers x {N_SAMP} iterations after "
          f"{N_BURN} burn-in, {time.time() - t_start:.0f} s")

    fd = np.load(HERE / "fd_chains.npz")
    joint = fd["chain1"]
    s_gb_mc, s_joint_mc = chain.std(axis=0), joint.std(axis=0)[:4]
    ratio_gb, ratio_joint = s4 / s_gb_mc, np.sqrt(np.diag(fd["C_fisher"]))[:4] / s_joint_mc
    print("\nsigma_Fisher(injection) / sigma_MCMC")
    print(f"  {'':10s}{'glitch absent':>15s}{'joint (Table 4)':>17s}")
    for i, n in enumerate(GB_NAMES):
        print(f"  {n:10s}{ratio_gb[i]:15.3f}{ratio_joint[i]:17.3f}")
    print("\nsampled width, glitch absent / joint: "
          + " ".join(f"{v:.3f}" for v in s_gb_mc / s_joint_mc))

    corr_gb = np.corrcoef(chain.T)
    corr_joint = np.corrcoef(joint.T)
    i_psi = GB_NAMES.index("psi")
    print("\ncorrelation of psi with log f0, log fdot, log A_gb")
    print("  joint chain:  " + " ".join(f"{corr_joint[i_psi, j]:+.3f}" for j in range(3)))
    print("  glitch absent:" + " ".join(f"{corr_gb[i_psi, j]:+.3f}" for j in range(3)))

    out = HERE / "gb_only.npz"
    np.savez_compressed(out, names=np.array(GB_NAMES), fisher_sigma_joint=s7,
                        fisher_sigma_gb_only=s4, chain_gb_only=chain,
                        fisher_over_mcmc_gb_only=ratio_gb,
                        fisher_over_mcmc_joint=ratio_joint,
                        corr_gb_only=corr_gb, corr_joint=corr_joint)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
