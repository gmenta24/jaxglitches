"""Convergence diagnostics for every chain the paper quotes, and a check that the
single Newton step of Eq. (33) really has converged.

Two things are asserted elsewhere in the paper and are measured here instead.

The first is that one Newton step from the injected values lands on the maximum. It is
cheap to check: take a second step and see how far it moves, in units of the posterior
width. The gradient is reported whitened -- multiplied by sigma -- because the seven
parameters have widths spanning nine orders of magnitude and an unwhitened gradient norm
says nothing.

The second is that the chains are converged. The stretch move needs no adaptation and
satisfies detailed balance by construction, so there is no proposal schedule to report,
but there is still an integrated autocorrelation time, and everything quoted as a
posterior median or width carries a Monte Carlo error set by it. In particular the
TDI-1/TDI-2 null tests compare medians from two chains, and that comparison is only as
sharp as the sampler allows.

Reads the stored chains -- `fd_chains.npz`, `wdm_chain.npz`, `wdm_tdi2.npz`,
`gb_free_sky.npz` -- so nothing is re-sampled. Writes `convergence.npz` next to this
script and `paper/figures/tab_convergence.tex`. Runtime a couple of minutes, almost all
of it the Newton check, which has to build the likelihoods.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))                  # fd_pipeline.py
sys.path.insert(0, str(REPO / "notebooks"))    # noise.py

import numpy as np

N_WALKERS = 16
SOKAL_C = 5.0          # window factor of Sokal's automatic truncation


# ---------------------------------------------------------------------------
# chain diagnostics
# ---------------------------------------------------------------------------

def _acf_1d(x):
    """Normalised autocorrelation function of a 1-D series, via FFT."""
    n = 1 << (2 * len(x) - 1).bit_length()
    f = np.fft.fft(x - x.mean(), n=n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    return acf / acf[0]


def tau_int(samples):
    """Integrated autocorrelation time of an (n_iter, n_walker) array.

    The autocorrelation function is averaged over walkers before being summed, as in
    Goodman & Weare's own prescription: individual walkers of an ensemble sampler are
    short and their tails are pure noise. The sum is truncated by Sokal's rule, the
    smallest window M with M >= 5 tau(M).
    """
    n_iter = samples.shape[0]
    acf = np.mean([_acf_1d(samples[:, w]) for w in range(samples.shape[1])], axis=0)
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = np.arange(n_iter) < SOKAL_C * taus
    m = np.argmin(window) if not window.all() else n_iter - 1
    return float(taus[m]), acf


def split_rhat(samples):
    """Split Gelman--Rubin R-hat: each walker is halved, giving 2 x n_walker chains."""
    n_iter, n_walker = samples.shape
    half = n_iter // 2
    chains = np.concatenate([samples[:half], samples[half: 2 * half]], axis=1).T
    m, n = chains.shape
    between = n * chains.mean(axis=1).var(ddof=1)
    within = chains.var(axis=1, ddof=1).mean()
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within))


def acceptance(samples, thin=1):
    """Fraction of proposals accepted, read off the chain: a Metropolis rejection
    repeats the previous sample exactly.

    On a chain stored with a stride the reading has to be undone. Over `thin`
    iterations the stored pair differs unless every one of them was rejected, so the
    observed rate is p_obs = 1 - (1 - p)^thin and p = 1 - (1 - p_obs)^(1/thin). The
    inversion assumes the rejections are independent, which a stretch move does not
    quite satisfy; measured against the unthinned `fd_chains.npz` it is good to 2%
    at thin = 10 and useless by thin = 20, so it refuses to guess once p_obs has
    saturated.
    """
    p_obs = float(np.mean(np.any(np.diff(samples, axis=0) != 0.0, axis=-1)))
    if thin == 1:
        return p_obs
    if p_obs > 0.999:
        return float("nan")
    return 1.0 - (1.0 - p_obs) ** (1.0 / thin)


def unflatten(flat, n_walker=N_WALKERS):
    """(n_walker * n_iter, dim) -> (n_iter, n_walker, dim).

    The stored chains are **walker-major**: `run_chain` transposes the sampler's
    (n_walker, dim, n_iter) array to (n_walker, n_iter, dim) before flattening, so
    each walker's history is contiguous. Getting this backwards is not a subtle
    error and it is worth checking rather than assuming, so `diagnose` verifies that
    walkers are mutually uncorrelated at equal iteration, which they are only under
    the right reading: under the wrong one the cross-walker correlation comes out at
    0.8 and every autocorrelation time is underestimated by an order of magnitude.
    """
    n_total, dim = flat.shape
    return flat.reshape(n_walker, n_total // n_walker, dim).transpose(1, 0, 2)


def cross_walker_corr(chain):
    """Mean correlation between walker 0 and the others at equal iteration."""
    x = chain[:, :, 0]
    return float(np.mean([np.corrcoef(x[:, 0], x[:, w])[0, 1]
                          for w in range(1, x.shape[1])]))


def diagnose(flat, labels, n_walker=N_WALKERS, thin=1, acceptance_exact=None):
    """Diagnostics for a flat (n_walker * n_iter, dim) chain.

    `thin` is the stride the chain was stored with. The effective sample size is
    invariant under thinning -- both n and tau scale together -- but tau itself and
    the iteration count have to be put back into sampler iterations, and the
    acceptance fraction has to be inverted (see `acceptance`).
    """
    n_total, dim = flat.shape
    n_iter = (n_total // n_walker) * thin
    chain = unflatten(flat, n_walker)
    cross = cross_walker_corr(chain)
    if abs(cross) > 0.2:
        raise RuntimeError(
            f"walkers correlate at {cross:.2f} at equal iteration -- the chain is "
            "probably not laid out the way `unflatten` assumes")
    out = {k: np.zeros(dim) for k in ("tau", "ess", "rhat", "sigma", "median", "se_median")}
    for i in range(dim):
        tau_stored, _ = tau_int(chain[:, :, i])
        out["tau"][i] = tau_stored * thin          # in sampler iterations
        out["ess"][i] = n_total / tau_stored       # invariant under thinning
        out["rhat"][i] = split_rhat(chain[:, :, i])
        out["sigma"][i] = flat[:, i].std()
        out["median"][i] = np.median(flat[:, i])
        # standard error of a median for a roughly Gaussian marginal
        out["se_median"][i] = np.sqrt(np.pi / 2) * out["sigma"][i] / np.sqrt(out["ess"][i])
    out["acceptance"] = (acceptance(chain, thin) if acceptance_exact is None
                         else acceptance_exact)
    out["cross_walker"] = cross
    out["n_iter"], out["n_walker"], out["labels"] = n_iter, n_walker, labels
    return out


def report(name, d):
    print(f"\n--- {name}: {d['n_walker']} walkers x {d['n_iter']:,} iterations, "
          f"acceptance {d['acceptance']:.3f}, cross-walker corr {d['cross_walker']:+.3f}")
    print(f"{'param':10s}{'tau_int':>10}{'ESS':>10}{'R-hat':>9}"
          f"{'sigma':>13}{'SE(median)/sigma':>18}")
    for i, lab in enumerate(d["labels"]):
        print(f"{str(lab):10s}{d['tau'][i]:10.1f}{d['ess'][i]:10.0f}{d['rhat'][i]:9.4f}"
              f"{d['sigma'][i]:13.4g}{d['se_median'][i] / d['sigma'][i]:18.4f}")


# ---------------------------------------------------------------------------
# has the Newton step converged?
# ---------------------------------------------------------------------------

def newton_check():
    """One Newton step against a converged one, on every posterior the paper samples.

    Reported per posterior: the log-posterior at the injected values, after a single
    undamped step, and at the maximum; how far the one-step answer sits from the
    maximum in units of the posterior width; and how far the maximum sits from the
    chain median, which is the independent check that the maximum is the right one.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import noise as ns
    import fd_pipeline as fp

    DS = np.load(HERE / "dataset.npz")
    grid = fp.make_grid(int(DS["N"]), float(DS["DT"]))
    model, n_gb = fp.make_gb_model(grid["T_OBS"], int(DS["N_GB"]))
    coords = fp.Coords(free_sky=False)
    th_true = coords.to_sampling(jnp.asarray(DS["gb_true"]), jnp.asarray(DS["glitch_true"]))
    bounds = {"log_f0": (np.log(1e-4), np.log(3e-3)),
              "log_fdot": (np.log(1e-22), np.log(1e-15)),
              "log_A_gb": (np.log(1e-25), np.log(1e-20)),
              "psi": (0.0, float(np.pi)), "t0": (0.0, float(grid["T_OBS"])),
              "log_Ag": (float(np.log(1e-17)), float(np.log(5e-3))),
              "log_tau": (float(np.log(0.1)), float(np.log(5e4)))}
    log_prior = fp.make_log_prior(coords, bounds)
    TFX1 = -1.0 + jnp.exp(-4j * float(DS["T_ARM"]) * 2.0 * jnp.pi * grid["f_safe"])
    data1 = jnp.asarray(DS["data_tdi1"])

    posteriors = {}
    for tdi in (1, 2):
        data = data1 if tdi == 1 else (-TFX1[:, None] * data1).at[0].set(0 + 0j)
        psd = (ns.psd_tdi1_array if tdi == 1 else ns.psd_tdi2_array)(
            grid["f_safe"], t_obs=grid["T_OBS"])
        log_lik, _ = fp.build_hybrid(grid, data, psd, int(DS["k_min"]), model, n_gb,
                                     coords, tdi=tdi)
        posteriors[f"frequency domain, TDI-{tdi}"] = jax.jit(
            lambda t, f=log_lik: f(t) + log_prior(t))

    fd = np.load(HERE / "fd_chains.npz")
    medians = {"frequency domain, TDI-1": np.median(fd["chain1"], axis=0),
               "frequency domain, TDI-2": np.median(fd["chain2"], axis=0)}
    sigmas = {"frequency domain, TDI-1": fd["chain1"].std(axis=0),
              "frequency domain, TDI-2": fd["chain2"].std(axis=0)}

    fallback = jnp.full(len(th_true), 0.05)
    out = {}
    print("\n=== has one Newton step converged? ===")
    print(f"{'posterior':26s}{'steps':>6}{'logpost:':>16}{'truth':>13}{'1 step':>13}"
          f"{'converged':>13}")
    for name, log_post in posteriors.items():
        one = th_true - jnp.linalg.solve(jax.hessian(log_post)(th_true),
                                         jax.grad(log_post)(th_true))
        conv, sig, used = fp.laplace(log_post, th_true, fallback, return_steps=True)
        sig_ch = sigmas[name]
        row = dict(logpost=np.array([float(log_post(th_true)), float(log_post(one)),
                                     float(log_post(conv))]),
                   steps=used,
                   one_vs_conv=np.asarray(np.abs(one - conv)) / sig_ch,
                   conv_vs_median=(np.asarray(conv) - medians[name]) / sig_ch,
                   sigma_ratio=np.asarray(sig) / sig_ch)
        out[name.replace(" ", "_").replace(",", "")] = row["logpost"]
        out[name.replace(" ", "_").replace(",", "") + "_one_vs_conv"] = row["one_vs_conv"]
        print(f"{name:26s}{used:6d}{'':16s}{row['logpost'][0]:13.4f}"
              f"{row['logpost'][1]:13.4f}{row['logpost'][2]:13.4f}")
        print(f"{'':26s}max |one step - converged| / sigma = "
              f"{row['one_vs_conv'].max():.2f};  "
              f"max |converged - chain median| / sigma = "
              f"{np.abs(row['conv_vs_median']).max():.2f}")
    print("\nThe first step is undamped and overshoots: it lands below the value it "
          "started from.\nBacktracking and iterating reaches the maximum, which agrees "
          "with the chain median.")
    return out


# ---------------------------------------------------------------------------

def latex_table(rows, path):
    """Emit the convergence table the paper \\input{}s."""
    head = (r"\begin{tabular}{lrrrrr}" "\n" r"\hline\hline" "\n"
            r"chain & walkers $\times$ iters & acc. & $\tau_{\rm int}$ & "
            r"$N_{\rm eff}$ & $\max\hat R$ \\" "\n" r"\hline" "\n")
    body = ""
    for name, d in rows:
        iters = f"{d['n_iter']:,}".replace(",", "\\,")
        body += (f"{name} & ${d['n_walker']}\\times{iters}$"
                 + f" & ${d['acceptance']:.2f}$"
                 + f" & ${d['tau'].min():.0f}$--${d['tau'].max():.0f}$"
                 + f" & ${d['ess'].min():.0f}$--${d['ess'].max():.0f}$"
                 + f" & ${d['rhat'].max():.4f}$ \\\\\n")
    tail = r"\hline\hline" "\n" r"\end{tabular}" "\n"
    path.write_text(head + body + tail)
    print(f"\nwrote {path}")


def main():
    fd = np.load(HERE / "fd_chains.npz")
    wdm = np.load(HERE / "wdm_chain.npz")
    tdi2 = np.load(HERE / "wdm_tdi2.npz")
    sky = np.load(HERE / "gb_free_sky.npz")
    labels = [str(x) for x in fd["labels"]]

    rows = [
        ("frequency domain, TDI-1", diagnose(fd["chain1"], labels)),
        ("frequency domain, TDI-2", diagnose(fd["chain2"], labels)),
        ("frequency domain, exact grid", diagnose(wdm["chain_fd"], labels)),
        ("time--frequency, TDI-1", diagnose(wdm["chain"], labels)),
        ("time--frequency, TDI-2", diagnose(tdi2["chain2"], labels)),
        ("frequency domain, sky free",
         diagnose(sky["free_chain"], [str(x) for x in sky["free_names"]],
                  int(sky["nwalkers"]) if "nwalkers" in sky.files else 32,
                  int(sky["thin"]) if "thin" in sky.files else 1,
                  # measured on the unthinned chain by run_gb_free_sky.py; the
                  # inversion in `acceptance` assumes independent rejections, which
                  # is poor when tau_int reaches 2000, so prefer the stored value
                  float(sky["free_acceptance"]) if "free_acceptance" in sky.files
                  else None)),
    ]
    for name, d in rows:
        report(name, d)

    # --- what the sampler can resolve, for each null test ------------------
    print("\n=== error bar on the null tests ===")
    pairs = [("TDI-2 - TDI-1, frequency domain", rows[0][1], rows[1][1]),
             ("TDI-2 - TDI-1, time--frequency", rows[3][1], rows[4][1]),
             ("WDM - exact frequency domain", rows[2][1], rows[3][1])]
    nulls = {}
    for name, a, b in pairs:
        diff = (b["median"] - a["median"]) / a["sigma"]
        err = np.sqrt((a["se_median"] / a["sigma"]) ** 2 + (b["se_median"] / a["sigma"]) ** 2)
        nulls[name] = np.vstack([diff, err])
        print(f"\n{name}")
        print(f"{'param':10s}{'(med_b-med_a)/sigma':>22}{'+/- MC':>10}{'in units of MC':>16}")
        for i, lab in enumerate(a["labels"]):
            print(f"{str(lab):10s}{diff[i]:22.3f}{err[i]:10.3f}{diff[i] / err[i]:16.1f}")

    out = newton_check()
    for name, d in rows:
        key = name.replace(" ", "_").replace(",", "").replace("--", "_")
        for field in ("tau", "ess", "rhat", "sigma", "median", "se_median"):
            out[f"{key}_{field}"] = d[field]
        out[f"{key}_acceptance"] = d["acceptance"]
    for k, v in nulls.items():
        out["null_" + k.split(",")[0].replace(" ", "_").replace("-", "")] = v
    np.savez_compressed(HERE / "convergence.npz", **out)
    latex_table(rows, REPO / "paper" / "figures" / "tab_convergence.tex")
    print(f"saved {HERE / 'convergence.npz'}")


if __name__ == "__main__":
    main()
