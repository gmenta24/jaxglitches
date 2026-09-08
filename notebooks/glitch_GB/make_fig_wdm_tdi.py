"""Redraw `paper/figures/fig_wdm_tdi.pdf` from the stored chains.

Reads `wdm_tdi2.npz` (written by `run_wdm_tdi2.py`) and `wdm_chain.npz` (written by
`glitch_and_gb_wdm.ipynb`), so the figure can be remade in seconds without re-running
the sampler -- the same arrangement as `make_corner_wdm_vs_fd.py`.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "paper" / "validation"))
from _style import FULL_IN, C, save  # noqa: E402

T = np.load(HERE / "wdm_tdi2.npz")
# Monte Carlo error on the median, from the integrated autocorrelation time if
# `run_convergence.py` has been run, otherwise the cruder walker-split estimate
# stored by `run_wdm_tdi2.py`.
MC = T["mc_error"]
if (HERE / "convergence.npz").exists():
    _c = np.load(HERE / "convergence.npz")
    _s = _c["time_frequency_TDI-1_sigma"]
    MC = np.sqrt((_c["time_frequency_TDI-1_se_median"] / _s) ** 2
                 + (_c["time_frequency_TDI-2_se_median"] / _s) ** 2)
DISPLAY = [r"$\log f_0$", r"$\log\dot f$", r"$\log\mathcal{A}$", r"$\psi$",
           r"$t_0$", r"$\log A_g$", r"$\log\tau$"]
N_GB_PAR = 4                      # the first four are the Galactic-binary block
x = np.arange(len(DISPLAY))
FLOOR = 1e-7                      # so that exact zeros stay on a log axis

fig, axes = plt.subplots(1, 2, figsize=(FULL_IN, 2.7))

# --- (a) how far apart are the two generations? ------------------------------
ax = axes[0]
series = [("noise-free data", T["clean_shift"], C["green"], "^", 4.5),
          ("stored realisation, Laplace", T["laplace_shift"], C["blue"], "o", 4.0),
          ("stored realisation, chains", T["med_shift"], C["red"], "s", 4.0)]
for label, v, col, mark, ms in series:
    ax.semilogy(x, np.maximum(np.abs(v), FLOOR), mark, ms=ms, color=col, label=label)
ax.semilogy(x, np.maximum(MC, FLOOR), "_", ms=11, color=C["grey"],
            label="Monte Carlo error on the median")
ax.axvline(N_GB_PAR - 0.5, color=C["grey"], lw=0.7, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(DISPLAY)
ax.set_ylim(3e-8, 3e3)
ax.set_ylabel(r"$|\theta^{(2)}-\theta^{(1)}|\,/\,\sigma$")
ax.legend(loc="upper left", fontsize=5.5)
ax.set_title("(a) the two generations compared", loc="left")

# --- (b) is the difference a bias or the noise? ------------------------------
ax = axes[1]
scatter = T["scatter"]
rng = np.random.default_rng(0)
for i in range(len(DISPLAY)):
    ax.plot(i + rng.uniform(-0.16, 0.16, scatter.shape[0]), scatter[:, i], ".",
            ms=3, color=C["grey"], alpha=0.8,
            label=f"{scatter.shape[0]} noise realisations" if i == 0 else None)
ax.plot(x, T["laplace_shift"], "o", ms=4.5, color=C["blue"],
        label="the realisation analysed")
ax.axhline(0.0, color=C["grey"], lw=0.7, ls=":")
ax.axvline(N_GB_PAR - 0.5, color=C["grey"], lw=0.7, ls=":")
ax.text(1.5, 1.45, "binary block", fontsize=6, color=C["grey"], ha="center")
ax.text(5.0, 1.45, "glitch block", fontsize=6, color=C["grey"], ha="center")
ax.set_ylim(-1.75, 1.75)
ax.set_xticks(x)
ax.set_xticklabels(DISPLAY)
ax.set_ylabel(r"$(\theta^{(2)}-\theta^{(1)})\,/\,\sigma$")
ax.legend(loc="lower left", fontsize=5.5)
ax.set_title("(b) over independent noise draws", loc="left")

save(fig, "fig_wdm_tdi",
     inputs=[HERE / "wdm_tdi2.npz", HERE / "wdm_chain.npz",
             HERE / "convergence.npz"])
print("\nsummary:")
for i, lab in enumerate(T["labels"]):
    print(f"  {str(lab):10s} clean {T['clean_shift'][i]:+9.1e}   "
          f"Laplace {T['laplace_shift'][i]:+7.3f}   chains {T['med_shift'][i]:+7.3f}   "
          f"MC {MC[i]:6.3f}   sigma ratio {T['sig_ratio'][i]:.4f}")
