"""fig_unmodelled.pdf -- what leaving the glitch out of the model costs, and the
excision the time--frequency representation makes available.

Reads `unmodelled.npz`, written by `run_unmodelled.py`. Computes nothing itself.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "paper" / "validation"))
from _style import FULL_IN, C, save  # noqa: E402

ANALYSES = ("fd_joint", "fd_gbonly", "wdm_split", "wdm_sub", "wdm_gbonly", "wdm_cut")
# Analyses whose noiseless bias is identically zero, so a log axis cannot show them,
# and the one that duplicates another curve exactly.
ZERO_BIAS = ("fd_joint", "wdm_sub")
DUPLICATE = "wdm_gbonly"                       # equals fd_gbonly to 1e-4 sigma
FLOOR = 3e-3                                   # bottom of the log axis in panel (c)

STYLE = {
    "fd_joint":   dict(color=C["blue"],   marker="o", ls="-",  label="FD, joint"),
    "fd_gbonly":  dict(color=C["red"],    marker="s", ls="-",  label="FD, glitch omitted"),
    "wdm_split":  dict(color=C["orange"], marker="^", ls="--", label="WDM, split as written"),
    "wdm_sub":    dict(color=C["cyan"],   marker="P", ls="-",
                       label=r"WDM, $h^{\rm gl}$ in $\mathcal{W}_{\rm GB}$ too"),
    "wdm_gbonly": dict(color=C["purple"], marker="v", ls=":",  label="WDM, glitch omitted"),
    "wdm_cut":    dict(color=C["green"],  marker="D", ls="-",  label="WDM, omitted, onset excised"),
}


def contour(ax, x, y, color, ls="-", lw=1.1, fill=False, bins=48):
    """1- and 2-sigma credible contours of a 2-D sample, lightly smoothed."""
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.1) ** 2)
    k /= k.sum()
    for ax_ in (0, 1):                      # separable Gaussian smoothing
        H = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax_, H)
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    lev = sorted({flat[np.searchsorted(csum, q)] for q in (0.865, 0.393)})
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    if fill:
        ax.contourf(xc, yc, H.T, levels=lev + [H.max() * 1.01], colors=color, alpha=0.18)
    ax.contour(xc, yc, H.T, levels=lev, colors=color, linestyles=ls, linewidths=lw)


def main():
    D = np.load(HERE / "unmodelled.npz", allow_pickle=True)
    n_cut, gl_rm, gb_kept = D["cut_info"]

    # explicit margins: the shared legend needs room the constrained layout
    # engine will not leave it
    fig = plt.figure(figsize=(FULL_IN, 5.6), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                  bottom=0.20, top=0.955, left=0.075, right=0.93)

    # ---- (a) where the glitch sits in the binary's window -------------------
    ax = fig.add_subplot(gs[0, 0])
    t = D["time_gb"] / 86400.0
    dt = t[1] - t[0]
    ax.step(t, D["p_gl_t"] / D["p_gl_t"].sum(), where="mid", color=C["red"], lw=1.0,
            label="glitch")
    ax.step(t, D["p_gb_t"] / D["p_gb_t"].sum(), where="mid", color=C["blue"], lw=1.0,
            label="binary")
    for k in D["cut_bins"]:
        ax.axvspan(t[k] - 0.5 * dt, t[k] + 0.5 * dt, color=C["green"], alpha=0.30, lw=0)
    ax.plot([], [], color=C["green"], alpha=0.45, lw=6, label="excised")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 5.0)
    ax.set_xlim(t[0] - dt, t[-1] + dt)
    ax.set_xlabel("time bin centre [d]")
    ax.set_ylabel(r"fraction of $\rho^{2}$ in $\mathcal{W}_{\rm GB}$")
    ax.set_title("(a) the binary's window, bin by bin", loc="left")
    ax.legend(loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.3,
              fontsize=6, borderaxespad=0.2)

    # ---- (b) the excision trade-off ----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    nb, _, gbk = D["trade"].T
    ok = D["trade_map_ok"].astype(bool)
    ax.plot(nb, D["trade_bias"], "-", color=C["purple"], lw=1.1, zorder=3,
            label="linearised")
    ax.plot(nb[ok], D["trade_map"][ok], "s", color=C["grey"], ms=3.5, mfc="none",
            zorder=4, label="posterior maximum")
    ax.plot(nb[~ok], D["trade_map"][~ok], "x", color=C["grey"], ms=4.5, zorder=4)
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.axvline(n_cut, color=C["green"], lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("time bins excised")
    ax.set_ylabel(r"$\max_i|\Delta\theta^i|/\sigma_i$")
    ax.set_title(r"(b) what the excision buys ($\rho_{\rm gl}=4\rho_{\rm crit}$)",
                 loc="left")
    ax.legend(loc="lower left", handlelength=1.6, fontsize=6)
    axr = ax.twinx()
    axr.plot(nb, 100 * (1 - gbk), color=C["blue"], lw=0.9, ls="-.")
    axr.set_ylabel(r"binary SNR lost [\%]", color=C["blue"], fontsize=7)
    axr.tick_params(axis="y", colors=C["blue"], labelsize=6)
    axr.grid(False)

    # ---- (c) the ladder -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    r = D["rungs"]
    m = r > 0
    for j, name in enumerate(ANALYSES):
        if name == DUPLICATE:
            continue
        st = STYLE[name]
        if name not in ZERO_BIAS:              # identically zero: nothing to show
            ax.plot(r[m], np.abs(D["z_clean"][m, j]).max(axis=1), st["ls"], lw=1.1,
                    color=st["color"])
        ax.plot(r[m], np.clip(np.abs(D["z_noisy"][m, j]).max(axis=1), FLOOR, None),
                st["marker"], ms=3.6, color=st["color"], mfc="none")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.axvline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 60.0)
    ax.set_xlabel(r"$\rho_{\rm gl}/\rho_{\rm crit}$")
    ax.set_ylabel(r"$\max_i|\hat\theta^i-\theta^i_{\rm true}|/\sigma_i$")
    ax.set_title("(c) how the bias grows", loc="left")
    ax.text(0.03, 0.94, "lines: noiseless\nsymbols: one noise draw", fontsize=6,
            transform=ax.transAxes, va="top")

    # ---- (d) the posteriors themselves -------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    sig = D["chain_fd_joint"].std(axis=0)[:4]
    tru = D["true_fd_joint"][:4]
    i, j = 3, 0                                     # psi against log f0
    for name in ("fd_joint", "wdm_sub", "wdm_cut", "fd_gbonly", "wdm_split"):
        ch = D[f"chain_{name}"]
        contour(ax, (ch[:, i] - tru[i]) / sig[i], (ch[:, j] - tru[j]) / sig[j],
                STYLE[name]["color"], ls=STYLE[name]["ls"],
                fill=(name == "fd_joint"))
    ax.plot(0, 0, "*", color="k", ms=9, zorder=6)
    ax.set_xlabel(r"$(\psi-\psi^{\rm true})/\sigma$")
    ax.set_ylabel(r"$(\log f_0-\log f_0^{\rm true})/\sigma$")
    ax.set_title(rf"(d) posteriors at $\rho_{{\rm gl}}={float(D['chain_rho_gl']):.0f}$",
                 loc="left")

    # ---- one legend for the whole figure ------------------------------------
    handles = [plt.Line2D([], [], color=STYLE[n]["color"], ls=STYLE[n]["ls"],
                          marker=STYLE[n]["marker"], mfc="none", ms=4, lw=1.1,
                          label=STYLE[n]["label"])
               for n in ANALYSES if n != DUPLICATE]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5,
               handlelength=2.2, columnspacing=1.6, frameon=False,
               bbox_to_anchor=(0.5, 0.012))

    save(fig, "fig_unmodelled", inputs=[HERE / "unmodelled.npz"])
    plt.close(fig)


if __name__ == "__main__":
    main()
