"""Export the two frequency-domain figures of the paper from the stored chains.

`fig_corner`  -- the joint posterior, split into the blocks that actually correlate
`fig_residual` -- the whitened residual after subtracting the posterior-median model

The corner plot
---------------
All seven parameters in one triangle, in units of (theta - theta_true) / sigma so that
widths spanning nine orders of magnitude share an axis. The cross-block panels carry no
structure -- every Galactic-binary/glitch correlation coefficient is below 0.02, against
0.94 within the binary block and 0.93 within the glitch block -- but they are drawn
anyway, because "the panels are empty" is a result and asserting it is weaker than
showing it. The correlation matrix goes in the free upper-right corner, where it reads
off the same statement as a number.

Requires `jaxgb` (the `joint` extra) for the residual only: the corner plot is built
from `fd_chains.npz` alone.

Usage
-----
    python make_fig_fd.py              # both figures
    python make_fig_fd.py --corner     # skip the residual, and with it jaxgb
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))                     # noise.py
sys.path.insert(0, os.path.join(REPO, "paper", "validation"))  # _style.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from scipy.ndimage import gaussian_filter

from _style import C, COL_IN, FULL_IN, save

DISP = {"log_f0": r"$\log f_0$", "log_fdot": r"$\log\dot f$",
        "log_A_gb": r"$\log\mathcal{A}$", "psi": r"$\psi$", "t0": r"$t_0$",
        "log_Ag": r"$\log A_g$", "log_tau": r"$\log\tau$"}
GB_BLOCK = ["log_f0", "log_fdot", "log_A_gb", "psi"]
GL_BLOCK = ["t0", "log_Ag", "log_tau"]
# Wide enough to hold the MCMC contours, which are centred on the noise-shifted
# maximum rather than on the truth -- the offset is the point of Sec. "Is the
# Fisher forecast trustworthy?" and must not be cropped.
LIM = 4.6


# ---------------------------------------------------------------------------
# corner
# ---------------------------------------------------------------------------

def _contour(ax, x, y, color, ls, lw, bins=60, smooth=1.0):
    """1 and 2 sigma contours of a 2-D sample cloud."""
    H, xe, ye = np.histogram2d(x, y, bins=bins,
                               range=[[-LIM, LIM], [-LIM, LIM]])
    H = gaussian_filter(H, smooth)
    if H.sum() <= 0:
        return
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    levels = [flat[np.searchsorted(csum, f)] for f in (0.393, 0.865)][::-1]
    if levels[0] >= levels[1]:
        return
    ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T,
               levels=levels, colors=color, linestyles=ls, linewidths=lw)


def _ellipse(ax, cov, sig_ref, i, j, color, lw):
    """1 and 2 sigma Fisher ellipses, standardised by the *MCMC* widths.

    Dividing the Fisher covariance by its own diagonal would force every ellipse to
    unit radius and throw away the comparison the figure exists to make: on these
    axes an ellipse wider than the contour means the forecast is wider than the
    posterior, which is the statement of Sec. "Is the Fisher forecast trustworthy?".
    """
    c = np.array([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])
    c = c / np.outer([sig_ref[i], sig_ref[j]], [sig_ref[i], sig_ref[j]])
    w, v = np.linalg.eigh(c)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    for k in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(max(w[-1], 0)),
                             2 * k * np.sqrt(max(w[0], 0)),
                             angle=ang, fill=False, edgecolor=color, lw=lw, ls="--",
                             alpha=0.95))


def _panels(fig, gs, order, idx, chains, covs, sig_ref, colors, n_gb_block):
    """Full lower-triangle corner over `order`, in units of sigma."""
    n = len(order)
    for r in range(n):
        for c in range(r + 1):
            ax = fig.add_subplot(gs[r, c])
            ir, ic = idx[order[r]], idx[order[c]]
            cross = (r < n_gb_block) != (c < n_gb_block)
            if cross:
                # the panels whose emptiness is the result of Sec. "Parameter recovery"
                ax.set_facecolor("#f2f2f2")
            if r == c:
                for k, (ch, col) in enumerate(zip(chains.values(), colors)):
                    ax.hist(ch[:, ir], bins=70, range=(-LIM, LIM), density=True,
                            histtype="step", color=col, lw=1.3 if k == 0 else 0.8)
                for k, (cv, col) in enumerate(zip(covs.values(), colors[2:])):
                    w = np.sqrt(cv[ir, ir]) / sig_ref[ir]   # Fisher width in MCMC units
                    t = np.linspace(-LIM, LIM, 200)
                    ax.plot(t, np.exp(-0.5 * (t / w) ** 2) / (w * np.sqrt(2 * np.pi)),
                            color=col, lw=1.3 if k == 0 else 0.8, ls="--")
                ax.set_yticks([])
                ax.axvline(0.0, color="k", lw=0.6, ls=":")
            else:
                for k, (ch, col) in enumerate(zip(chains.values(), colors)):
                    _contour(ax, ch[:, ic], ch[:, ir], col, "-", 1.2 if k == 0 else 0.7)
                for k, (cv, col) in enumerate(zip(covs.values(), colors[2:])):
                    _ellipse(ax, cv, sig_ref, ic, ir, col, 1.1 if k == 0 else 0.6)
                ax.axvline(0.0, color="k", lw=0.5, ls=":")
                ax.axhline(0.0, color="k", lw=0.5, ls=":")
                ax.set_ylim(-LIM, LIM)
                ax.set_yticks([-3, 0, 3])
            ax.set_xlim(-LIM, LIM)
            ax.set_xticks([-3, 0, 3])
            ax.tick_params(labelsize=5.5, length=2, pad=1.5)
            ax.grid(False)
            if r != n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(DISP[order[c]], fontsize=8, labelpad=2)
            if c != 0 or r == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(DISP[order[r]], fontsize=8, labelpad=2)


def make_corner(Z, out="fig_corner"):
    labels = [str(x) for x in Z["labels"]]
    idx = {n: i for i, n in enumerate(labels)}
    order = GB_BLOCK + GL_BLOCK
    assert sorted(order) == sorted(labels)
    th = Z["theta_true"]

    # one standardisation for everything on the figure -- the TDI-1 MCMC widths --
    # so that TDI-2 and the two Fisher forecasts can be read against it
    sig_ref = Z["chain1"].std(axis=0)
    ch1 = (Z["chain1"] - th) / sig_ref
    ch2 = (Z["chain2"] - th) / sig_ref
    R = np.corrcoef(Z["chain1"][:, [idx[n] for n in order]].T)

    chains = {"MCMC TDI-1": ch1, "MCMC TDI-2": ch2}
    covs = {"Fisher TDI-1": np.cov(Z["fish1"].T), "Fisher TDI-2": np.cov(Z["fish2"].T)}
    colors = [C["blue"], C["red"], C["green"], C["purple"]]
    n_gb = len(GB_BLOCK)

    fig = plt.figure(figsize=(FULL_IN, 6.4), constrained_layout=False)
    gs = fig.add_gridspec(7, 7, wspace=0.09, hspace=0.09, left=0.062,
                          right=0.988, bottom=0.068, top=0.988)
    _panels(fig, gs, order, idx, chains, covs, sig_ref, colors, n_gb)

    # --- legend, in the free space above the diagonal -----------------------
    axl = fig.add_subplot(gs[0:2, 2:5]); axl.axis("off")
    handles = ([plt.Line2D([], [], color=c, lw=1.3 if i == 0 else 0.9, label=k)
                for i, (k, c) in enumerate(zip(chains, colors))]
               + [plt.Line2D([], [], color=c, lw=1.3 if i == 0 else 0.9, ls="--",
                             label=k) for i, (k, c) in enumerate(zip(covs, colors[2:]))]
               + [Patch(facecolor="#f2f2f2", edgecolor="0.6",
                        label="cross-block panels")])
    axl.legend(handles=handles, fontsize=8, loc="center", ncol=1,
               frameon=False, handlelength=1.9, labelspacing=0.55,
               borderpad=0.2)

    # --- the correlation matrix, in the same free space ---------------------
    axC = fig.add_subplot(gs[0:3, 5:7])
    im = axC.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    axC.set_xticks(range(7)); axC.set_yticks(range(7))
    axC.set_xticklabels([DISP[n] for n in order], fontsize=6, rotation=90)
    axC.set_yticklabels([DISP[n] for n in order], fontsize=6)
    axC.tick_params(length=2, pad=1)
    axC.xaxis.tick_top()
    for i in range(7):
        for j in range(7):
            axC.text(j, i, f"{R[i, j]:.2f}".lstrip("0").replace("-0.", "-."),
                     ha="center", va="center", fontsize=4.6,
                     color="white" if abs(R[i, j]) > 0.55 else "black")
    for e in (n_gb - 0.5,):                     # the block boundary
        axC.axhline(e, color="k", lw=0.9)
        axC.axvline(e, color="k", lw=0.9)
    xmax = np.abs(R[:n_gb, n_gb:]).max()
    axC.set_xlabel(f"correlation, TDI-1\n" rf"cross-block $|\rho|\leq{xmax:.3f}$",
                   fontsize=8, labelpad=4)
    axC.grid(False)
    cb = fig.colorbar(im, ax=axC, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2)

    print(f"largest cross-block |rho| = {xmax:.4f}")
    save(fig, out, inputs=[os.path.join(HERE, "fd_chains.npz")])
    plt.close(fig)
    return xmax


# ---------------------------------------------------------------------------
# residual
# ---------------------------------------------------------------------------

def make_residual(out="fig_residual"):
    import jax.numpy as jnp
    import noise as ns
    import fd_pipeline as fp

    DS = np.load(os.path.join(HERE, "dataset.npz"))
    Z = np.load(os.path.join(HERE, "fd_chains.npz"))
    grid = fp.make_grid(int(DS["N"]), float(DS["DT"]))
    model, n_gb = fp.make_gb_model(grid["T_OBS"], int(DS["N_GB"]))
    co = fp.Coords(free_sky=False)
    freq = grid["freq"]

    # the second-generation stream is the same physical realisation seen through
    # h^(2) = (1 - D^4) h^(1), applied to signal and noise alike -- see the data-set
    # section -- so it is derived here rather than stored twice
    g2 = fp.gen2(freq)[:, None]
    data = {1: jnp.asarray(DS["data_tdi1"])}
    data[2] = data[1] * g2

    curves = {}
    for gen, chain_key in ((1, "chain1"), (2, "chain2")):
        psd = (ns.psd_tdi1_array if gen == 1 else ns.psd_tdi2_array)(
            grid["f_safe"], t_obs=grid["T_OBS"])
        th = jnp.median(jnp.asarray(Z[chain_key]), axis=0)
        gb8, g3 = co.to_physical(th)
        h = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"]) + fp.glitch_fd(g3, freq)
        if gen == 2:
            h = h * g2
        r = data[gen] - h
        curves[gen] = np.asarray(jnp.mean(jnp.abs(r[1:]) ** 2 / psd[1:], axis=1))
    f = np.asarray(freq[1:])

    fig, ax = plt.subplots(figsize=(COL_IN, COL_IN * 0.62))
    for gen, col, lw in ((1, C["blue"], 0.35), (2, C["orange"], 0.2)):
        ax.loglog(f, curves[gen], color=col, lw=lw, alpha=0.45)
    # running median in log-frequency bins, one curve per generation
    edges = np.geomspace(f[0], f[-1], 45)
    which = np.digitize(f, edges)
    for gen, col, lw in ((1, C["blue"], 1.9), (2, C["orange"], 1.0)):
        med = np.array([np.median(curves[gen][which == b]) if (which == b).any()
                        else np.nan for b in range(1, len(edges))])
        ax.loglog(0.5 * (edges[1:] + edges[:-1]), med, color=col, lw=lw,
                  label=f"TDI-{gen}, running median")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="noise floor")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|r|^{2}/S_{n}$")
    ax.set_ylim(2e-3, 3e2)
    ax.legend(fontsize=5.5, loc="lower right")
    ax.grid(alpha=0.3, which="both", lw=0.3)
    for gen in (1, 2):
        print(f"TDI-{gen}: mean whitened residual power = {curves[gen].mean():.4f}")
    save(fig, out, inputs=[os.path.join(HERE, "dataset.npz"),
                           os.path.join(HERE, "fd_chains.npz")])
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner", action="store_true", help="corner plot only")
    ap.add_argument("--residual", action="store_true", help="residual only")
    a = ap.parse_args()
    both = not (a.corner or a.residual)
    if both or a.corner:
        make_corner(np.load(os.path.join(HERE, "fd_chains.npz")))
    if both or a.residual:
        make_residual()
