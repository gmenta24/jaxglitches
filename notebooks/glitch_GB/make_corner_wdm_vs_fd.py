"""Three-way corner plot: WDM split likelihood, frequency-domain MCMC, FD Fisher.

Reads the chains persisted by the two analysis notebooks, so redrawing the figure
costs seconds rather than an hour of sampling:

    wdm_chain.npz   <- glitch_and_gb_wdm.ipynb, which samples both the split WDM
                       likelihood and the exact frequency-domain one on the same data

Everything is shown in units of the frequency-domain *chain* width about the injected
values. Normalising on the chain rather than on the Fisher is deliberate: the
(log f0, log fdot) block is a long curved degeneracy and its Fisher is ill-conditioned --
recomputing it on a grid 0.7% longer moves sigma(log fdot) by a factor 2.4 while leaving
the other five parameters unchanged to 1%. The chain width is the robust yardstick, and
it puts the number the reader wants -- the WDM/FD ratio -- straight on the axes.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, 'paper', 'validation'))
from _style import FULL_IN, C, save          # noqa: E402

wd = np.load(os.path.join(HERE, 'wdm_chain.npz'), allow_pickle=True)

theta_true = wd['theta_true']
sig_fd = wd['sig_fd']                         # exact FD Fisher, WDM grid
H_fd = wd['H_fd']
labels = [str(x) for x in wd['labels']]

# Both chains come from the same run on the same dataset.npz, so they share the grid,
# the injections and the noise draw. Their centres are therefore comparable, not just
# their widths -- which is the whole point of generating the data once.
chain_wdm = wd['chain']
chain_fd = wd['chain_fd']
print(f'WDM chain {chain_wdm.shape}, FD chain {chain_fd.shape}')

sig_ref = chain_fd.std(axis=0)                # frequency-domain chain width
z_wdm = (chain_wdm - theta_true) / sig_ref
z_fd = (chain_fd - theta_true) / sig_ref
Cf = (-np.linalg.inv(H_fd)) / np.outer(sig_ref, sig_ref)  # Fisher in the same units

disp = [r'$\log f_0$', r'$\log\dot f$', r'$\log\mathcal{A}$', r'$\psi$',
        r'$t_0$', r'$\log A_g$', r'$\log\tau$']


def ellipse(ax, C2, i, j, col, ls, lw=0.9):
    sub = np.array([[C2[j, j], C2[j, i]], [C2[i, j], C2[i, i]]])
    w, v = np.linalg.eigh(sub)
    w = np.maximum(w, 0.0)
    t = np.linspace(0, 2 * np.pi, 220)
    for k in (1, 2):
        xy = v @ (np.sqrt(w)[:, None] * np.array([np.cos(t), np.sin(t)])) * k
        ax.plot(xy[0], xy[1], color=col, lw=lw, ls=ls, zorder=4)


n, LIM = 7, 4.0
sets = [(z_fd, 'frequency domain, MCMC', C['blue'], '-'),
        (z_wdm, 'WDM split likelihood', C['green'], '-')]

fig, axes = plt.subplots(n, n, figsize=(FULL_IN, FULL_IN * 0.92))
grid = np.linspace(-LIM, LIM, 240)
for r in range(n):
    for c_ in range(n):
        ax = axes[r, c_]
        if c_ > r:
            ax.set_visible(False)
            continue
        if r == c_:
            for z, _, col, ls in sets:
                ax.hist(z[:, c_], bins=70, range=(-LIM, LIM), density=True,
                        histtype='step', color=col, lw=1.2, ls=ls)
            g = np.exp(-0.5 * grid ** 2 / Cf[c_, c_])
            ax.plot(grid, g / np.trapezoid(g, grid), color=C['grey'], lw=1.0, ls='--')
            ax.set_yticks([])
        else:
            for z, _, col, ls in sets:
                k = max(1, len(z) // 2500)
                ax.plot(z[::k, c_], z[::k, r], '.', ms=0.4, color=col, alpha=0.16,
                        rasterized=True)
            ellipse(ax, Cf, r, c_, C['grey'], '--')
            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.set_ylim(-LIM, LIM)
        ax.axvline(0, color='k', lw=0.5, ls=':')
        ax.set_xlim(-LIM, LIM)
        ax.tick_params(labelsize=5, pad=1)
        if r == n - 1:
            ax.set_xlabel(disp[c_], fontsize=7)
        else:
            ax.set_xticklabels([])
        if c_ == 0 and r > 0:
            ax.set_ylabel(disp[r], fontsize=7)
        else:
            ax.set_yticklabels([])

handles = [plt.Line2D([0], [0], color=col, lw=1.4, ls=ls, label=lbl)
           for _, lbl, col, ls in sets]
handles += [plt.Line2D([0], [0], color=C['grey'], lw=1.0, ls='--',
                       label='frequency domain, Fisher'),
            plt.Line2D([0], [0], color='k', lw=0.6, ls=':', label='injected')]
fig.legend(handles=handles, fontsize=7, loc='upper right',
           bbox_to_anchor=(0.99, 0.98), frameon=False)
fig.supxlabel(r'$(\theta-\theta_{\rm true})\,/\,\sigma_{\rm FD}$  (chain width)', fontsize=8)
save(fig, 'fig_wdm_corner')

# ── the numbers the figure is making ────────────────────────────────────────
print(f'\n{"parameter":11s}{"sigma FD chain":>16s}{"WDM/FD":>9s}{"Fisher/FD":>11s}')
for i, l in enumerate(labels):
    s_w = z_wdm[:, i].std()
    print(f'{l:11s}{sig_ref[i]:16.4g}{s_w:9.3f}{np.sqrt(Cf[i, i]):11.3f}')
print('\nWDM/FD > 1 means the split likelihood is less informative, as it must be;'
      '\nFisher/FD != 1 measures how non-Gaussian the frequency-domain posterior is.')
