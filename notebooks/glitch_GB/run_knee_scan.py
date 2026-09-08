"""Where does the glitch/binary separation break? A scan of f0 against the knee.

The fiducial configuration of `make_dataset.py` puts the Galactic binary at
f0 = 2 mHz and the glitch's spectral knee at f_knee = 1/2*pi*tau = 0.53 mHz, a factor
3.8 apart, and the joint fit separates them almost for free. That is a property of the
configuration, not of the method, and this script maps how far it extends: it walks f0
across the band at several tau, i.e. across the dimensionless separation

    x = f0 / f_knee = 2 pi f0 tau ,

and measures, at each point, the three things that can go wrong.

What is measured
----------------
1. `deg`  -- sigma(GB parameter, glitch marginalised) / sigma(GB parameter, no glitch
   in the model). What joint fitting *costs* the binary. Unity means free.
2. `xcorr` -- the largest posterior correlation coefficient between a binary and a
   glitch parameter. The quantity Sec. "Parameter recovery" reports as < 0.02 and
   Eq. (wdmsplit) assumes to be zero.
3. `beta` -- the Cutler-Vallisneri bias of the binary parameters when the glitch is
   *not* modelled, per unit glitch SNR:

       Delta theta^i = (F_GB^-1)^{ij} ( d_j h_GB | h_gl ),
       beta = max_i |Delta theta^i| / sigma_i / rho_gl .

   The critical glitch SNR at which an unmodelled glitch moves the binary by one
   standard deviation is then rho_crit = 1 / beta. This is the analogue, for a
   Galactic binary, of the rho ~ 70 threshold that Muratore et al. (2025) find for
   massive black hole binaries.

   The bias depends on the glitch's arrival time as well as on its spectrum, and
   sharply: across the 8 uHz the binary occupies, moving t0 rotates h_gl by a phase
   that is constant to a few per cent, so the projection onto each binary derivative
   is u_j cos(phi) - v_j sin(phi) with phi = -2 pi f0 t0. Quoting the bias at one
   arrival time would make rho_crit oscillate by a factor of a few between adjacent
   grid points for no physical reason. `beta` is therefore the maximum over phi --
   the worst arrival time, which is what Muratore et al. scan for explicitly -- and
   `beta_fid` is the value at the fiducial t0 = 400 s, for scale.
4. `eps` -- the fraction of the glitch's rho^2 that falls inside the 256-bin window
   the binary occupies. This is the number the WDM split of Eq. (wdmsplit) throws
   away when it notches the binary's channel out of the glitch window, so it governs
   the time-frequency factorisation as well as the frequency-domain one.

Why the first three do not depend on how loud anything is
---------------------------------------------------------
Write the joint Fisher in blocks, F = [[A, C], [C^T, B]], with A from the binary and B
from the glitch. In the sampling coordinates of `fd_pipeline.Coords` the amplitudes
enter as A ~ A_GB^2, B ~ Deltav^2, C ~ A_GB Deltav, so the Schur complement
A - C B^-1 C^T carries no factor of Deltav and the marginalised binary block is
independent of the glitch amplitude; the same cancellation makes the cross-block
correlation coefficient amplitude-free, and makes the bias exactly linear in Deltav
with a coefficient independent of A_GB. Items 1-3 are therefore functions of the
spectral geometry (f0, tau) alone -- which is what makes a two-dimensional scan the
complete answer rather than a slice of a four-dimensional one.

The chirp
---------
Holding fdot fixed while f0 moves over a factor 17 would inject a different physical
system at every point. fdot is instead scaled at fixed chirp mass,
fdot ~ f0^(11/3), anchored on the fiducial (2 mHz, 5e-17 Hz/s).

Usage
-----
    python run_knee_scan.py                  # the Fisher scan -> knee_scan.npz
    python run_knee_scan.py --mcmc           # confirm 3 points with real chains
    python run_knee_scan.py --figure         # -> paper/figures/fig_knee_scan.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))                      # noise.py
sys.path.insert(0, os.path.join(REPO, "paper", "validation"))   # _style.py

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import noise as ns
import fd_pipeline as fp

OUT = os.path.join(HERE, "knee_scan.npz")

# fiducial system, from make_dataset.py
F0_FID, FDOT_FID, A_GB_FID = 2.0e-3, 5.0e-17, 1.0e-21
PSI, RA, DEC, IOTA, PHI0 = float(np.pi / 4), 1.0, -0.5, 1.0, 0.0
T0_FID, TAU_FID, DELTAV_FID = 400.0, 300.0, 1.0e-11
RHO_GL_FID = 42.7

# The scan grid. tau reaches down to 1 s because that is where the LPF population
# lives -- its median duration is 0.5 s -- even though such a glitch is not
# individually measurable in band (Sec. "The spectral knee and why it matters"). Below
# a few seconds the knee is far above 3 mHz, the in-band waveform loses its tau
# dependence entirely, and every row collapses onto the same curve; tau = 1 s therefore
# stands in for the whole short-duration population.
TAUS = (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
# f0 runs from below LISA's nominal 0.1 mHz edge up to the Nyquist of the analysis
# grid, because the worst case turns out to sit at f0 ~ f_knee and the knee of the
# longest glitches is below 0.1 mHz. The sub-band points are reported but not quoted
# as the headline; see the section that uses this.
N_F0 = 16
F0_MIN, F0_MAX = 5.0e-5, 2.6e-3

GB_NAMES = ("log_f0", "log_fdot", "log_A_gb", "psi")

# Half-widths of the flat analysis prior on the four binary parameters, matching the
# P--P study of the paper. They enter rho_crit and nothing else: without them the map
# is meaningless at the low-frequency end, where fdot is not measurable at all --- at
# f0 = 0.11 mHz the likelihood width of log fdot is 5 500 times the prior range, the
# binary Fisher has condition number 1e21, and both the displacement and the sigma it
# is divided by are numerical noise. Adding the prior precision 1/h^2 turns the map
# into a statement about the posterior, which is also what the chains of `mcmc()`
# measure, at the cost of a weak dependence on the box (quantified by `--check`).
PRIOR_HALF_LOG_FDOT = float(np.log(10.0))
PRIOR_HALF_LOG_A = float(np.log(3.0))
PRIOR_HALF_PSI = 0.4
PRIOR_HALF_F0_BINS = 20.0


def prior_half(f0, df):
    """Half-widths of the flat prior on (log f0, log fdot, log A_GB, psi)."""
    return np.array([PRIOR_HALF_F0_BINS * df / f0, PRIOR_HALF_LOG_FDOT,
                     PRIOR_HALF_LOG_A, PRIOR_HALF_PSI])


def fdot_of(f0):
    """Chirp at fixed chirp mass: fdot ~ f0^(11/3), anchored on the fiducial."""
    return FDOT_FID * (f0 / F0_FID) ** (11.0 / 3.0)


# ---------------------------------------------------------------------------
# one point of the grid
# ---------------------------------------------------------------------------

def _setup(N, DT, n_gb):
    grid = fp.make_grid(int(N), float(DT))
    model, n_gb = fp.make_gb_model(grid["T_OBS"], int(n_gb))
    psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=grid["T_OBS"])
    return grid, model, n_gb, psd


def _exact_loglik(grid, data, psd, model, n_gb, co, with_glitch=True):
    """Exact fine-grid log-likelihood, DC bin excluded. Used only for curvature."""
    freq = grid["freq"]

    def ll(th):
        gb8, g3 = co.to_physical(th)
        h = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"])
        if with_glitch:
            h = h + fp.glitch_fd(g3, freq)
        r = (data - h)[1:]
        return -jnp.sum((r.real ** 2 + r.imag ** 2) / psd[1:])

    return jax.jit(ll)


def point(f0, tau, grid, model, n_gb, psd, deltav=DELTAV_FID, a_gb=A_GB_FID,
          t0=T0_FID):
    """Fisher-level diagnostics for one (f0, tau).

    `deltav` and `a_gb` set the SNRs that are reported; `deg`, `xcorr` and `beta` are
    independent of both (see the module docstring), which the `--check` mode verifies.
    `t0` sets the arrival time at which `bias_fid` is evaluated; `bias` is maximised
    over arrival phase and does not depend on it.
    """
    co = fp.Coords(free_sky=False)
    gb8 = jnp.array([f0, fdot_of(f0), a_gb, RA, DEC, PSI, IOTA, PHI0])
    g3 = jnp.array([t0, deltav, tau])
    th = co.to_sampling(gb8, g3)

    freq = grid["freq"]
    h_gb = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"])
    h_gl = fp.glitch_fd(g3, freq).at[0].set(0 + 0j)

    # (a|b) = 2 Re sum_{k>0} conj(a) b / S, the convention of jaxglitches.likelihood
    ip = lambda a, b: 2.0 * float(jnp.real(jnp.sum(jnp.conj(a[1:]) * b[1:] / psd[1:])))
    r2_gb, r2_gl = ip(h_gb, h_gb), ip(h_gl, h_gl)
    rho_gb, rho_gl = np.sqrt(r2_gb), np.sqrt(r2_gl)
    overlap = ip(h_gb, h_gl) / np.sqrt(r2_gb * r2_gl)

    # the window the binary occupies, and the glitch power inside it
    k0 = int(model.get_kmin(gb8[None, 0])[0])
    sl = slice(k0, k0 + n_gb)
    w = lambda a, b: 2.0 * float(
        jnp.real(jnp.sum(jnp.conj(a[sl]) * b[sl] / psd[sl])))
    r2_gl_win, r2_gb_win = w(h_gl, h_gl), w(h_gb, h_gb)
    eps = r2_gl_win / r2_gl
    # modulus, i.e. maximised over the glitch's arrival phase (see the docstring)
    ow = complex(jnp.sum(jnp.conj(h_gb[sl]) * h_gl[sl] / psd[sl]))
    overlap_win = 2.0 * abs(ow) / np.sqrt(r2_gb_win * r2_gl_win)

    # --- joint Fisher, from the curvature of the noiseless likelihood ---------
    data = h_gb + h_gl
    F = -np.asarray(jax.hessian(_exact_loglik(grid, data, psd, model, n_gb, co))(th))
    F = 0.5 * (F + F.T)
    Cov = np.linalg.inv(F)
    sig = np.sqrt(np.diag(Cov))
    corr = Cov / np.outer(sig, sig)
    xcorr = float(np.abs(corr[:4, 4:]).max())

    # binary alone: the 4x4 block, i.e. the glitch held fixed rather than marginalised.
    # `deg` and `xcorr` are pure likelihood statements and stay unregularised, which is
    # what makes them exactly amplitude-free; the bias below uses the prior-regularised
    # version, for the reason given at the top of this file.
    A = F[:4, :4]
    sig_only = np.sqrt(np.diag(np.linalg.inv(A)))
    deg = sig[:4] / sig_only
    half = prior_half(f0, grid["df"])
    Ap = A + np.diag(1.0 / half ** 2)
    sig_post = np.sqrt(np.diag(np.linalg.inv(Ap)))

    # --- Cutler-Vallisneri bias of a fit that omits the glitch ----------------
    # The two quadratures of the arrival phase: grad(log L) is linear in the unmodelled
    # residual, so injecting h_gl and i*h_gl returns the real and imaginary parts of
    # (d_j h_GB | h_gl) from one pair of gradient evaluations.
    grad_gb = lambda d: np.asarray(jax.grad(
        _exact_loglik(grid, d, psd, model, n_gb, co, with_glitch=False))(th))[:4]
    u = grad_gb(h_gb + h_gl)
    v = -grad_gb(h_gb + 1j * h_gl)
    au, av = np.linalg.solve(Ap, u), np.linalg.solve(Ap, v)
    phi = np.linspace(0.0, 2 * np.pi, 721)
    b_phi = (np.abs(np.cos(phi)[:, None] * au - np.sin(phi)[:, None] * av)
             / sig_post)                                   # (n_phi, 4)
    k = int(np.argmax(b_phi.max(axis=1)))
    bias = b_phi[k]
    beta = float(bias.max() / rho_gl)
    bias_fid = np.abs(au) / sig_post

    return dict(f0=f0, tau=tau, x=2 * np.pi * f0 * tau, rho_gb=rho_gb, rho_gl=rho_gl,
                overlap=overlap, overlap_win=overlap_win, eps=eps, k0=k0,
                xcorr=xcorr, deg=deg, bias=bias / rho_gl, beta=beta,
                bias_fid=bias_fid / rho_gl, beta_fid=float(bias_fid.max() / rho_gl),
                # phi is measured relative to the arrival time this point was
                # evaluated at, and the bias is periodic in t0 with period 1/f0
                t0_worst=float((t0 - phi[k] / (2 * np.pi * f0)) % (1.0 / f0)),
                rho_crit=1.0 / beta, sig=sig, sig_only=sig_only,
                sig_post=sig_post, prior_half=half,
                cond=float(np.linalg.cond(F)))


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------

def scan():
    DS = np.load(os.path.join(HERE, "dataset.npz"))
    grid, model, n_gb, psd = _setup(DS["N"], DS["DT"], DS["N_GB"])
    df = grid["df"]
    # keep the whole 256-bin window inside the band, with room for the +-20 df prior
    f0s = np.geomspace(F0_MIN, F0_MAX, N_F0)
    keys = ("x", "rho_gb", "rho_gl", "overlap", "overlap_win", "eps", "xcorr",
            "beta", "beta_fid", "rho_crit", "cond", "t0_worst")
    res = {k: np.zeros((len(TAUS), N_F0)) for k in keys}
    res["deg"] = np.zeros((len(TAUS), N_F0, 4))
    res["bias"] = np.zeros((len(TAUS), N_F0, 4))

    print(f"grid: {len(TAUS)} tau x {N_F0} f0,  df = {df:.3e} Hz, "
          f"window = {n_gb * df:.3e} Hz")
    print(f"{'tau':>7s}{'f_knee/mHz':>12s}{'f0/mHz':>9s}{'x=f0/fknee':>12s}"
          f"{'eps':>10s}{'|O|':>10s}{'xcorr':>9s}{'max deg':>9s}{'rho_crit':>11s}")
    for i, tau in enumerate(TAUS):
        for j, f0 in enumerate(f0s):
            r = point(float(f0), float(tau), grid, model, n_gb, psd)
            for k in keys:
                res[k][i, j] = r[k]
            res["deg"][i, j] = r["deg"]
            res["bias"][i, j] = r["bias"]
            print(f"{tau:7.0f}{1 / (2 * np.pi * tau) * 1e3:12.3f}{f0 * 1e3:9.3f}"
                  f"{r['x']:12.3f}{r['eps']:10.2e}{abs(r['overlap']):10.2e}"
                  f"{r['xcorr']:9.4f}{r['deg'].max():9.4f}{r['rho_crit']:11.3e}",
                  flush=True)

    np.savez_compressed(OUT, f0s=f0s, taus=np.asarray(TAUS),
                        gb_names=np.array(GB_NAMES), df=df, n_gb=n_gb,
                        **res)
    print(f"\nsaved {OUT}")
    return res


def extend(f0s=(6.0e-5, 8.0e-5, 1.0e-4, 1.3e-4), taus=(300.0, 1000.0, 3000.0)):
    """Does rho_crit turn over below the scan's lowest f0, or keep falling?

    The global minimum of `scan()` sits on the low-frequency edge for the two longest
    tau, which on its own does not distinguish a minimum from a boundary. This walks
    four more points below the edge; the fiducial configuration is evaluated too, since
    the scan grid does not land exactly on it.
    """
    DS = np.load(os.path.join(HERE, "dataset.npz"))
    grid, model, n_gb, psd = _setup(DS["N"], DS["DT"], DS["N_GB"])
    print(f"{'tau':>7s}{'f0/mHz':>9s}{'x':>9s}{'eps':>11s}{'xcorr':>9s}"
          f"{'max deg':>10s}{'rho_crit':>11s}")
    for tau in taus:
        for f0 in f0s:
            r = point(float(f0), float(tau), grid, model, n_gb, psd)
            print(f"{tau:7.0f}{f0*1e3:9.3f}{r['x']:9.3f}{r['eps']:11.2e}"
                  f"{r['xcorr']:9.4f}{r['deg'].max():10.5f}{r['rho_crit']:11.1f}",
                  flush=True)
    r = point(F0_FID, TAU_FID, grid, model, n_gb, psd)
    print(f"\nthe paper's injection: f0 = {F0_FID*1e3:g} mHz, tau = {TAU_FID:g} s, "
          f"x = {r['x']:.2f}")
    print(f"  eps = {r['eps']:.3e}, |O| = {abs(r['overlap']):.2e}, "
          f"|O|_win = {r['overlap_win']:.3f}, xcorr = {r['xcorr']:.4f}")
    print(f"  max deg = {r['deg'].max():.6f}, rho_crit = {r['rho_crit']:.1f}, "
          f"bias at rho_gl = 42.7 is {r['beta']*RHO_GL_FID:.4f} sigma")


def t0check(points=((2.0e-3, 300.0), (5.0e-4, 300.0), (1.6e-4, 1000.0)),
            n_t0=61, t0_max=None):
    """Is maximising over a constant arrival phase the same as maximising over t0?

    `point()` treats the arrival time as a constant phase rotation of h_gl across the
    binary's window, which is exact only while 2 pi (Delta f_win) t0 << 1. This walks
    t0 explicitly over a full cycle 1/f0 and past it, recomputing the bias from scratch
    at each value, and compares the largest displacement found with the two-quadrature
    prediction. If they agree the maximisation is doing what the text claims.
    """
    DS = np.load(os.path.join(HERE, "dataset.npz"))
    grid, model, n_gb, psd = _setup(DS["N"], DS["DT"], DS["N_GB"])
    co = fp.Coords(free_sky=False)
    for f0, tau in points:
        r = point(float(f0), float(tau), grid, model, n_gb, psd)
        gb8 = jnp.array([f0, fdot_of(f0), A_GB_FID, RA, DEC, PSI, IOTA, PHI0])
        h_gb = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"])
        A = None
        tmax = t0_max if t0_max else 3.0 / f0
        best, arg = 0.0, 0.0
        for t0 in np.linspace(0.0, tmax, n_t0):
            g3 = jnp.array([float(t0), DELTAV_FID, tau])
            h_gl = fp.glitch_fd(g3, grid["freq"]).at[0].set(0 + 0j)
            th = co.to_sampling(gb8, g3)
            if A is None:
                F = -np.asarray(jax.hessian(_exact_loglik(
                    grid, h_gb + h_gl, psd, model, n_gb, co))(th))
                A = (0.5 * (F + F.T)[:4, :4]
                     + np.diag(1.0 / prior_half(f0, grid["df"]) ** 2))
                sig_post = np.sqrt(np.diag(np.linalg.inv(A)))
            b = np.asarray(jax.grad(_exact_loglik(
                grid, h_gb + h_gl, psd, model, n_gb, co, with_glitch=False))(th))[:4]
            v = float((np.abs(np.linalg.solve(A, b)) / sig_post).max())
            if v > best:
                best, arg = v, float(t0)
        pred = r["beta"] * r["rho_gl"]
        print(f"f0 = {f0*1e3:6.3f} mHz, tau = {tau:6g} s:  "
              f"max over t0 in [0, {tmax:.3g}] s = {best:.4f} sigma at t0 = {arg:.4g} s;"
              f"  two-quadrature prediction {pred:.4f} sigma;  ratio {best/pred:.4f}",
              flush=True)


def check():
    """The amplitude-independence the module docstring claims, verified numerically."""
    DS = np.load(os.path.join(HERE, "dataset.npz"))
    grid, model, n_gb, psd = _setup(DS["N"], DS["DT"], DS["N_GB"])
    base = point(1.0e-3, 300.0, grid, model, n_gb, psd)
    print(f"{'variation':28s}{'rho_gl':>10s}{'rho_gb':>10s}{'xcorr':>10s}"
          f"{'max deg':>10s}{'beta':>12s}")
    rows = [("reference", {}),
            ("Deltav x 30", dict(deltav=30 * DELTAV_FID)),
            ("Deltav / 30", dict(deltav=DELTAV_FID / 30)),
            ("A_GB x 10", dict(a_gb=10 * A_GB_FID)),
            ("A_GB / 10", dict(a_gb=A_GB_FID / 10))]
    for name, kw in rows:
        r = point(1.0e-3, 300.0, grid, model, n_gb, psd, **kw)
        print(f"{name:28s}{r['rho_gl']:10.2f}{r['rho_gb']:10.1f}{r['xcorr']:10.5f}"
              f"{r['deg'].max():10.5f}{r['beta']:12.5e}")
    print(f"\nreference beta = {base['beta']:.6e}")


# ---------------------------------------------------------------------------
# confirmation with real chains
# ---------------------------------------------------------------------------
#
# The rho_crit map above is a linearisation: it assumes the bias is small enough that
# the binary's likelihood is quadratic over it, which is precisely the assumption that
# has to fail somewhere. It is tested on *noiseless* data -- d = h_GB + h_gl exactly --
# because then the offset of the posterior from the injection is the systematic error
# with no statistical scatter on top, and one chain per configuration suffices where a
# noisy version would need an ensemble.

def _gb_only(grid, data, psd, model, n_gb, gb8, buf=64):
    """Exact likelihood of a Galactic-binary-only model, on the window it occupies.

    Outside the window h_GB vanishes, so the omitted bins contribute a constant and the
    posterior is exact rather than approximate -- unlike the joint case, where the
    glitch is broadband and the coarse blocks of `build_hybrid` are needed.
    """
    k0 = int(model.get_kmin(gb8[None, 0])[0])
    k_lo = max(1, k0 - buf)
    n_win = n_gb + 2 * buf
    d_win, psd_win = data[k_lo:k_lo + n_win], psd[k_lo:k_lo + n_win]
    zero32 = jnp.zeros((), jnp.int32)

    @jax.jit
    def ll(th4):
        gb = jnp.stack([jnp.exp(th4[0]), jnp.exp(th4[1]), jnp.exp(th4[2]),
                        RA, DEC, th4[3], IOTA, PHI0])
        seg, k = fp.gb_segment(model, n_gb, gb)
        h = jax.lax.dynamic_update_slice(jnp.zeros((n_win, 3), jnp.complex128), seg,
                                         ((k - k_lo).astype(jnp.int32), zero32))
        r = d_win - h
        return -jnp.sum((r.real ** 2 + r.imag ** 2) / psd_win)

    return ll


def _rho_gb(f0, a_gb, grid, model, n_gb, psd):
    gb8 = jnp.array([f0, fdot_of(f0), a_gb, RA, DEC, PSI, IOTA, PHI0])
    h = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"])
    return float(np.sqrt(2.0 * float(jnp.real(
        jnp.sum(jnp.conj(h[1:]) * h[1:] / psd[1:])))))


def mcmc(nwalkers=16, nburn=3000, nsamp=12000, rho_gb_target=283.0):
    """Test Eq. (cvbias) against the exact displaced maximum, and against a chain.

    Two design points, both learned the hard way.

    The binary's amplitude is rescaled at every f0 to hold rho_GB fixed. Left at the
    fiducial value it would fall to rho_GB ~ 0 at the low-frequency end, where the
    noise is four orders of magnitude higher, and a fit to a source that is not there
    returns the prior rather than a biased posterior. Nothing is lost by rescaling:
    the bias in units of sigma is independent of A_GB (see the docstring), so this
    fixes the experiment without changing the quantity being measured.

    The primary comparison is with the exact *maximum* of the glitch-free posterior,
    found by the damped Newton iteration of `fp.laplace`, because that is what
    Eq. (cvbias) linearises. A chain is run as well, but its median is not the same
    object: the (log f0, log fdot) block is a long curved degeneracy along which the
    median sits some tenths of a sigma from the peak even with no glitch present at
    all. That offset is measured here on glitch-free data and subtracted, so that what
    is compared is the *shift* the glitch causes.
    """
    DS = np.load(os.path.join(HERE, "dataset.npz"))
    grid, model, n_gb, psd = _setup(DS["N"], DS["DT"], DS["N_GB"])
    df = grid["df"]
    S = np.load(OUT)
    f0s, rc = S["f0s"], S["rho_crit"]
    inb = f0s >= 1.0e-4                          # LISA's nominal low-frequency edge
    i, jj = np.unravel_index(np.argmin(rc[:, inb]), rc[:, inb].shape)
    j = int(np.flatnonzero(inb)[jj])
    f_worst, t_worst, r_worst = float(f0s[j]), float(S["taus"][i]), float(rc[i, j])
    print(f"worst in-band point of the scan: f0 = {f_worst*1e3:.3f} mHz, "
          f"tau = {t_worst:g} s, x = {2*np.pi*f_worst*t_worst:.3f}, "
          f"rho_crit = {r_worst:.1f}")

    cases = [("fiducial", F0_FID, TAU_FID, RHO_GL_FID),
             (r"worst point, $\rho_{\rm crit}$", f_worst, t_worst, r_worst),
             (r"worst point, $5\rho_{\rm crit}$", f_worst, t_worst, 5 * r_worst)]

    rows = []
    for name, f0, tau, rho_target in cases:
        a_gb = A_GB_FID * rho_gb_target / _rho_gb(f0, A_GB_FID, grid, model, n_gb, psd)
        ref = point(f0, tau, grid, model, n_gb, psd, a_gb=a_gb)
        deltav = DELTAV_FID * rho_target / ref["rho_gl"]
        t0 = ref["t0_worst"]                     # the arrival time that maximises it
        r = point(f0, tau, grid, model, n_gb, psd, deltav=deltav, a_gb=a_gb, t0=t0)

        gb8 = jnp.array([f0, fdot_of(f0), a_gb, RA, DEC, PSI, IOTA, PHI0])
        g3 = jnp.array([t0, deltav, tau])
        h_gb = fp.gb_fd_full(model, n_gb, gb8, grid["n_fine"])
        h_gl = fp.glitch_fd(g3, grid["freq"]).at[0].set(0 + 0j)

        th4 = jnp.array([jnp.log(f0), jnp.log(fdot_of(f0)), jnp.log(a_gb), PSI])
        half = jnp.asarray(prior_half(f0, df))
        lo4, hi4 = th4 - half, th4 + half

        @jax.jit
        def lp4(t, lo4=lo4, hi4=hi4):
            return jnp.where(jnp.all((t >= lo4) & (t <= hi4)), 0.0, -jnp.inf)

        sig_post = r["sig_post"]
        fb = jnp.asarray(sig_post)
        out = {}
        for tag, data in (("glitch", h_gb + h_gl), ("control", h_gb)):
            ll4 = _gb_only(grid, data, psd, model, n_gb, gb8)
            xhat, sg = fp.laplace(jax.jit(lambda t: ll4(t) + lp4(t)), th4, fb)
            ch = np.asarray(fp.run_chain(ll4, lp4, xhat, sg, 4, seed=11,
                                         nwalkers=nwalkers, nburn=nburn, nsamp=nsamp))
            out[tag] = ((np.asarray(xhat) - np.asarray(th4)) / sig_post,
                        (np.median(ch, axis=0) - np.asarray(th4)) / sig_post)

        newton = out["glitch"][0] - out["control"][0]
        median = out["glitch"][1] - out["control"][1]
        predicted = r["bias_fid"] * r["rho_gl"]
        rows.append(dict(name=name, f0=f0, tau=tau, t0=t0, r=r,
                         predicted=predicted, newton=newton, median=median,
                         control=out["control"][1]))
        print(f"\n{name}: f0 = {f0*1e3:.3f} mHz, tau = {tau:g} s, t0 = {t0:.4g} s, "
              f"rho_gl = {r['rho_gl']:.1f}, rho_GB = {r['rho_gb']:.1f}")
        print(f"  {'parameter':11s}{'predicted':>11s}{'maximum':>11s}"
              f"{'median':>11s}{'(control)':>11s}   units of sigma")
        for k, nm in enumerate(GB_NAMES):
            print(f"  {nm:11s}{predicted[k]:11.4f}{newton[k]:11.4f}"
                  f"{median[k]:11.4f}{out['control'][1][k]:11.4f}")

    np.savez_compressed(os.path.join(HERE, "knee_mcmc.npz"),
                        names=np.array([r["name"] for r in rows]),
                        f0=np.array([r["f0"] for r in rows]),
                        tau=np.array([r["tau"] for r in rows]),
                        t0=np.array([r["t0"] for r in rows]),
                        rho_gl=np.array([r["r"]["rho_gl"] for r in rows]),
                        rho_gb=np.array([r["r"]["rho_gb"] for r in rows]),
                        predicted=np.array([r["predicted"] for r in rows]),
                        measured=np.array([r["newton"] for r in rows]),
                        median=np.array([r["median"] for r in rows]),
                        control=np.array([r["control"] for r in rows]),
                        gb_names=np.array(GB_NAMES))
    print(f"\nsaved {os.path.join(HERE, 'knee_mcmc.npz')}")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def figure(out="fig_knee_scan"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _style import C, FULL_IN, save

    S = np.load(OUT)
    f0s, taus = S["f0s"], S["taus"]
    cmap = plt.get_cmap("viridis")
    cols = [cmap(v) for v in np.linspace(0.05, 0.92, len(taus))]
    mccols = [C["blue"], C["orange"], C["red"]]

    fig = plt.figure(figsize=(FULL_IN, 2.5), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.062, right=0.988,
                          bottom=0.20, top=0.90)
    axa, axb, axc = (fig.add_subplot(gs[0, k]) for k in range(3))

    for i, (tau, col) in enumerate(zip(taus, cols)):
        lab = rf"$\tau={tau:g}\,$s"
        axa.loglog(f0s * 1e3, S["eps"][i], color=col, lw=1.2, label=lab)
        axb.loglog(f0s * 1e3, S["rho_crit"][i], color=col, lw=1.2, label=lab)
        fk = 1.0 / (2 * np.pi * tau) * 1e3
        for ax in (axa, axb):
            if f0s[0] * 1e3 < fk < f0s[-1] * 1e3:
                ax.axvline(fk, color=col, lw=0.7, ls=":", alpha=0.8)

    # the fiducial configuration of the paper
    i_fid = int(np.argmin(np.abs(taus - TAU_FID)))
    j_fid = int(np.argmin(np.abs(f0s - F0_FID)))
    for ax, key in ((axa, "eps"), (axb, "rho_crit")):
        ax.plot(f0s[j_fid] * 1e3, S[key][i_fid, j_fid], "k*", ms=8, zorder=5)

    axa.set_xlabel(r"$f_{0}$ [mHz]")
    axa.set_ylabel(r"$\epsilon = \rho^{2}_{\rm gl,win}/\rho^{2}_{\rm gl}$")
    axa.set_title("(a) glitch power in the window", fontsize=7, loc="left")
    axa.legend(fontsize=5.0, ncol=2, loc="lower center", handlelength=1.4,
               columnspacing=0.8, labelspacing=0.25)

    # where the LPF population actually reaches: the top per cent of a year of
    # ordinary-run glitches, excluding the single catalogue outlier that would sit
    # three decades above the top of the axis
    lpf = os.path.join(HERE, "lpf_snr.npz")
    if os.path.exists(lpf):
        L = np.load(lpf)
        rho = L["boot_rho"][L["boot_deltav"] < 1e-9]
        axb.axhspan(np.percentile(rho, 99), rho.max(), color=C["orange"], alpha=0.18,
                    lw=0, zorder=0)
        axb.text(0.985, rho.max() * 1.25, "loudest $1\\%$ of LPF-like glitches",
                 transform=axb.get_yaxis_transform(), fontsize=5.2, va="bottom",
                 ha="right", color="#8a5000")
    axb.axhline(RHO_GL_FID, color="k", lw=0.8, ls="--")
    axb.text(0.97, RHO_GL_FID * 1.12, r"fiducial $\rho_{\rm gl}=42.7$",
             transform=axb.get_yaxis_transform(), fontsize=5.5, va="bottom", ha="right")
    axb.set_ylim(0.6 * RHO_GL_FID, 1.2e4)
    axb.set_xlabel(r"$f_{0}$ [mHz]")
    axb.set_ylabel(r"$\rho_{\rm crit}$")
    axb.set_title(r"(b) glitch SNR for a $1\sigma$ bias", fontsize=7, loc="left")

    # (c) the linearisation, against chains on noiseless data
    mc = os.path.join(HERE, "knee_mcmc.npz")
    if os.path.exists(mc):
        M = np.load(mc)
        # Two estimators, each valid where the other is not. Where the binary is well
        # measured the posterior maximum is well defined and is what Eq. (cvbias)
        # linearises; where a direction is prior-dominated the maximum slides along it
        # to the wall of the box and only the median is meaningful.
        mk = ["o", "s", "^"]
        for k, nm in enumerate([str(x) for x in M["names"]]):
            axc.plot(np.abs(M["predicted"][k]), np.abs(M["median"][k]), mk[k],
                     ms=4.5, color=mccols[k], mfc=mccols[k], alpha=0.85,
                     label=nm + ", median")
            axc.plot(np.abs(M["predicted"][k]), np.abs(M["measured"][k]), mk[k],
                     ms=4.5, color=mccols[k], mfc="none",
                     label=nm + ", maximum" if k == 0 else None)
        v = np.concatenate([np.abs(M[k]).ravel()
                            for k in ("predicted", "measured", "median")])
        v = v[v > 0]
        lim = [10 ** np.floor(np.log10(v.min() * 0.7)),
               10 ** np.ceil(np.log10(v.max() * 1.4))]
        axc.plot(lim, lim, color="k", lw=0.8, ls="--")
        axc.set_xscale("log"); axc.set_yscale("log")
        axc.set_xlim(*lim); axc.set_ylim(*lim)
        axc.legend(fontsize=4.6, loc="upper left", handlelength=1.2,
                   labelspacing=0.25, borderpad=0.2)
    axc.set_xlabel(r"predicted $|\Delta\theta|/\sigma$")
    axc.set_ylabel(r"measured $|\Delta\theta|/\sigma$")
    axc.set_title("(c) the linearisation, tested", fontsize=7, loc="left")

    for ax in (axa, axb, axc):
        ax.grid(alpha=0.3, which="both", lw=0.3)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    save(fig, out, inputs=[OUT, os.path.join(HERE, "knee_mcmc.npz")])
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify that deg/xcorr/beta do not depend on the amplitudes")
    ap.add_argument("--mcmc", action="store_true",
                    help="confirm the linearised bias with chains on noiseless data")
    ap.add_argument("--figure", action="store_true", help="draw fig_knee_scan")
    ap.add_argument("--t0check", action="store_true",
                    help="scan t0 explicitly against the two-quadrature maximisation")
    ap.add_argument("--extend", action="store_true",
                    help="a low-frequency leg below the scan grid, plus the fiducial")
    a = ap.parse_args()
    if a.check:
        check()
    elif a.mcmc:
        mcmc()
    elif a.t0check:
        t0check()
    elif a.extend:
        extend()
    elif a.figure:
        figure()
    else:
        scan()
