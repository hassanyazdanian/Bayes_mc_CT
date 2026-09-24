"""
Forward-model adequacy check on the experimental Talbot-Lau data: a round trip
measured intensities -> retrieval -> FBP -> forward model -> simulated
intensities -> the same retrieval -> FBP.

What it tests. The Bayesian formulation acts on retrieved sinograms and assumes
the linear post-retrieval channel models (T and D as line integrals, DPC as the
detector derivative of the phase line integral). Those hold by construction in
the synthetic study; here they are checked against measurement. The simulated
stepping curves are built from the MEASURED reference coefficients
(TLI_2D_forward.forward_with_measured_ref), so the source spectrum, grating
imperfections and beam profile cancel and the round trip isolates the
object-dependent part of the model.

Why it is not circular. The forward projection uses this project's own
quadrature fan-beam projector, while the images come from ASTRA's FBP, so the
two discretizations are independent; and the simulated intensities go through
the identical retrieval chain as the measurement, so retrieval, wrapping and
trimming errors are not hidden.

Consistency with the reconstruction pipeline. Retrieval, corrections, geometry,
projector kernel and FBP settings are imported from real_data/data_utils.py --
the same code the reported reconstructions use -- rather than re-specified here.
The one deliberate difference is the attenuation sign: data_utils flips the sign
of T to match the forward model's convention, which the simulated stack already
follows, so that flip is undone for the simulated data (see retrieve()).

Outputs (paper_roundtrip_outputs/):
    fig_roundtrip_stepping_curves.{pdf,png}
    fig_roundtrip_sinograms.{pdf,png}
    fig_roundtrip_fbp_images.{pdf,png}     (optional, --fbp_figure)
    roundtrip_metrics.csv, roundtrip_metrics_table.tex

Usage:
    python run_validate.py            # needs CUDA for ASTRA FBP
    python run_validate.py --n_sigma_trials 0   # skip noise floor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (BASE_DIR / "common", BASE_DIR / "real_data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data_utils as du  # noqa: E402
from TLI_2D_forward import TLInterferometryForward2D  # noqa: E402

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover
    skimage_ssim = None

OUT_DIR = Path(__file__).resolve().parent / "paper_roundtrip_outputs"

# Channels shown in the main-text figure: the three the likelihood uses.
PAPER_CHANNELS: Tuple[str, ...] = ("T", "DPC", "D")
# Channels in the table; P is the integrated phase, used only by delta's FBP.
REPORT_CHANNELS: Tuple[str, ...] = ("T", "DPC", "P", "D")

# Manuscript notation: A (attenuation), C (differential phase), D (dark-field),
# P (integrated phase, FBP baseline only). Internal keys stay T/DPC/P/D.
CHANNEL_LABELS = {"T": r"$A$", "DPC": r"$C$", "P": r"$P$", "D": r"$D$"}
IMAGE_LABELS = {"mu": r"$\mu$", "delta": r"$\delta$", "eps": r"$\epsilon$"}
SINO_FOR_IMAGE = {"mu": "T", "delta": "P", "eps": "D"}

# Phase-stepping motor positions -> phase: 0.2 um steps of a 2.0 um G2 period.
G2_STEP_UM, G2_PERIOD_UM = 0.2, 2.0


def set_paper_style() -> None:
    """Matplotlib settings matching the other figures of the paper."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def thin_ticks(ax, n_x: int = 5, n_y: int = 4) -> None:
    """Few, round tick values -- crowded axes are unreadable at column width."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=n_x, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_y, prune=None))


def add_colorbar(fig, im, ax, n_ticks: int = 3, **kwargs):
    cb = fig.colorbar(im, ax=ax, **kwargs)
    cb.locator = MaxNLocator(nbins=n_ticks)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    return cb


def support_range(img_a, img_b, mask=None, q=(1.0, 99.0)):
    """Display range from the reconstruction support only: outside it the FBP
    images are dominated by streaks, which would otherwise set the scale."""
    vals = np.concatenate([np.asarray(img_a)[mask].ravel() if mask is not None else np.asarray(img_a).ravel(),
                           np.asarray(img_b)[mask].ravel() if mask is not None else np.asarray(img_b).ravel()])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, q)
    return (float(lo), float(hi)) if hi > lo else (float(lo), float(lo) + 1.0)


def save_pub_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"[saved] {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def rel_err(sim: np.ndarray, real: np.ndarray, eps: float = 1e-12) -> float:
    sim, real = np.asarray(sim, dtype=np.float64), np.asarray(real, dtype=np.float64)
    return float(np.linalg.norm(sim - real) / (np.linalg.norm(real) + eps))


def nmae(sim: np.ndarray, real: np.ndarray, eps: float = 1e-12) -> float:
    sim, real = np.asarray(sim, dtype=np.float64), np.asarray(real, dtype=np.float64)
    return float(np.mean(np.abs(sim - real)) / (np.mean(np.abs(real)) + eps))


def ssim_metric(sim: np.ndarray, real: np.ndarray) -> float:
    a, b = np.asarray(sim, dtype=np.float64), np.asarray(real, dtype=np.float64)
    lo = min(float(np.nanmin(a)), float(np.nanmin(b)))
    hi = max(float(np.nanmax(a)), float(np.nanmax(b)))
    if not np.isfinite(hi - lo) or hi <= lo or skimage_ssim is None or a.ndim != 2:
        return float("nan")
    return float(skimage_ssim(a, b, data_range=hi - lo))


def metrics_for_pair(sim: np.ndarray, real: np.ndarray) -> Dict[str, float]:
    return {"rel_error": rel_err(sim, real), "nmae": nmae(sim, real), "ssim": ssim_metric(sim, real)}


def noise_floor(I_meas: np.ndarray, I_ref: np.ndarray, sinos: Dict[str, np.ndarray],
                n_trials: int) -> Dict[str, float]:
    """Relative retrieval noise per channel, sigma_channel / RMS(channel), with
    sigma propagated from the flat-field intensity noise exactly as the
    reconstruction pipeline estimates it. A round-trip discrepancy at or below
    this level cannot be distinguished from retrieval noise."""
    if n_trials <= 0:
        return {}
    sigma_I = du.estimate_sigma_I(I_ref)
    sig = du.propagate_sigma_montecarlo(I_meas, I_ref, sigma_I, n_trials=n_trials)
    out = {}
    for ch, key in (("T", "sigma_T"), ("DPC", "sigma_DPC"), ("D", "sigma_D")):
        rms = float(np.sqrt(np.mean(np.asarray(sinos[ch], dtype=np.float64) ** 2)))
        out[ch] = sig[key] / max(rms, 1e-30)
    print(f"  sigma_I = {sigma_I:.4g}; relative retrieval noise: "
          + ", ".join(f"{k}={v:.3g}" for k, v in out.items()))
    return out


# -----------------------------------------------------------------------------
# Retrieval and geometry (same code path as the reconstruction pipeline)
# -----------------------------------------------------------------------------

def retrieve(I_meas: np.ndarray, I_ref: np.ndarray, flip_T: bool, verbose: bool = False) -> Dict[str, np.ndarray]:
    """data_utils' retrieval + corrections. The measured data need the T sign
    flip that data_utils applies; the simulated stack is generated from the
    forward model and is already in that convention, so the flip is undone."""
    sinos = du.retrieve_and_correct_sinograms(I_meas, I_ref, verbose=verbose)
    if not flip_T:
        sinos = dict(sinos)
        sinos["T"] = -sinos["T"]
    return sinos


def crop_last_dim(a: np.ndarray, n: int) -> np.ndarray:
    m = a.shape[-1]
    start = (m - n) // 2
    return a[..., start:start + n]


def crop_to_common(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
                   keys=REPORT_CHANNELS) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    out_a, out_b = {}, {}
    for k in keys:
        n = min(a[k].shape[-1], b[k].shape[-1])
        out_a[k], out_b[k] = crop_last_dim(a[k], n), crop_last_dim(b[k], n)
    return out_a, out_b


def phi_from_steps(steps: np.ndarray) -> np.ndarray:
    return (2.0 * np.pi * np.asarray(steps, dtype=np.float32) * G2_STEP_UM / G2_PERIOD_UM).astype(np.float32)


def detector_mm(n_det: int) -> np.ndarray:
    """Detector coordinate at the isocentre, in mm."""
    u = (np.arange(n_det) - 0.5 * (n_det - 1)) * du.DET_PITCH
    return u * du.SAD / (du.SAD + du.OID) * 1e3


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def select_curve_locations(sinos: Dict[str, np.ndarray], n_angles: int, n_pix: int) -> List[Tuple[int, int, str]]:
    """One central object ray and one at the strongest DPC response."""
    loc1 = (min(20, n_angles - 1), min(400, n_pix - 1), "central object ray")
    dpc = np.asarray(sinos["DPC"])
    margin = min(20, max(0, dpc.shape[1] // 10))
    work = np.abs(dpc[:, margin:dpc.shape[1] - margin]) if dpc.shape[1] > 2 * margin else np.abs(dpc)
    ai, pj = np.unravel_index(int(np.nanargmax(work)), work.shape)
    loc2 = (int(ai), int(np.clip(pj + margin + du.LEFT_TRIM, 0, n_pix - 1)), "strongest DPC response")
    if loc2[:2] == loc1[:2]:
        loc2 = (n_angles // 2, n_pix // 2, "central ray")
    return [loc1, loc2]


def plot_stepping_curves(steps, I_meas_real, I_meas_sim, locations, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), constrained_layout=True)
    for ax, (ai, pi, label) in zip(axes, locations):
        ax.plot(steps, I_meas_real[ai, :, pi], "o-", label="measured")
        ax.plot(steps, I_meas_sim[ai, :, pi], "x--", label="simulated")
        ax.set_title(f"{label} (view {ai}, pixel {pi})")
        ax.set_xlabel("phase step")
        ax.set_ylabel("intensity (counts)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        thin_ticks(ax, n_x=5, n_y=4)
    save_pub_figure(fig, out_dir, "fig_roundtrip_stepping_curves")


def plot_sinogram_grid(sinos_real, sinos_sim, beta_deg, out_dir: Path,
                       channels=PAPER_CHANNELS, stem="fig_roundtrip_sinograms") -> None:
    """Measured / simulated / difference, one row per channel. Measured and
    simulated share a colorbar per row; the difference gets its own, symmetric
    about zero, so its amplitude can be read against the signal."""
    nrows = len(channels)
    fig, axes = plt.subplots(nrows, 3, figsize=(7.1, 1.75 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for r, ch in enumerate(channels):
        real, sim = np.asarray(sinos_real[ch]), np.asarray(sinos_sim[ch])
        res = sim - real
        u = detector_mm(real.shape[1])
        extent = [u[0], u[-1], float(beta_deg[0]), float(beta_deg[-1])]

        lo, hi = np.percentile(np.concatenate([real.ravel(), sim.ravel()]), [1.0, 99.0])
        lim = float(np.percentile(np.abs(res), 99.0)) or 1.0

        ims = []
        for c, (arr, cmap, vlim) in enumerate(((real, "viridis", (lo, hi)),
                                               (sim, "viridis", (lo, hi)),
                                               (res, "coolwarm", (-lim, lim)))):
            ax = axes[r, c]
            ims.append(ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, extent=extent,
                                 vmin=vlim[0], vmax=vlim[1], interpolation="nearest", rasterized=True))
            if r == 0:
                ax.set_title(["measured", "simulated", "difference"][c])
            if r == nrows - 1:
                ax.set_xlabel("detector position (mm)")
            else:
                ax.set_xticklabels([])
            thin_ticks(ax, n_x=4, n_y=4)
            if c == 0:
                ax.set_ylabel(f"{CHANNEL_LABELS.get(ch, ch)}\nview angle (deg)")
            else:
                ax.set_yticklabels([])

        add_colorbar(fig, ims[0], [axes[r, 0], axes[r, 1]], fraction=0.035, pad=0.01)
        add_colorbar(fig, ims[2], axes[r, 2], fraction=0.046, pad=0.02)

    save_pub_figure(fig, out_dir, stem)


def plot_fbp_grid(real_imgs, sim_imgs, out_dir: Path) -> None:
    """FBP triplets. Display ranges are taken from inside the reconstruction
    support: outside it the images are streaks, and letting them set the scale
    is what made the earlier version unreadable."""
    n = np.asarray(real_imgs["mu"]).shape[0]
    c = np.linspace(-1.0, 1.0, n)
    Y, X = np.meshgrid(c, c, indexing="ij")
    support = (X ** 2 + Y ** 2) <= du.SUPPORT_R ** 2

    fig, axes = plt.subplots(3, 3, figsize=(7.1, 6.6), constrained_layout=True)
    for r, ch in enumerate(("mu", "delta", "eps")):
        real, sim = np.asarray(real_imgs[ch]), np.asarray(sim_imgs[ch])
        res = sim - real
        lo, hi = support_range(real, sim, mask=support)
        lim = float(np.percentile(np.abs(res[support]), 99.0)) or 1.0
        ims = []
        for c_i, (arr, cmap, vlim) in enumerate(((real, "gray", (lo, hi)),
                                                 (sim, "gray", (lo, hi)),
                                                 (res, "coolwarm", (-lim, lim)))):
            ax = axes[r, c_i]
            ims.append(ax.imshow(arr, origin="lower", cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                                 interpolation="nearest", rasterized=True))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(["measured", "simulated", "difference"][c_i])
            if c_i == 0:
                ax.set_ylabel(IMAGE_LABELS[ch])
        add_colorbar(fig, ims[0], [axes[r, 0], axes[r, 1]], fraction=0.035, pad=0.01)
        add_colorbar(fig, ims[2], axes[r, 2], fraction=0.046, pad=0.02)
    save_pub_figure(fig, out_dir, "fig_roundtrip_fbp_images")


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------

def write_metrics_outputs(out_dir: Path, sino_metrics: Dict[str, Dict[str, float]],
                          image_metrics: Dict[str, Dict[str, float]],
                          intensity_metric: float, noise: Dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "roundtrip_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("group,quantity,rel_error,nmae,ssim,relative_noise,rel_error_over_noise,"
                "rel_error_over_noise_floor\n")
        f.write(f"intensity,I_meas,{intensity_metric:.6g},,,,,\n")
        for ch in REPORT_CHANNELS:
            m = sino_metrics[ch]
            n = noise.get(ch, float("nan"))
            ok = n == n and n > 0
            f.write(f"sinogram,{ch},{m['rel_error']:.6g},{m['nmae']:.6g},{m['ssim']:.6g},"
                    f"{n:.6g},{m['rel_error'] / n if ok else float('nan'):.6g},"
                    f"{m['rel_error'] / (np.sqrt(2.0) * n) if ok else float('nan'):.6g}\n")
        for ch in ("mu", "delta", "eps"):
            m = image_metrics[ch]
            f.write(f"image,{ch},{m['rel_error']:.6g},{m['nmae']:.6g},{m['ssim']:.6g},,,\n")
    print(f"[saved] {csv_path}")

    tex_path = out_dir / "roundtrip_metrics_table.tex"
    has_noise = any(ch in noise for ch in REPORT_CHANNELS)
    with open(tex_path, "w") as f:
        f.write("% Generated by forward_validate/run_validate.py\n")
        f.write("\\begin{table}[!t]\n\\centering\n")
        f.write("\\caption{Round-trip consistency for the full-angle experimental data. $\\rho$ is\n"
                "the relative discrepancy between measured and simulated quantities and $n$ the\n"
                "relative retrieval noise propagated from the flat-field scans. Measured and\n"
                "simulated data carry independent noise, so an exact model would still give\n"
                "$\\rho=\\sqrt{2}\\,n$; the last row reports $\\rho/(\\sqrt{2}n)$, the excess over\n"
                "that floor. The noise floor is propagated for the three channels entering the\n"
                "likelihood; $P$ is obtained from $C$ by integration and serves only the FBP\n"
                "baseline.}\n")
        f.write("\\label{tab:real:roundtrip}\n")
        f.write("\\begin{tabular}{l" + "c" * len(REPORT_CHANNELS) + "}\n\\hline\n")
        f.write("Sinogram & " + " & ".join(CHANNEL_LABELS[c] for c in REPORT_CHANNELS) + r" \\" + "\n\\hline\n")
        # No SSIM row for the sinograms: both stacks are noise-dominated, and
        # unlike rho, SSIM has no noise floor to be judged against. It stays in
        # the CSV, and in the image table where delta's value is meaningful.
        f.write("$\\rho$ & " + " & ".join(f"{sino_metrics[c]['rel_error']:.3g}" for c in REPORT_CHANNELS)
                + r" \\" + "\n")
        if has_noise:
            f.write("Retrieval noise $n$ & "
                    + " & ".join(f"{noise[c]:.3g}" if c in noise else "--" for c in REPORT_CHANNELS)
                    + r" \\" + "\n")
            f.write("$\\rho/(\\sqrt{2}n)$ & "
                    + " & ".join(f"{sino_metrics[c]['rel_error'] / (np.sqrt(2.0) * noise[c]):.3g}"
                                 if c in noise else "--" for c in REPORT_CHANNELS)
                    + r" \\" + "\n")
        f.write("\\hline\n\\end{tabular}\n\n\\vspace{0.5em}\n\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write("FBP image & $\\mu$ & $\\delta$ & $\\epsilon$ " + r"\\" + "\n\\hline\n")
        f.write("$\\rho$ & " + " & ".join(f"{image_metrics[c]['rel_error']:.3g}"
                                          for c in ("mu", "delta", "eps")) + r" \\" + "\n")
        f.write("SSIM & " + " & ".join(f"{image_metrics[c]['ssim']:.3g}"
                                       for c in ("mu", "delta", "eps")) + r" \\" + "\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"[saved] {tex_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--n_sigma_trials", type=int, default=50,
                    help="Monte Carlo trials for the retrieval-noise floor; 0 skips it.")
    ap.add_argument("--fbp_figure", action="store_true", help="Also save the FBP image grid.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_paper_style()
    out_dir = Path(args.out_dir)
    device, dtype = torch.device(args.device), torch.float32
    print(f"Using device: {device}; projector kernel: {du.PROJECTOR_INTERP}")

    # 1. Measured intensities and the pipeline's own retrieval -------------
    data = du.load_real_intensities()
    I_meas, I_ref = data["I_meas"], data["I_ref"]
    steps, angles = data["steps"], data["angles"]
    n_angles, n_phase, n_det_raw = I_meas.shape

    sinos_real = retrieve(I_meas, I_ref, flip_T=True, verbose=True)
    beta_deg = du.make_beta_deg(angles, n_angles)
    L_phys = du.geometry_for(n_det_raw)

    # 2. Reference noise level --------------------------------------------
    print("\nRetrieval noise floor:")
    noise = noise_floor(I_meas, I_ref, sinos_real, args.n_sigma_trials)

    # 3. FBP triplet from the measured sinograms ---------------------------
    print("\nFBP from measured sinograms...")
    real_imgs = {ch: du.astra_fbp_real(sinos_real[SINO_FOR_IMAGE[ch]], beta_deg, L_phys)
                 for ch in ("mu", "delta", "eps")}

    # 4. Forward model on the measured reference ---------------------------
    # Same projector the reconstructions use (data_utils.build_projector:
    # kernel, quadrature and detector offset), at the RAW detector width so the
    # simulated stack passes through the identical trimming.
    projector, _ = du.build_projector(n_det_raw, L_phys, beta_deg, device, dtype)
    model = TLInterferometryForward2D(projector, I0=1.0, vis=0.3)  # I0/vis unused here
    out = model.forward_with_measured_ref(
        real_imgs["mu"], real_imgs["delta"], real_imgs["eps"],
        I_ref_meas=I_ref, phi=phi_from_steps(steps),
    )
    I_meas_sim = np.transpose(out.I_meas.detach().cpu().numpy(), (0, 2, 1))
    I_ref_sim = np.transpose(out.I_ref.detach().cpu().numpy(), (0, 2, 1))
    del projector

    intensity_rho = rel_err(I_meas_sim, I_meas)
    print(f"\nIntensity-level discrepancy: rho = {intensity_rho:.4g}")

    # 5. Identical retrieval of the simulated stack ------------------------
    sinos_sim = retrieve(I_meas_sim, I_ref_sim, flip_T=False)
    sinos_real_c, sinos_sim_c = crop_to_common(sinos_real, sinos_sim)
    print("  sign check (should agree): median T measured "
          f"{np.median(sinos_real_c['T']):+.3e}, simulated {np.median(sinos_sim_c['T']):+.3e}")

    print("\nSinogram round-trip metrics:")
    sino_metrics = {}
    for ch in REPORT_CHANNELS:
        sino_metrics[ch] = metrics_for_pair(sinos_sim_c[ch], sinos_real_c[ch])
        extra = ""
        if ch in noise and noise[ch] > 0:
            extra = (f"  ({sino_metrics[ch]['rel_error'] / noise[ch]:.2f}x the retrieval noise; "
                     f"{sino_metrics[ch]['rel_error'] / (np.sqrt(2.0) * noise[ch]):.2f}x the "
                     f"two-realization floor)")
        print(f"  {ch:4s}: rho={sino_metrics[ch]['rel_error']:.4g}  "
              f"NMAE={sino_metrics[ch]['nmae']:.4g}  SSIM={sino_metrics[ch]['ssim']:.4g}{extra}")

    # 6. FBP from the simulated sinograms ----------------------------------
    print("\nFBP from simulated sinograms...")
    sim_imgs = {ch: du.astra_fbp_real(sinos_sim[SINO_FOR_IMAGE[ch]], beta_deg, L_phys)
                for ch in ("mu", "delta", "eps")}
    image_metrics = {ch: metrics_for_pair(sim_imgs[ch], real_imgs[ch]) for ch in ("mu", "delta", "eps")}
    print("Image round-trip metrics:")
    for ch in ("mu", "delta", "eps"):
        print(f"  {ch:5s}: rho={image_metrics[ch]['rel_error']:.4g}  SSIM={image_metrics[ch]['ssim']:.4g}")

    # 7. Figures and tables -------------------------------------------------
    locations = select_curve_locations(sinos_real_c, n_angles, n_det_raw)
    plot_stepping_curves(np.arange(1, n_phase + 1), I_meas, I_meas_sim, locations, out_dir)
    plot_sinogram_grid(sinos_real_c, sinos_sim_c, beta_deg, out_dir, channels=PAPER_CHANNELS)
    if args.fbp_figure:
        plot_fbp_grid(real_imgs, sim_imgs, out_dir)
    write_metrics_outputs(out_dir, sino_metrics, image_metrics, intensity_rho, noise)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
