#%%
"""
Generate synthetic multi-contrast TLI (Talbot-Lau interferometry) data.

Phantom geometry lives in ``phantoms.py`` (disk, inclusion, shepp,
multicontrast). This script builds a phantom, runs the fan-beam forward
model and phase-stepping simulation, adds noise, optionally retrieves
sinograms from the noisy intensities, and saves everything under ``./obs``
next to this file.

Usage: edit the settings in ``main()`` at the bottom of this file and run
the script (or the cell) directly. No command-line arguments are used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Local imports
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
COMMON_DIR = BASE_DIR / "common"
for p in (BASE_DIR, COMMON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phantoms import (  # noqa: E402
    MULTICONTRAST_ANNOTATIONS,
    load_phantom_npz,
    make_phantom,
    plot_phantom,
    save_phantom_npz,
    set_publication_style,
)
from TLI_2D_forward import TLInterferometryForward2D, radon_fanbeam  # noqa: E402
from phase_stepping import ColumnPhaseSteppingProcessor, PhaseSteppingConfig  # noqa: E402


# -----------------------------------------------------------------------------
# Noise models
# -----------------------------------------------------------------------------

def add_gaussian_noise(I: np.ndarray, noise_level: float = 0.05, rng=None) -> Tuple[np.ndarray, float]:
    """Add white Gaussian noise with relative level ``noise_level``."""
    if rng is None:
        rng = np.random.default_rng()
    I = np.asarray(I, dtype=np.float32)
    sigma = noise_level * np.linalg.norm(I.ravel()) / np.sqrt(I.size)
    noisy = I + sigma * rng.standard_normal(I.shape)
    return noisy.astype(np.float32), float(sigma)


def add_poisson_noise(I: np.ndarray, noise_level: float = 0.05, rng=None) -> Tuple[np.ndarray, float]:
    """Add scaled Poisson noise with approximate relative level ``noise_level``."""
    if rng is None:
        rng = np.random.default_rng()
    I = np.clip(np.asarray(I, dtype=np.float32), 0.0, None)
    mean_I = float(np.mean(I))
    if mean_I <= 0.0:
        raise ValueError("Mean intensity must be positive for Poisson noise.")

    target_mean_counts = 1.0 / (noise_level ** 2)
    scale = target_mean_counts / mean_I
    noisy_counts = rng.poisson(I * scale)
    noisy = noisy_counts / scale
    sigma_est = float(np.std(noisy - I))
    return noisy.astype(np.float32), sigma_est


def add_noise(I: np.ndarray, noise_type: str = "gaussian", noise_level: float = 0.05, rng=None):
    if noise_type == "none":
        return np.asarray(I, dtype=np.float32).copy(), 0.0
    if noise_type == "gaussian":
        return add_gaussian_noise(I, noise_level=noise_level, rng=rng)
    if noise_type == "poisson":
        return add_poisson_noise(I, noise_level=noise_level, rng=rng)
    raise ValueError("noise_type must be one of {'none', 'gaussian', 'poisson'}.")


# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------

def save_h5(path: str | Path, data: Mapping[str, np.ndarray], meta: Mapping[str, object]) -> Path:
    """Save arrays as datasets and metadata as HDF5 attributes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                f.attrs[key] = arr.item()
            else:
                f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        for key, value in meta.items():
            if value is None:
                continue
            f.attrs[key] = value
    return path


def save_figure(fig, prefix: str | Path, dpi: int = 600) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=dpi)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _add_colorbar(fig, ax, im, label: Optional[str] = None):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    if label:
        cbar.set_label(label)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.7)
    cbar.outline.set_linewidth(0.7)
    return cbar


def _format_sino_axis(ax, n_angles: int, n_det: int) -> None:
    ax.set_xlabel("Detector bin")
    ax.set_ylabel("View index")
    ax.set_xticks([0, n_det // 2, n_det - 1])
    ax.set_yticks([0, n_angles // 2, n_angles - 1])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def plot_sinograms(
    sinos: Mapping[str, np.ndarray],
    save_prefix: Optional[str | Path] = None,
    show: bool = True,
):
    """Plot T, P, D, and DPC sinograms in a journal-ready 2x2 panel."""
    set_publication_style()
    order = ["T", "P", "D", "DPC"]
    titles = [r"$T=R\mu$", r"$P=R\delta$", r"$D=R\epsilon$", r"DPC"]

    T = np.asarray(sinos["T"])
    n_angles, n_det = T.shape
    extent = [0, n_det - 1, 0, n_angles - 1]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for ax, key, title in zip(axes.ravel(), order, titles):
        im = ax.imshow(
            np.asarray(sinos[key]),
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(title)
        _format_sino_axis(ax, n_angles, n_det)
        _add_colorbar(fig, ax, im)

    if save_prefix is not None:
        save_figure(fig, save_prefix)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_intensity_snapshots(
    I_meas: np.ndarray,
    I_ref: np.ndarray,
    save_prefix: Optional[str | Path] = None,
    show: bool = True,
):
    """Plot reference/measured intensity images at representative phase steps."""
    set_publication_style()
    I_meas = np.asarray(I_meas)
    I_ref = np.asarray(I_ref)
    n_angles, n_phase, n_det = I_meas.shape
    mid_step = n_phase // 2
    extent = [0, n_det - 1, 0, n_angles - 1]

    panels = [
        (I_ref[:, 0, :], r"$I_{\mathrm{ref}}$, step 0"),
        (I_meas[:, 0, :], r"$I_{\mathrm{meas}}$, step 0"),
        (I_meas[:, mid_step, :], rf"$I_{{\mathrm{{meas}}}}$, step {mid_step}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    for ax, (img, title) in zip(axes, panels):
        im = ax.imshow(img, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="nearest")
        ax.set_title(title)
        _format_sino_axis(ax, n_angles, n_det)
        _add_colorbar(fig, ax, im)

    if save_prefix is not None:
        save_figure(fig, save_prefix)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_phase_stepping_curves(
    I_meas: np.ndarray,
    I_ref: np.ndarray,
    beta_deg: np.ndarray,
    phi: np.ndarray,
    save_prefix: Optional[str | Path] = None,
    show: bool = True,
):
    """Plot representative phase-stepping curves with clean axes and legends."""
    set_publication_style()
    I_meas = np.asarray(I_meas)
    I_ref = np.asarray(I_ref)
    beta_deg = np.asarray(beta_deg).ravel()
    phi = np.asarray(phi).ravel()

    n_angles, n_phase, n_det = I_meas.shape
    if phi.size != n_phase:
        raise ValueError(f"phi has length {phi.size}, but intensity stack has {n_phase} phase steps.")

    angle_ids = [max(0, n_angles // 4), max(0, n_angles // 2)]
    det_ids = [max(0, n_det // 3), max(0, 2 * n_det // 3)]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

    det0 = n_det // 2
    for j, a in enumerate(angle_ids):
        axes[0].plot(phi, I_meas[a, :, det0], "o-", markersize=3.0, linewidth=1.0,
                     label=rf"meas., $\beta={beta_deg[a]:.1f}^\circ$")
        axes[0].plot(phi, I_ref[a, :, det0], "--", color="0.25", linewidth=1.0,
                     label="ref." if j == 0 else None)
    axes[0].set_title(rf"Fixed detector bin {det0}")
    axes[0].set_xlabel(r"Phase step $\phi$ [rad]")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.25, linewidth=0.5)
    axes[0].legend(frameon=False)

    a0 = n_angles // 4
    for j, d in enumerate(det_ids):
        axes[1].plot(phi, I_meas[a0, :, d], "o-", markersize=3.0, linewidth=1.0,
                     label=rf"meas., det. {d}")
        axes[1].plot(phi, I_ref[a0, :, d], "--", color="0.25", linewidth=1.0,
                     label="ref." if j == 0 else None)
    axes[1].set_title(rf"Fixed angle $\beta={beta_deg[a0]:.1f}^\circ$")
    axes[1].set_xlabel(r"Phase step $\phi$ [rad]")
    axes[1].set_ylabel("Intensity")
    axes[1].grid(True, alpha=0.25, linewidth=0.5)
    axes[1].legend(frameon=False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    if save_prefix is not None:
        save_figure(fig, save_prefix)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# -----------------------------------------------------------------------------
# Main data-generation pipeline
# -----------------------------------------------------------------------------

def build_phantom(
    phantom: str,
    N: int,
    supersample: int,
    phantom_npz: Optional[str | Path] = None,
) -> Dict[str, np.ndarray]:
    """Load a phantom from disk, or generate one of the named phantoms
    ("disk", "inclusion", "shepp", "multicontrast") from ``phantoms.py``."""
    if phantom_npz is not None:
        ph = load_phantom_npz(phantom_npz)
        missing = {"mu", "delta", "eps"} - set(ph)
        if missing:
            raise ValueError(f"Phantom file is missing keys: {sorted(missing)}")
        if ph["mu"].shape != (N, N):
            raise ValueError(f"Expected phantom shape {(N, N)}, got {ph['mu'].shape}. Adjust N or phantom file.")
        return ph
    return make_phantom(phantom, N=N, supersample=supersample)


def generate_synthetic_data(
    phantom: str = "inclusion",
    N: int = 128,
    supersample: int = 2,
    phantom_npz: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    n_angles: int = 180,
    n_det: int = 128,
    n_quad: int = 182,
    L_phys: float = 0.06,
    DSO_phys: float = 0.20,
    DOD_phys: float = 0.20,
    det_width_phys: float = 0.12,
    det_center_offset_phys: float = 0.0,
    I0: float = 1.0,
    vis: float = 0.3,
    n_phase: int = 10,
    noise_type: str = "gaussian",
    noise_level: float = 0.05,
    seed: int = 0,
    save_retrieved_sinos: bool = True,
    retrieval_left_trim: int = 0,
    retrieval_right_trim: int = 0,
    retrieval_bg: int = 50,
    plot: bool = True,
    show: bool = True,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Generate and save a synthetic multi-contrast TLI dataset. All outputs
    (the HDF5 dataset and every figure) are written under ``out_dir``, which
    defaults to an ``obs`` folder next to this script."""
    out_dir = Path(out_dir) if out_dir is not None else BASE_DIR / "obs"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Phantom
    # ------------------------------------------------------------------
    ph = build_phantom(phantom, N=N, supersample=supersample, phantom_npz=phantom_npz)
    mu = np.asarray(ph["mu"], dtype=np.float32)
    delta = np.asarray(ph["delta"], dtype=np.float32)
    eps = np.asarray(ph["eps"], dtype=np.float32)
    labels = np.asarray(ph.get("labels", np.zeros_like(mu, dtype=np.uint8)))

    save_phantom_npz(out_dir / f"{phantom}_phantom_{N}.npz", ph)

    # ------------------------------------------------------------------
    # 2. Forward model
    # ------------------------------------------------------------------
    torch_device = torch.device(device)
    dtype = torch.float32

    projector = radon_fanbeam(
        N_detect=n_det,
        N_pix=N,
        L_phys=L_phys,
        N_quad=n_quad,
        DSO_phys=DSO_phys,
        DOD_phys=DOD_phys,
        det_width_phys=det_width_phys,
        det_center_offset_phys=det_center_offset_phys,
        device=torch_device,
        dtype=dtype,
    )

    beta = torch.linspace(0.0, 2.0 * torch.pi, n_angles + 1, dtype=dtype, device=torch_device)[:-1]
    projector.set_view_angles(beta)

    model = TLInterferometryForward2D(projector, I0=I0, vis=vis)
    out = model.forward(mu, delta, eps, n_phase=n_phase)

    T = out.T.detach().cpu().numpy().astype(np.float32)
    P = out.P.detach().cpu().numpy().astype(np.float32)
    D = out.D.detach().cpu().numpy().astype(np.float32)
    DPC = out.DPC.detach().cpu().numpy().astype(np.float32)
    phi = out.phi.detach().cpu().numpy().astype(np.float32)
    beta_deg = np.asarray(projector.beta_deg, dtype=np.float32)

    # forward() returns (angle, detector, phase); convert to the
    # phase-stepping processor convention (angle, phase step, detector).
    I_meas_true = np.transpose(out.I_meas.detach().cpu().numpy(), (0, 2, 1)).astype(np.float32)
    I_ref_true = np.transpose(out.I_ref.detach().cpu().numpy(), (0, 2, 1)).astype(np.float32)

    dpc_abs_max = float(np.abs(DPC).max())
    if dpc_abs_max > np.pi:
        print("Warning: |DPC| exceeds pi; phase wrapping may affect retrieval.")

    # ------------------------------------------------------------------
    # 3. Add noise
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    T_obs, sigma_T = add_noise(T, noise_type=noise_type, noise_level=noise_level, rng=rng)
    P_obs, sigma_P = add_noise(P, noise_type=noise_type, noise_level=noise_level, rng=rng)
    D_obs, sigma_D = add_noise(D, noise_type=noise_type, noise_level=noise_level, rng=rng)
    DPC_obs, sigma_DPC = add_noise(DPC, noise_type=noise_type, noise_level=noise_level, rng=rng)
    I_meas, sigma_meas = add_noise(I_meas_true, noise_type=noise_type, noise_level=noise_level, rng=rng)
    I_ref, sigma_ref = add_noise(I_ref_true, noise_type=noise_type, noise_level=noise_level, rng=rng)

    # ------------------------------------------------------------------
    # 4. Optional retrieval from noisy intensities
    # ------------------------------------------------------------------
    retrieved = None
    if save_retrieved_sinos:
        pixel_size = det_width_phys / max(n_det - 1, 1) * DSO_phys / (DSO_phys + DOD_phys)
        processor = ColumnPhaseSteppingProcessor(
            PhaseSteppingConfig(
                pixel_size=pixel_size,
                harmonic=1,
                wrap_phase=True,
                norm_phase=True,
                bg=retrieval_bg,
                negate_integrated_phase=True,
                apply_neg_log_T=True,
                apply_neg_log_D=True,
                neglog_eps=1e-6,
                left_trim=retrieval_left_trim,
                right_trim=retrieval_right_trim,
            )
        )
        retrieved = processor.build_sinograms(I_meas=I_meas, I_ref=I_ref)

    # ------------------------------------------------------------------
    # 5. Save HDF5 dataset
    # ------------------------------------------------------------------
    data: Dict[str, np.ndarray] = {
        "mu": mu,
        "delta": delta,
        "eps": eps,
        "labels": labels,
        "T": T,
        "P": P,
        "D": D,
        "DPC": DPC,
        "T_obs": T_obs,
        "P_obs": P_obs,
        "D_obs": D_obs,
        "DPC_obs": DPC_obs,
        "I_meas_true": I_meas_true,
        "I_ref_true": I_ref_true,
        "I_meas": I_meas,
        "I_ref": I_ref,
        "beta_deg": beta_deg,
        "phi": phi,
        "sigma_T": np.asarray(sigma_T, dtype=np.float32),
        "sigma_P": np.asarray(sigma_P, dtype=np.float32),
        "sigma_D": np.asarray(sigma_D, dtype=np.float32),
        "sigma_DPC": np.asarray(sigma_DPC, dtype=np.float32),
    }
    if retrieved is not None:
        data.update({
            "T_rec": retrieved["T"],
            "P_rec": retrieved["P"],
            "D_rec": retrieved["D"],
            "DPC_rec": retrieved["DPC"],
        })

    meta = {
        "mode": "generate" if phantom_npz is None else "load",
        "phantom_type": phantom,
        "phantom_npz": str(phantom_npz) if phantom_npz is not None else "generated_from_phantoms.py",
        "N": int(N),
        "supersample": int(supersample),
        "n_angles": int(n_angles),
        "n_det": int(n_det),
        "n_quad": int(n_quad),
        "L_phys": float(L_phys),
        "DSO_phys": float(DSO_phys),
        "DOD_phys": float(DOD_phys),
        "det_width_phys": float(det_width_phys),
        "det_center_offset_phys": float(det_center_offset_phys),
        "I0": float(I0),
        "vis": float(vis),
        "n_phase": int(n_phase),
        "noise_model": noise_type,
        "noise_level": float(noise_level),
        "sigma_meas": float(sigma_meas),
        "sigma_ref": float(sigma_ref),
        "seed": int(seed),
        "dpc_abs_max": dpc_abs_max,
        "intensity_layout": "(n_angles, n_steps, n_pix)",
        "beta_deg_note": "Stored in projector/Astra-matching convention",
        "retrieval_left_trim": int(retrieval_left_trim),
        "retrieval_right_trim": int(retrieval_right_trim),
        "retrieval_bg": int(retrieval_bg),
    }

    h5_path = save_h5(out_dir / "synthetic_data.h5", data, meta)

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    if plot:
        annotations = MULTICONTRAST_ANNOTATIONS if phantom == "multicontrast" else None
        plot_phantom(ph, fig_dir / "phantom_truth", annotations=annotations, show=show)
        plot_sinograms({"T": T, "P": P, "D": D, "DPC": DPC}, fig_dir / "ideal_sinograms", show=show)
        plot_sinograms({"T": T_obs, "P": P_obs, "D": D_obs, "DPC": DPC_obs}, fig_dir / "noisy_sinograms", show=show)
        if retrieved is not None:
            plot_sinograms(retrieved, fig_dir / "retrieved_from_noisy_intensities", show=show)
        plot_intensity_snapshots(I_meas, I_ref, fig_dir / "intensity_snapshots", show=show)
        plot_phase_stepping_curves(I_meas, I_ref, beta_deg, phi, fig_dir / "phase_stepping_curves", show=show)

    # Console summary.
    print(f"Saved synthetic dataset: {h5_path}")
    print(f"Saved figures to: {fig_dir}")
    print(f"mu/delta/eps shape: {mu.shape}")
    print(f"T/P/D/DPC shape: {T.shape}")
    print(f"I_meas shape: {I_meas.shape}  layout=(angle, phase, detector)")
    print(f"sigma_T={sigma_T:.4g}, sigma_P={sigma_P:.4g}, sigma_D={sigma_D:.4g}, sigma_DPC={sigma_DPC:.4g}")
    print(f"sigma_meas={sigma_meas:.4g}, sigma_ref={sigma_ref:.4g}")

    return data


# -----------------------------------------------------------------------------
# SETTINGS -- edit the values below directly, then run the script or cell
# -----------------------------------------------------------------------------

def main() -> None:
    generate_synthetic_data(
        phantom="inclusion",           # "disk", "inclusion", "shepp", or "multicontrast"
        N=128,
        supersample=2,
        phantom_npz=None,              # optional phantom .npz file to load instead of generating
        out_dir=None,                  # defaults to ./obs next to this script
        n_angles=180,
        n_det=128,
        n_quad=182,
        L_phys=0.06,
        DSO_phys=0.20,
        DOD_phys=0.20,
        det_width_phys=0.12,
        det_center_offset_phys=0.0,
        I0=1.0,
        vis=0.3,
        n_phase=10,
        noise_type="gaussian",         # "none", "gaussian", or "poisson"
        noise_level=0.05,
        seed=0,
        save_retrieved_sinos=True,     # run phase retrieval from noisy intensities
        retrieval_left_trim=0,
        retrieval_right_trim=0,
        retrieval_bg=50,
        plot=True,                     # save figures
        show=True,                     # also display figures interactively
        device="cpu",
    )


if __name__ == "__main__":
    main()

# %%
