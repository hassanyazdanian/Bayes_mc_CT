"""
Compare FBP, channel-wise TV, and joint-TV (JTV) reconstructions of
(mu, delta, eps) across four limited-data scenarios:

  1. full        -- all angles, all phase steps
  2. sparse_angle -- reduced projection angles, full phase steps
  3. sparse_step  -- full angles, reduced phase steps
  4. combined     -- reduced angles AND reduced phase steps

Regularization strength (alpha_tv, alpha_joint) is held fixed across all
four scenarios, matching the paper's own convention (Table II/III: fixed
alpha_J across N_theta=361 and N_theta=30).

The FBP baseline uses ASTRA's FBP_CUDA (ram-lak) via common/fanbeam_astra.py,
which requires a CUDA-enabled ASTRA build (astra.astra.use_cuda() must be
True). This script will raise a RuntimeError at the first FBP call on a
CPU-only machine.

Outputs (all under obs/scenarios/):
  <scenario>/comparison_true_fbp_tv_jtv.png   -- 3x4 grid (Fig. 2 layout)
  <scenario>/metrics_table.png                -- per-channel RE/SSIM/PSNR (Table I layout)
  summary_metrics.png                         -- RE and SSIM vs. scenario, all methods/channels
  summary_metrics.csv                         -- same data, machine-readable
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
for p in (BASE_DIR, COMMON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from TLI_2D_forward import radon_fanbeam  # noqa: E402
from fanbeam_astra import FanbeamGeometry, FanbeamReconstructor2D, ReconConfig  # noqa: E402
from scenario_utils import make_sparse_angle_indices, simulate_and_retrieve  # noqa: E402
from map_tv_jtv import compute_metrics, reconstruct_map  # noqa: E402
from phantoms import PHANTOM_NAMES, make_phantom  # noqa: E402
from create_data import calibrate_phantom_scale  # noqa: E402
import time

METHODS = ["FBP", "TV", "JTV"]
CHANNELS = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]

# Fixed regularization across all scenarios, tuned against ground truth
# (T/DPC/D likelihood, 512^2 bilinear simulation, 256^2 nearest
# reconstruction, noise 1/4.7/4.8 %, N_ITER=1000):
#  - tv_beta (recon/sweep_tv_beta.py --stage alpha_beta, full
#    scan): beta in [1e-4, 1e-3] ties on RE (worst channel within 3.4% of its
#    best); beta=3e-4 taken -- 1e-3 leaves visible background texture in eps,
#    and 3e-4 is 3-10x less stiff than 1e-4.
#  - alpha_mu/delta/eps (--stage alpha_beta --betas 3e-4, run on all four
#    scenarios): each minimizes RE averaged over the scenarios, and also the
#    worst-case excess over each scenario's own optimum (mu +1.5%, delta
#    +2.4%, eps +3.9%). The full-scan optima (177.8/1778/100) over-regularize
#    limited data -- alpha_delta=1778 doubles delta's RE on combined.
#  - lambda0 (--stage lambda0, full scan and combined): largest value leaving
#    the MAP unchanged in every channel and scenario -- on combined, 1 costs
#    delta +0.24% RE and 5 already +1.4% (the full scan tolerates up to 10).
#    The alpha and alpha_joint sweeps ran at lambda0=5; the difference is
#    below their resolution.
#  - alpha_joint (tune_alpha_joint.py --scenario <each>): minimizes
#    the summed per-channel RE change vs TV averaged over the four scenarios
#    (-0.77%; every scenario improves, worst -0.23%); the gain is mostly eps
#    (RE -1.3%, SSIM +0.008..0.013). 58.17 and above degrade the sparse
#    scenarios.
ALPHA_MU = 100
ALPHA_DELTA = 316.2
ALPHA_EPS = 56.23
ALPHA_JOINT = 29.55
TV_BETA = 3e-4
LAMBDA0 = 1.0
N_ITER = 1000

# Data are SIMULATED on a SIM_UPSAMPLE x finer grid (512^2) with a bilinear
# projector and RECONSTRUCTED on the N x N grid (256^2) with nearest,
# deliberately:
#  - the ground truth must not depend on the reconstruction kernel: the
#    phantom contrast is calibrated from max|DPC| of a trial projection, and
#    nearest's pixel staircase inflates that maximum;
#  - generating and inverting data with the identical discrete operator is
#    an inverse crime -- which for DPC is severe: the detector derivative
#    amplifies every discretization difference;
#  - nearest matches the real-data pipeline and is ~2.9x cheaper per gradient.
# The cost is a DPC model mismatch of ~24% of the DPC signal (~5 sigma_DPC on
# the full scan), versus ~11% even for a 256^2 bilinear reconstruction.
# The ground truth used for metrics is the 512^2 phantom averaged to 256^2.
SIM_INTERP = "bilinear"
RECON_INTERP = "nearest"
SIM_UPSAMPLE = 2

# Per-channel noise, relative to the RMS of each noiseless sinogram. Photon
# statistics of phase stepping predict noise in the retrieved phase and
# dark-field exceeding that of attenuation by sqrt(2)/V and sqrt(1 + 2/V^2)
# (Weber et al. 2011; Chabior et al. 2011): 4.71 and 4.82 at V = 0.3. Those
# are ratios of absolute noise standard deviations; since the phantom's
# channel amplitudes are in normalized units, they are adopted here as ratios
# of the relative levels.
NOISE_LEVEL = {"T": 0.01, "DPC": 0.047, "D": 0.048}
RETRIEVAL_BG = 5
SEED = 0

# FBP baseline uses ASTRA's FBP_CUDA -- requires a CUDA-enabled ASTRA build
# (astra.astra.use_cuda() must be True). CPU-only ASTRA builds only support
# SIRT/SART/CGLS for fan-beam data; see common/fanbeam_astra.py.
FBP_FILTER = "ram-lak"

N_SPARSE_ANGLES = 30
N_SPARSE_PHASE = 4

# Set to one of phantoms.PHANTOM_NAMES ("disk", "inclusion", "shepp",
# "multicontrast") to reconstruct a freshly-built phantom instead of the one
# baked into obs/synthetic_data.h5. Geometry (N, n_det, DSO_phys, ...) is
# still taken from the h5 file; only mu/delta/eps are replaced. None keeps
# the dataset's own phantom.
PHANTOM_NAME: Optional[str] = "multicontrast"
PHANTOM_SUPERSAMPLE = 2
PHANTOM_TARGET_DPC_MAX = 0.8 * np.pi


def synchronize_device(device) -> None:
    """Synchronize CUDA before/after timing GPU operations."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_dataset(h5_path: Path) -> Dict:
    with h5py.File(h5_path, "r") as f:
        d = {
            "mu": f["mu"][:], "delta": f["delta"][:], "eps": f["eps"][:],
            "beta_deg": f["beta_deg"][:],
            "N": int(f.attrs["N"]), "n_det": int(f.attrs["n_det"]), "n_quad": int(f.attrs["n_quad"]),
            "L_phys": float(f.attrs["L_phys"]), "DSO_phys": float(f.attrs["DSO_phys"]),
            "DOD_phys": float(f.attrs["DOD_phys"]), "det_width_phys": float(f.attrs["det_width_phys"]),
            "det_center_offset_phys": float(f.attrs["det_center_offset_phys"]),
            "I0": float(f.attrs["I0"]), "vis": float(f.attrs["vis"]),
            "n_phase_full": int(f.attrs["n_phase"]),
        }
    d["n_angles_full"] = len(d["beta_deg"])
    return d


def override_phantom(d: Dict, phantom_name: str, supersample: int, target_dpc_max: float, device, dtype) -> Dict:
    """Replace the phantom with a freshly-built one on the SIM_UPSAMPLE x finer
    simulation grid, DPC-wrap calibrated on that grid with the simulation
    projector (see calibrate_phantom_scale). d['mu'/'delta'/'eps'] become its
    block average on the N x N reconstruction grid (the ground truth for
    metrics); d['mu_sim'/...] and d['N_sim'] hold the simulation phantom."""
    if phantom_name not in PHANTOM_NAMES:
        raise ValueError(f"phantom_name={phantom_name!r} must be one of {PHANTOM_NAMES}")

    N, N_sim = d["N"], SIM_UPSAMPLE * d["N"]
    ph = make_phantom(phantom_name, N=N_sim, supersample=supersample)
    calib_projector = build_projector(d, np.arange(d["n_angles_full"]), device, dtype, interp=SIM_INTERP, N=N_sim)
    ph, scale = calibrate_phantom_scale(ph, calib_projector, I0=d["I0"], vis=d["vis"], target_dpc_max=target_dpc_max)
    del calib_projector
    print(f"Phantom override: '{phantom_name}' on {N_sim}^2 (supersample={supersample}), "
          f"contrast_scale={scale:.4g}; ground truth = {N_sim}^2 averaged to {N}^2")

    d = dict(d)
    d["N_sim"] = N_sim
    for ch in ("mu", "delta", "eps"):
        sim = np.asarray(ph[ch], dtype=np.float32)
        d[f"{ch}_sim"] = sim
        d[ch] = sim.reshape(N, SIM_UPSAMPLE, N, SIM_UPSAMPLE).mean(axis=(1, 3)).astype(np.float32)
    return d


def build_projector(d: Dict, angle_idx: np.ndarray, device, dtype, interp: str = RECON_INTERP,
                    N: Optional[int] = None) -> radon_fanbeam:
    """Projector for the given views; N defaults to the reconstruction grid."""
    N = d["N"] if N is None else int(N)
    projector = radon_fanbeam(
        N_detect=d["n_det"], N_pix=N, L_phys=d["L_phys"],
        N_quad=d["n_quad"] if N == d["N"] else int(round(N * np.sqrt(2))),
        DSO_phys=d["DSO_phys"], DOD_phys=d["DOD_phys"], det_width_phys=d["det_width_phys"],
        det_center_offset_phys=d["det_center_offset_phys"], device=device, dtype=dtype,
        interp=interp,
    )
    beta_rad = torch.as_tensor(np.deg2rad(d["beta_deg"][angle_idx]), dtype=dtype, device=device)
    projector.set_view_angles(beta_rad)
    return projector


def det_spacing_of(d: Dict) -> float:
    """Detector sample spacing at the isocenter (the DPC derivative step)."""
    return d["det_width_phys"] / (d["n_det"] - 1) * d["DSO_phys"] / (d["DSO_phys"] + d["DOD_phys"])


def simulate_data(d: Dict, angle_idx: np.ndarray, n_phase: int, device, dtype, seed: int = SEED):
    """Simulate and retrieve one scenario's noisy T/DPC/D (+ integrated P) on the
    simulation grid with the simulation projector. Every script that needs
    synthetic data goes through here, so MAP, tuning and NUTS see identical data."""
    N_sim = d.get("N_sim", d["N"])
    fields = [d.get(f"{ch}_sim", d[ch]) for ch in ("mu", "delta", "eps")]
    sim_projector = build_projector(d, angle_idx, device, dtype, interp=SIM_INTERP, N=N_sim)
    noisy, sigma, _clean = simulate_and_retrieve(
        *fields, sim_projector, d["I0"], d["vis"], n_phase, d["n_phase_full"],
        pixel_size=det_spacing_of(d), noise_level=NOISE_LEVEL, bg=RETRIEVAL_BG, seed=seed,
    )
    del sim_projector
    return noisy, sigma


def astra_fbp(sino: np.ndarray, beta_deg: np.ndarray, d: Dict) -> np.ndarray:
    """FBP baseline via ASTRA's FBP_CUDA (ram-lak). Requires a CUDA-enabled
    ASTRA build; raises RuntimeError otherwise (see fanbeam_astra.py)."""
    det_pitch = d["det_width_phys"] / max(d["n_det"] - 1, 1)
    geom = FanbeamGeometry(SAD=d["DSO_phys"], SDD=d["DSO_phys"] + d["DOD_phys"], det_pitch=det_pitch, L_phys=d["L_phys"])
    cfg = ReconConfig(N=d["N"], method="FBP", fbp_filter=FBP_FILTER)
    R = FanbeamReconstructor2D(geom, cfg)
    R.set_angles_deg(beta_deg)
    R.set_sinogram(sino)
    rec = R.reconstruct()
    # ASTRA's returned array uses row 0 = y-max, the opposite of this
    # project's own convention (row 0 = y-min, e.g. phantoms.py's _grid and
    # this project's mu/delta/eps arrays). Flip here so the array itself --
    # not just the plot -- is directly comparable (compute_metrics does a
    # raw, unregistered numpy comparison against mu_true/delta_true/eps_true).
    return np.flipud(rec).copy()


# -----------------------------------------------------------------------------
# One scenario
# -----------------------------------------------------------------------------

def run_scenario(name: str, d: Dict, n_angles: int, n_phase: int, device="cpu", dtype=torch.float32) -> Dict:
    print(f"\n=== Scenario: {name} (n_angles={n_angles}, n_phase={n_phase}) ===")

    angle_idx = make_sparse_angle_indices(d["n_angles_full"], n_angles)
    noisy, sigma = simulate_data(d, angle_idx, n_phase, device, dtype)
    projector = build_projector(d, angle_idx, device, dtype, interp=RECON_INTERP)
    det_spacing = det_spacing_of(d)

    recon = {"FBP": {}, "TV": {}, "JTV": {}}

    runtime = {
        "FBP": np.nan,
        "TV": np.nan,
        "JTV": np.nan,
    }

    fbp_runtime_by_channel = {}

    # TV/JTV first: zero-init (standard cold start), decoupled from FBP so
    # they don't depend on ASTRA/CUDA at all -- lets this run to completion
    # on a CPU-only machine even though FBP (below) requires CUDA. JTV
    # warm-starts from TV's result.

    synchronize_device(device)
    t0 = time.perf_counter()

    tv_out = reconstruct_map(
        noisy["T"], noisy["DPC"], noisy["D"], projector, d["N"],
        sigma["T"], sigma["DPC"], sigma["D"], det_spacing,
        alpha_mu=ALPHA_MU, alpha_delta=ALPHA_DELTA, alpha_eps=ALPHA_EPS, alpha_joint=0.0,
        tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER,
        device=device, dtype=dtype,
    )
    synchronize_device(device)
    runtime["TV"] = (time.perf_counter() - t0) * 1000.0
    print(f"  TV runtime:  {runtime['TV']:.2f} ms")

    for ch in ("mu", "delta", "eps"):
        recon["TV"][ch] = tv_out[ch]

    synchronize_device(device)
    t0 = time.perf_counter()

    jtv_out = reconstruct_map(
        noisy["T"], noisy["DPC"], noisy["D"], projector, d["N"],
        sigma["T"], sigma["DPC"], sigma["D"], det_spacing,
        alpha_mu=ALPHA_MU, alpha_delta=ALPHA_DELTA, alpha_eps=ALPHA_EPS, alpha_joint=ALPHA_JOINT,
        tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER,
        # init_mu=tv_out["mu"], init_delta=tv_out["delta"], init_eps=tv_out["eps"],
        device=device, dtype=dtype,
    )
    synchronize_device(device)
    runtime["JTV"] = (time.perf_counter() - t0) * 1000.0
    print(f"  JTV runtime: {runtime['JTV']:.2f} ms")

    for ch in ("mu", "delta", "eps"):
        recon["JTV"][ch] = jtv_out[ch]

    # FBP last -- requires CUDA (see astra_fbp). Caught per-channel so a
    # CPU-only run still returns complete, saveable TV/JTV results for all
    # four scenarios instead of crashing on the first FBP call; recon["FBP"]
    # entries stay None until run on a CUDA machine.
    beta_deg_scenario = np.asarray(projector.beta_deg, dtype=np.float32)

    fbp_total = 0.0
    for name_ch, sino_key in [("mu", "T"), ("delta", "P"), ("eps", "D")]:
        try:
            t0 = time.perf_counter()
            recon["FBP"][name_ch] = astra_fbp(noisy[sino_key], beta_deg_scenario, d)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            fbp_runtime_by_channel[name_ch] = elapsed_ms
            fbp_total += elapsed_ms

            print(f"  FBP {name_ch:5s} runtime: {elapsed_ms:.2f} ms")

        except RuntimeError as e:
            print(f"  FBP {name_ch}: skipped ({e})")
            recon["FBP"][name_ch] = None
            fbp_runtime_by_channel[name_ch] = np.nan

    if any(np.isfinite(v) for v in fbp_runtime_by_channel.values()):
        runtime["FBP"] = fbp_total

    metrics = {}
    for method in METHODS:
        for ch in ("mu", "delta", "eps"):
            if recon[method][ch] is None:
                continue
            re, s, p = compute_metrics(d[ch], recon[method][ch])
            metrics[(method, ch)] = {"RE": re, "SSIM": s, "PSNR": p}
            print(f"  {method:4s} {ch:5s}: RE={re:.4f} SSIM={s:.4f} PSNR={p:.2f}")

    return {"name": name, "n_angles": n_angles, "n_phase": n_phase, "recon": recon, "metrics": metrics, "runtime": runtime, "fbp_runtime_by_channel": fbp_runtime_by_channel}


def save_recon_arrays(d: Dict, result: Dict, save_path: Path) -> None:
    """Save this scenario's raw reconstructed arrays (mu/delta/eps for each
    method that succeeded) plus ground truth, so figures can be regenerated
    later without recomputing -- e.g. after running FBP on a CUDA machine."""
    arrays = {"mu_true": d["mu"], "delta_true": d["delta"], "eps_true": d["eps"]}
    for method in METHODS:
        for ch in ("mu", "delta", "eps"):
            img = result["recon"][method][ch]
            if img is not None:
                arrays[f"{method}_{ch}"] = img
    np.savez(save_path, **arrays)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_comparison_grid(d: Dict, result: Dict, save_path: Path, cmap: str = "gray") -> None:
    """3x4 grid: rows = mu/delta/eps, cols = True/FBP/TV/JTV (paper Fig. 2 layout)."""
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.2))
    col_titles = ["True", "FBP", "TV", "JTV"]

    for i, (ch, label) in enumerate(CHANNELS):
        true_img = d[ch]
        imgs = [true_img] + [result["recon"][m][ch] for m in METHODS]
        vmax = max(float(np.clip(im, 0, None).max()) for im in imgs if im is not None)
        vmax = max(vmax, 1e-12)
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(label, fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            if img is None:
                ax.text(0.5, 0.5, "N/A\n(needs CUDA)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                if i == 0:
                    ax.set_title(col_titles[j], fontsize=13)
                continue
            # All four columns share the same array convention (row 0 =
            # y-min, per phantoms.py's _grid), so they must all use the same
            # origin -- treating True differently here was a leftover from
            # the old reference script and displayed it upside down.
            im = ax.imshow(np.clip(img, 0, None), cmap=cmap, origin="lower", vmin=0.0, vmax=vmax)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=13)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{result['name']}  (angles={result['n_angles']}, phase steps={result['n_phase']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table(result: Dict, save_path: Path) -> None:
    """Channel x Method table of reconstruction metrics and runtime."""
    rows = []

    for ch, label in CHANNELS:
        for method in METHODS:
            m = result["metrics"].get((method, ch))

            # Runtime
            rt = result["runtime"].get(method, np.nan)

            rt_str = f"{rt:.2f}" if np.isfinite(rt) else "N/A"

            if m is None:
                rows.append([
                    label,
                    method,
                    "N/A",
                    "N/A",
                    "N/A",
                    rt_str,
                ])
                continue

            rows.append([
                label,
                method,
                f"{m['RE']:.4f}",
                f"{m['SSIM']:.4f}",
                f"{m['PSNR']:.2f}",
                rt_str,
            ])

    fig, ax = plt.subplots(
        figsize=(6.7, 0.35 * len(rows) + 1.0)
    )
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Channel",
            "Method",
            "RE",
            "SSIM",
            "PSNR",
            "Runtime [ms]",
        ],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    ax.set_title(
        f"{result['name']}: reconstruction metrics",
        fontsize=11,
        pad=14,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results: List[Dict], save_path: Path, csv_path: Path) -> None:
    """RE and SSIM vs. scenario, grouped by method, one subplot per channel."""
    scenario_names = [r["name"] for r in results]
    colors = {"FBP": "tab:gray", "TV": "tab:blue", "JTV": "tab:red"}

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    x = np.arange(len(scenario_names))
    width = 0.25

    csv_lines = ["scenario,channel,method,RE,SSIM,PSNR,runtime_ms"]
    for col, (ch, label) in enumerate(CHANNELS):
        for row, metric_name in enumerate(["RE", "SSIM"]):
            ax = axes[row, col]
            for k, method in enumerate(METHODS):
                vals = [r["metrics"].get((method, ch), {}).get(metric_name, np.nan) for r in results]
                if all(np.isnan(v) for v in vals):
                    continue
                ax.bar(x + (k - 1) * width, vals, width, label=method, color=colors[method])
            ax.set_xticks(x)
            ax.set_xticklabels(scenario_names, rotation=20, ha="right", fontsize=8)
            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(metric_name)
            if row == 0 and col == 2:
                ax.legend(fontsize=8, frameon=False)

        for r in results:
            for method in METHODS:
                m = r["metrics"].get((method, ch))
                if m is None:
                    continue
                # csv_lines.append(f"{r['name']},{ch},{method},{m['RE']:.6f},{m['SSIM']:.6f},{m['PSNR']:.4f}")
                rt = r["runtime"].get(method, np.nan)

                csv_lines.append(
                    f"{r['name']},{ch},{method},"
                    f"{m['RE']:.6f},{m['SSIM']:.6f},{m['PSNR']:.4f},"
                    f"{rt:.6f}"
                )
    fig.suptitle("FBP vs. TV vs. JTV across limited-data scenarios", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path.write_text("\n".join(csv_lines) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    d = load_dataset(BASE_DIR / "obs" / "synthetic_data.h5")

    out_dir = BASE_DIR / "obs" / "scenarios"
    if PHANTOM_NAME is not None:
        d = override_phantom(d, PHANTOM_NAME, PHANTOM_SUPERSAMPLE, PHANTOM_TARGET_DPC_MAX, device, dtype)
        out_dir = out_dir / PHANTOM_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("full", d["n_angles_full"], d["n_phase_full"]),
        ("sparse_angle", N_SPARSE_ANGLES, d["n_phase_full"]),
        ("sparse_step", d["n_angles_full"], N_SPARSE_PHASE),
        ("combined", N_SPARSE_ANGLES, N_SPARSE_PHASE),
    ]

    results = []
    for name, n_angles, n_phase in scenarios:
        result = run_scenario(name, d, n_angles, n_phase, device=device, dtype=dtype)
        results.append(result)

        scen_dir = out_dir / name
        scen_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_grid(d, result, scen_dir / "comparison_true_fbp_tv_jtv.png")
        plot_metrics_table(result, scen_dir / "metrics_table.png")
        save_recon_arrays(d, result, scen_dir / "recon_arrays.npz")

    plot_summary(results, out_dir / "summary_metrics.png", out_dir / "summary_metrics.csv")
    print(f"\nAll scenario outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
