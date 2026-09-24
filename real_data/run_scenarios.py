"""
FBP vs. TV vs. JTV comparison on real Talbot-Lau data across four
limited-data scenarios, analogous to recon/run_scenarios.py for synthetic
data:

  1. full         -- all 361 angles, all 10 phase steps
  2. sparse_angle -- reduced angles, full phase steps
  3. sparse_step  -- full angles, reduced phase steps
  4. combined     -- reduced angles AND reduced phase steps

Regularization (alpha_mu/delta/eps/joint) is held fixed across all four
scenarios, matching the synthetic pipeline's own convention -- these are
the values from cv_alpha_real.py / cv_alpha_real_joint.py's held-out-angle
cross-validation on the full scan, visually confirmed via run_full.py.

Real data cannot resimulate phase-stepping intensities at an arbitrary
n_phase the way synthetic's scenario_utils.simulate_and_retrieve does --
the 10 recorded phase steps are fixed measurements, not a forward model we
control. sparse_step therefore subsamples the *recorded* steps rather than
resimulating, and the subsample count must evenly divide 10 (else it isn't
evenly spaced over one grating period, corrupting the FFT-based harmonic
retrieval -- the same leakage problem synthetic's resimulation approach was
specifically built to avoid). N_SPARSE_STEPS=5 (every other step) is used
here, not synthetic's N_SPARSE_PHASE=4.

No ground truth, so -- like run_full.py -- comparison is via sinogram
data-fit residual per channel/method, not RE/SSIM/PSNR.

Outputs (real_data/obs/scenarios/<scenario>/):
  comparison_fbp_tv_jtv.png, metrics_table.png, recon_arrays.npz
Outputs (real_data/obs/scenarios/):
  summary_metrics.png/csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

import data_utils as du
from map_real import make_direct_predict, make_dpc_predict, reconstruct_map_real

COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from scenario_utils import make_sparse_angle_indices  # noqa: E402

METHODS = ["FBP", "TV", "JTV"]
CHANNELS = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]

# Locked regularization (see module docstring).
ALPHA_MU = 5.62
# ALPHA_DELTA = 17.78
ALPHA_DELTA = 10
ALPHA_EPS = 1.78
ALPHA_JOINT = 1.0
TV_BETA = 3e-2
LAMBDA0 = 1e-2
N_ITER = 150

N_SPARSE_ANGLES = 30    
N_SPARSE_STEPS = 5  # must evenly divide n_phase_full (10) -- see module docstring

OUT_DIR = Path(__file__).resolve().parent / "obs" / "scenarios"


def synchronize_device(device) -> None:
    """Synchronize CUDA before/after timing GPU operations."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sparse_step_indices(n_phase_full: int, n_keep: int) -> np.ndarray:
    if n_phase_full % n_keep != 0:
        raise ValueError(
            f"n_keep={n_keep} must evenly divide n_phase_full={n_phase_full} "
            "(real data's recorded phase steps can't be resimulated at an "
            "arbitrary spacing the way synthetic data can)."
        )
    return np.arange(0, n_phase_full, n_phase_full // n_keep)


def run_scenario(name: str, angle_idx: np.ndarray, step_idx: np.ndarray, data: Dict, L_phys: float,
                  field_scale: Dict[str, float], device, dtype) -> Dict:
    print(f"\n=== Scenario: {name} (n_angles={len(angle_idx)}, n_phase={len(step_idx)}) ===")

    # sigma follows the scenario's stepping: fewer steps retrieve noisier
    # sinograms, and the likelihood has to know that (see sigma_for_stepping).
    sigma_ff = du.sigma_for_stepping(data["I_meas"], data["I_ref"], step_idx)
    sigma_T, sigma_DPC, sigma_D = sigma_ff["sigma_T"], sigma_ff["sigma_DPC"], sigma_ff["sigma_D"]
    print(f"  sigma_T={sigma_T:.4e}  sigma_DPC={sigma_DPC:.4e}  sigma_D={sigma_D:.4e}")

    I_meas = data["I_meas"][angle_idx][:, step_idx, :]
    I_ref = data["I_ref"][angle_idx][:, step_idx, :]
    sinos = du.retrieve_and_correct_sinograms(I_meas, I_ref)
    T_real, P_real, DPC_real, D_real = sinos["T"], sinos["P"], sinos["DPC"], sinos["D"]

    n_angles, n_det_eff = T_real.shape
    beta_deg = du.make_beta_deg(data["angles"][angle_idx], n_angles)
    projector, det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg, device, dtype)

    # FBP is the baseline column only, never a source of field_scale or
    # sigma: under sparse_angle it is badly streaked, and an FBP-residual
    # noise estimate would absorb that reconstruction error as measurement
    # noise (see data_utils).
    synchronize_device(device)
    t0 = time.perf_counter()
    mu_fbp = du.astra_fbp_real(T_real, beta_deg, L_phys)
    delta_fbp = du.astra_fbp_real(P_real, beta_deg, L_phys)
    eps_fbp = du.astra_fbp_real(D_real, beta_deg, L_phys)
    synchronize_device(device)
    runtime = {"FBP": (time.perf_counter() - t0) * 1000.0}
    print(f"  FBP runtime:  {runtime['FBP']:.2f} ms")

    with torch.no_grad():
        predict_T = make_direct_predict(projector)
        predict_delta = make_dpc_predict(projector, det_spacing)
        predict_D = make_direct_predict(projector)
        T0 = predict_T(torch.as_tensor(mu_fbp, dtype=dtype, device=device)).cpu().numpy()
        DPC0 = predict_delta(torch.as_tensor(delta_fbp, dtype=dtype, device=device)).cpu().numpy()
        D0 = predict_D(torch.as_tensor(eps_fbp, dtype=dtype, device=device)).cpu().numpy()

    print("Running TV (alpha_joint=0)...")
    synchronize_device(device)
    t0 = time.perf_counter()
    tv_out = reconstruct_map_real(
        T_real, DPC_real, D_real, projector, det_spacing, du.N_RECON,
        sigma_T, sigma_DPC, sigma_D,
        alpha_mu=ALPHA_MU, alpha_delta=ALPHA_DELTA, alpha_eps=ALPHA_EPS, alpha_joint=0.0,
        field_scale_mu=field_scale["mu"], field_scale_delta=field_scale["delta"], field_scale_eps=field_scale["eps"],
        tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER, support_R=du.SUPPORT_R, device=device, dtype=dtype,
    )
    synchronize_device(device)
    runtime["TV"] = (time.perf_counter() - t0) * 1000.0
    print(f"  TV runtime:  {runtime['TV']:.2f} ms")

    print("Running JTV...")
    # Zero-init, same as TV -- not warm-started from TV's result, matching
    # the independent-initialization principle settled on for synthetic data
    # (a warm start would give JTV an unfair head start from TV's basin).
    synchronize_device(device)
    t0 = time.perf_counter()
    jtv_out = reconstruct_map_real(
        T_real, DPC_real, D_real, projector, det_spacing, du.N_RECON,
        sigma_T, sigma_DPC, sigma_D,
        alpha_mu=ALPHA_MU, alpha_delta=ALPHA_DELTA, alpha_eps=ALPHA_EPS, alpha_joint=ALPHA_JOINT,
        field_scale_mu=field_scale["mu"], field_scale_delta=field_scale["delta"], field_scale_eps=field_scale["eps"],
        tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER,
        support_R=du.SUPPORT_R, device=device, dtype=dtype,
    )
    synchronize_device(device)
    runtime["JTV"] = (time.perf_counter() - t0) * 1000.0
    print(f"  JTV runtime: {runtime['JTV']:.2f} ms")

    recon = {
        "FBP": {"mu": mu_fbp, "delta": delta_fbp, "eps": eps_fbp},
        "TV": {"mu": tv_out["mu"], "delta": tv_out["delta"], "eps": tv_out["eps"]},
        "JTV": {"mu": jtv_out["mu"], "delta": jtv_out["delta"], "eps": jtv_out["eps"]},
    }

    def data_fit(pred, obs, sigma):
        return float(np.linalg.norm(pred - obs) / sigma / np.sqrt(obs.size))

    metrics = {
        ("FBP", "mu"): data_fit(T0, T_real, sigma_T),
        ("FBP", "delta"): data_fit(DPC0, DPC_real, sigma_DPC),
        ("FBP", "eps"): data_fit(D0, D_real, sigma_D),
        ("TV", "mu"): data_fit(tv_out["T_pred"], T_real, sigma_T),
        ("TV", "delta"): data_fit(tv_out["DPC_pred"], DPC_real, sigma_DPC),
        ("TV", "eps"): data_fit(tv_out["D_pred"], D_real, sigma_D),
        ("JTV", "mu"): data_fit(jtv_out["T_pred"], T_real, sigma_T),
        ("JTV", "delta"): data_fit(jtv_out["DPC_pred"], DPC_real, sigma_DPC),
        ("JTV", "eps"): data_fit(jtv_out["D_pred"], D_real, sigma_D),
    }
    for (method, ch), v in metrics.items():
        print(f"  {method:4s} {ch:5s}: data_fit={v:.4f}")

    return {"name": name, "n_angles": n_angles, "n_phase": len(step_idx), "recon": recon, "metrics": metrics, "runtime": runtime}


def plot_comparison_grid(result: Dict, save_path: Path, cmap: str = "gray") -> None:
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.5))
    for i, (ch, label) in enumerate(CHANNELS):
        imgs = [result["recon"][m][ch] for m in METHODS]
        vmax = max(float(np.clip(im, 0, None).max()) for im in imgs)
        vmax = max(vmax, 1e-12)
        for j, (method, img) in enumerate(zip(METHODS, imgs)):
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(label, fontsize=13)
            if i == 0:
                ax.set_title(method, fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            im = ax.imshow(np.clip(img, 0, None), cmap=cmap, origin="lower", vmin=0.0, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"Real data, {result['name']} (angles={result['n_angles']}, phase steps={result['n_phase']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table(result: Dict, save_path: Path) -> None:
    rows = [
        [label, method, f"{result['metrics'][(method, ch)]:.4f}", f"{result['runtime'][method]:.2f}"]
        for ch, label in CHANNELS for method in METHODS
    ]
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(rows) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Channel", "Method", "Data-fit residual", "Runtime [ms]"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title(f"{result['name']}: sinogram data-fit residual\n(no ground truth -- not RE/SSIM/PSNR)", fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_recon_arrays(result: Dict, save_path: Path) -> None:
    arrays = {}
    for method in METHODS:
        for ch, _ in CHANNELS:
            arrays[f"{method}_{ch}"] = result["recon"][method][ch]
    np.savez(save_path, **arrays)


def plot_summary(results: List[Dict], save_path: Path, csv_path: Path) -> None:
    scenario_names = [r["name"] for r in results]
    colors = {"FBP": "tab:gray", "TV": "tab:blue", "JTV": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(len(scenario_names))
    width = 0.25
    csv_lines = ["scenario,channel,method,data_fit_residual,runtime_ms"]
    for col, (ch, label) in enumerate(CHANNELS):
        ax = axes[col]
        for k, method in enumerate(METHODS):
            vals = [r["metrics"][(method, ch)] for r in results]
            ax.bar(x + (k - 1) * width, vals, width, label=method, color=colors[method])
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=12)
        if col == 0:
            ax.set_ylabel("data-fit residual")
        if col == 2:
            ax.legend(fontsize=8, frameon=False)
        for r in results:
            for method in METHODS:
                csv_lines.append(f"{r['name']},{ch},{method},{r['metrics'][(method, ch)]:.6f},{r['runtime'][method]:.2f}")
    fig.suptitle("Real data: FBP vs. TV vs. JTV across limited-data scenarios", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    csv_path.write_text("\n".join(csv_lines) + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = du.load_real_intensities()
    n_angles_full = data["I_meas"].shape[0]
    n_phase_full = data["I_meas"].shape[1]
    L_phys = du.geometry_for(data["I_ref"].shape[-1])

    full_angles = np.arange(n_angles_full)
    full_steps = np.arange(n_phase_full)
    sparse_angles = make_sparse_angle_indices(n_angles_full, N_SPARSE_ANGLES)
    sparse_steps = sparse_step_indices(n_phase_full, N_SPARSE_STEPS)

    # field_scale is estimated ONCE from the full scan's FBP and shared by all
    # four scenarios: it is the normalization the priors act through, so
    # re-deriving it from a scenario's own (streaked) FBP would make a given
    # alpha mean different things in different scenarios. sigma is estimated
    # per scenario instead -- it depends on the stepping, not on FBP.
    print("\nEstimating field_scale once from the full scan...")
    full_sinos = du.retrieve_and_correct_sinograms(data["I_meas"], data["I_ref"])
    full_beta_deg = du.make_beta_deg(data["angles"], n_angles_full)
    full_mu_fbp = du.astra_fbp_real(full_sinos["T"], full_beta_deg, L_phys)
    full_delta_fbp = du.astra_fbp_real(full_sinos["P"], full_beta_deg, L_phys)
    full_eps_fbp = du.astra_fbp_real(full_sinos["D"], full_beta_deg, L_phys)
    field_scale = {
        "mu": du.robust_scale(full_mu_fbp), "delta": du.robust_scale(full_delta_fbp), "eps": du.robust_scale(full_eps_fbp),
    }
    print(f"  field_scale: {field_scale}")

    scenarios: List[Tuple[str, np.ndarray, np.ndarray]] = [
        ("full", full_angles, full_steps),
        ("sparse_angle", sparse_angles, full_steps),
        ("sparse_step", full_angles, sparse_steps),
        ("combined", sparse_angles, sparse_steps),
    ]

    results = []
    for name, angle_idx, step_idx in scenarios:
        result = run_scenario(name, angle_idx, step_idx, data, L_phys, field_scale, device, dtype)
        results.append(result)

        scen_dir = OUT_DIR / name
        scen_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_grid(result, scen_dir / "comparison_fbp_tv_jtv.png")
        plot_metrics_table(result, scen_dir / "metrics_table.png")
        save_recon_arrays(result, scen_dir / "recon_arrays.npz")

    plot_summary(results, OUT_DIR / "summary_metrics.png", OUT_DIR / "summary_metrics.csv")
    print(f"\nAll scenario outputs saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
