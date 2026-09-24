"""
FBP vs. TV vs. JTV comparison on the real Talbot-Lau dataset, full-angle scan
(361 angles, all 10 phase steps) -- the real-data counterpart of
recon/run_scenarios.py's "full" scenario. Sparse-angle real-data scenarios
come later, once this is validated (see module docstring precedent in
recon/run_scenarios.py).

No ground truth, so unlike synthetic there is no True column and no RE/SSIM/
PSNR -- comparison is via sinogram data-fit residual (||pred-obs||/sigma)
per channel/method, the meaningful ground-truth-free quality signal.

Outputs (real_data/obs/full/):
  comparison_fbp_tv_jtv.png   -- 3x3 grid (rows mu/delta/eps, cols FBP/TV/JTV)
  metrics_table.png           -- per-channel/method data-fit residual (Table I layout, no RE/SSIM/PSNR)
  recon_arrays.npz            -- raw reconstructed arrays, for later figure regeneration
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

import data_utils as du
from map_real import make_direct_predict, make_dpc_predict, reconstruct_map_real

METHODS = ["FBP", "TV", "JTV"]
CHANNELS = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]

# Full cross-validated configuration (cv_alpha_real.py + cv_alpha_real_joint.py,
# held-out-angle CV, re-run after fixing the FBP flip bug in data_utils.py's
# astra_fbp_real -- channelwise picks were stable across that fix, alpha_joint
# was derived fresh from the corrected sigma). alpha_joint=1.0 is ~200x the
# empirically-validated 5e-3 -- not yet visually verified, same "check the
# actual image before trusting the number" discipline as the channelwise picks.
ALPHA_MU = 5.62
# ALPHA_DELTA = 17.78
ALPHA_DELTA = 10.0
ALPHA_EPS = 1.78
ALPHA_JOINT = 1.0
TV_BETA = 3e-2
LAMBDA0 = 1e-2
N_ITER = 150

OUT_DIR = Path(__file__).resolve().parent / "obs" / "full"


def run() -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = du.load_real_intensities()
    sinos = du.retrieve_and_correct_sinograms(data["I_meas"], data["I_ref"])
    T_real, P_real, DPC_real, D_real = sinos["T"], sinos["P"], sinos["DPC"], sinos["D"]

    n_angles, n_det_eff = T_real.shape
    beta_deg = du.make_beta_deg(data["angles"], n_angles)
    L_phys = du.geometry_for(data["I_ref"].shape[-1])
    print(f"\nGeometry: n_angles={n_angles}  n_det_eff={n_det_eff}  L_phys={L_phys:.6e}")

    projector, det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg, device, dtype)

    print("\nRunning ASTRA FBP baselines...")
    mu_fbp = du.astra_fbp_real(T_real, beta_deg, L_phys)
    delta_fbp = du.astra_fbp_real(P_real, beta_deg, L_phys)
    eps_fbp = du.astra_fbp_real(D_real, beta_deg, L_phys)

    field_scale = {"mu": du.robust_scale(mu_fbp), "delta": du.robust_scale(delta_fbp), "eps": du.robust_scale(eps_fbp)}
    print("Field scales:", {k: f"{v:.4e}" for k, v in field_scale.items()})

    with torch.no_grad():
        predict_T = make_direct_predict(projector)
        predict_delta = make_dpc_predict(projector, det_spacing)
        predict_D = make_direct_predict(projector)
        T0 = predict_T(torch.as_tensor(mu_fbp, dtype=dtype, device=device)).cpu().numpy()
        DPC0 = predict_delta(torch.as_tensor(delta_fbp, dtype=dtype, device=device)).cpu().numpy()
        D0 = predict_D(torch.as_tensor(eps_fbp, dtype=dtype, device=device)).cpu().numpy()

    # Flat-field-based sigma, not FBP-residual -- see data_utils.py.
    sigma_I = du.estimate_sigma_I(data["I_ref"])
    sigma_ff = du.propagate_sigma_montecarlo(data["I_meas"], data["I_ref"], sigma_I, n_trials=50, seed=0)
    sigma_T, sigma_DPC, sigma_D = sigma_ff["sigma_T"], sigma_ff["sigma_DPC"], sigma_ff["sigma_D"]
    print(f"sigma_I (flat-field) = {sigma_I:.4e}")
    print(f"sigma_T={sigma_T:.4e}  sigma_DPC={sigma_DPC:.4e}  sigma_D={sigma_D:.4e}")

    print("\nRunning TV (alpha_joint=0)...")
    tv_out = reconstruct_map_real(
        T_real, DPC_real, D_real, projector, det_spacing, du.N_RECON,
        sigma_T, sigma_DPC, sigma_D,
        alpha_mu=ALPHA_MU, alpha_delta=ALPHA_DELTA, alpha_eps=ALPHA_EPS, alpha_joint=0.0,
        field_scale_mu=field_scale["mu"], field_scale_delta=field_scale["delta"], field_scale_eps=field_scale["eps"],
        tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER, support_R=du.SUPPORT_R, device=device, dtype=dtype,
    )

    print("\nRunning JTV...")
    # Zero-init, same as TV -- not warm-started from TV's result, so JTV isn't
    # given a head start from TV's already-converged basin. Matches the
    # independent-initialization principle settled on for synthetic data.
    jtv_out = reconstruct_map_real(
        T_real, DPC_real, D_real, projector, det_spacing, du.N_RECON,
        sigma_T, sigma_DPC, sigma_D,
        alpha_mu=ALPHA_MU, alpha_delta=ALPHA_DELTA, alpha_eps=ALPHA_EPS, alpha_joint=ALPHA_JOINT,
        field_scale_mu=field_scale["mu"], field_scale_delta=field_scale["delta"], field_scale_eps=field_scale["eps"],
        tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER,
        support_R=du.SUPPORT_R, device=device, dtype=dtype,
    )

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
    print("\nPer-channel/method data-fit residual (RMS, sigma-normalized):")
    for (method, ch), v in metrics.items():
        print(f"  {method:4s} {ch:5s}: {v:.4f}")

    plot_comparison_grid(recon, OUT_DIR / "comparison_fbp_tv_jtv.png")
    plot_metrics_table(metrics, OUT_DIR / "metrics_table.png")
    save_recon_arrays(recon, OUT_DIR / "recon_arrays.npz")
    print(f"\nSaved outputs under: {OUT_DIR}")
    return {"recon": recon, "metrics": metrics}


def plot_comparison_grid(recon: Dict, save_path: Path, cmap: str = "gray") -> None:
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.5))
    for i, (ch, label) in enumerate(CHANNELS):
        imgs = [recon[m][ch] for m in METHODS]
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
    fig.suptitle("Real data, full scan: FBP vs. TV vs. JTV", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table(metrics: Dict, save_path: Path) -> None:
    rows = [[label, method, f"{metrics[(method, ch)]:.4f}"] for ch, label in CHANNELS for method in METHODS]
    fig, ax = plt.subplots(figsize=(4.5, 0.35 * len(rows) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Channel", "Method", "Data-fit residual"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title("Real data, full scan: sinogram data-fit residual\n(no ground truth -- not RE/SSIM/PSNR)", fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_recon_arrays(recon: Dict, save_path: Path) -> None:
    arrays = {}
    for method in METHODS:
        for ch, _ in CHANNELS:
            arrays[f"{method}_{ch}"] = recon[method][ch]
    np.savez(save_path, **arrays)


if __name__ == "__main__":
    run()
