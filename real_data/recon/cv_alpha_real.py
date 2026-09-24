"""
Cross-validation-based regularization tuning for real data: hold out a
subset of projection angles, fit TV on the remaining angles at each
candidate alpha, and score by how well the reconstruction *predicts* the
held-out angles' actual measured sinogram.

Why this instead of L-curve or the discrepancy principle: both of those
rely on estimated sigma_T/DPC/D either directly (discrepancy principle) or
indirectly (L-curve's curvature calculation still normalizes by sigma on
one axis). At the time this was first tried, sigma was estimated from
FBP-forward residuals, and we found both criteria agreed with each other
on much higher alpha than this project's own empirically-validated values
-- and eps's data-fit never even reached its discrepancy-principle target
no matter how much it was regularized, which meant sigma_eps (and
plausibly the others) was overestimated, most likely because FBP-forward
residuals contain FBP's own systematic reconstruction error, not just
measurement noise. sigma is now estimated from the flat-field reference
measurement instead (data_utils.estimate_sigma_I/propagate_sigma_montecarlo
-- reconstruction-independent, no FBP involved), which is more accurate,
but held-out angle prediction error doesn't depend on sigma being right at
all regardless -- sigma is only used internally to weight the optimizer's
data term, as a FIXED per-channel constant held constant across the whole
sweep, so it cannot bias which alpha comes out on top of the CV score.

This also directly matches what matters for the sparse-angle scenarios
coming next: an alpha chosen for how well it predicts angles it never saw
is a much more relevant criterion than one chosen from a curve shape.

Usage: python cv_alpha_real.py
Outputs (obs/tuning_cv/): channelwise_cv.png/.csv; recommended values
printed at the end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
for _p in (COMMON_DIR, BASE_DIR, BASE_DIR / "recon", BASE_DIR / "uq"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_utils as du
from map_real import make_direct_predict, make_dpc_predict, reconstruct_single_channel_real

OUT_DIR = BASE_DIR / "obs" / "tuning_cv"

CHANNELWISE_SWEEP = np.logspace(-4.0, 3.0, 29)
CHANNELS = ["mu", "delta", "eps"]
TV_BETA = 3e-2
LAMBDA0 = 1e-2
N_ITER = 150

HOLDOUT_STRIDE = 5  # every 5th angle held out (~20%)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = du.load_real_intensities()
    sinos = du.retrieve_and_correct_sinograms(data["I_meas"], data["I_ref"])
    T_real, P_real, DPC_real, D_real = sinos["T"], sinos["P"], sinos["DPC"], sinos["D"]

    n_angles_full, n_det_eff = T_real.shape
    beta_deg_full = du.make_beta_deg(data["angles"], n_angles_full)
    L_phys = du.geometry_for(data["I_ref"].shape[-1])

    test_idx = np.arange(0, n_angles_full, HOLDOUT_STRIDE)
    train_idx = np.setdiff1d(np.arange(n_angles_full), test_idx)
    print(f"\nHeld-out CV split: {len(train_idx)} train angles, {len(test_idx)} held-out angles "
          f"(every {HOLDOUT_STRIDE}th)")

    train_projector, train_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full[train_idx], device, dtype)
    test_projector, test_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full[test_idx], device, dtype)

    # Field scale and sigma: from the FULL-data FBP/residual, same as the
    # other tuning scripts. This is a coarse, robust normalization/optimizer
    # weighting convenience, held FIXED across the whole sweep -- not part
    # of what's under test -- so using the full dataset here isn't the kind
    # of leakage that would bias the CV comparison between alphas.
    print("\nRunning ASTRA FBP baselines (full data, for field scale/sigma only)...")
    mu_fbp = du.astra_fbp_real(T_real, beta_deg_full, L_phys)
    delta_fbp = du.astra_fbp_real(P_real, beta_deg_full, L_phys)
    eps_fbp = du.astra_fbp_real(D_real, beta_deg_full, L_phys)
    field_scale = {"mu": du.robust_scale(mu_fbp), "delta": du.robust_scale(delta_fbp), "eps": du.robust_scale(eps_fbp)}

    full_projector, full_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full, device, dtype)
    with torch.no_grad():
        predict_T_full = make_direct_predict(full_projector)
        predict_delta_full = make_dpc_predict(full_projector, full_det_spacing)
        predict_D_full = make_direct_predict(full_projector)
        T0 = predict_T_full(torch.as_tensor(mu_fbp, dtype=dtype, device=device)).cpu().numpy()
        DPC0 = predict_delta_full(torch.as_tensor(delta_fbp, dtype=dtype, device=device)).cpu().numpy()
        D0 = predict_D_full(torch.as_tensor(eps_fbp, dtype=dtype, device=device)).cpu().numpy()
    # Flat-field-based sigma, not FBP-residual -- see data_utils.py module
    # docstring / estimate_sigma_I. Independent of FBP/reconstruction quality.
    sigma_I = du.estimate_sigma_I(data["I_ref"])
    sigma_ff = du.propagate_sigma_montecarlo(data["I_meas"], data["I_ref"], sigma_I, n_trials=50, seed=0)
    sigma = {"mu": sigma_ff["sigma_T"], "delta": sigma_ff["sigma_DPC"], "eps": sigma_ff["sigma_D"]}
    print("Field scales:", {k: f"{v:.4e}" for k, v in field_scale.items()})
    print(f"sigma_I (flat-field) = {sigma_I:.4e}")
    print("Sigma (optimizer weighting only, not the CV metric):", {k: f"{v:.4e}" for k, v in sigma.items()})

    obs_train = {"mu": T_real[train_idx], "delta": DPC_real[train_idx], "eps": D_real[train_idx]}
    obs_test = {"mu": T_real[test_idx], "delta": DPC_real[test_idx], "eps": D_real[test_idx]}
    predict_train = {
        "mu": make_direct_predict(train_projector),
        "delta": make_dpc_predict(train_projector, train_det_spacing),
        "eps": make_direct_predict(train_projector),
    }
    predict_test = {
        "mu": make_direct_predict(test_projector),
        "delta": make_dpc_predict(test_projector, test_det_spacing),
        "eps": make_direct_predict(test_projector),
    }

    print("\n=== Cross-validated per-channel TV sweep ===")
    curves = {ch: {"alpha": [], "train_data_fit": [], "test_relerr": []} for ch in CHANNELS}

    for ch in CHANNELS:
        print(f"--- channel: {ch} ---")
        for a in CHANNELWISE_SWEEP:
            out = reconstruct_single_channel_real(
                obs_train[ch], predict_train[ch], du.N_RECON, sigma[ch], alpha_tv=a,
                field_scale=field_scale[ch], tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER,
                support_R=du.SUPPORT_R, device=device, dtype=dtype,
            )
            train_data_fit = float(np.linalg.norm(out["pred"] - obs_train[ch]) / sigma[ch])

            with torch.no_grad():
                img_t = torch.as_tensor(out["img"], dtype=dtype, device=device)
                pred_test = predict_test[ch](img_t).cpu().numpy()
            test_relerr = float(np.linalg.norm(pred_test - obs_test[ch]) / (np.linalg.norm(obs_test[ch]) + 1e-12))

            curves[ch]["alpha"].append(a)
            curves[ch]["train_data_fit"].append(train_data_fit)
            curves[ch]["test_relerr"].append(test_relerr)
            print(f"  alpha={a:9.3g}  train_data_fit={train_data_fit:.4e}  held_out_relerr={test_relerr:.6f}")

    best_alpha = {}
    for ch in CHANNELS:
        relerr = np.asarray(curves[ch]["test_relerr"])
        idx = int(np.argmin(relerr))
        best_alpha[ch] = float(curves[ch]["alpha"][idx])
        print(f"[{ch}] CV-optimal: alpha={best_alpha[ch]:.4g}  (held_out_relerr={relerr[idx]:.6f})")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for j, ch in enumerate(CHANNELS):
        ax = axes[j]
        ax.semilogx(curves[ch]["alpha"], curves[ch]["test_relerr"], "o-")
        idx = list(curves[ch]["alpha"]).index(best_alpha[ch])
        ax.axvline(best_alpha[ch], color="r", linestyle="--", label=f"CV-optimal ({best_alpha[ch]:.3g})")
        ax.set_xlabel("alpha")
        ax.set_ylabel("held-out relative error")
        ax.set_title(f"{ch}: cross-validation")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "channelwise_cv.png", dpi=180)
    plt.close(fig)

    with open(OUT_DIR / "channelwise_cv.csv", "w") as f:
        f.write("channel,alpha,train_data_fit,held_out_relerr\n")
        for ch in CHANNELS:
            for i in range(len(curves[ch]["alpha"])):
                f.write(f"{ch},{curves[ch]['alpha'][i]:.6g},{curves[ch]['train_data_fit'][i]:.6g},{curves[ch]['test_relerr'][i]:.6f}\n")

    print("\n=== Recommended values (cross-validation) ===")
    print(f"ALPHA_MU = {best_alpha['mu']:.4g}")
    print(f"ALPHA_DELTA = {best_alpha['delta']:.4g}")
    print(f"ALPHA_EPS = {best_alpha['eps']:.4g}")
    print(f"\nAll outputs saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
