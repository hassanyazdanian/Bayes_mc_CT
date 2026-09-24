"""
alpha_joint sweep for the sparse-angle regime, scored by SSIM/RE against the
full-scan TV reconstruction (used as a trusted stand-in for ground truth --
real data has none, and 361 angles is well-determined enough to serve as
one), not by held-out sinogram data-fit.

Why: held-out data-fit (cv_alpha_real_joint.py, cv_alpha_real_joint_sparse.py)
structurally penalizes regularization, since priors are supposed to trade
fit for prior belief -- it can tell us whether we're over/under-fitting the
noisy data, not whether the resulting image is actually better. Both of
those held-out sweeps picked alpha_joint~2.15 regardless of whether the
held-out split came from the full scan or the sparse-angle pool itself --
suggestive that data-fit isn't the right axis to find where JTV's image-
domain benefit shows up. This mirrors exactly why synthetic data used
RE/SSIM against ground truth rather than data-fit for its own alpha_joint
selection; the full-scan TV reconstruction plays the role ground truth
played there.

Uses ALL 30 sparse angles (not a train/test split) -- the reference is an
independent image, not held-out data, so there's no leakage concern; this
also matches exactly what run_scenarios.py's sparse_angle scenario does.

Usage:
    python tune_alpha_joint_ssim_sparse.py --alpha_mu 5.62 --alpha_delta 10 --alpha_eps 1.78

Outputs (obs/tuning_cv/): joint_ssim_sparse.csv, joint_ssim_sparse.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_utils as du
from map_real import reconstruct_map_real
from cv_alpha_real import OUT_DIR, TV_BETA, LAMBDA0, N_ITER
from cv_alpha_real_joint import JOINT_SWEEP

RECON_DIR = Path(__file__).resolve().parent.parent / "recon"
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))
from map_tv_jtv import compute_metrics  # noqa: E402

COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from scenario_utils import make_sparse_angle_indices  # noqa: E402

N_SPARSE_ANGLES = 30  # matches run_scenarios.py
CHANNELS = ["mu", "delta", "eps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_mu", type=float, required=True)
    parser.add_argument("--alpha_delta", type=float, required=True)
    parser.add_argument("--alpha_eps", type=float, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = du.load_real_intensities()
    sinos_full = du.retrieve_and_correct_sinograms(data["I_meas"], data["I_ref"], verbose=False)
    n_angles_full, n_det_eff = sinos_full["T"].shape
    beta_deg_full = du.make_beta_deg(data["angles"], n_angles_full)
    L_phys = du.geometry_for(data["I_ref"].shape[-1])

    mu_fbp = du.astra_fbp_real(sinos_full["T"], beta_deg_full, L_phys)
    delta_fbp = du.astra_fbp_real(sinos_full["P"], beta_deg_full, L_phys)
    eps_fbp = du.astra_fbp_real(sinos_full["D"], beta_deg_full, L_phys)
    field_scale = {"mu": du.robust_scale(mu_fbp), "delta": du.robust_scale(delta_fbp), "eps": du.robust_scale(eps_fbp)}

    sigma_I = du.estimate_sigma_I(data["I_ref"])
    sigma_ff = du.propagate_sigma_montecarlo(data["I_meas"], data["I_ref"], sigma_I, n_trials=50, seed=0)
    sigma = {"mu": sigma_ff["sigma_T"], "delta": sigma_ff["sigma_DPC"], "eps": sigma_ff["sigma_D"]}
    print(f"sigma_I (flat-field) = {sigma_I:.4e}")

    fixed = {"mu": args.alpha_mu, "delta": args.alpha_delta, "eps": args.alpha_eps}
    print(f"Fixed (channelwise CV-optimal): alpha_mu={fixed['mu']:.4g}, alpha_delta={fixed['delta']:.4g}, alpha_eps={fixed['eps']:.4g}")

    # ------------------------------------------------------------------
    # Reference: full-scan TV (alpha_joint=0) -- our stand-in for ground
    # truth. Not JTV, to avoid any dependence on the alpha_joint being tuned.
    # ------------------------------------------------------------------
    print("\nReconstructing full-scan TV reference...")
    full_projector, full_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full, device, dtype)
    ref_out = reconstruct_map_real(
        sinos_full["T"], sinos_full["DPC"], sinos_full["D"], full_projector, full_det_spacing, du.N_RECON,
        sigma["mu"], sigma["delta"], sigma["eps"],
        alpha_mu=fixed["mu"], alpha_delta=fixed["delta"], alpha_eps=fixed["eps"], alpha_joint=0.0,
        field_scale_mu=field_scale["mu"], field_scale_delta=field_scale["delta"], field_scale_eps=field_scale["eps"],
        tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER, support_R=du.SUPPORT_R, device=device, dtype=dtype,
    )
    reference = {"mu": ref_out["mu"], "delta": ref_out["delta"], "eps": ref_out["eps"]}

    # ------------------------------------------------------------------
    # Sparse-angle projector (all N_SPARSE_ANGLES, matching run_scenarios.py)
    # ------------------------------------------------------------------
    sparse_idx = make_sparse_angle_indices(n_angles_full, N_SPARSE_ANGLES)
    print(f"\nSparse-angle pool: {len(sparse_idx)} angles")
    sparse_projector, sparse_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full[sparse_idx], device, dtype)
    T_sparse = sinos_full["T"][sparse_idx]
    DPC_sparse = sinos_full["DPC"][sparse_idx]
    D_sparse = sinos_full["D"][sparse_idx]

    print("\n=== alpha_joint sweep, scored by SSIM/RE against full-scan TV ===")
    rows: List[Dict] = []
    tv_out = None
    for aj in JOINT_SWEEP:
        out = reconstruct_map_real(
            T_sparse, DPC_sparse, D_sparse, sparse_projector, sparse_det_spacing, du.N_RECON,
            sigma["mu"], sigma["delta"], sigma["eps"],
            alpha_mu=fixed["mu"], alpha_delta=fixed["delta"], alpha_eps=fixed["eps"], alpha_joint=aj,
            field_scale_mu=field_scale["mu"], field_scale_delta=field_scale["delta"], field_scale_eps=field_scale["eps"],
            tv_beta=TV_BETA, lambda0=LAMBDA0, n_iter=N_ITER,
            init_mu=(tv_out["mu"] if tv_out is not None else None),
            init_delta=(tv_out["delta"] if tv_out is not None else None),
            init_eps=(tv_out["eps"] if tv_out is not None else None),
            support_R=du.SUPPORT_R, device=device, dtype=dtype,
        )
        if aj == 0.0:
            tv_out = out

        row = {"alpha_joint": aj}
        line = f"  alpha_joint={aj:9.3g}  "
        re_sum = 0.0
        ssim_sum = 0.0
        for ch in CHANNELS:
            re, ssim, psnr = compute_metrics(reference[ch], out[ch])
            row[f"RE_{ch}"] = re
            row[f"SSIM_{ch}"] = ssim
            row[f"PSNR_{ch}"] = psnr
            re_sum += re
            ssim_sum += ssim
            line += f"{ch}: RE={re:.4f} SSIM={ssim:.4f}  "
        row["RE_total"] = re_sum
        row["SSIM_mean"] = ssim_sum / len(CHANNELS)
        print(line + f" | RE_total={re_sum:.4f} SSIM_mean={row['SSIM_mean']:.4f}")
        rows.append(row)

    with open(OUT_DIR / "joint_ssim_sparse.csv", "w") as f:
        header = ["alpha_joint"] + [f"{m}_{ch}" for ch in CHANNELS for m in ("RE", "SSIM", "PSNR")] + ["RE_total", "SSIM_mean"]
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]:.6g}" for k in header) + "\n")

    aj_vals = np.asarray([r["alpha_joint"] for r in rows])
    re_total = np.asarray([r["RE_total"] for r in rows])
    ssim_mean = np.asarray([r["SSIM_mean"] for r in rows])
    best_re_idx = int(np.argmin(re_total))
    best_ssim_idx = int(np.argmax(ssim_mean))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ch in CHANNELS:
        axes[0].semilogx(aj_vals, [r[f"RE_{ch}"] for r in rows], "o-", label=ch)
        axes[1].semilogx(aj_vals, [r[f"SSIM_{ch}"] for r in rows], "o-", label=ch)
    axes[0].axvline(aj_vals[best_re_idx], color="r", linestyle="--", label=f"RE-optimal ({aj_vals[best_re_idx]:.3g})")
    axes[1].axvline(aj_vals[best_ssim_idx], color="r", linestyle="--", label=f"SSIM-optimal ({aj_vals[best_ssim_idx]:.3g})")
    axes[0].set_xlabel("alpha_joint"); axes[0].set_ylabel("RE (vs. full-scan TV)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("alpha_joint"); axes[1].set_ylabel("SSIM (vs. full-scan TV)"); axes[1].legend(fontsize=8)
    fig.suptitle("Sparse-angle alpha_joint sweep, scored against full-scan TV reference")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "joint_ssim_sparse.png", dpi=180)
    plt.close(fig)

    print(f"\nRE-optimal alpha_joint    = {aj_vals[best_re_idx]:.4g}  (RE_total={re_total[best_re_idx]:.4f})")
    print(f"SSIM-optimal alpha_joint  = {aj_vals[best_ssim_idx]:.4g}  (SSIM_mean={ssim_mean[best_ssim_idx]:.4f})")
    print(f"Saved: {OUT_DIR / 'joint_ssim_sparse.csv'}")
    print(f"Saved: {OUT_DIR / 'joint_ssim_sparse.png'}")


if __name__ == "__main__":
    main()
