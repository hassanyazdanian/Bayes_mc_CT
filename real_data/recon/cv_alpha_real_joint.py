"""
Cross-validated alpha_joint sweep: same held-out-angle methodology as
cv_alpha_real.py, extended to the joint-TV coupling term. Channelwise
alphas are held fixed at the values used in the paper (from
cv_alpha_real.py, with alpha_delta at the lower end of its flat held-out
range); alpha_joint is swept, and each candidate is scored by held-out
prediction error (summed across mu/delta/eps) on angles never used for
fitting.

The held-out error is nearly flat in alpha_joint (it varies by less than
0.1 % over [0, 2.15]), so it does not select alpha_joint; the paper's
value comes from tune_alpha_joint_ssim_sparse.py.

Usage:
    python cv_alpha_real_joint.py --alpha_mu 5.62 --alpha_delta 10 --alpha_eps 1.78

Outputs (obs/tuning_cv/): joint_cv.csv, joint_cv.png
"""
from __future__ import annotations

import argparse
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
from map_real import make_direct_predict, make_dpc_predict, reconstruct_map_real
from cv_alpha_real import OUT_DIR, TV_BETA, LAMBDA0, N_ITER, HOLDOUT_STRIDE

JOINT_SWEEP = np.concatenate([[0.0], np.logspace(-4.0, 3.0, 22)])


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
    sinos = du.retrieve_and_correct_sinograms(data["I_meas"], data["I_ref"])
    T_real, P_real, DPC_real, D_real = sinos["T"], sinos["P"], sinos["DPC"], sinos["D"]

    n_angles_full, n_det_eff = T_real.shape
    beta_deg_full = du.make_beta_deg(data["angles"], n_angles_full)
    L_phys = du.geometry_for(data["I_ref"].shape[-1])

    test_idx = np.arange(0, n_angles_full, HOLDOUT_STRIDE)
    train_idx = np.setdiff1d(np.arange(n_angles_full), test_idx)
    print(f"CV split: {len(train_idx)} train, {len(test_idx)} held-out angles")

    train_projector, train_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full[train_idx], device, dtype)
    test_projector, test_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full[test_idx], device, dtype)

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
    # Flat-field-based sigma, not FBP-residual -- see data_utils.py.
    sigma_I = du.estimate_sigma_I(data["I_ref"])
    sigma_ff = du.propagate_sigma_montecarlo(data["I_meas"], data["I_ref"], sigma_I, n_trials=50, seed=0)
    sigma = {"mu": sigma_ff["sigma_T"], "delta": sigma_ff["sigma_DPC"], "eps": sigma_ff["sigma_D"]}
    print(f"sigma_I (flat-field) = {sigma_I:.4e}")
    print("Sigma:", {k: f"{v:.4e}" for k, v in sigma.items()})

    fixed = {"mu": args.alpha_mu, "delta": args.alpha_delta, "eps": args.alpha_eps}
    print(f"Fixed (CV-optimal): alpha_mu={fixed['mu']:.4g}, alpha_delta={fixed['delta']:.4g}, alpha_eps={fixed['eps']:.4g}")

    T_train, DPC_train, D_train = T_real[train_idx], DPC_real[train_idx], D_real[train_idx]
    T_test, DPC_test, D_test = T_real[test_idx], DPC_real[test_idx], D_real[test_idx]
    predict_test = {
        "mu": make_direct_predict(test_projector),
        "delta": make_dpc_predict(test_projector, test_det_spacing),
        "eps": make_direct_predict(test_projector),
    }

    rows: List[Dict] = []
    tv_out = None
    for aj in JOINT_SWEEP:
        out = reconstruct_map_real(
            T_train, DPC_train, D_train, train_projector, train_det_spacing, du.N_RECON,
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

        with torch.no_grad():
            mu_t = torch.as_tensor(out["mu"], dtype=dtype, device=device)
            delta_t = torch.as_tensor(out["delta"], dtype=dtype, device=device)
            eps_t = torch.as_tensor(out["eps"], dtype=dtype, device=device)
            T_pred_test = predict_test["mu"](mu_t).cpu().numpy()
            DPC_pred_test = predict_test["delta"](delta_t).cpu().numpy()
            D_pred_test = predict_test["eps"](eps_t).cpu().numpy()

        relerr_mu = float(np.linalg.norm(T_pred_test - T_test) / (np.linalg.norm(T_test) + 1e-12))
        relerr_delta = float(np.linalg.norm(DPC_pred_test - DPC_test) / (np.linalg.norm(DPC_test) + 1e-12))
        relerr_eps = float(np.linalg.norm(D_pred_test - D_test) / (np.linalg.norm(D_test) + 1e-12))
        relerr_total = relerr_mu + relerr_delta + relerr_eps

        row = {"alpha_joint": aj, "relerr_mu": relerr_mu, "relerr_delta": relerr_delta, "relerr_eps": relerr_eps, "relerr_total": relerr_total}
        print(f"  alpha_joint={aj:9.3g}  relerr: mu={relerr_mu:.4f} delta={relerr_delta:.4f} eps={relerr_eps:.4f}  total={relerr_total:.4f}")
        rows.append(row)

    with open(OUT_DIR / "joint_cv.csv", "w") as f:
        f.write("alpha_joint,relerr_mu,relerr_delta,relerr_eps,relerr_total\n")
        for row in rows:
            f.write(f"{row['alpha_joint']:.6g},{row['relerr_mu']:.6f},{row['relerr_delta']:.6f},{row['relerr_eps']:.6f},{row['relerr_total']:.6f}\n")

    aj_vals = np.asarray([r["alpha_joint"] for r in rows])
    total = np.asarray([r["relerr_total"] for r in rows])
    best_idx = int(np.argmin(total))
    best_alpha_joint = float(aj_vals[best_idx])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [("relerr_mu", "mu"), ("relerr_delta", "delta"), ("relerr_eps", "eps"), ("relerr_total", "total")]:
        vals = [r[key] for r in rows]
        ax.semilogx(aj_vals, vals, "o-", label=label, linewidth=2 if key == "relerr_total" else 1)
    ax.axvline(best_alpha_joint, color="r", linestyle="--", label=f"CV-optimal ({best_alpha_joint:.3g})")
    ax.set_xlabel("alpha_joint")
    ax.set_ylabel("held-out relative error")
    ax.legend(fontsize=8)
    ax.set_title(f"Joint-TV cross-validation (alpha_mu={fixed['mu']:.3g}, alpha_delta={fixed['delta']:.3g}, alpha_eps={fixed['eps']:.3g})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "joint_cv.png", dpi=180)
    plt.close(fig)

    print(f"\nALPHA_JOINT (cross-validation) = {best_alpha_joint:.4g}  (total relerr={total[best_idx]:.4f})")
    print(f"Saved: {OUT_DIR / 'joint_cv.csv'}")
    print(f"Saved: {OUT_DIR / 'joint_cv.png'}")


if __name__ == "__main__":
    main()
