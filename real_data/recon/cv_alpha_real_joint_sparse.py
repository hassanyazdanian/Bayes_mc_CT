"""
Cross-validated alpha_joint sweep evaluated WITHIN the sparse-angle data
regime, not the full 361-angle scan.

Why: cv_alpha_real_joint.py's held-out split comes from the full scan
(289 train / 72 test angles), where JTV's joint-TV curve turned out nearly
flat across 7 orders of magnitude of alpha_joint -- the full scan is
well-determined enough that channels don't need to borrow structure from
each other, so there was no real signal there for CV to lock onto (matches
the synthetic-data finding that JTV's benefit is specifically a
limited-data phenomenon). Confirmed by the resulting run_scenarios.py
comparison: JTV came out essentially tied with (sometimes marginally worse
than) TV under sparse_angle/combined -- alpha_joint=2.15 just happened to
sit on that flat plateau, not at a point where coupling does real work.

This script instead: takes the SAME N_SPARSE_ANGLES=30-angle pool
run_scenarios.py's sparse_angle scenario actually uses (make_sparse_angle_
indices on the full 361), then further holds out a fraction of THAT pool
as test angles, fits on the rest, and scores alpha_joint by held-out
prediction error within this genuinely sparse regime -- where cross-channel
coupling is actually supposed to matter.

alpha_mu/delta/eps are held fixed at their (full-scan-CV-derived) values,
matching the fixed-regularization-across-scenarios convention. sigma and
field_scale are also the same global, full-scan-derived values used
everywhere else (see data_utils.py) -- not re-derived from the sparse
subset, for the same reasons run_scenarios.py fixed them globally.

Usage:
    python cv_alpha_real_joint_sparse.py --alpha_mu 5.62 --alpha_delta 10 --alpha_eps 1.78

Outputs (obs/tuning_cv/): joint_cv_sparse.csv, joint_cv_sparse.png
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

BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
for _p in (COMMON_DIR, BASE_DIR, BASE_DIR / "recon", BASE_DIR / "uq"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_utils as du
from map_real import make_direct_predict, make_dpc_predict, reconstruct_map_real
from cv_alpha_real import OUT_DIR, TV_BETA, LAMBDA0, N_ITER, HOLDOUT_STRIDE
from cv_alpha_real_joint import JOINT_SWEEP

from scenario_utils import make_sparse_angle_indices  # noqa: E402

N_SPARSE_ANGLES = 30  # matches run_scenarios.py


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

    # The same 30-angle pool run_scenarios.py's sparse_angle scenario uses.
    sparse_pool = make_sparse_angle_indices(n_angles_full, N_SPARSE_ANGLES)
    # Within that pool, hold out every HOLDOUT_STRIDE-th as test.
    test_pos = np.arange(0, len(sparse_pool), HOLDOUT_STRIDE)
    train_pos = np.setdiff1d(np.arange(len(sparse_pool)), test_pos)
    train_idx = sparse_pool[train_pos]
    test_idx = sparse_pool[test_pos]
    print(f"Sparse-angle pool: {len(sparse_pool)} angles -> {len(train_idx)} train, {len(test_idx)} held-out")

    train_projector, train_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full[train_idx], device, dtype)
    test_projector, test_det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg_full[test_idx], device, dtype)

    # Global field_scale/sigma, same values used everywhere else (see
    # data_utils.py / run_scenarios.py's module docstring for why these are
    # not re-derived from the sparse subset).
    mu_fbp = du.astra_fbp_real(sinos_full["T"], beta_deg_full, L_phys)
    delta_fbp = du.astra_fbp_real(sinos_full["P"], beta_deg_full, L_phys)
    eps_fbp = du.astra_fbp_real(sinos_full["D"], beta_deg_full, L_phys)
    field_scale = {"mu": du.robust_scale(mu_fbp), "delta": du.robust_scale(delta_fbp), "eps": du.robust_scale(eps_fbp)}

    sigma_I = du.estimate_sigma_I(data["I_ref"])
    sigma_ff = du.propagate_sigma_montecarlo(data["I_meas"], data["I_ref"], sigma_I, n_trials=50, seed=0)
    sigma = {"mu": sigma_ff["sigma_T"], "delta": sigma_ff["sigma_DPC"], "eps": sigma_ff["sigma_D"]}
    print(f"sigma_I (flat-field) = {sigma_I:.4e}")
    print("Sigma:", {k: f"{v:.4e}" for k, v in sigma.items()})

    fixed = {"mu": args.alpha_mu, "delta": args.alpha_delta, "eps": args.alpha_eps}
    print(f"Fixed (channelwise CV-optimal): alpha_mu={fixed['mu']:.4g}, alpha_delta={fixed['delta']:.4g}, alpha_eps={fixed['eps']:.4g}")

    T_train = sinos_full["T"][train_idx]
    DPC_train = sinos_full["DPC"][train_idx]
    D_train = sinos_full["D"][train_idx]
    T_test = sinos_full["T"][test_idx]
    DPC_test = sinos_full["DPC"][test_idx]
    D_test = sinos_full["D"][test_idx]
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

    with open(OUT_DIR / "joint_cv_sparse.csv", "w") as f:
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
    ax.set_ylabel("held-out relative error (within sparse-angle pool)")
    ax.legend(fontsize=8)
    ax.set_title(f"Sparse-angle joint-TV CV (n_pool={len(sparse_pool)}, train={len(train_idx)}, test={len(test_idx)})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "joint_cv_sparse.png", dpi=180)
    plt.close(fig)

    print(f"\nALPHA_JOINT (sparse-angle cross-validation) = {best_alpha_joint:.4g}  (total relerr={total[best_idx]:.4f})")
    print(f"Saved: {OUT_DIR / 'joint_cv_sparse.csv'}")
    print(f"Saved: {OUT_DIR / 'joint_cv_sparse.png'}")


if __name__ == "__main__":
    main()
