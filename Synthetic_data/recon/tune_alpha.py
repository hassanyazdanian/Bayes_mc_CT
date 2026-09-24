"""
Tune per-channel TV regularization (alpha_mu, alpha_delta, alpha_eps) and
then the joint-TV coupling weight (alpha_joint), on the multicontrast
phantom's full-data scenario (180 angles, 10 phase steps) -- matching the
paper's own convention of tuning once and holding fixed across scenarios
(Table II/III: fixed alpha_J across N_theta=361 and N_theta=30).

Runs entirely on CPU: TV/JTV are self-contained (no FBP/CUDA dependency).

Step 1 tunes each channel with its own independent single-channel optimizer
(reconstruct_single_channel), NOT by sweeping alpha_mu=alpha_delta=alpha_eps
through the joint 3-channel optimizer. The two are not equivalent: at
alpha_joint=0 the *objective* is exactly separable (T depends only on mu, P
only on delta, D only on eps, no cross terms), but running all three
through one LBFGS instance still couples their *optimization dynamics* --
L-BFGS's quasi-Newton memory and shared strong-Wolfe line search operate on
the full concatenated parameter vector. Verified empirically: with matched
alpha_mu=alpha_delta=alpha_eps=31.62, delta reconstructs to RE=0.101; with
alpha_mu/alpha_eps changed to very different values (177.8/100) but
alpha_delta held at the same 31.62, delta's result changes to RE=0.153 --
a purely numerical-optimizer artifact, since the loss has no term coupling
delta to mu or eps. Single-channel optimization avoids this entirely.

Selection criterion: RE-minimizing (against the known phantom), NOT the
L-curve corner. The L-curve corner (max curvature of log(prior norm) vs.
log(data-fit residual)) is also computed and plotted for reference -- it's
the data-driven method that needs no ground truth and would transfer to
real data -- but empirically it picks 3-10x more regularization than the
RE-minimizing value across all three channels in this synthetic setting,
where ground truth *is* available, so RE-minimizing is what's actually used
to fix alpha_mu/delta/eps for Step 2 and to produce the ALPHA_* recommendation
printed at the end.

Step 2 (alpha_joint) is NOT separable, so each sweep point is necessarily a
genuine joint 3-channel optimization -- this matches what run_scenarios.py
actually runs, so no analogous artifact applies there.

Usage: python tune_alpha.py
Outputs (written to obs/tuning/): channelwise_sweep.png/.csv,
joint_sweep.png/.csv; recommended values printed at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
for p in (BASE_DIR, COMMON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from map_tv_jtv import compute_metrics, reconstruct_map, reconstruct_single_channel, tv_prior  # noqa: E402
import run_scenarios as rs  # noqa: E402

OUT_DIR = BASE_DIR / "obs" / "tuning"

CHANNELWISE_SWEEP = np.logspace(-1.0, 5.0, 25)  # 1e-1 .. 1e5
JOINT_SWEEP = np.concatenate([[0.0], np.logspace(0.0, 5.0, 18)])

CHANNELS = ["mu", "delta", "eps"]
SINO_KEYS = {"mu": "T", "delta": "P", "eps": "D"}


def l_curve_corner(data_fit: np.ndarray, prior_norm: np.ndarray) -> int:
    """Index of max discrete curvature on the log-log (data_fit, prior_norm)
    curve -- the standard L-curve corner criterion."""
    x = np.log(np.clip(data_fit, 1e-12, None))
    y = np.log(np.clip(prior_norm, 1e-12, None))
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx ** 2 + dy ** 2 + 1e-12, 1.5)
    curvature[0] = curvature[-1] = -np.inf  # endpoints are never the corner
    return int(np.argmax(curvature))


def prior_norm_value(img: np.ndarray, tv_beta: float, device, dtype) -> float:
    t = torch.as_tensor(img, dtype=dtype, device=device)
    return float(tv_prior(t, beta=tv_beta).detach().cpu())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d = rs.load_dataset(BASE_DIR / "obs" / "synthetic_data.h5")
    d = rs.override_phantom(d, rs.PHANTOM_NAME, rs.PHANTOM_SUPERSAMPLE, rs.PHANTOM_TARGET_DPC_MAX, device, dtype)

    angle_idx = np.arange(d["n_angles_full"])
    noisy, sigma = rs.simulate_data(d, angle_idx, d["n_phase_full"], device, dtype)
    projector = rs.build_projector(d, angle_idx, device, dtype, interp=rs.RECON_INTERP)
    det_spacing = rs.det_spacing_of(d)
    truth = {"mu": d["mu"], "delta": d["delta"], "eps": d["eps"]}
    obs_arr = {"mu": noisy["T"], "delta": noisy["DPC"], "eps": noisy["D"]}
    sig_val = {"mu": sigma["T"], "delta": sigma["DPC"], "eps": sigma["D"]}

    # ------------------------------------------------------------------
    # Step 1: independent per-channel TV sweep (own optimizer per channel)
    # ------------------------------------------------------------------
    print("=== Step 1: independent per-channel TV sweep ===")
    curves = {ch: {"alpha": [], "data_fit": [], "prior_norm": [], "RE": [], "SSIM": []} for ch in CHANNELS}

    for ch in CHANNELS:
        print(f"--- channel: {ch} ---")
        for a in CHANNELWISE_SWEEP:
            out = reconstruct_single_channel(
                obs_arr[ch], projector, d["N"], sig_val[ch], alpha_tv=a,
                tv_beta=rs.TV_BETA, lambda0=rs.LAMBDA0, n_iter=rs.N_ITER, device=device, dtype=dtype,
                dpc_det_spacing=det_spacing if ch == "delta" else None,
            )
            data_fit = float(np.linalg.norm(out["pred"] - obs_arr[ch]) / sig_val[ch])
            pnorm = prior_norm_value(out["img"], rs.TV_BETA, device, dtype)
            re, s, _p = compute_metrics(truth[ch], out["img"])
            curves[ch]["alpha"].append(a)
            curves[ch]["data_fit"].append(data_fit)
            curves[ch]["prior_norm"].append(pnorm)
            curves[ch]["RE"].append(re)
            curves[ch]["SSIM"].append(s)
            print(f"  alpha={a:9.3g}  RE={re:.4f} SSIM={s:.4f}")

    best_alpha = {}
    for ch in CHANNELS:
        data_fit = np.asarray(curves[ch]["data_fit"])
        pnorm = np.asarray(curves[ch]["prior_norm"])
        corner = l_curve_corner(data_fit, pnorm)
        re_argmin = int(np.argmin(curves[ch]["RE"]))
        best_alpha[ch] = float(curves[ch]["alpha"][re_argmin])
        print(f"[{ch}] RE-minimizing (used): alpha={best_alpha[ch]:.4g} (RE={curves[ch]['RE'][re_argmin]:.4f})  |  "
              f"L-curve corner (reference only): alpha={curves[ch]['alpha'][corner]:.4g} "
              f"(RE={curves[ch]['RE'][corner]:.4f}, SSIM={curves[ch]['SSIM'][corner]:.4f})")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, ch in enumerate(CHANNELS):
        data_fit = np.asarray(curves[ch]["data_fit"])
        pnorm = np.asarray(curves[ch]["prior_norm"])
        lcurve_idx = l_curve_corner(data_fit, pnorm)
        re_idx = list(curves[ch]["alpha"]).index(best_alpha[ch])

        ax = axes[0, j]
        ax.loglog(curves[ch]["data_fit"], curves[ch]["prior_norm"], "o-")
        ax.plot(curves[ch]["data_fit"][re_idx], curves[ch]["prior_norm"][re_idx], "g*", markersize=16, label="RE-optimal (used)")
        ax.plot(curves[ch]["data_fit"][lcurve_idx], curves[ch]["prior_norm"][lcurve_idx], "r^", markersize=10, label="L-curve corner (ref.)")
        ax.set_xlabel("data-fit residual (||pred-obs||/sigma)")
        ax.set_ylabel("TV prior norm")
        ax.set_title(f"{ch}: L-curve")
        ax.legend(fontsize=7)

        ax2 = axes[1, j]
        ax2.semilogx(curves[ch]["alpha"], curves[ch]["RE"], "o-", label="RE")
        ax2.axvline(best_alpha[ch], color="g", linestyle=":", label="RE-optimal (used)")
        ax2.axvline(curves[ch]["alpha"][lcurve_idx], color="r", linestyle="--", label="L-curve corner (ref.)")
        ax2.set_xlabel("alpha")
        ax2.set_ylabel("RE")
        ax2.set_title(f"{ch}: RE vs alpha")
        ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "channelwise_sweep.png", dpi=180)
    plt.close(fig)

    with open(OUT_DIR / "channelwise_sweep.csv", "w") as f:
        f.write("channel,alpha,data_fit,prior_norm,RE,SSIM\n")
        for ch in CHANNELS:
            for i in range(len(curves[ch]["alpha"])):
                f.write(f"{ch},{curves[ch]['alpha'][i]:.6g},{curves[ch]['data_fit'][i]:.6g},"
                        f"{curves[ch]['prior_norm'][i]:.6g},{curves[ch]['RE'][i]:.6f},{curves[ch]['SSIM'][i]:.6f}\n")

    # ------------------------------------------------------------------
    # Step 2: alpha_joint sweep (genuine joint optimizer -- matches what
    # run_scenarios.py actually runs, so plugging in the step-1 alphas here
    # is expected to shift results slightly from the single-channel numbers
    # above; that shift is real optimizer coupling, not a bug -- see module
    # docstring).
    # ------------------------------------------------------------------
    print("\n=== Step 2: alpha_joint sweep (alpha_mu/delta/eps fixed, joint optimizer) ===")
    print(f"Fixed: alpha_mu={best_alpha['mu']:.4g}, alpha_delta={best_alpha['delta']:.4g}, alpha_eps={best_alpha['eps']:.4g}")

    # Zero-init at every alpha_joint, matching run_scenarios.py and
    # nuts_synthetic.py (see tune_alpha_joint_reoptimal.py).
    joint_rows: List[Dict] = []
    for aj in JOINT_SWEEP:
        out = reconstruct_map(
            noisy["T"], noisy["DPC"], noisy["D"], projector, d["N"],
            sigma["T"], sigma["DPC"], sigma["D"], det_spacing,
            alpha_mu=best_alpha["mu"], alpha_delta=best_alpha["delta"], alpha_eps=best_alpha["eps"], alpha_joint=aj,
            tv_beta=rs.TV_BETA, lambda0=rs.LAMBDA0, n_iter=rs.N_ITER,
            device=device, dtype=dtype,
        )
        row = {"alpha_joint": aj}
        line = f"  alpha_joint={aj:9.3g}  "
        for ch in CHANNELS:
            re, s, _p = compute_metrics(truth[ch], out[ch])
            row[f"RE_{ch}"] = re
            row[f"SSIM_{ch}"] = s
            line += f"{ch}: RE={re:.4f} SSIM={s:.4f}  "
        print(line)
        joint_rows.append(row)

    with open(OUT_DIR / "joint_sweep.csv", "w") as f:
        f.write("alpha_joint," + ",".join(f"RE_{c},SSIM_{c}" for c in CHANNELS) + "\n")
        for row in joint_rows:
            f.write(f"{row['alpha_joint']:.6g}," + ",".join(f"{row[f'RE_{c}']:.6f},{row[f'SSIM_{c}']:.6f}" for c in CHANNELS) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    aj_vals = [r["alpha_joint"] for r in joint_rows]
    for ch in CHANNELS:
        axes[0].plot(aj_vals, [r[f"RE_{ch}"] for r in joint_rows], "o-", label=ch)
        axes[1].plot(aj_vals, [r[f"SSIM_{ch}"] for r in joint_rows], "o-", label=ch)
    axes[0].set_xscale("symlog"); axes[0].set_xlabel("alpha_joint"); axes[0].set_ylabel("RE"); axes[0].legend()
    axes[1].set_xscale("symlog"); axes[1].set_xlabel("alpha_joint"); axes[1].set_ylabel("SSIM"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "joint_sweep.png", dpi=180)
    plt.close(fig)

    print("\n=== Recommended per-channel values (independent single-channel, RE-minimizing) ===")
    print(f"ALPHA_MU = {best_alpha['mu']:.4g}")
    print(f"ALPHA_DELTA = {best_alpha['delta']:.4g}")
    print(f"ALPHA_EPS = {best_alpha['eps']:.4g}")
    print(f"See {OUT_DIR / 'joint_sweep.png'} to pick ALPHA_JOINT.")
    print(f"\nAll outputs saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
