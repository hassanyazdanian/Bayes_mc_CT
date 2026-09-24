"""
Re-run only the alpha_joint sweep (tune_alpha.py's Step 2), with alpha_mu/
delta/eps FIXED at given RE-optimal values instead of tune_alpha.py's
default L-curve-corner picks. Avoids repeating the expensive Step 1
per-channel sweep when only the channelwise alphas' *selection criterion*
changed, not the underlying curves.

--scenario tunes alpha_joint on limited data instead of the full scan. The
coupling exists to transfer structure between channels when the data do not
determine each channel on their own, so the full scan can be uninformative
about it (as with the real data, where alpha_joint was tuned on sparse_angle).

Usage:
    python tune_alpha_joint_reoptimal.py --alpha_mu 100 --alpha_delta 31.62 --alpha_eps 56.23
    python tune_alpha_joint_reoptimal.py --alpha_mu 100 --alpha_delta 31.62 --alpha_eps 56.23 --scenario sparse_angle

Outputs (obs/tuning/): joint_sweep_reoptimal[_<scenario>].csv/.png (no suffix for full)
"""
from __future__ import annotations

import argparse
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

from map_tv_jtv import compute_metrics, reconstruct_map  # noqa: E402
import run_scenarios as rs  # noqa: E402
from tune_alpha import JOINT_SWEEP, OUT_DIR, CHANNELS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_mu", type=float, required=True)
    parser.add_argument("--alpha_delta", type=float, required=True)
    parser.add_argument("--alpha_eps", type=float, required=True)
    parser.add_argument("--scenario", default="full", choices=("full", "sparse_angle", "sparse_step", "combined"))
    args = parser.parse_args()
    tag = "" if args.scenario == "full" else f"_{args.scenario}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d = rs.load_dataset(BASE_DIR / "obs" / "synthetic_data.h5")
    d = rs.override_phantom(d, rs.PHANTOM_NAME, rs.PHANTOM_SUPERSAMPLE, rs.PHANTOM_TARGET_DPC_MAX, device, dtype)

    n_angles, n_phase = {
        "full": (d["n_angles_full"], d["n_phase_full"]),
        "sparse_angle": (rs.N_SPARSE_ANGLES, d["n_phase_full"]),
        "sparse_step": (d["n_angles_full"], rs.N_SPARSE_PHASE),
        "combined": (rs.N_SPARSE_ANGLES, rs.N_SPARSE_PHASE),
    }[args.scenario]
    angle_idx = rs.make_sparse_angle_indices(d["n_angles_full"], n_angles)
    noisy, sigma = rs.simulate_data(d, angle_idx, n_phase, device, dtype)
    projector = rs.build_projector(d, angle_idx, device, dtype, interp=rs.RECON_INTERP)
    print(f"scenario={args.scenario} (n_angles={n_angles}, n_phase={n_phase})")
    print(f"tv_beta={rs.TV_BETA:g}  lambda0={rs.LAMBDA0:g}  noise={rs.NOISE_LEVEL}  "
          f"simulate={d.get('N_sim', d['N'])}^2 {rs.SIM_INTERP}, reconstruct={d['N']}^2 {rs.RECON_INTERP}")
    det_spacing = rs.det_spacing_of(d)
    truth = {"mu": d["mu"], "delta": d["delta"], "eps": d["eps"]}

    fixed = {"mu": args.alpha_mu, "delta": args.alpha_delta, "eps": args.alpha_eps}
    print(f"Fixed (RE-optimal): alpha_mu={fixed['mu']:.4g}, alpha_delta={fixed['delta']:.4g}, alpha_eps={fixed['eps']:.4g}")

    # Zero-init at every alpha_joint, matching run_scenarios.py and
    # nuts_synthetic.py (JTV is not warm-started there). With a fixed LBFGS
    # budget the start point changes the result, so tuning from a TV warm
    # start would select alpha_joint for a reconstruction that is never run.
    joint_rows: List[Dict] = []
    for aj in JOINT_SWEEP:
        out = reconstruct_map(
            noisy["T"], noisy["DPC"], noisy["D"], projector, d["N"],
            sigma["T"], sigma["DPC"], sigma["D"], det_spacing,
            alpha_mu=fixed["mu"], alpha_delta=fixed["delta"], alpha_eps=fixed["eps"], alpha_joint=aj,
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

    with open(OUT_DIR / f"joint_sweep_reoptimal{tag}.csv", "w") as f:
        f.write("alpha_joint," + ",".join(f"RE_{c},SSIM_{c}" for c in CHANNELS) + "\n")
        for row in joint_rows:
            f.write(f"{row['alpha_joint']:.6g}," + ",".join(f"{row[f'RE_{c}']:.6f},{row[f'SSIM_{c}']:.6f}" for c in CHANNELS) + "\n")

    re_sum = [sum(r[f"RE_{c}"] for c in CHANNELS) for r in joint_rows]
    i_best = int(np.argmin(re_sum))
    print(f"\nSummed-RE minimum: alpha_joint={joint_rows[i_best]['alpha_joint']:.4g} "
          f"(sum RE {re_sum[i_best]:.6f} vs {re_sum[0]:.6f} at alpha_joint=0)"
          + ("  <-- AT GRID EDGE, extend JOINT_SWEEP" if i_best == len(joint_rows) - 1 else ""))
    ssim_mean = [np.mean([r[f"SSIM_{c}"] for c in CHANNELS]) for r in joint_rows]
    j_best = int(np.argmax(ssim_mean))
    print(f"Mean-SSIM maximum: alpha_joint={joint_rows[j_best]['alpha_joint']:.4g} "
          f"(mean SSIM {ssim_mean[j_best]:.6f} vs {ssim_mean[0]:.6f} at alpha_joint=0)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    aj_vals = [r["alpha_joint"] for r in joint_rows]
    for ch in CHANNELS:
        axes[0].plot(aj_vals, [r[f"RE_{ch}"] for r in joint_rows], "o-", label=ch)
        axes[1].plot(aj_vals, [r[f"SSIM_{ch}"] for r in joint_rows], "o-", label=ch)
    axes[0].set_xscale("symlog"); axes[0].set_xlabel("alpha_joint"); axes[0].set_ylabel("RE"); axes[0].legend()
    axes[1].set_xscale("symlog"); axes[1].set_xlabel("alpha_joint"); axes[1].set_ylabel("SSIM"); axes[1].legend()
    fig.suptitle(f"{args.scenario}: alpha_mu={fixed['mu']:.4g}, alpha_delta={fixed['delta']:.4g}, "
                 f"alpha_eps={fixed['eps']:.4g} (fixed)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"joint_sweep_reoptimal{tag}.png", dpi=180)
    plt.close(fig)

    print(f"\nSaved: {OUT_DIR / f'joint_sweep_reoptimal{tag}.csv'}")
    print(f"Saved: {OUT_DIR / f'joint_sweep_reoptimal{tag}.png'}")


if __name__ == "__main__":
    main()
