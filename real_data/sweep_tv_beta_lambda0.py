"""
Sweep tv_beta x lambda0, scoring every MAP against the FULL-ANGLE FBP.

Why these two parameters, and why this reference:

  tv_beta sets the smoothed-TV kink width. The TV term's curvature is
      f''(g) = beta^2 / (g^2 + beta^2)^(3/2),  peaking at alpha/beta for g=0.
  So the stiffness lives in FLAT regions, not at edges (at an edge g >> beta
  and f'' ~ beta^2/g^3 ~ 0). In a 512^2 image with a small object, almost
  every pixel is flat, so alpha/tv_beta essentially sets lambda_max: a stripe
  mode on a flat plateau predicts 4*alpha_delta/tv_beta = 4.0e5 against a
  measured lambda_max of 3.9e5. Raising tv_beta lowers that peak.

  lambda0 works from the other end. The L2 amplitude term contributes a
  constant curvature lambda0 per pixel, which puts a floor under the
  flattest, least-constrained directions -- it lifts lambda_min rather than
  lowering lambda_max. Both reduce the condition number, which is what sets
  NUTS trajectory length (leapfrog steps ~ sqrt(lambda_max/lambda_min)).

  The full-angle (361-view) FBP is the reference because it is independent of
  every prior choice being swept -- it uses no prior at all -- and it is the
  best-sampled reconstruction available for this data.

READ THE METRICS WITH CARE. FBP is unregularized and noisy, so scoring
against it rewards LESS regularization: a perfectly denoised MAP is
"different from FBP" precisely because it removed noise FBP kept. Prefer
SSIM (structural) over RE (dominated by the noise mismatch), and read both
as "does this still reconstruct the same object", not as a quality ranking.
The alpha/tv_beta column is the sampling-side cost of the same choice.

Usage:
    python sweep_tv_beta_lambda0.py --scenario sparse_angle
    python sweep_tv_beta_lambda0.py --betas 1e-4 1e-3 1e-2 --lambda0s 1e-6 1e0 1e2
    python sweep_tv_beta_lambda0.py --priors tv jtv --montage_channel eps

Outputs (obs/uq/prior_sweep/): metrics_<prior>.csv, heatmaps_<prior>.png,
montage_<prior>_<channel>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nuts_real import SCENARIOS, setup_problem
import data_utils as du

def _recon_dir(base: Path) -> Path:
    """Locate the shared prior/metric module (map_tv_jtv.py).

    It lived in <project>/recon/ before the synthetic pipeline was moved into
    Synthetic_data/; accept either layout so this file works in both.
    """
    for candidate in (base / "recon", base / "Synthetic_data" / "recon"):
        if (candidate / "map_tv_jtv.py").exists():
            return candidate
    return base / "recon"


RECON_DIR = _recon_dir(Path(__file__).resolve().parent.parent)
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))
from map_tv_jtv import compute_metrics  # noqa: E402

CHANNELS = ["mu", "delta", "eps"]
OUT_DIR = Path(__file__).resolve().parent / "obs" / "uq" / "prior_sweep"

# Mean block curvature of the data term, measured at the MAP for
# sparse_angle (mu 1.18e4, delta 1.55e4, eps 1.22e4). Used only to put the
# alpha/tv_beta numbers on a meaningful scale in the printout.
DATA_CURVATURE = 1.2e4


def full_angle_fbp(device, dtype):
    """The 361-view FBP, in all three channels. Prior-independent reference."""
    data = du.load_real_intensities()
    sinos = du.retrieve_and_correct_sinograms(data["I_meas"], data["I_ref"], verbose=False)
    beta_deg = du.make_beta_deg(data["angles"], data["I_meas"].shape[0])
    L_phys = du.geometry_for(data["I_ref"].shape[-1])
    return {
        "mu": du.astra_fbp_real(sinos["T"], beta_deg, L_phys),
        "delta": du.astra_fbp_real(sinos["P"], beta_deg, L_phys),
        "eps": du.astra_fbp_real(sinos["D"], beta_deg, L_phys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="sparse_angle", choices=SCENARIOS)
    parser.add_argument("--betas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--lambda0s", type=float, nargs="+", default=[1e-6, 1e-2, 1e0, 1e2],
                         help="The L2 term contributes curvature lambda0 per pixel against a data "
                              "curvature of ~1.2e4, so values below ~1 cannot move the MAP; they can "
                              "still lift lambda_min for the sampler.")
    parser.add_argument("--priors", type=str, nargs="+", default=["tv"], choices=["tv", "jtv"])
    parser.add_argument("--montage_channel", type=str, default="delta", choices=CHANNELS)
    parser.add_argument("--projector", type=str, default=None, choices=("bilinear", "nearest"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {args.device}")

    print("\nBuilding the full-angle FBP reference (361 views)...")
    ref = full_angle_fbp(args.device, dtype)
    for ch in CHANNELS:
        print(f"  {ch:5s}: range [{ref[ch].min():.4g}, {ref[ch].max():.4g}]")

    n_runs = len(args.priors) * len(args.betas) * len(args.lambda0s)
    print(f"\n{n_runs} MAP solves ({len(args.priors)} prior(s) x {len(args.betas)} betas "
          f"x {len(args.lambda0s)} lambda0s)")

    for prior_tag in args.priors:
        aj = 0.0 if prior_tag == "tv" else None
        maps = {}
        rows = []
        for b in args.betas:
            for l0 in args.lambda0s:
                print(f"\n=== {prior_tag}: tv_beta={b:g}, lambda0={l0:g} ===")
                prob = setup_problem(args.scenario, alpha_joint=aj, tv_beta=b, lambda0=l0,
                                     projector_interp=args.projector, device=args.device,
                                     dtype=dtype, verbose=False)
                m = prob["map_out"]
                maps[(b, l0)] = m
                row = {"tv_beta": b, "lambda0": l0,
                       "kink_mu": prob["alpha_mu"] / b,
                       "kink_delta": prob["alpha_delta"] / b,
                       "kink_eps": prob["alpha_eps"] / b}
                line = f"  kink alpha/beta: mu={row['kink_mu']:.3g} delta={row['kink_delta']:.3g} " \
                       f"eps={row['kink_eps']:.3g}  (data ~{DATA_CURVATURE:.1g})\n  "
                for ch in CHANNELS:
                    re, ssim, _ = compute_metrics(ref[ch], m[ch])
                    row[f"RE_{ch}"], row[f"SSIM_{ch}"] = re, ssim
                    line += f"{ch}: RE={re:.4f} SSIM={ssim:.4f}   "
                print(line)
                rows.append(row)

        # ---------------- CSV ----------------
        keys = (["tv_beta", "lambda0", "kink_mu", "kink_delta", "kink_eps"]
                + [f"{m}_{ch}" for ch in CHANNELS for m in ("RE", "SSIM")])
        csv = OUT_DIR / f"metrics_{prior_tag}.csv"
        with open(csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.6g}" for k in keys) + "\n")
        print(f"\nsaved: {csv}")

        # ---------------- heatmaps ----------------
        nb, nl = len(args.betas), len(args.lambda0s)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        for j, ch in enumerate(CHANNELS):
            for i, metric in enumerate(("RE", "SSIM")):
                grid = np.array([[next(r[f"{metric}_{ch}"] for r in rows
                                       if r["tv_beta"] == b and r["lambda0"] == l0)
                                  for l0 in args.lambda0s] for b in args.betas])
                ax = axes[i, j]
                im = ax.imshow(grid, cmap="viridis" if metric == "SSIM" else "magma_r",
                               aspect="auto", origin="lower")
                ax.set_xticks(range(nl)); ax.set_xticklabels([f"{v:g}" for v in args.lambda0s], fontsize=8)
                ax.set_yticks(range(nb)); ax.set_yticklabels([f"{v:g}" for v in args.betas], fontsize=8)
                ax.set_xlabel("lambda0", fontsize=9); ax.set_ylabel("tv_beta", fontsize=9)
                ax.set_title(f"{ch}: {metric} vs full-angle FBP", fontsize=10)
                for y in range(nb):
                    for x in range(nl):
                        ax.text(x, y, f"{grid[y, x]:.3f}", ha="center", va="center",
                                fontsize=7, color="w")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f"{args.scenario} / {prior_tag}: MAP vs full-angle FBP "
                     f"(SSIM higher = better; RE lower = closer to FBP, incl. its noise)", fontsize=12)
        fig.tight_layout()
        out = OUT_DIR / f"heatmaps_{prior_tag}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}")

        # ---------------- image montage ----------------
        ch = args.montage_channel
        fig, axes = plt.subplots(nb, nl + 1, figsize=(2.4 * (nl + 1), 2.4 * nb), squeeze=False)
        vmax = max(float(np.clip(ref[ch], 0, None).max()),
                   max(float(np.clip(maps[(b, l0)][ch], 0, None).max())
                       for b in args.betas for l0 in args.lambda0s))
        for y, b in enumerate(args.betas):
            for x, l0 in enumerate(args.lambda0s):
                ax = axes[y][x]
                ax.imshow(np.clip(maps[(b, l0)][ch], 0, None), cmap="gray",
                          origin="lower", vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if y == 0:
                    ax.set_title(f"lambda0={l0:g}", fontsize=9)
                if x == 0:
                    ax.set_ylabel(f"beta={b:g}", fontsize=9)
            ax = axes[y][nl]
            ax.imshow(np.clip(ref[ch], 0, None), cmap="gray", origin="lower", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if y == 0:
                ax.set_title("full-angle FBP", fontsize=9, color="tab:red")
        fig.suptitle(f"{args.scenario} / {prior_tag}: {ch} MAP across the sweep, "
                     f"vs the full-angle FBP reference", fontsize=12)
        fig.tight_layout()
        out = OUT_DIR / f"montage_{prior_tag}_{ch}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
