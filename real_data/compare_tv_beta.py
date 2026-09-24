"""
MAP A/B gate: does raising tv_beta change the MAP reconstruction?

Why this exists: the 2x200 sparse-angle NUTS run failed to mix (median ESS
5.1, max R-hat 7.8) while the old reference script's run on the SAME data,
dimension, likelihood, and tv_beta mixed beautifully (median ESS 322, max
R-hat 1.05). The one large difference is alpha: the CV-tuned values are
36-500x the reference script's, so the smoothed-TV kink stiffness
alpha/tv_beta went from a benign 200-500 (2-4% of the data curvature) to
5.6e4-1e5 (up to 8x the data curvature) -- a dominant, state-dependent
curvature that no fixed preconditioner can absorb. The proposed fix is
tv_beta 1e-4 -> ~1e-2 (the posterior's own gradient-fluctuation scale,
sqrt(2/h_data) ~ 1.3e-2), restoring alpha/beta to the proven-good range
while staying 20-50x below edge gradients.

That fix changes the prior, hence (slightly) the MAP: gradients below beta
get a quadratic (Huber) penalty instead of a linear one. This script
measures exactly how much, for TV and JTV, before the new beta is adopted
anywhere. If RE between the two betas' MAPs is small (~1% or less) and the
difference images show only flat-region texture (no edge changes), the new
beta is safe to adopt globally; a large difference means stop and rethink.

Usage:
    python compare_tv_beta.py --scenario sparse_angle --betas 1e-4 1e-2

Outputs (obs/uq/tv_beta_check/): map_diff_tv.png, map_diff_jtv.png
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
OUT_DIR = Path(__file__).resolve().parent / "obs" / "uq" / "tv_beta_check"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="sparse_angle", choices=SCENARIOS)
    parser.add_argument("--betas", type=float, nargs=2, default=[1e-4, 1e-2],
                         help="Reference tv_beta and candidate tv_beta, in that order.")
    parser.add_argument("--projector", type=str, default="bilinear", choices=("bilinear", "nearest"),
                             help="Override data_utils.PROJECTOR_INTERP for this run.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b_ref, b_new = args.betas

    maps = {}
    for prior_tag, aj in (("tv", 0.0), ("jtv", None)):
        for b in (b_ref, b_new):
            print(f"=== MAP: {prior_tag}, tv_beta={b:g} ===")
            prob = setup_problem(args.scenario, alpha_joint=aj, tv_beta=b,
                                 device=args.device, dtype=dtype,  projector_interp = args.projector, verbose=False)
            maps[(prior_tag, b)] = prob["map_out"]

    print(f"\n=== MAP difference, tv_beta {b_ref:g} -> {b_new:g} ===")
    print("(RE/SSIM of the candidate-beta MAP scored against the reference-beta MAP;")
    print(" RE ~ 0.01 or below and SSIM ~ 1 => beta change is safe to adopt)")
    for prior_tag in ("tv", "jtv"):
        a = maps[(prior_tag, b_ref)]
        c = maps[(prior_tag, b_new)]
        line = f"  {prior_tag:4s}: "
        for ch in CHANNELS:
            re, ssim, _ = compute_metrics(a[ch], c[ch])
            line += f"{ch}: RE={re:.4f} SSIM={ssim:.4f}   "
        print(line)

        fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.5))
        for i, ch in enumerate(CHANNELS):
            ref_img = np.clip(a[ch], 0, None)
            new_img = np.clip(c[ch], 0, None)
            diff = np.abs(a[ch] - c[ch])
            vmax = max(float(ref_img.max()), float(new_img.max()), 1e-12)
            panels = [
                (ref_img, f"beta={b_ref:g}", "gray", vmax),
                (new_img, f"beta={b_new:g}", "gray", vmax),
                (diff, "|difference|", "magma", max(float(diff.max()), 1e-12)),
            ]
            for j, (img, title, cmap, vm) in enumerate(panels):
                ax = axes[i, j]
                im = ax.imshow(img, cmap=cmap, origin="lower", vmin=0.0, vmax=vm)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(title, fontsize=12)
                if j == 0:
                    ax.set_ylabel(ch, fontsize=12)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=7)
        fig.suptitle(f"{args.scenario} / {prior_tag}: MAP sensitivity to tv_beta", fontsize=13)
        fig.tight_layout()
        out = OUT_DIR / f"map_diff_{prior_tag}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
