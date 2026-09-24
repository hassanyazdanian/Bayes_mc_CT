"""
Posterior mean/std (paper eqs. 25-27) from NUTS samples produced by
nuts_real.py, plus a comparison-grid figure per channel: MAP, posterior
mean, posterior std, |MAP - mean|.

Differs from uq/post_process_nuts.py only in dropping the "True" column and
the |True - mean| error column -- real data has no ground truth. |MAP -
mean| is kept as a self-consistency diagnostic instead: MAP is the mode,
posterior mean is the mean, and for these smoothed TV/JTV posteriors they
should be close if NUTS actually converged and the posterior isn't badly
skewed; a large discrepancy is itself a useful red flag even without truth
to compare against.

posterior_mean_std/load_samples are reused as-is from uq/post_process_nuts.py
-- that function only touches stat["meta"]/stat["samples"], nothing
synthetic-specific.

Usage:
    python post_process_real.py --scenario full --prior jtv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
UQ_DIR = BASE_DIR.parent / "uq"
if str(UQ_DIR) not in sys.path:
    sys.path.insert(0, str(UQ_DIR))

from post_process_nuts import load_samples, posterior_mean_std  # noqa: E402

CHANNELS = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]


def plot_uq_grid(stat: Dict, post: Dict[str, np.ndarray], save_path: Path) -> None:
    """4 columns: MAP, posterior mean, posterior std, |MAP - mean|; one row per channel."""
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8.5))
    col_titles = ["MAP", "Posterior mean", "Posterior std", "|MAP - mean|"]

    for i, (ch, label) in enumerate(CHANNELS):
        map_img = stat["map"][ch]
        mean_img = post[f"{ch}_mean"]
        std_img = post[f"{ch}_std"]
        err_img = np.abs(map_img - mean_img)

        imgs = [map_img, mean_img, std_img, err_img]
        vmax_main = max(float(np.clip(im, 0, None).max()) for im in (map_img, mean_img))
        vmax_main = max(vmax_main, 1e-12)
        vmax_std = max(float(std_img.max()), 1e-12)
        vmax_err = max(float(err_img.max()), 1e-12)

        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=13)
            if j < 2:
                im = ax.imshow(np.clip(img, 0, None), cmap="gray", origin="lower", vmin=0.0, vmax=vmax_main)
            elif j == 2:
                im = ax.imshow(img, cmap="magma", origin="lower", vmin=0.0, vmax=vmax_std)
            else:
                im = ax.imshow(img, cmap="magma", origin="lower", vmin=0.0, vmax=vmax_err)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=13)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

    meta = stat["meta"]
    fig.suptitle(
        f"Real data / {meta['scenario']} / {meta['prior_tag']}: NUTS posterior UQ "
        f"(chains={meta['num_chains']}, samples/chain={meta['num_samples']}, "
        f"warmup={meta['warmup_steps']})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="full")
    parser.add_argument("--prior", type=str, default="jtv", choices=("tv", "jtv"))
    parser.add_argument("--pickle_path", type=str, default=None)
    args = parser.parse_args()

    scen_dir = BASE_DIR / "obs" / "uq" / args.scenario / args.prior
    pickle_path = Path(args.pickle_path) if args.pickle_path else scen_dir / "nuts_samples.pickle"

    stat = load_samples(pickle_path)
    post = posterior_mean_std(stat)
    np.savez(scen_dir / "posterior_mean_std.npz", **post)
    plot_uq_grid(stat, post, scen_dir / "uq_grid.png")


if __name__ == "__main__":
    main()
