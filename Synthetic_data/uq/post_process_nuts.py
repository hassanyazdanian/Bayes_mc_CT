"""
Posterior mean/std (paper eqs. 25-27) from NUTS samples produced by
nuts_synthetic.py, plus a comparison-grid figure per channel: True, MAP,
posterior mean, posterior std, |True - mean|.

Usage:
    python post_process_nuts.py --phantom inclusion --scenario full
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CHANNELS = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]


def load_samples(pickle_path: Path) -> Dict:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def posterior_mean_std(stat: Dict) -> Dict[str, np.ndarray]:
    """Pixel-wise posterior mean/std per channel (paper eqs. 25-27), in
    physical units (un-normalized by field_scale, scattered back through the
    support mask -- pixels outside the mask are exactly 0 in every sample)."""
    meta = stat["meta"]
    N, n_active, R = meta["N"], meta["n_active"], meta["support_R"]

    y = np.linspace(-1.0, 1.0, N)
    x = np.linspace(-1.0, 1.0, N)
    Y, X = np.meshgrid(y, x, indexing="ij")
    mask = (X ** 2 + Y ** 2) <= (R ** 2)

    z = stat["samples"]["z"].numpy()  # (K, 3*n_active)
    out: Dict[str, np.ndarray] = {}
    for i, ch in enumerate(("mu", "delta", "eps")):
        scale = meta[f"field_scale_{ch}"]
        z_ch = z[:, i * n_active:(i + 1) * n_active] * scale  # (K, n_active), physical units
        img = np.zeros((z_ch.shape[0], N, N), dtype=np.float64)
        img[:, mask] = z_ch
        out[f"{ch}_mean"] = img.mean(axis=0)
        out[f"{ch}_std"] = img.std(axis=0, ddof=1)
    return out


def plot_uq_grid(stat: Dict, post: Dict[str, np.ndarray], save_path: Path) -> None:
    """5 columns: True, MAP, posterior mean, posterior std, |True - mean|; one row per channel."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.5))
    col_titles = ["True", "MAP", "Posterior mean", "Posterior std", "|True - mean|"]

    for i, (ch, label) in enumerate(CHANNELS):
        true_img = stat["true"][ch]
        map_img = stat["map"][ch]
        mean_img = post[f"{ch}_mean"]
        std_img = post[f"{ch}_std"]
        err_img = np.abs(true_img - mean_img)

        imgs = [true_img, map_img, mean_img, std_img, err_img]
        vmax_main = max(float(np.clip(im, 0, None).max()) for im in (true_img, map_img, mean_img))
        vmax_main = max(vmax_main, 1e-12)
        vmax_std = max(float(std_img.max()), 1e-12)
        vmax_err = max(float(err_img.max()), 1e-12)

        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=13)
            if j < 3:
                im = ax.imshow(np.clip(img, 0, None), cmap="gray", origin="lower", vmin=0.0, vmax=vmax_main)
            elif j == 3:
                im = ax.imshow(img, cmap="magma", origin="lower", vmin=0.0, vmax=vmax_std)
            else:
                im = ax.imshow(img, cmap="magma", origin="lower", vmin=0.0, vmax=vmax_err)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=13)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

    meta = stat["meta"]
    fig.suptitle(
        f"{meta['phantom_name']} / {meta['scenario']}: NUTS posterior UQ "
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
    parser.add_argument("--phantom", type=str, default="inclusion")
    parser.add_argument("--scenario", type=str, default="full")
    parser.add_argument("--prior", type=str, default="jtv", choices=("tv", "jtv"))
    parser.add_argument("--pickle_path", type=str, default=None)
    args = parser.parse_args()

    scen_dir = BASE_DIR / "uq" / "obs" / args.phantom / args.scenario / args.prior
    pickle_path = Path(args.pickle_path) if args.pickle_path else scen_dir / "nuts_samples.pickle"

    stat = load_samples(pickle_path)
    post = posterior_mean_std(stat)
    np.savez(scen_dir / "posterior_mean_std.npz", **post)
    plot_uq_grid(stat, post, scen_dir / "uq_grid.png")


if __name__ == "__main__":
    main()
