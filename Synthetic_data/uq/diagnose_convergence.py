"""
Spatial R-hat/ESS diagnostic map: where in the image is MCMC convergence
poor? nuts_synthetic.py's built-in diagnostic only checks a random 2000-dim
subset and reports min/median/max -- this computes R-hat/ESS for every
active pixel and scatters it back into (mu, delta, eps) images so bad
dimensions can be located spatially (e.g. background/void vs. object edges).

Usage:
    python diagnose_convergence.py --pickle_path obs/multicontrast/combined/tv/nuts_samples.pickle
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyro.ops.stats import effective_sample_size, gelman_rubin

CHANNELS = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]

# pyro's effective_sample_size (_cummin) allocates an intermediate that
# scales as O(num_samples^2 * dim) -- fine for the few hundred samples we
# used early on, but at num_samples=2200 a single call over all ~147k dims
# tries to allocate ~700GB and OOMs. Chunk over the dimension axis instead
# (same numerics, pyro's own trusted implementation, just called in batches
# small enough to fit in memory) so this still covers every pixel rather
# than falling back to nuts_synthetic.py's random 2000-dim subset.
CHUNK_SIZE = 2000


def chunked_ess_rhat(z_grouped: torch.Tensor, chunk_size: int = CHUNK_SIZE):
    dim = z_grouped.shape[-1]
    ess_parts, rhat_parts = [], []
    for start in range(0, dim, chunk_size):
        chunk = z_grouped[..., start:start + chunk_size]
        ess_parts.append(effective_sample_size(chunk))
        rhat_parts.append(gelman_rubin(chunk))
    return torch.cat(ess_parts), torch.cat(rhat_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    pickle_path = Path(args.pickle_path)
    with open(pickle_path, "rb") as f:
        stat = pickle.load(f)

    meta = stat["meta"]
    N, n_active, R = meta["N"], meta["n_active"], meta["support_R"]
    num_chains = meta["num_chains"]
    print(f"phantom={meta['phantom_name']} scenario={meta['scenario']} prior={meta.get('prior_tag')}")
    print(f"num_chains={num_chains} num_samples={meta['num_samples']} warmup={meta['warmup_steps']}")

    if num_chains < 2:
        raise SystemExit("Need num_chains >= 2 for R-hat/ESS.")

    z_grouped = stat["samples"]["z_grouped"]  # (chains, samples, dim)
    print(f"z_grouped shape: {tuple(z_grouped.shape)}  (computing over all {z_grouped.shape[-1]} dims)")

    ess, rhat = chunked_ess_rhat(z_grouped)
    print(f"ESS:  min={ess.min():.2f} median={ess.median():.2f} max={ess.max():.2f}")
    print(f"Rhat: min={rhat.min():.4f} median={rhat.median():.4f} max={rhat.max():.4f}")

    y = np.linspace(-1.0, 1.0, N)
    x = np.linspace(-1.0, 1.0, N)
    Y, X = np.meshgrid(y, x, indexing="ij")
    mask = (X ** 2 + Y ** 2) <= (R ** 2)

    ess_np = ess.numpy()
    rhat_np = rhat.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, (ch, label) in enumerate(CHANNELS):
        ess_ch = ess_np[i * n_active:(i + 1) * n_active]
        rhat_ch = rhat_np[i * n_active:(i + 1) * n_active]

        ess_img = np.zeros((N, N))
        ess_img[mask] = ess_ch
        rhat_img = np.ones((N, N))
        rhat_img[mask] = rhat_ch

        ax = axes[0, i]
        im = ax.imshow(ess_img, cmap="viridis", origin="lower", vmin=0)
        ax.set_title(f"{label}: ESS (median={np.median(ess_ch):.1f})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1, i]
        im = ax.imshow(rhat_img, cmap="magma", origin="lower", vmin=1.0, vmax=min(2.0, max(1.05, np.percentile(rhat_ch, 99))))
        ax.set_title(f"{label}: R-hat (median={np.median(rhat_ch):.3f})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"{meta['phantom_name']} / {meta['scenario']} / {meta.get('prior_tag')}: "
        f"per-pixel ESS / R-hat ({num_chains} chains x {meta['num_samples']} samples)",
        fontsize=12,
    )
    fig.tight_layout()
    save_path = Path(args.save_path) if args.save_path else pickle_path.parent / "convergence_map.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
