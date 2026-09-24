"""Loading NUTS sample archives and reducing them to posterior summaries.

Shared by both studies: samples are stored the same way either way -- a flat
vector over the support mask, normalized by field_scale -- so the reduction to
pixel-wise mean and standard deviation is identical.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np


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
