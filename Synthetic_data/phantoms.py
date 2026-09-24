"""
Synthetic phantom library for limited-data multi-contrast CT.

Each phantom returns attenuation (mu), refraction (delta), and dark-field
(eps) fields of shape (N, N), plus an integer region ``labels`` map.

Available phantoms (see ``PHANTOMS``):
    disk          - single homogeneous disk (sanity check).
    inclusion     - disk body with a few contrast inclusions.
    shepp         - classic Shepp-Logan ellipse layout, multi-contrast.
    multicontrast - full structural phantom for joint Bayesian reconstruction
                     (TV-friendly regions, shared-edge inclusions, channel-
                     unique features, and a detectability ladder for UQ).

Anti-inverse-crime: phantoms are painted at ``supersample`` x resolution and
average-pooled down to ``N``, so the data-generation grid differs from the
reconstruction grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _rot(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, deg: float):
    t = np.deg2rad(deg)
    Xr = (X - cx) * np.cos(t) + (Y - cy) * np.sin(t)
    Yr = -(X - cx) * np.sin(t) + (Y - cy) * np.cos(t)
    return Xr, Yr


def _ellipse(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, a: float, b: float, deg: float = 0.0) -> np.ndarray:
    Xr, Yr = _rot(X, Y, cx, cy, deg)
    return (Xr / a) ** 2 + (Yr / b) ** 2 <= 1.0


def _ring_segment(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, r0: float, r1: float, th0_deg: float, th1_deg: float) -> np.ndarray:
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    TH = np.degrees(np.arctan2(Y - cy, X - cx))
    return (R >= r0) & (R <= r1) & (TH >= th0_deg) & (TH <= th1_deg)


def _grid(N: int, supersample: int) -> Tuple[np.ndarray, np.ndarray]:
    n = N * supersample
    c = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0 / n
    return np.meshgrid(c, c)  # X, Y with row = y, col = x


def _pool(A: np.ndarray, N: int, supersample: int) -> np.ndarray:
    return A.reshape(N, supersample, N, supersample).mean(axis=(1, 3)).astype(np.float32)


def _finalize(N: int, supersample: int, mu: np.ndarray, delta: np.ndarray, eps: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "mu": _pool(mu, N, supersample),
        "delta": _pool(delta, N, supersample),
        "eps": _pool(eps, N, supersample),
        "labels": labels[::supersample, ::supersample].copy(),
    }


def _paint(N: int, supersample: int, regions: List[Tuple[np.ndarray, str, int]], materials: Mapping[str, Tuple[float, float, float]]) -> Dict[str, np.ndarray]:
    """Rasterize a list of (mask, material_key, label_code) regions, later entries on top."""
    n = N * supersample
    mu = np.zeros((n, n))
    delta = np.zeros((n, n))
    eps = np.zeros((n, n))
    labels = np.zeros((n, n), dtype=np.uint8)
    for mask, key, code in regions:
        m, d, e = materials[key]
        mu[mask], delta[mask], eps[mask] = m, d, e
        labels[mask] = code
    return _finalize(N, supersample, mu, delta, eps, labels)


# -----------------------------------------------------------------------------
# Shared material palette (mu, delta, eps), normalized units, body = 1.0
# -----------------------------------------------------------------------------

MATERIALS: Mapping[str, Tuple[float, float, float]] = {
    "body":        (1.00, 1.00, 0.20),  # PMMA-like disc, shared outer edge
    "rodA":        (1.50, 1.35, 0.30),  # shared edges, strong in all channels
    "rodB":        (0.65, 0.80, 0.10),  # shared edges, negative contrast
    "foam":        (0.92, 0.96, 1.20),  # eps-unique; weak in mu/delta
    "delta_only":  (1.00, 1.30, 0.20),  # mu-matched, delta-unique
    "lc_ellipse":  (1.06, 1.06, 0.20),  # large low-contrast feature for UQ
    "ladder":      (1.25, 1.25, 0.45),  # detectability series for UQ + JTV
    "crack":       (1.03, 1.05, 0.90),  # thin eps-dominant feature
}

LABEL_CODES: Mapping[int, str] = {
    0: "background",
    1: "body",
    2: "rodA_shared",
    3: "rodB_shared",
    4: "foam_eps_dominant",
    5: "delta_only",
    6: "low_contrast_ellipse",
    7: "detectability_ladder",
    8: "thin_eps_defect",
}

LADDER_RADII = [0.055, 0.042, 0.030, 0.021, 0.013]
LADDER_X = [-0.30, -0.15, 0.00, 0.15, 0.30]
LADDER_Y = -0.55


# -----------------------------------------------------------------------------
# Phantoms
# -----------------------------------------------------------------------------

def disk_phantom(N: int = 512, supersample: int = 2) -> Dict[str, np.ndarray]:
    """Single homogeneous disk on an empty background."""
    X, Y = _grid(N, supersample)
    regions = [(_ellipse(X, Y, 0.0, 0.0, 0.80, 0.80), "body", 1)]
    return _paint(N, supersample, regions, MATERIALS)


def inclusion_phantom(N: int = 512, supersample: int = 2) -> Dict[str, np.ndarray]:
    """Homogeneous disk body with three inclusions of distinct contrast."""
    X, Y = _grid(N, supersample)
    regions = [
        (_ellipse(X, Y, 0.0, 0.0, 0.80, 0.80), "body", 1),
        (_ellipse(X, Y, -0.30, 0.20, 0.16, 0.16), "rodA", 2),
        (_ellipse(X, Y, 0.30, 0.15, 0.14, 0.14), "rodB", 3),
        (_ellipse(X, Y, -0.05, -0.30, 0.13, 0.13), "foam", 4),
    ]
    return _paint(N, supersample, regions, MATERIALS)


def multicontrast_phantom(N: int = 512, supersample: int = 2) -> Dict[str, np.ndarray]:
    """Full structural phantom: shared-edge inclusions, channel-unique
    features, a low-contrast ellipse, a detectability ladder, and a thin
    eps-dominant crack (see ``LABEL_CODES``)."""
    X, Y = _grid(N, supersample)
    regions = [
        (_ellipse(X, Y, 0.0, 0.0, 0.82, 0.82), "body", 1),
        (_ellipse(X, Y, 0.0, 0.06, 0.30, 0.20, -10), "lc_ellipse", 6),
        (_ellipse(X, Y, -0.40, 0.38, 0.15, 0.22, 25), "rodA", 2),
        (_ellipse(X, Y, 0.42, 0.38, 0.14, 0.14), "rodB", 3),
        (_ellipse(X, Y, -0.44, -0.28, 0.16, 0.16), "foam", 4),
        (_ellipse(X, Y, 0.42, -0.30, 0.14, 0.14), "delta_only", 5),
        *[(_ellipse(X, Y, x0, LADDER_Y, r, r), "ladder", 7) for x0, r in zip(LADDER_X, LADDER_RADII)],
        (_ring_segment(X, Y, 0.0, 0.06, 0.615, 0.625, -20, 20), "crack", 8),
    ]
    return _paint(N, supersample, regions, MATERIALS)


# Classic Shepp-Logan ellipse geometry, extended with per-ellipse (mu, delta,
# eps) amplitudes. mu matches the standard table; delta/eps are independent
# amplitude sets so the three channels carry genuinely different contrast.
# Columns: a, b, x0, y0, phi_deg, A_mu, A_delta, A_eps
SHEPP_ELLIPSES: List[Tuple[float, float, float, float, float, float, float, float]] = [
    (0.6900, 0.9200,  0.00,  0.0000,   0,  1.00,  0.95,  0.30),
    (0.6624, 0.8740,  0.00, -0.0184,   0, -0.80, -0.75, -0.24),
    (0.1100, 0.3100,  0.22,  0.0000, -18,  0.20,  0.10,  0.35),
    (0.1600, 0.4100, -0.22,  0.0000,  18,  0.20,  0.10,  0.35),
    (0.2100, 0.2500,  0.00,  0.3500,   0,  0.10,  0.05,  0.40),
    (0.0460, 0.0460,  0.00,  0.1000,   0,  0.10,  0.30,  0.05),
    (0.0460, 0.0460,  0.00, -0.1000,   0,  0.10,  0.30,  0.05),
    (0.0460, 0.0230, -0.08, -0.6050,   0,  0.10,  0.30,  0.05),
    (0.0230, 0.0230,  0.00, -0.6060,   0,  0.10,  0.30,  0.05),
    (0.0230, 0.0460,  0.06, -0.6050,   0,  0.10,  0.30,  0.05),
]


def shepp_logan_phantom(N: int = 512, supersample: int = 2) -> Dict[str, np.ndarray]:
    """Classic Shepp-Logan ellipse layout with multi-contrast amplitudes."""
    X, Y = _grid(N, supersample)
    n = N * supersample
    mu = np.zeros((n, n))
    delta = np.zeros((n, n))
    eps = np.zeros((n, n))
    labels = np.zeros((n, n), dtype=np.uint8)
    for code, (a, b, x0, y0, phi, A_mu, A_delta, A_eps) in enumerate(SHEPP_ELLIPSES, start=1):
        mask = _ellipse(X, Y, x0, y0, a, b, phi)
        mu[mask] += A_mu
        delta[mask] += A_delta
        eps[mask] += A_eps
        labels[mask] = code
    mu = np.clip(mu, 0.0, None)
    delta = np.clip(delta, 0.0, None)
    eps = np.clip(eps, 0.0, None)
    return _finalize(N, supersample, mu, delta, eps, labels)


PHANTOMS = {
    "disk": disk_phantom,
    "inclusion": inclusion_phantom,
    "shepp": shepp_logan_phantom,
    "multicontrast": multicontrast_phantom,
}
PHANTOM_NAMES = tuple(PHANTOMS)


def make_phantom(name: str = "multicontrast", N: int = 512, supersample: int = 2) -> Dict[str, np.ndarray]:
    """Build a phantom by name. See ``PHANTOM_NAMES`` for valid choices."""
    if name not in PHANTOMS:
        raise ValueError(f"Unknown phantom '{name}'. Choose from {PHANTOM_NAMES}.")
    return PHANTOMS[name](N=N, supersample=supersample)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

def load_phantom_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a phantom saved by :func:`save_phantom_npz`."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_phantom_npz(path: str | Path, phantom: Mapping[str, np.ndarray]) -> Path:
    """Save phantom arrays to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in phantom.items()})
    return path


# -----------------------------------------------------------------------------
# Publication-quality plotting
# -----------------------------------------------------------------------------

def set_publication_style(font_size: float = 9.0, tick_size: float = 8.0) -> None:
    """Set Matplotlib parameters suitable for journal figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": tick_size,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _format_image_axis(ax, xlabel: str = r"$x$", ylabel: str = r"$y$") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _add_colorbar(fig, ax, im, label: Optional[str] = None):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    if label:
        cbar.set_label(label)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.7)
    cbar.outline.set_linewidth(0.7)
    return cbar


# Compact annotations for the multicontrast phantom's mu panel (region labels).
MULTICONTRAST_ANNOTATIONS = [
    (-0.40, 0.4, "A", 0),
    (0.42, 0.4, "B", 0),
    (-0.44, -0.30, "F", 0),
    (0.42, -0.30, "P", 0),
    (0.0, 0.06, "L", 0),
    (0.0, -0.74, "ladder", 0),
    (0.72, 0.03, "crack", 90),
]


def plot_phantom(
    phantom: Mapping[str, np.ndarray],
    save_prefix: Optional[str | Path] = None,
    annotations: Optional[List[Tuple[float, float, str, float]]] = None,
    show: bool = False,
    dpi: int = 600,
):
    """Plot the mu/delta/eps channels of any phantom in a publication-ready
    layout. Saves ``<save_prefix>.pdf`` and ``.png`` when given."""
    import matplotlib.pyplot as plt

    set_publication_style()

    keys = ["mu", "delta", "eps"]
    titles = [r"$\mu$ (attenuation)", r"$\delta$ (refraction)", r"$\epsilon$ (dark-field)"]
    cbar_labels = [r"$\mu$", r"$\delta$", r"$\epsilon$"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    extent = [-1.0, 1.0, -1.0, 1.0]

    for ax, key, title, cblabel in zip(axes, keys, titles, cbar_labels):
        im = ax.imshow(np.asarray(phantom[key]), origin="lower", extent=extent, cmap="gray", interpolation="nearest")
        ax.set_title(title)
        _format_image_axis(ax)
        _add_colorbar(fig, ax, im, cblabel)

    if annotations:
        for x, y, text, rot in annotations:
            axes[0].text(
                x, y, text, color="green", ha="center", va="center", fontsize=6.5,
                rotation=rot, rotation_mode="anchor",
                bbox=dict(boxstyle="round,pad=0.12", fc="none", ec="none", alpha=0.45),
            )

    if save_prefix is not None:
        save_prefix = Path(save_prefix)
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_prefix.with_suffix(".pdf"))
        fig.savefig(save_prefix.with_suffix(".png"), dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
