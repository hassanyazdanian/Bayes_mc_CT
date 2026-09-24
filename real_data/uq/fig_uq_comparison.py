"""
Paper figure: posterior mean / posterior std for REAL multi-contrast CT data,
comparing TV vs JTV across four acquisition scenarios.

Real data have no ground truth, so the synthetic |posterior mean - truth|
(error) column is intentionally omitted.

Layout
------
Rows are grouped by scenario:
    (a) Full data
    (b) Sparse angles
    (c) Sparse steps
    (d) Combined undersampling

Within each scenario there are two rows:
    TV
    JTV

Columns are grouped by channel and quantity:
    mu    : Mean | Std
    delta : Mean | Std
    eps   : Mean | Std

Each image has its own compact horizontal colorbar underneath. Each colorbar
has two ticks: 0 and the maximum value of that individual image.

Expected real-data tree (default)
---------------------------------
    <project>/real_data/obs/uq/
        full/
            tv/posterior_mean_std.npz
            jtv/posterior_mean_std.npz
        sparse_angle/
            tv/posterior_mean_std.npz
            jtv/posterior_mean_std.npz
        sparse_step/
            tv/posterior_mean_std.npz
            jtv/posterior_mean_std.npz
        combined/
            tv/posterior_mean_std.npz
            jtv/posterior_mean_std.npz

The script also searches recursively inside each scenario/prior if the .npz
file is one level deeper.

Usage
-----
    python fig_uq_comparison.py

or, if automatic path detection does not match your project layout:

    python fig_uq_comparison.py --uq_root /path/to/real_data/obs/uq

Output
------
    <uq_root>/UQ_compare_real_data.pdf
    <uq_root>/UQ_compare_real_data.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter


import sys
BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
for _p in (COMMON_DIR, BASE_DIR, BASE_DIR / "recon", BASE_DIR / "uq"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCENARIOS: List[Tuple[str, str]] = [
    ("full", "(a) Full data"),
    ("sparse_angle", "(b) Sparse angles"),
    ("sparse_step", "(c) Sparse steps"),
    ("combined", "(d) Combined undersampling"),
]

PRIORS: List[Tuple[str, str]] = [
    ("tv", "TV"),
    ("jtv", "JTV"),
]

CHANNELS: List[Tuple[str, str]] = [
    ("mu", r"$\mu$"),
    ("delta", r"$\delta$"),
    ("eps", r"$\epsilon$"),
]

QUANTITIES: List[Tuple[str, str]] = [
    ("mean", "Mean"),
    ("std", "Std"),
]

CMAPS = {
    "mean": "gray",
    "std": "magma",
}

# 6 image columns + 2 narrow spacers between channel groups:
# mu(mean,std) | gap | delta(mean,std) | gap | eps(mean,std)
COL_WIDTH_RATIOS = [
    0.52, 0.52,
    0.15,
    0.52, 0.52,
    0.15,
    0.52, 0.52,
]
IMAGE_COLS = [0, 1, 3, 4, 6, 7]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def find_default_uq_root() -> Path:
    """Try common placements of real_data/obs/uq relative to this script."""
    here = BASE_DIR
    candidates = [
        here / "real_data" / "obs" / "uq",
        here.parent / "real_data" / "obs" / "uq",
        here.parent.parent / "real_data" / "obs" / "uq",
        here / "obs" / "uq",                 # if script itself is in real_data/
        here.parent / "obs" / "uq",          # if script is in real_data/<subdir>/
    ]
    for path in candidates:
        if path.exists():
            return path

    # Return the most likely path so any error message is informative.
    return here.parent / "real_data" / "obs" / "uq"


def resolve_stats_file(uq_root: Path, scenario: str, prior: str) -> Path:
    """Find posterior_mean_std.npz for one scenario/prior."""
    direct = uq_root / scenario / prior / "posterior_mean_std.npz"
    if direct.exists():
        return direct

    scenario_dir = uq_root / scenario
    if scenario_dir.exists():
        # Allow one or more extra nesting levels while requiring the prior name
        # somewhere in the path.
        matches = [
            p for p in scenario_dir.rglob("posterior_mean_std.npz")
            if prior.lower() in {part.lower() for part in p.parts}
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple posterior files found for scenario={scenario!r}, "
                f"prior={prior!r}:\n  " + "\n  ".join(str(p) for p in matches)
            )

    raise FileNotFoundError(
        f"Could not find posterior statistics for scenario={scenario!r}, "
        f"prior={prior!r}.\nExpected first at:\n  {direct}"
    )


def load_posterior_stats(uq_root: Path) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Load posterior mean/std maps for all four scenarios and both priors."""
    data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for scen_key, _ in SCENARIOS:
        data[scen_key] = {}

        for prior_key, _ in PRIORS:
            fpath = resolve_stats_file(uq_root, scen_key, prior_key)
            arr = np.load(fpath)

            print(f"Loaded {scen_key:12s} / {prior_key:3s}: {fpath}")
            data[scen_key][prior_key] = {}

            for ch, _ in CHANNELS:
                mean_key = f"{ch}_mean"
                std_key = f"{ch}_std"

                if mean_key not in arr or std_key not in arr:
                    raise KeyError(
                        f"Missing keys in {fpath}. Expected {mean_key!r} and "
                        f"{std_key!r}. Available keys: {list(arr.keys())}"
                    )

                data[scen_key][prior_key][mean_key] = np.asarray(arr[mean_key])
                data[scen_key][prior_key][std_key] = np.asarray(arr[std_key])

    return data


def image_vmax(img: np.ndarray, quantity: str) -> float:
    """Maximum display value for one individual image."""
    arr = np.asarray(img, dtype=float)
    if quantity == "mean":
        arr = np.clip(arr, 0, None)

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.01

    return float(arr.max())


def channel_quantity_ranges(data: Dict, q: float = 99.5) -> Dict[str, Dict[str, float]]:
    """One display range per channel and quantity, shared by every scenario and
    both priors.

    Two things depend on this. The TV-vs-JTV comparison is only fair on a common
    scale -- with per-image ranges the prior with the smaller standard deviation
    is drawn brighter, i.e. backwards. And the growth of the posterior standard
    deviation under undersampling is only visible if the panels share a scale.
    A high percentile rather than the maximum keeps a few outlying pixels from
    setting the range.
    """
    ranges: Dict[str, Dict[str, float]] = {}
    for ch, _ in CHANNELS:
        ranges[ch] = {}
        for qty, _ in QUANTITIES:
            vals = []
            for scen_key, _ in SCENARIOS:
                for prior_key, _ in PRIORS:
                    img = data.get(scen_key, {}).get(prior_key, {}).get(f"{ch}_{qty}")
                    if img is None:
                        continue
                    arr = np.asarray(img, dtype=float)
                    arr = np.clip(arr, 0, None) if qty == "mean" else arr
                    vals.append(arr[np.isfinite(arr)].ravel())
            ranges[ch][qty] = max(float(np.percentile(np.concatenate(vals), q)), 1e-12) if vals else 1.0
    return ranges


def make_uq_comparison_figure(
    uq_root: Path,
    out_stem: str = "UQ_compare_real_data",
) -> None:
    data = load_posterior_stats(uq_root)
    ranges = channel_quantity_ranges(data)
    uq_root.mkdir(parents=True, exist_ok=True)

    fig_width = 7.1
    fig_height = 10.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Four scenario panels. hspace controls separation BETWEEN scenarios.
    outer_gs = fig.add_gridspec(
        4, 1,
        left=0.07,
        right=0.985,
        bottom=0.035,
        top=0.878,
        hspace=0.2,
    )

    panel_axes: Dict[str, List[List[plt.Axes]]] = {}
    cbar_axes_grid: Dict[str, List[plt.Axes]] = {}
    colorbar_count = 0

    for s, (scen_key, scen_title) in enumerate(SCENARIOS):
        # image row, spacer, image row, spacer. Only the last panel's spacer
        # carries colorbars: the scales are shared by every panel, so repeating
        # them eight times is clutter (and it collided with the panel titles).
        last_panel = s == len(SCENARIOS) - 1
        inner_gs = outer_gs[s, 0].subgridspec(
            4, len(COL_WIDTH_RATIOS),
            height_ratios=[0.85, 0.02, 0.85, 0.10 if last_panel else 0.001],
            width_ratios=COL_WIDTH_RATIOS,
            wspace=0.1,
            hspace=0.001,
        )

        rows_for_this_panel: List[List[plt.Axes]] = []
        panel_cbar_axes: List[plt.Axes] = []

        for r, (prior_key, prior_label) in enumerate(PRIORS):
            row_axes: List[plt.Axes] = []
            image_col_counter = 0
            image_row = 2 * r
            cbar_row = 2 * r + 1

            for col in range(len(COL_WIDTH_RATIOS)):
                ax = fig.add_subplot(inner_gs[image_row, col])
                slot_ax = fig.add_subplot(inner_gs[cbar_row, col])
                slot_ax.set_axis_off()

                ax.set_xticks([])
                ax.set_yticks([])

                if col not in IMAGE_COLS:
                    ax.axis("off")
                    continue

                draw_cbar = last_panel and r == len(PRIORS) - 1
                cax = slot_ax.inset_axes([0.15, 0.55, 0.62, 0.22]) if draw_cbar else None

                group_idx = image_col_counter // 2
                within_group = image_col_counter % 2
                ch, _ = CHANNELS[group_idx]
                qty, _ = QUANTITIES[within_group]

                img = data[scen_key][prior_key][f"{ch}_{qty}"]
                vmax = ranges[ch][qty]
                cmap = CMAPS[qty]
                disp = np.clip(img, 0, None) if qty == "mean" else img

                ax.imshow(
                    disp,
                    cmap=cmap,
                    origin="lower",
                    vmin=0.0,
                    vmax=vmax,
                )

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.6)

                if col == IMAGE_COLS[0]:
                    ax.set_ylabel(
                        prior_label,
                        fontsize=11,
                        rotation=90,
                        labelpad=8,
                    )

                if draw_cbar:
                    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap=cmap)
                    sm.set_array([])

                    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
                    cbar.set_ticks([0.0, vmax])
                    if ch == "delta":
                        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
                    else:
                        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                    cbar.ax.tick_params(labelsize=6, length=0, pad=1)
                    cbar.outline.set_linewidth(0.5)

                    colorbar_count += 1
                    panel_cbar_axes.append(cax)

                row_axes.append(ax)
                image_col_counter += 1

            rows_for_this_panel.append(row_axes)

        panel_axes[scen_key] = rows_for_this_panel
        cbar_axes_grid[scen_key] = panel_cbar_axes

    fig.canvas.draw()

    # ------------------------------------------------------------------
    # Scenario frames and titles
    # ------------------------------------------------------------------
    for scen_key, scen_title in SCENARIOS:
        rows = panel_axes[scen_key]
        image_boxes = [ax.get_position() for row in rows for ax in row]
        colorbar_boxes = [ax.get_position() for ax in cbar_axes_grid[scen_key]]
        all_boxes = image_boxes + colorbar_boxes

        x0 = min(b.x0 for b in all_boxes)
        x1 = max(b.x1 for b in all_boxes)
        y0 = min(b.y0 for b in all_boxes)
        y1 = max(b.y1 for b in image_boxes)

        pad_x = 0.008
        pad_y_bottom = 0.010
        pad_y_top = 0.025

        rect = patches.Rectangle(
            (x0 - pad_x, y0 - pad_y_bottom),
            (x1 - x0) + 2 * pad_x,
            (y1 - y0) + pad_y_bottom + pad_y_top,
            transform=fig.transFigure,
            fill=False,
            edgecolor="black",
            linewidth=0.9,
            zorder=10,
        )
        fig.add_artist(rect)

        fig.text(
            x0 - pad_x + 0.006,
            y1 + pad_y_top - 0.008,
            scen_title,
            ha="left",
            va="top",
            fontsize=10.0,
            fontweight="semibold",
        )

    # ------------------------------------------------------------------
    # Global Mean / Std labels
    # ------------------------------------------------------------------
    top_row_axes = panel_axes[SCENARIOS[0][0]][0]
    top_boxes = [ax.get_position() for ax in top_row_axes]

    header_y = 0.92
    header_h = 0.015

    # Just under the channel header boxes, clear of the panel titles.
    y_col = header_y - 0.014
    for i, box in enumerate(top_boxes):
        _, qty_label = QUANTITIES[i % 2]
        fig.text(
            0.5 * (box.x0 + box.x1),
            y_col,
            qty_label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            zorder=25,
        )

    # ------------------------------------------------------------------
    # Shaded channel-header boxes: mu / delta / epsilon
    # ------------------------------------------------------------------
    mu_left, mu_right = top_boxes[0].x0, top_boxes[1].x1
    delta_left, delta_right = top_boxes[2].x0, top_boxes[3].x1
    eps_left, eps_right = top_boxes[4].x0, top_boxes[5].x1

    channel_boxes = [
        (mu_left, mu_right, r"$\mu$"),
        (delta_left, delta_right, r"$\delta$"),
        (eps_left, eps_right, r"$\epsilon$"),
    ]

    for x0, x1, label in channel_boxes:
        rect = patches.Rectangle(
            (x0, header_y),
            x1 - x0,
            header_h,
            transform=fig.transFigure,
            facecolor="0.92",
            edgecolor="black",
            linewidth=0.9,
            zorder=20,
        )
        fig.add_artist(rect)

        fig.text(
            0.5 * (x0 + x1),
            header_y + 0.5 * header_h,
            label,
            ha="center",
            va="center",
            fontsize=12,
            zorder=21,
        )

    expected = len(CHANNELS) * len(QUANTITIES)  # one shared set at the foot
    print(f"Colorbars created: {colorbar_count} (expected {expected})")

    for ext in ("pdf", "png"):
        out_path = uq_root / f"{out_stem}.{ext}"
        fig.savefig(
            out_path,
            dpi=300 if ext == "png" else None,
            bbox_inches="tight",
        )
        print(f"Saved: {out_path}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uq_root",
        type=str,
        default=None,
        help="Path to real_data/obs/uq. If omitted, the script tries to find it.",
    )
    parser.add_argument(
        "--out_stem",
        type=str,
        default="UQ_compare_real_data",
    )
    args = parser.parse_args()

    uq_root = Path(args.uq_root).resolve() if args.uq_root else find_default_uq_root()

    if not uq_root.exists():
        raise FileNotFoundError(
            f"Real-data UQ root not found:\n  {uq_root}\n"
            "Pass it explicitly, e.g.\n"
            "  python fig_uq_comparison.py --uq_root /path/to/real_data/obs/uq"
        )

    print(f"Using real-data UQ root: {uq_root}")
    make_uq_comparison_figure(
        uq_root=uq_root,
        out_stem=args.out_stem,
    )


if __name__ == "__main__":
    main()
