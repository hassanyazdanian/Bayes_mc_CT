"""
Paper figure: True phantom vs. FBP/TV/JTV reconstructions across all four
limited-data scenarios, as a single figure sized for one column of a two-column template.


Times 12pt, one colorbar per row (4 ticks, at the right), row labels
(True/FBP/TV/JTV) at the left rotated, each scenario's 3-row block framed
with a border and a left-aligned header.

Reads obs/scenarios/<scenario>/recon_arrays.npz, written by
run_scenarios.py's save_recon_arrays(). FBP entries are None until run on a
CUDA machine (see run_scenarios.py's module docstring) -- this script shows
a placeholder panel for any missing method/channel and otherwise proceeds
normally, so it can be re-run as-is once FBP results are available.


"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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

SCEN_DIR = BASE_DIR / "obs" / "scenarios" 

SCENARIOS: List[Tuple[str, str]] = [
    ("full", "(a) Full data"),
    ("sparse_angle", "(b) Sparse angles"),
    ("sparse_step", "(c) Sparse steps"),
    ("combined", "(d) Combined undersampling"),
]

METHODS = ["FBP", "TV", "JTV"]
CHANNELS: List[Tuple[str, str]] = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]


CHANNEL_UNITS = {
    "mu": r"$\mathrm{mm}^{-1}$",
    "delta": r"$-$",              # dimensionless
    "eps": r"$\mathrm{mm}^{-1}$",
}

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


def load_data() -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    data: Dict[str, Dict[str, Optional[np.ndarray]]] = {}

    for scen_key, _ in SCENARIOS:
        arr = np.load(SCEN_DIR / scen_key / "recon_arrays.npz")

        for method in METHODS:
            row = {}

            for ch, _ in CHANNELS:
                key = f"{method}_{ch}"
                row[ch] = np.asarray(arr[key]) if key in arr else None

            data[f"{scen_key}_{method}"] = row

    return data

def image_display_vmax(img: np.ndarray, q: float = 99.5) -> float:
    """
    Robust display upper limit for one image.
    Uses a percentile instead of the absolute max so that noise/outliers
    do not destroy contrast.
    """
    vals = np.clip(img, 0, None)
    vmax = float(np.percentile(vals, q))
    return max(vmax, 1e-12)


def prior_display_vmax(data: Dict, scen_key: str, ch: str, q: float = 99.5) -> float:
    """One display range for TV and JTV within a panel and channel.

    The point of the figure is the TV-vs-JTV comparison, so the two must share
    a scale: with per-image scaling a prior that produces smaller values is
    displayed brighter, which is exactly backwards. FBP keeps its own range,
    since its dynamic range is 5-20x larger and sharing one would leave the
    Bayesian reconstructions black.
    """
    vmaxes = []
    for method in ("TV", "JTV"):
        img = data.get(f"{scen_key}_{method}", {}).get(ch)
        if img is not None:
            vmaxes.append(image_display_vmax(np.clip(img, 0, None), q=q))
    return max(vmaxes) if vmaxes else 1.0


def MAP_comparison() -> None:
    data = load_data()

    # Full-width figure for a two-column IEEE layout
    fig_width = 7.1
    fig_height = 8.2
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Outer 2x2 layout:
    # [0,0] Full data       [0,1] Sparse angle
    # [1,0] Sparse steps    [1,1] Combined
    outer_gs = fig.add_gridspec(
        2, 2,
        left=0.07,
        right=0.98,
        bottom=0.05,
        top=0.96,
        wspace=0.18,
        hspace=0.20,
    )

    # Keep axes for each scenario panel
    axes_grid = {}
    cbar_axes_grid = {}

    for p, (scen_key, scen_title) in enumerate(SCENARIOS):
        panel_row, panel_col = divmod(p, 2)

        # Inner 6x3 grid:
        # image rows    = 0, 2, 4
        # colorbar rows = 1, 3, 5
        inner_gs = outer_gs[panel_row, panel_col].subgridspec(
            6, 3,
            height_ratios=[1, 0.15, 1, 0.15, 1, 0.15],
            wspace=0.06,
            hspace=0.08,
        )

        panel_axes = []
        panel_cbar_axes = []

        # --------------------------------------------------------------
        # Plot the 3x3 images and their individual colorbars
        # --------------------------------------------------------------
        for r, method in enumerate(METHODS):
            row_axes = []

            for c, (ch, ch_label) in enumerate(CHANNELS):

                # Image axis: rows 0, 2, 4
                ax = fig.add_subplot(inner_gs[2 * r, c])

                # Colorbar slot: rows 1, 3, 5
                slot_ax = fig.add_subplot(inner_gs[2 * r + 1, c])
                slot_ax.set_axis_off()

                # Actual compact colorbar
                cax = slot_ax.inset_axes([0.15, 0.82, 0.72, 0.25])

                row_key = f"{scen_key}_{method}"
                img = data[row_key][ch]

                # Remove image-axis ticks from EVERY image
                ax.set_xticks([])
                ax.set_yticks([])

                # Image border
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.6)

                if img is None:
                    ax.text(
                        0.5, 0.5, "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="0.5",
                    )

                    vmax = 1.0

                    sm = ScalarMappable(
                        norm=Normalize(vmin=0.0, vmax=vmax),
                        cmap="gray",
                    )
                    sm.set_array([])

                else:
                    img_disp = np.clip(img, 0, None)

                    vmax = (image_display_vmax(img_disp, q=99.5) if method == "FBP"
                            else prior_display_vmax(data, scen_key, ch, q=99.5))

                    ax.imshow(
                        img_disp,
                        cmap="gray",
                        origin="lower",
                        vmin=0.0,
                        vmax=vmax,
                    )

                    sm = ScalarMappable(
                        norm=Normalize(
                            vmin=0.0,
                            vmax=vmax,
                        ),
                        cmap="gray",
                    )
                    sm.set_array([])

                # Channel titles: only above first row
                if r == 0:
                    ax.set_title(
                        ch_label,
                        fontsize=11,
                        pad=3,
                    )

                # Method labels: only on first column
                if c == 0:
                    ax.set_ylabel(
                        method,
                        fontsize=11,
                        rotation=90,
                        labelpad=8,
                    )

                # Individual colorbar for this image
                cbar = fig.colorbar(
                    sm,
                    cax=cax,
                    orientation="horizontal",
                )

                cbar.set_ticks(
                    np.linspace(0.0, vmax, 3)
                )

                cbar.ax.xaxis.set_major_formatter(
                    FormatStrFormatter("%.2f")
                )

                cbar.ax.tick_params(
                    labelsize=6,
                    length=0,
                    pad=1,
                )

                cbar.outline.set_linewidth(0.5)

                row_axes.append(ax)
                panel_cbar_axes.append(cax)

            panel_axes.append(row_axes)

        # Save all 9 image axes and 9 colorbar axes for THIS scenario
        axes_grid[scen_key] = panel_axes
        cbar_axes_grid[scen_key] = panel_cbar_axes

    # ------------------------------------------------------------------
    # Add one frame and one title per 3x3 scenario panel
    # ------------------------------------------------------------------
    fig.canvas.draw()

    for scen_key, scen_title in SCENARIOS:
        panel_axes = axes_grid[scen_key]

        # Collect all axes positions in this 3x3 panel
        image_boxes = [
            ax.get_position()
            for row in panel_axes
            for ax in row
        ]

        colorbar_boxes = [
            ax.get_position()
            for ax in cbar_axes_grid[scen_key]
        ]

        all_boxes = image_boxes + colorbar_boxes

        x0 = min(b.x0 for b in all_boxes)
        x1 = max(b.x1 for b in all_boxes)
        y0 = min(b.y0 for b in all_boxes)

        # Top is determined by image axes
        y1 = max(b.y1 for b in image_boxes)

        # Small padding around the 3x3 image block
        pad_x = 0.008
        pad_y_bottom = 0.015
        pad_y_top = 0.05   # includes space so title sits nicely above images

        # Draw outer frame
        rect = patches.Rectangle(
            (x0 - pad_x, y0 - pad_y_bottom),
            (x1 - x0) + 2 * pad_x,
            (y1 - y0) + pad_y_bottom + pad_y_top,
            transform=fig.transFigure,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
        )
        fig.add_artist(rect)

        # Add panel title, left-aligned near the top border
        fig.text(
            x0 - pad_x + 0.006,
            y1 + pad_y_top - 0.006,
            scen_title,
            ha="left",
            va="top",
            fontsize=10.0,
            fontweight="semibold",
        )


    # Temporary save / show stage
    for ext in ("pdf", "png"):
        out_path = SCEN_DIR / f"MAP_compare_real.{ext}"
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    MAP_comparison()