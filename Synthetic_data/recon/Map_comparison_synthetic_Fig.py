"""
Paper figures: FBP/TV/JTV reconstructions across the four limited-data
scenarios (MAP_comparison: a 2x2 arrangement of framed 3x3 panels -- rows
FBP/TV/JTV, columns mu/delta/eps -- sized for the full width of a two-column
template), and the true phantom (save_true_phantom_figure, 1x3).

Every image of a channel, in every scenario and for every method, uses the
same grey scale: [0, max of the ground truth] for that channel. Values
outside it saturate. A shared scale is what makes contrast loss and noise
comparable across methods and scenarios -- per-image scaling hides both. Each
panel carries one colorbar per channel, under its last (JTV) row.

Reads obs/scenarios/<PHANTOM_NAME>/<scenario>/recon_arrays.npz, written by
run_scenarios.py's save_recon_arrays(). FBP entries are None until run on a
CUDA machine (see run_scenarios.py's module docstring) -- this script shows
a placeholder panel for any missing method/channel and otherwise proceeds
normally, so it can be re-run as-is once FBP results are available.

Usage: python paper_figure_13row.py
Output: obs/scenarios/<PHANTOM_NAME>/paper_figure_13row.pdf (+ .png)
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

BASE_DIR = Path(__file__).resolve().parent.parent
PHANTOM_NAME = "multicontrast"  # must match run_scenarios.py's PHANTOM_NAME
SCEN_DIR = BASE_DIR / "obs" / "scenarios" / PHANTOM_NAME

SCENARIOS: List[Tuple[str, str]] = [
    ("full", "(a) Full data"),
    ("sparse_angle", "(b) Sparse angles"),
    ("sparse_step", "(c) Sparse steps"),
    ("combined", "(d) Combined undersampling"),
]

METHODS = ["FBP", "TV", "JTV"]
CHANNELS: List[Tuple[str, str]] = [("mu", r"$\mu$"), ("delta", r"$\delta$"), ("eps", r"$\epsilon$")]

MULTICONTRAST_ANNOTATIONS = [
    (-0.40, 0.40, "A", 0),
    (0.42, 0.40, "B", 0),
    (-0.44, -0.30, "F", 0),
    (0.42, -0.30, "P", 0),
    (0.00, 0.06, "L", 0),
    (0.00, -0.74, "ladder", 0),
    (0.72, 0.03, "crack", 90),
]

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

def load_true_phantom() -> Dict[str, np.ndarray]:
    """
    Load the true multicontrast phantom from the 'full' scenario file.
    """
    arr = np.load(SCEN_DIR / "full" / "recon_arrays.npz")
    return {
        "mu": np.asarray(arr["mu_true"]),
        "delta": np.asarray(arr["delta_true"]),
        "eps": np.asarray(arr["eps_true"]),
    }

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

def channel_display_vmax(true_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Upper end of each channel's shared grey scale: the ground-truth maximum."""
    return {ch: max(float(np.clip(true_data[ch], 0, None).max()), 1e-12) for ch, _ in CHANNELS}

def save_true_phantom_figure(
    annotate_mu: bool = True,
    annotations=MULTICONTRAST_ANNOTATIONS,
    out_stem: str = "true_phantom_1x3",
) -> None:
    """
    Generate and save a 1x3 figure of the true phantom (mu, delta, eps),
    with one horizontal colorbar beneath each panel.

    If annotate_mu=True, compact region labels are added to the mu panel.
    """
    true_data = load_true_phantom()
    vmax_ch = channel_display_vmax(true_data)

    fig = plt.figure(figsize=(6.8, 2.8))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 0.08],
        left=0.07,
        right=0.98,
        bottom=0.14,
        top=0.88,
        wspace=0.18,
        hspace=0.10,
    )

    # Use physical-style coordinates so annotations can be placed naturally
    extent = [-1.0, 1.0, -1.0, 1.0]

    image_axes = []
    cbar_axes = []

    for c, (ch, ch_label) in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[0, c])
        cax = fig.add_subplot(gs[1, c])

        img = np.clip(true_data[ch], 0, None)
        vmax = vmax_ch[ch]

        im = ax.imshow(
            img,
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=vmax,
        )

        ax.set_title(ch_label, fontsize=14, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        # Optional annotations only on the mu panel
        if ch == "mu" and annotate_mu and annotations is not None:
            for x, y, label, rotation in annotations:
                ax.text(
                    x, y, label,
                    ha="center",
                    va="center",
                    rotation=rotation,
                    fontsize=8,
                    # fontweight="bold",
                    color="green",
                    bbox=dict(
                        facecolor="none",
                        edgecolor="none",
                        alpha=0.65,
                        pad=0.2,
                    ),
                )

        # Horizontal colorbar under each panel
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks(np.linspace(0.0, vmax, 3))
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # cbar.set_label(
        #     CHANNEL_UNITS[ch],
        #     fontsize=8,
        #     labelpad=2,
        # )
        cbar.ax.tick_params(labelsize=8, length=2, pad=1)
        cbar.outline.set_linewidth(0.6)

        image_axes.append(ax)
        cbar_axes.append(cax)

    for ext in ("pdf", "png"):
        out_path = SCEN_DIR / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)

def MAP_comparison() -> None:
    data = load_data()
    vmax_ch = channel_display_vmax(load_true_phantom())

    # Full-width figure for a two-column IEEE layout
    fig_width = 7.1
    fig_height = 7.6
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

        # Inner 4x3 grid:
        # image rows    = 0, 1, 2 (FBP, TV, JTV)
        # colorbar row  = 3       (one per channel, shared by the column)
        inner_gs = outer_gs[panel_row, panel_col].subgridspec(
            4, 3,
            height_ratios=[1, 1, 1, 0.15],
            wspace=0.06,
            hspace=0.08,
        )

        panel_axes = []
        panel_cbar_axes = []

        # --------------------------------------------------------------
        # Plot the 3x3 images on each channel's ground-truth grey scale
        # --------------------------------------------------------------
        for r, method in enumerate(METHODS):
            row_axes = []

            for c, (ch, ch_label) in enumerate(CHANNELS):

                ax = fig.add_subplot(inner_gs[r, c])

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
                else:
                    ax.imshow(
                        img,
                        cmap="gray",
                        origin="lower",
                        vmin=0.0,
                        vmax=vmax_ch[ch],
                    )

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

                row_axes.append(ax)

            panel_axes.append(row_axes)

        # --------------------------------------------------------------
        # One colorbar per channel, under the last (JTV) row
        # --------------------------------------------------------------
        for c, (ch, _) in enumerate(CHANNELS):
            slot_ax = fig.add_subplot(inner_gs[3, c])
            slot_ax.set_axis_off()
            cax = slot_ax.inset_axes([0.1, 0.85, 0.82, 0.25])

            sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax_ch[ch]), cmap="gray")
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cbar.set_ticks(np.linspace(0.0, vmax_ch[ch], 3))
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            cbar.ax.tick_params(labelsize=6, length=0, pad=1)
            cbar.outline.set_linewidth(0.5)

            panel_cbar_axes.append(cax)

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
        out_path = SCEN_DIR / f"MAP_compare_synthetic.{ext}"
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    MAP_comparison()
    save_true_phantom_figure()