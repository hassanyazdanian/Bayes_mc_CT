"""
Paper figure: posterior mean / posterior std / absolute posterior-mean error
for synthetic multi-contrast CT UQ results, comparing TV vs JTV across four
acquisition scenarios as one full-page-width figure.

Layout
------
Rows are grouped by scenario:
    (a) Full data
    (b) Sparse angles
    (c) Sparse steps
    (d) Combined undersampling

Within each scenario there are two image rows:
    TV
    JTV

Columns are grouped by channel and quantity:
    mu    : Mean | Std | z
    delta : Mean | Std | z
    eps   : Mean | Std | z

where z = |posterior mean - truth| / posterior std is the error measured in
units of the posterior's own standard deviation: z ~ 1 where the posterior is
calibrated, z >> 1 where it is overconfident. It replaces the absolute error
map, which is recoverable as std x z, because the two carry the same
information while z is the one that cannot be read off the other panels.

Mean and std use one display range per channel, shared by all scenarios and
priors so that panels are comparable; z uses a fixed logarithmic diverging
scale centred on 1. Each image has its own compact horizontal colorbar
underneath, following the style of fig_map_comparison.py.

Expected directory structure
----------------------------
Assuming this script lives in the project root/code directory similarly to
fig_map_comparison.py:

    BASE_DIR/
        obs/scenarios/<PHANTOM_NAME>/full/recon_arrays.npz
        uq/obs/<PHANTOM_NAME>/<scenario>/tv/posterior_mean_std.npz
        uq/obs/<PHANTOM_NAME>/<scenario>/jtv/posterior_mean_std.npz

Usage
-----
    python fig_uq_comparison.py
or
    python fig_uq_comparison.py --phantom multicontrast

Output
------
    uq/obs/<PHANTOM_NAME>/UQ_compare_synthetic.pdf
    uq/obs/<PHANTOM_NAME>/UQ_compare_synthetic.png
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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter

PHANTOM_NAME = "multicontrast"

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
    ("z", r"$z$"),  # defined in the caption: |posterior mean - truth| / posterior std
]

CMAPS = {
    "mean": "gray",
    "std": "magma",
    "z": "coolwarm",
}

# z is shown on a logarithmic diverging scale centred on 1 (the calibrated
# value): blue = the posterior is wider than its error, red = overconfident.
# The absolute error map is recoverable as std x z.
Z_VMIN, Z_VMAX = 0.25, 4.0
SUPPORT_R = 0.98

# 9 image columns + 2 narrow spacer columns between channel groups.
COL_WIDTH_RATIOS = [1, 1, 1, 0.18, 1, 1, 1, 0.18, 1, 1, 1]
IMAGE_COLS = [0, 1, 2, 4, 5, 6, 8, 9, 10]

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


def robust_vmax(arrays: List[np.ndarray], q: float = 99.5) -> float:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        vals.append(a.ravel())
    if not vals:
        return 1.0
    vals = np.concatenate(vals)
    vals = np.clip(vals, 0, None)
    vmax = float(np.percentile(vals, q))
    return max(vmax, 1e-12)


def load_true_phantom(base_dir: Path, phantom_name: str) -> Dict[str, np.ndarray]:
    recon_path = base_dir / "obs" / "scenarios" / phantom_name / "full" / "recon_arrays.npz"
    if not recon_path.exists():
        raise FileNotFoundError(
            f"Could not find true phantom file:\n  {recon_path}\n"
            "Expected keys: mu_true, delta_true, eps_true"
        )

    arr = np.load(recon_path)
    return {
        "mu": np.asarray(arr["mu_true"]),
        "delta": np.asarray(arr["delta_true"]),
        "eps": np.asarray(arr["eps_true"]),
    }


def load_posterior_stats(base_dir: Path, phantom_name: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Load posterior mean/std from saved .npz files and compute abs. error."""
    uq_root = base_dir / "uq" / "obs" / phantom_name
    true_data = load_true_phantom(base_dir, phantom_name)

    data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    missing = []

    for scen_key, _ in SCENARIOS:
        data[scen_key] = {}
        for prior_key, _ in PRIORS:
            fpath = uq_root / scen_key / prior_key / "posterior_mean_std.npz"
            if not fpath.exists():
                missing.append(str(fpath))
                continue

            arr = np.load(fpath)
            data[scen_key][prior_key] = {}

            for ch, _ in CHANNELS:
                mean_key = f"{ch}_mean"
                std_key = f"{ch}_std"
                if mean_key not in arr or std_key not in arr:
                    raise KeyError(
                        f"Missing required keys in {fpath}: expected {mean_key} and {std_key}"
                    )

                mean_img = np.asarray(arr[mean_key])
                std_img = np.asarray(arr[std_key])
                err_img = np.abs(true_data[ch] - mean_img)

                # z = error in units of the posterior's own standard deviation,
                # undefined outside the reconstruction support.
                support = support_mask(mean_img.shape[0])
                z_img = np.full_like(err_img, np.nan, dtype=float)
                z_img[support] = err_img[support] / np.maximum(std_img[support], 1e-30)

                data[scen_key][prior_key][f"{ch}_mean"] = mean_img
                data[scen_key][prior_key][f"{ch}_std"] = std_img
                data[scen_key][prior_key][f"{ch}_err"] = err_img
                data[scen_key][prior_key][f"{ch}_z"] = z_img

    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(
            "Some posterior_mean_std.npz files are missing:\n"
            f"{msg}"
        )

    return data


def support_mask(n: int) -> np.ndarray:
    """Circular reconstruction support, as used by the sampler."""
    c = np.linspace(-1.0, 1.0, n)
    Y, X = np.meshgrid(c, c, indexing="ij")
    return (X ** 2 + Y ** 2) <= SUPPORT_R ** 2


def channel_ranges(data: Dict, true_data: Dict[str, np.ndarray], q: float = 99.5) -> Dict[str, Dict[str, float]]:
    """One display range per channel and quantity, shared by all scenarios and
    priors, so that panels are comparable: the mean uses the ground-truth
    maximum, the standard deviation the q-th percentile over every panel."""
    ranges: Dict[str, Dict[str, float]] = {}
    for ch, _ in CHANNELS:
        stds = [data[s][p][f"{ch}_std"] for s, _ in SCENARIOS for p, _ in PRIORS if p in data[s]]
        ranges[ch] = {
            "mean": max(float(np.clip(true_data[ch], 0, None).max()), 0.01),
            "std": robust_vmax(stds, q=q),
        }
    return ranges

def make_uq_comparison_figure(
    base_dir: Path,
    phantom_name: str = PHANTOM_NAME,
    out_stem: str = "UQ_compare_synthetic",
    q: float = 99.5,
) -> None:
    true_data = load_true_phantom(base_dir, phantom_name)
    data = load_posterior_stats(base_dir, phantom_name)
    ranges = channel_ranges(data, true_data, q=99.9)

    out_dir = base_dir / "uq" / "obs" / phantom_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_width = 7.1
    fig_height = 9.2
    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_gs = fig.add_gridspec(
        4, 1,
        left=0.07,
        right=0.985,
        bottom=0.045,
        top=0.94,
        hspace=0.025,
    )

    panel_axes: Dict[str, List[List[plt.Axes]]] = {}
    cbar_axes_grid: Dict[str, List[plt.Axes]] = {}
    # formatter = tick_formatter()
    colorbar_count = 0
    print("Mode: one colorbar row at the foot of the figure; scales shared by all panels")

    for s, (scen_key, scen_title) in enumerate(SCENARIOS):
        # For each scenario:
        # rows 0 and 2 -> image rows (TV, JTV)
        # rows 1 and 3 -> spacers; the last panel's row 3 carries the only
        # colorbars, since every panel shares the same per-channel scales.
        last_panel = s == len(SCENARIOS) - 1
        inner_gs = outer_gs[s, 0].subgridspec(
            4, len(COL_WIDTH_RATIOS),
            height_ratios=[0.8, 0.04, 0.8, 0.1 if last_panel else 0.04],
            width_ratios=COL_WIDTH_RATIOS,
            wspace=0.04,
            hspace=0.0,
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

                # Colorbars only under the last panel's JTV row: every panel
                # shares the same per-channel scales.
                draw_cbar = last_panel and r == len(PRIORS) - 1
                cax = slot_ax.inset_axes([0.15, 1.5, 0.62, 0.28]) if draw_cbar else None

                group_idx = image_col_counter // 3
                within_group = image_col_counter % 3
                ch, _ = CHANNELS[group_idx]
                qty, _ = QUANTITIES[within_group]

                img = data[scen_key][prior_key][f"{ch}_{qty}"]
                cmap = plt.get_cmap(CMAPS[qty]).copy()
                cmap.set_bad("black")
                if qty == "z":
                    norm = LogNorm(vmin=Z_VMIN, vmax=Z_VMAX)
                    disp = np.clip(img, Z_VMIN, Z_VMAX)
                else:
                    norm = Normalize(vmin=0.0, vmax=ranges[ch][qty])
                    disp = np.clip(img, 0, None) if qty == "mean" else img

                ax.imshow(
                    disp,
                    cmap=cmap,
                    origin="lower",
                    norm=norm,
                )

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.6)

                if col == IMAGE_COLS[0]:
                    ax.set_ylabel(prior_label, fontsize=11, rotation=90, labelpad=8)

                if draw_cbar:
                    sm = ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
                    if qty == "z":
                        cbar.ax.minorticks_off()
                        cbar.set_ticks([Z_VMIN, 1.0, Z_VMAX])
                        cbar.ax.set_xticklabels(["0.25", "1", "4"])
                    else:
                        cbar.set_ticks([0.0, ranges[ch][qty]])
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

    # Scenario frames and titles
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
        pad_y_top = 0.0250

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

    # Shared top labels, positioned from first scenario / first row
    top_row_axes = panel_axes[SCENARIOS[0][0]][0]
    top_boxes = [ax.get_position() for ax in top_row_axes]

    y_col = max(b.y1 for b in top_boxes) + 0.030
    for i, box in enumerate(top_boxes):
        _, qty_label = QUANTITIES[i % 3]
        fig.text(
            0.5 * (box.x0 + box.x1),
            y_col,
            qty_label,
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    # y_grp = y_col + 0.020
    # for g, (_, ch_label) in enumerate(CHANNELS):
    #     group_boxes = top_boxes[3 * g: 3 * g + 3]
    #     x_center = 0.5 * (group_boxes[0].x0 + group_boxes[-1].x1)
    #     fig.text(
    #         x_center,
    #         y_grp,
    #         ch_label,
    #         ha="center",
    #         va="bottom",
    #         fontsize=12.0,
    #         fontweight="semibold",
    #     )

    
   
    # --------------------------------------------------------------
    # Three-column framed header for mu, delta, epsilon
    # --------------------------------------------------------------
    
    # Left/right limits of the entire 9-column image block
    x_left  = top_boxes[0].x0
    x_right = top_boxes[-1].x1
    
    # Vertical position and height of the header box
    header_y = 0.97
    header_h = 0.018
    
    # Boundaries of the three channel groups
    mu_left     = top_boxes[0].x0
    mu_right    = top_boxes[2].x1
    
    delta_left  = top_boxes[3].x0
    delta_right = top_boxes[5].x1
    
    eps_left    = top_boxes[6].x0
    eps_right   = top_boxes[8].x1
    
    
    # Draw one box for each channel group
    channel_boxes = [
        (mu_left,    mu_right,    r"$\mu$"),
        (delta_left, delta_right, r"$\delta$"),
        (eps_left,   eps_right,   r"$\epsilon$"),
    ]
    
    for x0, x1, label in channel_boxes:
    
        rect = patches.Rectangle(
            (x0, header_y),
            x1 - x0,
            header_h,
            transform=fig.transFigure,
            # fill=False,
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
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out_path}")
    
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phantom", type=str, default=PHANTOM_NAME)
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help=(
            "Project base directory. If omitted, uses two levels above this file, "
            "matching the convention in your existing plotting scripts."
        ),
    )
    parser.add_argument("--out_stem", type=str, default="UQ_compare_synthetic")
    parser.add_argument(
        "--quantile", type=float, default=99.5,
        help="Percentile used for robust shared display ranges."
    )
    args = parser.parse_args()

    if args.base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(args.base_dir).resolve()

    make_uq_comparison_figure(
        base_dir=base_dir,
        phantom_name=args.phantom,
        out_stem=args.out_stem,
        q=args.quantile,
    )


if __name__ == "__main__":
    main()
