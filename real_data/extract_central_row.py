"""Extract the slice data used in the paper from the raw Talbot-Lau scan.

Source: PC_PMMA_Phantom/raw.zip of the open dataset
    H. Mäkinen, "Data for paper: Bayesian Joint Reconstruction and Uncertainty
    Quantification for Limited-Data Multi-Contrast X-ray CT", University of Helsinki,
    2026, https://doi.org/10.23729/fd-769dae34-0245-386f-a856-92e1adc60f18 (CC BY 4.0),
    repackaged from scan pmma_stick_phantom/scan22 of
    https://doi.org/10.23729/60132ae3-1ce3-41eb-935f-d0721ad464aa.

The unpacked folder holds, for each of the 361 angles (0-360 deg in 1-deg steps) and 10 phase
steps, an object image imAAAA_SS.smv and a reference (flat-field) image refAAAA_SS.smv,
each an SMV file with a 512-byte header and 864 x 1536 uint16 pixels. Its scan.csv lists,
in the pix_offset column, the random sample offset in pixels applied at each angle to
reduce ring artefacts.

The slice is the transaxial detector line through the detector centre, which is image
column 768 (the rotation axis runs along the image rows). Rows 10-852 (843 pixels) are
kept, shifted at each angle by pix_offset so that the sample stays fixed in the window;
object and reference images use the same window.

Usage:
    python extract_central_row.py /path/to/raw [output_dir]

Writes I_meas_central.npy and I_ref_central.npy (361 x 10 x 843, float32), angles.npy
(angle indices 1-361) and steps.npy (step indices 1-10) to output_dir, by default obs/
next to this script. Needs numpy only.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

N_ANGLES, N_STEPS = 361, 10
COLUMN = 768                    # detector centre along the rotation axis
ROW_START, ROW_END = 10, 853    # 843 pixels; leaves room for offsets of up to 10 pixels


def read_smv(path):
    """Read an SMV image, taking header size, byte order and shape from its header."""
    with open(path, "rb") as f:
        head = f.read(512).decode("ascii", "ignore")

        def key(k):
            return re.search(rf"{k}=([^;]*);", head).group(1).strip()

        order = "<" if key("BYTE_ORDER").startswith("little") else ">"
        n_col, n_row = int(key("SIZE1")), int(key("SIZE2"))
        f.seek(int(key("HEADER_BYTES")))
        img = np.fromfile(f, dtype=order + "u2", count=n_row * n_col)
    return img.reshape(n_row, n_col)


def read_offsets(raw_dir):
    with open(Path(raw_dir) / "scan.csv", newline="") as f:
        return np.array([int(row["pix_offset"]) for row in csv.DictReader(f)])


def extract(raw_dir, prefix, offsets, angles=range(1, N_ANGLES + 1)):
    """Stack (angle, step, row) of the central-column window for im or ref images."""
    n = ROW_END - ROW_START
    out = np.empty((len(angles), N_STEPS, n), dtype=np.float32)
    for i, a in enumerate(angles):
        r0 = ROW_START + offsets[a - 1]
        for s in range(1, N_STEPS + 1):
            img = read_smv(Path(raw_dir) / f"{prefix}{a:04d}_{s:02d}.smv")
            out[i, s - 1] = img[r0:r0 + n, COLUMN]
    return out


def main():
    raw_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = read_offsets(raw_dir)
    assert len(offsets) == N_ANGLES and np.abs(offsets).max() <= ROW_START
    np.save(out_dir / "I_meas_central.npy", extract(raw_dir, "im", offsets))
    np.save(out_dir / "I_ref_central.npy", extract(raw_dir, "ref", offsets))
    np.save(out_dir / "angles.npy", np.arange(1, N_ANGLES + 1))
    np.save(out_dir / "steps.npy", np.arange(1, N_STEPS + 1))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
