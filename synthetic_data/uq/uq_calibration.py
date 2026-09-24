"""
Is the posterior uncertainty calibrated? Scores the posterior mean and
pixel-wise standard deviation against the known ground truth, for every
scenario and prior.

The posterior std map and the error map |mean - truth| peak in the same places
(object and inclusion boundaries), but the figure's colorbars differ by nearly
an order of magnitude there, so the question is quantitative: by how much does
the posterior under- or over-state its own error?

Reported per channel, over the reconstruction support:

  err/std        RMS(mean - truth) / RMS(std). 1 = the posterior's spread
                 matches its error on average; > 1 = overconfident.
  z within 1,2,3 fraction of pixels with |mean - truth| <= k*std. For a
                 calibrated Gaussian posterior these are 68.3 / 95.4 / 99.7 %.
  RE mean/MAP    relative error of the posterior mean and of the MAP, for
                 whether sampling also improves the point estimate.

Edges and flat regions are reported separately: TV's uncertainty is
concentrated at boundaries, and pooling the two hides which one is
miscalibrated. A pixel is an "edge" pixel if the true image's gradient
magnitude there exceeds 5% of its maximum, dilated by one pixel.

Usage:
    python uq_calibration.py                       # all scenarios, both priors
    python uq_calibration.py --scenarios combined  # subset
Outputs: obs/<phantom>/uq_calibration.csv (+ printed table)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
RECON_DIR = BASE_DIR / "recon"
for p in (BASE_DIR, COMMON_DIR, RECON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scenario_utils import make_reconstruction_mask  # noqa: E402

CHANNELS = ["mu", "delta", "eps"]
SCENARIOS = ["full", "sparse_angle", "sparse_step", "combined"]
PRIORS = ["tv", "jtv"]


def edge_mask(truth: np.ndarray, rel_thresh: float = 0.05) -> np.ndarray:
    """Pixels on a true edge: |grad| above rel_thresh of its max, dilated by 1."""
    gy, gx = np.gradient(truth.astype(np.float64))
    g = np.hypot(gx, gy)
    m = g > rel_thresh * g.max()
    out = m.copy()
    for ax in (0, 1):
        for s in (1, -1):
            out |= np.roll(m, s, axis=ax)
    return out


def stats_for(mean: np.ndarray, std: np.ndarray, truth: np.ndarray, sel: np.ndarray) -> Dict[str, float]:
    err = np.abs(mean - truth)[sel]
    s = std[sel]
    z = err / np.maximum(s, 1e-30)
    return {
        "rms_err": float(np.sqrt(np.mean(err ** 2))),
        "rms_std": float(np.sqrt(np.mean(s ** 2))),
        "err_over_std": float(np.sqrt(np.mean(err ** 2)) / max(np.sqrt(np.mean(s ** 2)), 1e-30)),
        "within_1s": float(np.mean(z <= 1.0)),
        "within_2s": float(np.mean(z <= 2.0)),
        "within_3s": float(np.mean(z <= 3.0)),
    }


def rel_err(x: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(x - truth) / (np.linalg.norm(truth) + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", default="multicontrast")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--priors", nargs="+", default=PRIORS)
    ap.add_argument("--support_R", type=float, default=0.98)
    ap.add_argument("--edge_thresh", type=float, default=0.05)
    args = ap.parse_args()

    uq_root = BASE_DIR / "uq" / "obs" / args.phantom
    scen_root = BASE_DIR / "obs" / "scenarios" / args.phantom

    rows = []
    print(f"{'scenario':13s} {'prior':4s} {'ch':5s} {'region':5s} "
          f"{'RMS err':>9s} {'RMS std':>9s} {'err/std':>8s} {'<1s':>6s} {'<2s':>6s} {'<3s':>6s} "
          f"{'RE mean':>8s} {'RE MAP':>8s}")
    for scen in args.scenarios:
        arr = np.load(scen_root / scen / "recon_arrays.npz")
        truth = {ch: np.asarray(arr[f"{ch}_true"]) for ch in CHANNELS}
        support = make_reconstruction_mask(truth["mu"].shape[0], R=args.support_R).numpy()
        for prior in args.priors:
            f = uq_root / scen / prior / "posterior_mean_std.npz"
            if not f.exists():
                print(f"  missing: {f}")
                continue
            post = np.load(f)
            map_key = {"tv": "TV", "jtv": "JTV"}[prior]
            for ch in CHANNELS:
                mean, std = np.asarray(post[f"{ch}_mean"]), np.asarray(post[f"{ch}_std"])
                edges = edge_mask(truth[ch], args.edge_thresh) & support
                regions = {"all": support, "edge": edges, "flat": support & ~edges}
                re_mean = rel_err(mean, truth[ch])
                re_map = rel_err(np.asarray(arr[f"{map_key}_{ch}"]), truth[ch]) \
                    if f"{map_key}_{ch}" in arr else float("nan")
                for region, sel in regions.items():
                    st = stats_for(mean, std, truth[ch], sel)
                    rows.append({"scenario": scen, "prior": prior, "channel": ch, "region": region,
                                 **st, "n_pix": int(sel.sum()), "RE_mean": re_mean, "RE_MAP": re_map})
                    print(f"{scen:13s} {prior:4s} {ch:5s} {region:5s} "
                          f"{st['rms_err']:9.5f} {st['rms_std']:9.5f} {st['err_over_std']:8.2f} "
                          f"{100 * st['within_1s']:5.1f}% {100 * st['within_2s']:5.1f}% {100 * st['within_3s']:5.1f}% "
                          f"{re_mean:8.4f} {re_map:8.4f}")

    if rows:
        keys = ["scenario", "prior", "channel", "region", "n_pix", "rms_err", "rms_std", "err_over_std",
                "within_1s", "within_2s", "within_3s", "RE_mean", "RE_MAP"]
        out = uq_root / "uq_calibration.csv"
        with open(out, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
        print(f"\nsaved: {out}")
        print("Calibrated posterior: err/std ~ 1 and 68/95/99.7% within 1/2/3 std.")


if __name__ == "__main__":
    main()
