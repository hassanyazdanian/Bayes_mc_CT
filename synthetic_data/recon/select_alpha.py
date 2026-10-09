"""
Select the per-channel TV weights from the alpha_beta sweeps.

sweep_tv_beta.py --stage alpha_beta reports the RE-optimal alpha for ONE
acquisition condition. The reported weights are not any one of those: they
minimize RE averaged over all four conditions, because a weight tuned on full
data over-regularizes the undersampled ones (the full-scan optimum
alpha_delta=1778 doubles delta's RE on combined). Run the sweep for every
scenario, then this.

Usage:
    python select_alpha.py                    # all four scenarios, shared alpha grid
    python select_alpha.py --metric SSIM      # maximize mean SSIM instead
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

OUT_DIR = Path(__file__).resolve().parent.parent / "obs" / "tuning" / "beta_sweep"
CHANNELS = ["mu", "delta", "eps"]
SCENARIOS = ["full", "sparse_angle", "sparse_step", "combined"]


def grid_path(scenario: str) -> Path:
    tag = "" if scenario == "full" else f"_{scenario}"
    return OUT_DIR / f"alpha_beta_grid{tag}.csv"


def load(scenario: str, metric: str) -> Dict[str, Dict[float, float]]:
    """{channel: {alpha: metric}} for one scenario."""
    out: Dict[str, Dict[float, float]] = {ch: {} for ch in CHANNELS}
    with open(grid_path(scenario)) as f:
        for r in csv.DictReader(f):
            out[r["channel"]][float(r["alpha"])] = float(r[metric])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--metric", default="RE", choices=("RE", "SSIM"))
    args = ap.parse_args()

    missing = [s for s in args.scenarios if not grid_path(s).exists()]
    if missing:
        raise SystemExit(
            "Missing sweep output for: " + ", ".join(missing) + "\nRun first:\n"
            + "\n".join(f"  python sweep_tv_beta.py --stage alpha_beta --betas 3e-4 --scenario {s}"
                        for s in missing))

    better = min if args.metric == "RE" else max
    worse = max if args.metric == "RE" else min   # excess is positive for RE, negative for SSIM
    per = {s: load(s, args.metric) for s in args.scenarios}

    print(f"Selecting alpha by mean {args.metric} over: {', '.join(args.scenarios)}\n")
    for ch in CHANNELS:
        curves = {s: per[s][ch] for s in args.scenarios}
        alphas = sorted(set.intersection(*(set(c) for c in curves.values())))
        if not alphas:
            raise SystemExit(f"{ch}: the scenarios share no alpha grid points.")

        mean = {a: sum(curves[s][a] for s in curves) / len(curves) for a in alphas}
        a_star = better(mean, key=mean.get)

        own = {s: better(curves[s], key=curves[s].get) for s in curves}
        excess = {s: 100.0 * (curves[s][a_star] / curves[s][own[s]] - 1.0) for s in curves}
        worst = worse(excess.values())

        print(f"{ch:6s} alpha = {a_star:<9g} mean {args.metric}={mean[a_star]:.4f}   "
              f"worst excess over a scenario's own optimum {worst:+.1f}%")
        print("       per-scenario optima: "
              + "  ".join(f"{s}={own[s]:g} ({excess[s]:+.1f}%)" for s in args.scenarios))
    print(f"\nSet these as ALPHA_MU/DELTA/EPS in {Path('synthetic_data/recon/run_scenarios.py')}.")


if __name__ == "__main__":
    main()
