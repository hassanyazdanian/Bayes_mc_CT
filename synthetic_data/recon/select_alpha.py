"""
Select the prior weights from the per-scenario sweeps.

sweep_tv_beta.py and tune_alpha_joint.py each report the optimum for ONE
acquisition condition, and those optima disagree. The reported weights are none
of them: they are chosen across all four conditions, because a weight tuned on
full data over-regularizes the undersampled ones (the full-scan optimum
alpha_delta=1778 doubles delta's RE on combined). Run the sweeps for every
scenario, then this.

Stage "channel" (alpha_beta_grid*.csv): the alpha with the lowest mean RE per
channel.

Stage "joint" (joint_sweep_reoptimal*.csv): the alpha_joint with the largest
mean improvement in summed RE over alpha_joint=0, measured per scenario as a
relative change so the scenarios are weighted equally. A candidate that
degrades any scenario is flagged, since the coupling is meant to help
everywhere.

Usage:
    python select_alpha.py                    # channel weights
    python select_alpha.py --stage joint      # coupling weight
    python select_alpha.py --metric SSIM      # channel stage, by mean SSIM
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

OUT_DIR = Path(__file__).resolve().parent.parent / "obs" / "tuning"
CHANNELS = ["mu", "delta", "eps"]
SCENARIOS = ["full", "sparse_angle", "sparse_step", "combined"]


def tag_of(scenario: str) -> str:
    return "" if scenario == "full" else f"_{scenario}"


def channel_grid(scenario: str) -> Path:
    return OUT_DIR / "beta_sweep" / f"alpha_beta_grid{tag_of(scenario)}.csv"


def joint_grid(scenario: str) -> Path:
    return OUT_DIR / f"joint_sweep_reoptimal{tag_of(scenario)}.csv"


def require(paths: Dict[str, Path], hint: str) -> None:
    missing = [s for s, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit("Missing sweep output for: " + ", ".join(missing)
                         + "\nRun first:\n" + "\n".join(hint.format(s=s) for s in missing))


def select_channel(scenarios, metric: str) -> None:
    paths = {s: channel_grid(s) for s in scenarios}
    require(paths, "  python sweep_tv_beta.py --stage alpha_beta --betas 3e-4 --scenario {s}")

    curves: Dict[str, Dict[str, Dict[float, float]]] = {}
    for s, p in paths.items():
        per = {ch: {} for ch in CHANNELS}
        with open(p) as f:
            for r in csv.DictReader(f):
                per[r["channel"]][float(r["alpha"])] = float(r[metric])
        curves[s] = per

    better = min if metric == "RE" else max
    worse = max if metric == "RE" else min

    print(f"Selecting alpha by mean {metric} over: {', '.join(scenarios)}\n")
    for ch in CHANNELS:
        c = {s: curves[s][ch] for s in scenarios}
        alphas = sorted(set.intersection(*(set(v) for v in c.values())))
        if not alphas:
            raise SystemExit(f"{ch}: the scenarios share no alpha grid points.")
        mean = {a: sum(c[s][a] for s in c) / len(c) for a in alphas}
        a_star = better(mean, key=mean.get)
        own = {s: better(c[s], key=c[s].get) for s in c}
        excess = {s: 100.0 * (c[s][a_star] / c[s][own[s]] - 1.0) for s in c}
        print(f"{ch:6s} alpha = {a_star:<9g} mean {metric}={mean[a_star]:.4f}   "
              f"worst excess over a scenario's own optimum {worse(excess.values()):+.1f}%")
        print("       per-scenario optima: "
              + "  ".join(f"{s}={own[s]:g} ({excess[s]:+.1f}%)" for s in scenarios))
    print("\nSet these as ALPHA_MU/DELTA/EPS in synthetic_data/recon/run_scenarios.py.")


def select_joint(scenarios) -> None:
    paths = {s: joint_grid(s) for s in scenarios}
    require(paths, "  python tune_alpha_joint.py --scenario {s} "
                   "--alpha_mu 100 --alpha_delta 316.2 --alpha_eps 56.23")

    summed: Dict[str, Dict[float, float]] = {}
    for s, p in paths.items():
        with open(p) as f:
            summed[s] = {float(r["alpha_joint"]): sum(float(r[f"RE_{ch}"]) for ch in CHANNELS)
                         for r in csv.DictReader(f)}

    alphas = sorted(set.intersection(*(set(v) for v in summed.values())))
    change = {a: {s: 100.0 * (summed[s][a] / summed[s][0.0] - 1.0) for s in scenarios}
              for a in alphas}
    mean = {a: sum(change[a].values()) / len(scenarios) for a in alphas}
    a_star = min(mean, key=mean.get)

    print(f"Selecting alpha_joint by mean change in summed RE vs alpha_joint=0, "
          f"over: {', '.join(scenarios)}\n")
    print(f"{'alpha_joint':>12s} {'mean':>8s}   " + "  ".join(f"{s:>13s}" for s in scenarios))
    for a in alphas:
        if a == 0.0:
            continue
        mark = "  <-- selected" if a == a_star else ""
        print(f"{a:12g} {mean[a]:+7.2f}%   "
              + "  ".join(f"{change[a][s]:+12.2f}%" for s in scenarios) + mark)

    worst = max(change[a_star].values())
    degraded = [s for s, v in change[a_star].items() if v > 0]
    print(f"\nalpha_joint = {a_star:g}: mean {mean[a_star]:+.2f}%, worst scenario {worst:+.2f}%")
    print("  every scenario improves" if not degraded
          else "  WARNING, degrades: " + ", ".join(degraded))
    print(f"\nSet this as ALPHA_JOINT in synthetic_data/recon/run_scenarios.py.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="channel", choices=("channel", "joint"))
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--metric", default="RE", choices=("RE", "SSIM"),
                    help="Channel stage only.")
    args = ap.parse_args()
    if args.stage == "channel":
        select_channel(args.scenarios, args.metric)
    else:
        select_joint(args.scenarios)


if __name__ == "__main__":
    main()
