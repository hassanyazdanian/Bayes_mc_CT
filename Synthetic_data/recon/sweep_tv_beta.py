"""
Tune the smoothed-TV width (TV_BETA), the amplitude-prior weight (LAMBDA0)
and the per-channel TV weights (ALPHA_MU/DELTA/EPS) for the synthetic
phantom, scoring every MAP against the known ground truth.

Same questions as real_data/sweep_tv_beta_lambda0.py, with two differences.
Ground truth replaces the full-angle FBP surrogate. And alpha and beta are
swept JOINTLY, because they trade off: below |grad| ~ beta the smoothed TV
sqrt(g^2 + beta^2) is quadratic with curvature alpha/beta, so changing beta
at fixed alpha also changes the effective smoothing. Each beta here gets its
own RE-optimal alpha per channel, so the beta curve compares the best
reconstruction each beta can achieve. Single-channel solves at N=256 make
the 2-D grid affordable.

Stage "alpha_beta": for each tv_beta and channel, sweep alpha with the
channel's own optimizer (map_tv_jtv.reconstruct_single_channel -- see its
docstring for why not reconstruct_map at alpha_joint=0), lambda0 fixed.

Stage "lambda0": at the chosen tv_beta and per-channel alphas, sweep lambda0.
The L2 term adds curvature lambda0 in every direction, bounding the
posterior's smallest curvature from below (helps NUTS), and leaves the MAP
unchanged until it competes with the data. Take the largest value that does
not degrade RE.

Units. The synthetic priors act on the PHYSICAL fields (map_tv_jtv.py), so
tv_beta is in units of the field gradient; the real-data priors act on
z = x / field_scale. Dividing tv_beta by the field scale (printed at startup)
gives the comparable number: the real-data choice beta_z = 0.03 corresponds
to tv_beta ~ 5e-3 for the multicontrast phantom.

Stiffness. alpha/tv_beta is the TV curvature in flat regions -- what limits
the NUTS step size. Divided by the mean data-term curvature
(mean diag(A^T A) / sigma^2, Hutchinson estimate) it is unit-free, which
makes candidate betas comparable across channels. It is not comparable with
the real-data DATA_CURVATURE constant, which was measured differently.

Tuning uses the same noise realization (run_scenarios.SEED) that the reported
full-scan results use, i.e. oracle tuning as in tune_alpha.py; pass --seed to
tune on an independent realization instead.

Usage:
    python sweep_tv_beta.py --stage alpha_beta
    python sweep_tv_beta.py --stage lambda0 --tv_beta 3e-3 \
        --alpha_mu 1000 --alpha_delta 300 --alpha_eps 500

Outputs (obs/tuning/beta_sweep/; non-full scenarios add a _<scenario> suffix):
    alpha_beta_grid.csv, alpha_beta_best.csv, alpha_beta_curves.png,
    alpha_beta_heatmaps.png, alpha_beta_montage.png,
    lambda0_sweep.csv, lambda0_sweep.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
RECON_DIR = Path(__file__).resolve().parent
for p in (BASE_DIR, COMMON_DIR, RECON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_scenarios as rs  # noqa: E402
from map_tv_jtv import compute_metrics, reconstruct_single_channel  # noqa: E402
from scenario_utils import make_reconstruction_mask, make_sparse_angle_indices, vec_to_img  # noqa: E402
from TLI_2D_forward import dpc_from_phase_sino  # noqa: E402

CHANNELS = ["mu", "delta", "eps"]
SINO_KEYS = {"mu": "T", "delta": "DPC", "eps": "D"}
OUT_DIR = BASE_DIR / "obs" / "tuning" / "beta_sweep"


def scenario_sizes(d: Dict, scenario: str) -> Tuple[int, int]:
    return {
        "full": (d["n_angles_full"], d["n_phase_full"]),
        "sparse_angle": (rs.N_SPARSE_ANGLES, d["n_phase_full"]),
        "sparse_step": (d["n_angles_full"], rs.N_SPARSE_PHASE),
        "combined": (rs.N_SPARSE_ANGLES, rs.N_SPARSE_PHASE),
    }[scenario]


def setup(scenario: str, seed: int, device, dtype) -> Dict:
    """Phantom, simulated data (SIM_INTERP) and reconstruction projector (RECON_INTERP)."""
    d = rs.load_dataset(BASE_DIR / "obs" / "synthetic_data.h5")
    d = rs.override_phantom(d, rs.PHANTOM_NAME, rs.PHANTOM_SUPERSAMPLE, rs.PHANTOM_TARGET_DPC_MAX, device, dtype)

    n_angles, n_phase = scenario_sizes(d, scenario)
    angle_idx = make_sparse_angle_indices(d["n_angles_full"], n_angles)
    noisy, sigma = rs.simulate_data(d, angle_idx, n_phase, device, dtype, seed=seed)
    projector = rs.build_projector(d, angle_idx, device, dtype, interp=rs.RECON_INTERP)

    print(f"Scenario: {scenario} (n_angles={n_angles}, n_phase={n_phase}), seed={seed}, noise={rs.NOISE_LEVEL}")
    print(f"Simulate: {d.get('N_sim', d['N'])}^2 {rs.SIM_INTERP}; reconstruct: {d['N']}^2 {rs.RECON_INTERP}; "
          f"likelihood T/DPC/D")

    truth = {ch: d[ch] for ch in CHANNELS}
    field_scale = {ch: float(np.percentile(np.abs(truth[ch]), 99)) for ch in CHANNELS}
    return {
        "d": d, "projector": projector, "truth": truth, "field_scale": field_scale,
        "det_spacing": rs.det_spacing_of(d),
        "obs": {ch: noisy[SINO_KEYS[ch]] for ch in CHANNELS},
        "sigma": {ch: sigma[SINO_KEYS[ch]] for ch in CHANNELS},
    }


def mean_data_curvature(projector, N: int, sigma: Dict[str, float], det_spacing: float, device, dtype,
                        n_probe: int = 8) -> Dict[str, float]:
    """mean diag(G^T G) / sigma^2 over the reconstruction mask, by Hutchinson
    probes, with G = A for mu/eps and G = -d/du A (the DPC operator) for delta."""
    mask = make_reconstruction_mask(N, R=0.98, device=device)
    n_active = int(mask.sum().item())
    gen = torch.Generator().manual_seed(0)
    trace = {"A": 0.0, "DPC": 0.0}
    with torch.no_grad():
        for _ in range(n_probe):
            v = (torch.randint(0, 2, (n_active,), generator=gen) * 2 - 1).to(device=device, dtype=dtype)
            s = projector.make_sinogram(vec_to_img(v, mask), return_physical=True)
            trace["A"] += float(torch.sum(s ** 2))
            trace["DPC"] += float(torch.sum(dpc_from_phase_sino(s, det_spacing=det_spacing) ** 2))
    op = {"mu": "A", "delta": "DPC", "eps": "A"}
    return {ch: trace[op[ch]] / n_probe / n_active / sigma[ch] ** 2 for ch in CHANNELS}


def solve(prob: Dict, ch: str, alpha: float, tv_beta: float, lambda0: float, device, dtype) -> Tuple[np.ndarray, float, float, float]:
    out = reconstruct_single_channel(
        prob["obs"][ch], prob["projector"], prob["d"]["N"], prob["sigma"][ch], alpha_tv=alpha,
        tv_beta=tv_beta, lambda0=lambda0, n_iter=rs.N_ITER, device=device, dtype=dtype,
        dpc_det_spacing=prob["det_spacing"] if ch == "delta" else None,
    )
    re, s, p = compute_metrics(prob["truth"][ch], out["img"])
    return out["img"], re, s, p


# -----------------------------------------------------------------------------
# Stage 1: alpha x beta
# -----------------------------------------------------------------------------

def stage_alpha_beta(prob: Dict, args, data_curv: Dict[str, float], device, dtype) -> None:
    betas = list(args.betas)
    alphas = np.logspace(args.alpha_min_exp, args.alpha_max_exp, args.n_alpha)
    n_solves = len(betas) * len(CHANNELS) * len(alphas)
    print(f"\n=== Stage alpha_beta: {len(betas)} betas x {len(CHANNELS)} channels x {len(alphas)} alphas "
          f"= {n_solves} solves (lambda0={args.lambda0:g}) ===")

    grid_rows, best = [], {}
    best_img = {}
    t_start = time.perf_counter()
    for b in betas:
        for ch in CHANNELS:
            res = []
            for a in alphas:
                img, re, s, p = solve(prob, ch, a, b, args.lambda0, device, dtype)
                kink = a / b
                grid_rows.append({"tv_beta": b, "channel": ch, "alpha": a, "RE": re, "SSIM": s, "PSNR": p,
                                  "kink": kink, "kink_over_data": kink / data_curv[ch]})
                res.append((re, s, p, a, img))
            i_re = int(np.argmin([r[0] for r in res]))
            i_ss = int(np.argmax([r[1] for r in res]))
            re, s, p, a, img = res[i_re]
            at_edge = i_re in (0, len(alphas) - 1)
            best[(b, ch)] = {"tv_beta": b, "channel": ch, "alpha": a, "RE": re, "SSIM": s, "PSNR": p,
                             "alpha_ssim_max": res[i_ss][3], "SSIM_max": res[i_ss][1],
                             "kink": a / b, "kink_over_data": a / b / data_curv[ch],
                             "at_grid_edge": at_edge}
            best_img[(b, ch)] = img
            print(f"  beta={b:8.2g} (beta/scale={b / prob['field_scale'][ch]:.2g})  {ch:5s}: "
                  f"alpha*={a:9.4g}  RE={re:.4f} SSIM={s:.4f} PSNR={p:.2f}  "
                  f"alpha/beta={a / b:.3g} (/data {a / b / data_curv[ch]:.2g})"
                  + ("  <-- ALPHA AT GRID EDGE, widen --alpha_*_exp" if at_edge else ""))
        print(f"  [elapsed {(time.perf_counter() - t_start) / 60:.1f} min]")

    # ---------------- CSVs ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["tv_beta", "channel", "alpha", "RE", "SSIM", "PSNR", "kink", "kink_over_data"]
    with open(OUT_DIR / f"alpha_beta_grid{args.tag}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in grid_rows:
            f.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    bkeys = ["tv_beta", "channel", "alpha", "RE", "SSIM", "PSNR", "alpha_ssim_max", "SSIM_max",
             "kink", "kink_over_data", "at_grid_edge"]
    with open(OUT_DIR / f"alpha_beta_best{args.tag}.csv", "w") as f:
        f.write(",".join(bkeys) + "\n")
        for b in betas:
            for ch in CHANNELS:
                r = best[(b, ch)]
                f.write(",".join(str(r[k]) if isinstance(r[k], (str, bool, np.bool_)) else f"{r[k]:.6g}"
                                 for k in bkeys) + "\n")

    # ---------------- suggestion ----------------
    # Largest beta (softest prior, easiest to sample) whose best RE is within
    # --beta_re_tol of that channel's best over all betas, in every channel.
    re_star = {ch: min(best[(b, ch)]["RE"] for b in betas) for ch in CHANNELS}
    ok = [b for b in betas if all(best[(b, ch)]["RE"] <= (1 + args.beta_re_tol) * re_star[ch] for ch in CHANNELS)]
    print("\n=== Best achievable RE per beta (RE-optimal alpha per channel) ===")
    print(f"  {'beta':>8s}" + "".join(f"  {ch + ' RE':>10s} {ch + ' alpha*':>12s}" for ch in CHANNELS))
    for b in betas:
        line = f"  {b:8.2g}"
        for ch in CHANNELS:
            r = best[(b, ch)]
            mark = "*" if r["RE"] == re_star[ch] else " "
            line += f"  {r['RE']:9.4f}{mark} {r['alpha']:12.4g}"
        print(line)
    if ok:
        b_sug = max(ok)
        print(f"\nSuggested tv_beta = {b_sug:g}  (largest beta within {100 * args.beta_re_tol:.0f}% of every channel's best RE)")
        print("Check alpha_beta_montage.png for background texture before accepting it. Next:")
        print(f"  python sweep_tv_beta.py --stage lambda0 --tv_beta {b_sug:g} "
              + " ".join(f"--alpha_{ch} {best[(b_sug, ch)]['alpha']:.4g}" for ch in CHANNELS))
    else:
        print(f"\nNo single beta is within {100 * args.beta_re_tol:.0f}% of every channel's best RE; "
              "channels disagree -- inspect alpha_beta_curves.png.")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True)
    for j, ch in enumerate(CHANNELS):
        rows = [best[(b, ch)] for b in betas]
        axes[0, j].semilogx(betas, [r["RE"] for r in rows], "o-")
        axes[0, j].set_title(ch)
        axes[1, j].semilogx(betas, [r["SSIM"] for r in rows], "o-")
        axes[2, j].loglog(betas, [r["alpha"] for r in rows], "o-", label="alpha* (RE)")
        axes[2, j].loglog(betas, [r["alpha_ssim_max"] for r in rows], "s--", label="alpha* (SSIM)")
        axes[2, j].set_xlabel("tv_beta (physical units)")
        axes[2, j].legend(fontsize=7)
    axes[0, 0].set_ylabel("RE at alpha*")
    axes[1, 0].set_ylabel("SSIM at alpha*")
    axes[2, 0].set_ylabel("alpha*")
    fig.suptitle("Best achievable reconstruction per tv_beta (alpha re-optimized per channel)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"alpha_beta_curves{args.tag}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for j, ch in enumerate(CHANNELS):
        grid = np.array([[next(r["RE"] for r in grid_rows
                               if r["tv_beta"] == b and r["channel"] == ch and r["alpha"] == a)
                          for a in alphas] for b in betas])
        im = axes[j].imshow(np.log10(grid), aspect="auto", origin="lower", cmap="magma_r")
        axes[j].set_xticks(range(0, len(alphas), 4))
        axes[j].set_xticklabels([f"{a:.0e}" for a in alphas[::4]], fontsize=7)
        axes[j].set_yticks(range(len(betas)))
        axes[j].set_yticklabels([f"{b:g}" for b in betas], fontsize=8)
        for y, b in enumerate(betas):
            axes[j].plot(int(np.argmin(grid[y])), y, "c*", markersize=10)
        axes[j].set_xlabel("alpha"); axes[j].set_ylabel("tv_beta"); axes[j].set_title(f"{ch}: log10 RE")
        fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"alpha_beta_heatmaps{args.tag}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(betas) + 1, 3, figsize=(7.5, 2.4 * (len(betas) + 1)), squeeze=False)
    for j, ch in enumerate(CHANNELS):
        vmax = float(prob["truth"][ch].max())
        panels = [(prob["truth"][ch], "truth")] + [(best_img[(b, ch)], f"beta={b:g}") for b in betas]
        for i, (img, lab) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(np.clip(img, 0, None), cmap="gray", origin="lower", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(ch)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.suptitle("RE-optimal reconstruction per tv_beta", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"alpha_beta_montage{args.tag}.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved under: {OUT_DIR}")


# -----------------------------------------------------------------------------
# Stage 2: lambda0
# -----------------------------------------------------------------------------

def stage_lambda0(prob: Dict, args, data_curv: Dict[str, float], device, dtype) -> None:
    if args.tv_beta is None or None in (args.alpha_mu, args.alpha_delta, args.alpha_eps):
        raise SystemExit("--stage lambda0 needs --tv_beta, --alpha_mu, --alpha_delta, --alpha_eps")
    alpha = {"mu": args.alpha_mu, "delta": args.alpha_delta, "eps": args.alpha_eps}
    lambda0s = sorted(args.lambda0s)
    print(f"\n=== Stage lambda0: tv_beta={args.tv_beta:g}, alphas={alpha}, {len(lambda0s)} values ===")

    rows = []
    for ch in CHANNELS:
        for l0 in lambda0s:
            _img, re, s, p = solve(prob, ch, alpha[ch], args.tv_beta, l0, device, dtype)
            l0_z = l0 * prob["field_scale"][ch] ** 2
            rows.append({"lambda0": l0, "channel": ch, "RE": re, "SSIM": s, "PSNR": p,
                         "lambda0_z": l0_z, "lambda0_over_data": l0 / data_curv[ch]})
            print(f"  {ch:5s} lambda0={l0:8.2g} (z-units {l0_z:.2g}, /data {l0 / data_curv[ch]:.2g}): "
                  f"RE={re:.4f} SSIM={s:.4f} PSNR={p:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["lambda0", "channel", "RE", "SSIM", "PSNR", "lambda0_z", "lambda0_over_data"]
    with open(OUT_DIR / f"lambda0_sweep{args.tag}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")

    # Largest lambda0 whose RE stays within --lambda0_re_tol of the smallest
    # lambda0's RE, in every channel (lambda0 is shared across channels).
    base = {ch: next(r["RE"] for r in rows if r["channel"] == ch and r["lambda0"] == lambda0s[0]) for ch in CHANNELS}
    ok = [l0 for l0 in lambda0s
          if all(next(r["RE"] for r in rows if r["channel"] == ch and r["lambda0"] == l0)
                 <= (1 + args.lambda0_re_tol) * base[ch] for ch in CHANNELS)]
    print(f"\nSuggested LAMBDA0 = {max(ok):g}  (largest value with RE within "
          f"{100 * args.lambda0_re_tol:.1f}% of lambda0={lambda0s[0]:g} in every channel)")
    if max(ok) == lambda0s[-1]:
        print("  (that is the top of the grid -- extend --lambda0s to find where it starts to bite)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ch in CHANNELS:
        rr = [r for r in rows if r["channel"] == ch]
        axes[0].semilogx(lambda0s, [r["RE"] / base[ch] for r in rr], "o-", label=ch)
        axes[1].semilogx(lambda0s, [r["SSIM"] for r in rr], "o-", label=ch)
    axes[0].axhline(1 + args.lambda0_re_tol, color="k", ls=":", lw=1)
    axes[0].set_xlabel("lambda0 (physical units)"); axes[0].set_ylabel("RE / RE at smallest lambda0"); axes[0].legend()
    axes[1].set_xlabel("lambda0 (physical units)"); axes[1].set_ylabel("SSIM"); axes[1].legend()
    fig.suptitle(f"lambda0 sweep at tv_beta={args.tv_beta:g}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"lambda0_sweep{args.tag}.png", dpi=180)
    plt.close(fig)
    print(f"Saved under: {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("alpha_beta", "lambda0"))
    parser.add_argument("--scenario", default="full", choices=("full", "sparse_angle", "sparse_step", "combined"),
                        help="Synthetic convention: tune on the full scan, hold fixed across scenarios.")
    parser.add_argument("--seed", type=int, default=rs.SEED)
    parser.add_argument("--betas", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    parser.add_argument("--alpha_min_exp", type=float, default=0.0)
    parser.add_argument("--alpha_max_exp", type=float, default=6.0)
    parser.add_argument("--n_alpha", type=int, default=25)
    parser.add_argument("--lambda0", type=float, default=1e-6, help="Held fixed in stage alpha_beta.")
    parser.add_argument("--beta_re_tol", type=float, default=0.02)
    parser.add_argument("--tv_beta", type=float, default=None)
    parser.add_argument("--alpha_mu", type=float, default=None)
    parser.add_argument("--alpha_delta", type=float, default=None)
    parser.add_argument("--alpha_eps", type=float, default=None)
    parser.add_argument("--lambda0s", type=float, nargs="+", default=list(np.logspace(-6, 3, 10)))
    parser.add_argument("--lambda0_re_tol", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.tag = "" if args.scenario == "full" else f"_{args.scenario}"

    device, dtype = torch.device(args.device), torch.float32
    print(f"Using device: {device}")
    prob = setup(args.scenario, args.seed, device, dtype)

    data_curv = mean_data_curvature(prob["projector"], prob["d"]["N"], prob["sigma"], prob["det_spacing"], device, dtype)
    for ch in CHANNELS:
        print(f"  {ch:5s}: field scale (p99 |truth|)={prob['field_scale'][ch]:.4g}  sigma={prob['sigma'][ch]:.3e}  "
              f"mean data curvature={data_curv[ch]:.3e}")

    if args.stage == "alpha_beta":
        stage_alpha_beta(prob, args, data_curv, device, dtype)
    else:
        stage_lambda0(prob, args, data_curv, device, dtype)


if __name__ == "__main__":
    main()
