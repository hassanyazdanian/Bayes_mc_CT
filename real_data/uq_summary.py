"""
Collect every sampling number the paper quotes into one CSV, so the UQ section
can be checked against a file instead of against scattered logs.

Per scenario, prior and channel it reports:

  ESS / R-hat   per-pixel effective sample size and Gelman-Rubin statistic over
                ALL sampled dimensions (not a random subset): min, median, and
                the 99th percentile of R-hat, which is what the text calls the
                "upper percentile".
  mean_std      mean posterior standard deviation over the reconstruction
                support, the quantity behind the "JTV lowers sigma by x%" claims.
  rho_post_mean noise-normalized sinogram residual at the posterior mean, the
                counterpart of the MAP data-fit table. sigma comes from the
                pickle's own metadata, so it is the value the sampler used.

Run-level columns (step size, acceptance probability, runtime, and the prior
parameters) are repeated on each of the channel rows.

The adapted step size and acceptance probability are not stored in the pickle --
they exist only in the progress bar -- so they are parsed from the run log when
one is given or found next to the pickle (nuts_<prior>.log).

Usage:
    python uq_summary.py                        # all scenarios, both priors
    python uq_summary.py --scenarios combined --priors tv
    python uq_summary.py --skip_rho             # no projector, no data reload
Output: real_data/obs/uq/uq_summary.csv
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "common"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data_utils as du  # noqa: E402
from diagnose_convergence import chunked_ess_rhat  # noqa: E402
from nuts_real import scenario_indices  # noqa: E402
from TLI_2D_forward import dpc_from_phase_sino  # noqa: E402

CHANNELS = ["mu", "delta", "eps"]
SINO_KEY = {"mu": "T", "delta": "DPC", "eps": "D"}
SIGMA_KEY = {"mu": "sigma_T", "delta": "sigma_DPC", "eps": "sigma_D"}
SCENARIOS = ["full", "sparse_angle", "sparse_step", "combined"]
PRIORS = ["tv", "jtv"]

STEP_RE = re.compile(r"step size=([0-9.eE+-]+).*?acc\. prob=([0-9.]+)")


def parse_log(path: Optional[Path]) -> Dict[str, float]:
    """Final adapted step size and acceptance probability per chain, from the
    tqdm progress bar in a tee'd run log."""
    if path is None or not path.exists():
        return {}
    text = path.read_text(errors="ignore").replace("\r", "\n")
    hits = STEP_RE.findall(text)
    if not hits:
        return {}
    steps = [float(s) for s, _ in hits[-2:]]
    accs = [float(a) for _, a in hits[-2:]]
    return {"step_size_adapted": float(np.mean(steps)), "acc_prob": float(np.mean(accs))}


def support_mask(n: int, r: float) -> np.ndarray:
    c = np.linspace(-1.0, 1.0, n)
    Y, X = np.meshgrid(c, c, indexing="ij")
    return (X ** 2 + Y ** 2) <= r ** 2


def scenario_observations(scenario: str, device, dtype) -> Tuple[Dict[str, np.ndarray], object, float]:
    """Measured sinograms and projector for one scenario, built exactly as
    nuts_real.setup_problem builds them (same subsetting, retrieval, kernel)."""
    data = du.load_real_intensities()
    n_angles_full, n_phase_full = data["I_meas"].shape[0], data["I_meas"].shape[1]
    L_phys = du.geometry_for(data["I_ref"].shape[-1])
    angle_idx, step_idx = scenario_indices(n_angles_full, n_phase_full, scenario)
    sinos = du.retrieve_and_correct_sinograms(
        data["I_meas"][angle_idx][:, step_idx, :], data["I_ref"][angle_idx][:, step_idx, :], verbose=False)
    beta_deg = du.make_beta_deg(data["angles"][angle_idx], len(angle_idx))
    projector, det_spacing = du.build_projector(sinos["T"].shape[1], L_phys, beta_deg, device, dtype)
    return sinos, projector, det_spacing


def posterior_mean_std(stat: Dict, meta: Dict) -> Dict[str, np.ndarray]:
    """Posterior mean/std per channel, in physical units, from the samples."""
    z = stat["samples"]["z_grouped"]           # (chains, samples, dim)
    z = z.reshape(-1, z.shape[-1])
    n_active, N = meta["n_active"], meta["N"]
    mask = support_mask(N, meta["support_R"])
    out = {}
    for i, ch in enumerate(CHANNELS):
        scale = meta[f"field_scale_{ch}"]
        block = z[:, i * n_active:(i + 1) * n_active] * scale
        for tag, vec in (("mean", block.mean(dim=0)), ("std", block.std(dim=0, unbiased=True))):
            img = np.zeros((N, N), dtype=np.float64)
            img[mask] = vec.numpy()
            out[f"{ch}_{tag}"] = img
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uq_root", type=str, default=str(HERE / "obs" / "uq"))
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--priors", nargs="+", default=PRIORS)
    ap.add_argument("--skip_rho", action="store_true", help="Skip the posterior-mean data residual.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    uq_root = Path(args.uq_root)
    device, dtype = torch.device(args.device), torch.float32
    obs_cache: Dict[str, Tuple] = {}
    rows: List[Dict] = []

    for scenario in args.scenarios:
        for prior in args.priors:
            pkl = uq_root / scenario / prior / "nuts_samples.pickle"
            if not pkl.exists():
                print(f"  missing: {pkl}")
                continue
            print(f"\n=== {scenario} / {prior} ===")
            with open(pkl, "rb") as f:
                stat = pickle.load(f)
            meta, diag = stat["meta"], stat.get("diagnostics", {})

            ess, rhat = chunked_ess_rhat(stat["samples"]["z_grouped"])
            ess, rhat = ess.numpy(), rhat.numpy()
            post = posterior_mean_std(stat, meta)
            mask = support_mask(meta["N"], meta["support_R"])

            log_info = parse_log(pkl.parent.parent / f"nuts_{prior}.log")
            chain_times = diag.get("chain_times", [])
            run_cols = {
                "n_chains": meta["num_chains"], "n_samples": meta["num_samples"],
                "warmup": meta["warmup_steps"], "max_tree_depth": meta["max_tree_depth"],
                "alpha_mu": meta["alpha_mu"], "alpha_delta": meta["alpha_delta"],
                "alpha_eps": meta["alpha_eps"], "alpha_joint": meta["alpha_joint"],
                "tv_beta": meta["tv_beta"], "lambda0": meta["lambda0"],
                "chain_minutes": float(np.mean(chain_times) / 60) if chain_times else float("nan"),
                "step_size_adapted": log_info.get("step_size_adapted", float("nan")),
                "acc_prob": log_info.get("acc_prob", float("nan")),
            }

            rho = {}
            if not args.skip_rho:
                if scenario not in obs_cache:
                    obs_cache[scenario] = scenario_observations(scenario, device, dtype)
                sinos, projector, det_spacing = obs_cache[scenario]
                for ch in CHANNELS:
                    x = torch.as_tensor(post[f"{ch}_mean"], dtype=dtype, device=device)
                    with torch.no_grad():
                        s = projector.make_sinogram(x, return_physical=True)
                        pred = (-dpc_from_phase_sino(s, det_spacing=det_spacing) if ch == "delta" else s)
                    y = sinos[SINO_KEY[ch]]
                    rho[ch] = float(np.linalg.norm(pred.cpu().numpy() - y)
                                    / meta[SIGMA_KEY[ch]] / np.sqrt(y.size))

            n_active = meta["n_active"]
            for i, ch in enumerate(CHANNELS):
                sl = slice(i * n_active, (i + 1) * n_active)
                row = {
                    "scenario": scenario, "prior": prior, "channel": ch,
                    "ess_min": float(ess[sl].min()), "ess_median": float(np.median(ess[sl])),
                    "rhat_median": float(np.median(rhat[sl])), "rhat_p99": float(np.percentile(rhat[sl], 99)),
                    "rhat_max": float(rhat[sl].max()),
                    "mean_std": float(post[f"{ch}_std"][mask].mean()),
                    "rho_post_mean": rho.get(ch, float("nan")),
                    "sigma": meta[SIGMA_KEY[ch]], **run_cols,
                }
                rows.append(row)
                print(f"  {ch:6s} ESS med {row['ess_median']:8.1f} (min {row['ess_min']:6.1f})  "
                      f"Rhat med {row['rhat_median']:.4f} p99 {row['rhat_p99']:.4f} max {row['rhat_max']:.4f}  "
                      f"std {row['mean_std']:.4g}  rho {row['rho_post_mean']:.3f}")
            del stat

    if not rows:
        raise SystemExit("No pickles found.")
    keys = list(rows[0].keys())
    out = Path(args.out) if args.out else uq_root / "uq_summary.csv"
    with open(out, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
