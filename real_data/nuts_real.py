"""
NUTS posterior sampling for real Talbot-Lau data, on the same smoothed
joint-TV posterior used for the real-data MAP pipeline (map_real.py) and the
four limited-data scenarios (run_scenarios.py).

Mirrors uq/nuts_synthetic.py, adapted for real data's established
conventions (see run_scenarios.py / data_utils.py docstrings for why each is
the validated choice, not an arbitrary port):

  - Likelihood on T/DPC/D (not T/P/D) -- map_real.py's validated choice.
  - Priors evaluated on the *normalized* field z = x/field_scale, not the
    physical field -- map_real.py's convention. The potential below
    reproduces reconstruct_map_real's loss term-for-term, so its mode
    coincides exactly with the real-data MAP at the same alpha values. This
    has been verified numerically against the original reference pipeline
    (my_paper/noether/real_data/TV/): identical potential value, gradients
    agreeing to 1.8e-07, at a common point.
  - field_scale AND sigma are estimated ONCE from the full scan and reused
    across all scenarios (matching run_scenarios.py) -- NOT re-derived per
    scenario, which collapsed delta to near-zero under sparse angles.
  - support_R defaults to real data's SUPPORT_R=0.5 (data_utils.py).
  - No ground truth: the output pickle has no "true" key. Use
    post_process_real.py; uq/diagnose_convergence.py works unchanged.
  - MAP initialization from a zero-init reconstruct_map_real call at this
    prior's own alpha_joint -- not warm-started from another prior's MAP.

The forward-projector interpolation kernel comes from
data_utils.PROJECTOR_INTERP so that MAP and UQ cannot silently use different
forward models; --projector overrides it for one run.

NOTE ON --device: this defaults to CUDA when available. An earlier version
defaulted to "cpu" (copied from nuts_synthetic.py), and because nothing
passed --device explicitly, every run silently used the CPU -- roughly 100x
slower, which is what made sampling look intractable.

Usage:
    python nuts_real.py --scenario sparse_angle --alpha_joint 0 \
        --num_chains 2 --num_samples 200 --warmup_steps 200 --max_tree_depth 6

Quick smoke test (confirm per-iteration cost before a long run):
    python nuts_real.py --scenario sparse_angle \
        --num_chains 1 --num_samples 5 --warmup_steps 40 --max_tree_depth 6

Output: real_data/obs/uq/<scenario>/<tv|jtv>/nuts_samples.pickle
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from pyro.infer.mcmc import MCMC, NUTS
from pyro.ops.stats import effective_sample_size, gelman_rubin

BASE_DIR = Path(__file__).resolve().parent
COMMON_DIR = BASE_DIR.parent / "common"
def _recon_dir(base: Path) -> Path:
    """Locate the shared prior/metric module (map_tv_jtv.py).

    It lived in <project>/recon/ before the synthetic pipeline was moved into
    Synthetic_data/; accept either layout so this file works in both.
    """
    for candidate in (base / "recon", base / "Synthetic_data" / "recon"):
        if (candidate / "map_tv_jtv.py").exists():
            return candidate
    return base / "recon"


RECON_DIR = _recon_dir(BASE_DIR.parent)
# Insertion order matters: recon/ has its own run_scenarios.py (the synthetic
# one) -- BASE_DIR (real_data/) must end up at sys.path[0] so the import below
# resolves to real_data/run_scenarios.py, not recon/'s.
for p in (RECON_DIR, COMMON_DIR, BASE_DIR):
    if str(p) in sys.path:
        sys.path.remove(str(p))
    sys.path.insert(0, str(p))

import data_utils as du  # noqa: E402
from map_real import reconstruct_map_real  # noqa: E402
from map_tv_jtv import joint_tv_prior, l2_amplitude_prior, tv_prior  # noqa: E402
from TLI_2D_forward import dpc_from_phase_sino  # noqa: E402
from scenario_utils import img_to_vec, make_reconstruction_mask, make_sparse_angle_indices, vec_to_img  # noqa: E402
from run_scenarios import (  # noqa: E402
    ALPHA_DELTA,
    ALPHA_EPS,
    ALPHA_JOINT,
    ALPHA_MU,
    LAMBDA0,
    N_ITER,
    N_SPARSE_ANGLES,
    N_SPARSE_STEPS,
    TV_BETA,
    sparse_step_indices,
)

SCENARIOS = ("full", "sparse_angle", "sparse_step", "combined")
CHANNELS = ("mu", "delta", "eps")


def scenario_indices(n_angles_full: int, n_phase_full: int, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    full_angles = np.arange(n_angles_full)
    full_steps = np.arange(n_phase_full)
    if scenario == "full":
        return full_angles, full_steps
    if scenario == "sparse_angle":
        return make_sparse_angle_indices(n_angles_full, N_SPARSE_ANGLES), full_steps
    if scenario == "sparse_step":
        return full_angles, sparse_step_indices(n_phase_full, N_SPARSE_STEPS)
    if scenario == "combined":
        return (make_sparse_angle_indices(n_angles_full, N_SPARSE_ANGLES),
                sparse_step_indices(n_phase_full, N_SPARSE_STEPS))
    raise ValueError(f"scenario={scenario!r} must be one of {SCENARIOS}")


# ============================================================
# Problem setup
# ============================================================

def setup_problem(
    scenario: str,
    alpha_joint: Optional[float] = None,
    support_R: float = du.SUPPORT_R,
    tv_beta: Optional[float] = None,
    alpha_mu: Optional[float] = None,
    alpha_delta: Optional[float] = None,
    alpha_eps: Optional[float] = None,
    lambda0: Optional[float] = None,
    projector_interp: Optional[str] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> Dict:
    """Data, sigma, field_scale, projector, MAP initialization, mask, and the
    posterior's potential function.

    Every regularization parameter is overridable so the same code can be run
    at a different operating point (e.g. re-tuned alphas after a tv_beta
    change) without editing run_scenarios.py.
    """
    device_t = torch.device(device)
    N = du.N_RECON

    # ----------------------------------------------------------
    # 1. Data, and the global field_scale from the FULL scan
    # ----------------------------------------------------------
    data = du.load_real_intensities()
    n_angles_full = data["I_meas"].shape[0]
    n_phase_full = data["I_meas"].shape[1]
    L_phys = du.geometry_for(data["I_ref"].shape[-1])

    if verbose:
        print("\nEstimating field_scale once from the full scan...")
    full_sinos = du.retrieve_and_correct_sinograms(data["I_meas"], data["I_ref"], verbose=verbose)
    full_beta_deg = du.make_beta_deg(data["angles"], n_angles_full)
    full_mu_fbp = du.astra_fbp_real(full_sinos["T"], full_beta_deg, L_phys)
    full_delta_fbp = du.astra_fbp_real(full_sinos["P"], full_beta_deg, L_phys)
    full_eps_fbp = du.astra_fbp_real(full_sinos["D"], full_beta_deg, L_phys)
    field_scale = {
        "mu": du.robust_scale(full_mu_fbp),
        "delta": du.robust_scale(full_delta_fbp),
        "eps": du.robust_scale(full_eps_fbp),
    }
    sigma_I = du.estimate_sigma_I(data["I_ref"])
    if verbose:
        print(f"  field_scale: {field_scale}")
        print(f"  sigma_I={sigma_I:.4e}")

    # ----------------------------------------------------------
    # 2. Scenario-specific sinograms, sigma and projector
    # ----------------------------------------------------------
    angle_idx, step_idx = scenario_indices(n_angles_full, n_phase_full, scenario)
    if verbose:
        print(f"\nScenario: {scenario} (n_angles={len(angle_idx)}, n_phase={len(step_idx)})")

    # sigma follows the scenario's stepping: fewer steps retrieve noisier
    # sinograms, and the likelihood has to know that (see sigma_for_stepping).
    sigma_ff = du.sigma_for_stepping(data["I_meas"], data["I_ref"], step_idx)
    sigma = {"mu": sigma_ff["sigma_T"], "delta": sigma_ff["sigma_DPC"], "eps": sigma_ff["sigma_D"]}
    if verbose:
        print(f"  sigma ({len(step_idx)} phase steps): {sigma}")

    I_meas = data["I_meas"][angle_idx][:, step_idx, :]
    I_ref = data["I_ref"][angle_idx][:, step_idx, :]
    sinos = du.retrieve_and_correct_sinograms(I_meas, I_ref, verbose=verbose)
    T_real, DPC_real, D_real = sinos["T"], sinos["DPC"], sinos["D"]

    n_angles, n_det_eff = T_real.shape
    beta_deg = du.make_beta_deg(data["angles"][angle_idx], n_angles)
    # Resolve the kernel BEFORE building, and never pass None onward: the
    # two data_utils.py variants differ (one defines PROJECTOR_INTERP and
    # treats interp=None as "use it", the other hardcodes "nearest" as the
    # default and forwards whatever it gets straight to radon_fanbeam, which
    # rejects None). Resolving here makes this file work with either.
    resolved_interp = (projector_interp if projector_interp is not None
                       else getattr(du, "PROJECTOR_INTERP", "nearest"))
    projector, det_spacing = du.build_projector(n_det_eff, L_phys, beta_deg, device_t, dtype,
                                                interp=resolved_interp)
    if verbose:
        print(f"  projector interpolation: {resolved_interp}")

    T_obs_t = torch.as_tensor(T_real, dtype=dtype, device=device_t)
    DPC_obs_t = torch.as_tensor(DPC_real, dtype=dtype, device=device_t)
    D_obs_t = torch.as_tensor(D_real, dtype=dtype, device=device_t)

    # ----------------------------------------------------------
    # 3. MAP at this prior's own alpha_joint: the NUTS initialization
    # ----------------------------------------------------------
    resolved_alpha_joint = ALPHA_JOINT if alpha_joint is None else alpha_joint
    resolved_tv_beta = TV_BETA if tv_beta is None else float(tv_beta)
    a_mu = ALPHA_MU if alpha_mu is None else float(alpha_mu)
    a_delta = ALPHA_DELTA if alpha_delta is None else float(alpha_delta)
    a_eps = ALPHA_EPS if alpha_eps is None else float(alpha_eps)
    resolved_lambda0 = LAMBDA0 if lambda0 is None else float(lambda0)
    prior_tag = "tv" if resolved_alpha_joint == 0 else "jtv"

    if verbose:
        print(f"\nPrior: {prior_tag} (alpha_mu={a_mu:g}, alpha_delta={a_delta:g}, "
              f"alpha_eps={a_eps:g}, alpha_joint={resolved_alpha_joint:g}, "
              f"tv_beta={resolved_tv_beta:g}, lambda0={resolved_lambda0:g})")
        print("Computing MAP (zero-init) for NUTS initialization...")
    map_out = reconstruct_map_real(
        T_real, DPC_real, D_real, projector, det_spacing, N,
        sigma["mu"], sigma["delta"], sigma["eps"],
        alpha_mu=a_mu, alpha_delta=a_delta, alpha_eps=a_eps, alpha_joint=resolved_alpha_joint,
        field_scale_mu=field_scale["mu"], field_scale_delta=field_scale["delta"],
        field_scale_eps=field_scale["eps"],
        tv_beta=resolved_tv_beta, lambda0=resolved_lambda0, n_iter=N_ITER, support_R=support_R,
        device=device_t, dtype=dtype,
    )

    # ----------------------------------------------------------
    # 4. Support mask and the normalized MAP (z = x/field_scale)
    # ----------------------------------------------------------
    mask = make_reconstruction_mask(N, R=support_R, device=device_t)
    n_active = int(mask.sum().item())
    dim = 3 * n_active
    if verbose:
        print(f"\nSupport mask: R={support_R}, active pixels={n_active}/{N * N}, sampling dim={dim}")

    def to_z(img: np.ndarray, scale: float) -> torch.Tensor:
        return img_to_vec(torch.as_tensor(img, dtype=dtype, device=device_t), mask) / scale

    z0 = torch.cat([to_z(map_out[ch], field_scale[ch]) for ch in CHANNELS], dim=0)

    # ----------------------------------------------------------
    # 5. Potential: negative log posterior, on the normalized fields --
    #    term-for-term identical to reconstruct_map_real's loss.
    # ----------------------------------------------------------
    def potential(params):
        z = params["z"]
        z_mu = vec_to_img(z[:n_active], mask)
        z_delta = vec_to_img(z[n_active:2 * n_active], mask)
        z_eps = vec_to_img(z[2 * n_active:], mask)

        mu = field_scale["mu"] * z_mu
        delta = field_scale["delta"] * z_delta
        eps = field_scale["eps"] * z_eps

        T_pred = projector.make_sinogram(mu, return_physical=True)
        P_pred = projector.make_sinogram(delta, return_physical=True)
        DPC_pred = -dpc_from_phase_sino(P_pred, det_spacing=det_spacing)
        D_pred = projector.make_sinogram(eps, return_physical=True)

        loss_data = (
            0.5 / sigma["mu"] ** 2 * torch.sum((T_pred - T_obs_t) ** 2)
            + 0.5 / sigma["delta"] ** 2 * torch.sum((DPC_pred - DPC_obs_t) ** 2)
            + 0.5 / sigma["eps"] ** 2 * torch.sum((D_pred - D_obs_t) ** 2)
        )
        prior_cw = (
            a_mu * tv_prior(z_mu, resolved_tv_beta)
            + a_delta * tv_prior(z_delta, resolved_tv_beta)
            + a_eps * tv_prior(z_eps, resolved_tv_beta)
        )
        prior_jtv = resolved_alpha_joint * joint_tv_prior(z_mu, z_delta, z_eps, resolved_tv_beta)
        prior_amp = l2_amplitude_prior(z_mu, z_delta, z_eps, resolved_lambda0)
        return loss_data + prior_cw + prior_jtv + prior_amp

    return {
        "potential": potential,
        "z0": z0,
        "map_out": map_out,
        "field_scale": field_scale,
        "sigma": sigma,
        "sigma_I": sigma_I,
        "det_spacing": det_spacing,
        "projector": projector,
        "mask": mask,
        "n_active": n_active,
        "dim": dim,
        "N": N,
        "n_angles": int(n_angles),
        "n_phase": int(len(step_idx)),
        "angle_idx": angle_idx,
        "step_idx": step_idx,
        "alpha_mu": a_mu,
        "alpha_delta": a_delta,
        "alpha_eps": a_eps,
        "alpha_joint": resolved_alpha_joint,
        "tv_beta": resolved_tv_beta,
        "lambda0": resolved_lambda0,
        "projector_interp": resolved_interp,
        "prior_tag": prior_tag,
        "support_R": support_R,
        "device": device_t,
        "dtype": dtype,
    }


# ============================================================
# Main NUTS routine
# ============================================================

def run_nuts_real(
    saving_path: Path,
    scenario: str,
    alpha_joint: Optional[float] = None,
    support_R: float = du.SUPPORT_R,
    tv_beta: Optional[float] = None,
    alpha_mu: Optional[float] = None,
    alpha_delta: Optional[float] = None,
    alpha_eps: Optional[float] = None,
    lambda0: Optional[float] = None,
    projector_interp: Optional[str] = None,
    init_pickle: Optional[str] = None,
    num_chains: int = 2,
    num_samples: int = 200,
    warmup_steps: int = 200,
    max_tree_depth: int = 6,
    step_size: float = 1e-2,
    adapt_step_size: bool = True,
    adapt_mass_matrix: bool = True,
    jitter_scale: float = 1e-3,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict:
    prob = setup_problem(scenario, alpha_joint=alpha_joint, support_R=support_R, tv_beta=tv_beta,
                         alpha_mu=alpha_mu, alpha_delta=alpha_delta, alpha_eps=alpha_eps,
                         lambda0=lambda0, projector_interp=projector_interp,
                         device=device, dtype=dtype)
    potential, z0 = prob["potential"], prob["z0"]
    n_active, dim = prob["n_active"], prob["dim"]

    with torch.no_grad():
        print(f"\nPotential at MAP initialization: {potential({'z': z0}).item():.6e}")

    # ----------------------------------------------------------
    # NUTS chains. Jitter is drawn per chain, so the chains start from
    # genuinely distinct points and R-hat measures what it should.
    # ----------------------------------------------------------
    # Chain continuation. Restarting from the MAP re-traverses a transient
    # already paid for: delta relaxes with tau ~1265 samples here, so a run
    # that ended mid-relaxation is far closer to equilibrium than the mode is.
    # Chain i resumes from chain i's own last sample, keeping the chains
    # independent so R-hat still compares genuinely separate trajectories.
    # Step size is re-adapted during warmup rather than carried over.
    resume = None
    if init_pickle is not None:
        with open(init_pickle, "rb") as f:
            prev = pickle.load(f)
        prev_meta = prev["meta"]
        if int(prev_meta["n_active"]) != n_active:
            raise ValueError(f"init_pickle n_active={prev_meta['n_active']} != {n_active}")
        resume = prev["samples"]["z_grouped"]
        print(f"\nResuming from {init_pickle}: {resume.shape[0]} chain(s) x {resume.shape[1]} samples")
        for i, ch in enumerate(CHANNELS):
            sl = slice(i * n_active, (i + 1) * n_active)
            print(f"  {ch:5s}: ||z|| MAP={float(z0[sl].norm()):8.2f}  ->  resume="
                  + "/".join(f"{float(resume[c, -1, sl].norm()):.2f}" for c in range(resume.shape[0])))

    all_samples = []
    chain_times = []
    for chain in range(num_chains):
        print(f"\nRunning chain {chain + 1}/{num_chains}...")
        start = time.time()
        if resume is None:
            init = {"z": z0 + jitter_scale * torch.randn_like(z0)}
        else:
            src = resume[min(chain, resume.shape[0] - 1), -1]
            init = {"z": src.to(dtype=prob["dtype"], device=prob["device"]).clone()}

        kernel = NUTS(
            potential_fn=potential,
            max_tree_depth=max_tree_depth,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            step_size=step_size,
        )
        mcmc = MCMC(kernel=kernel, num_chains=1, num_samples=num_samples,
                    warmup_steps=warmup_steps, initial_params=init)
        mcmc.run()

        runtime = time.time() - start
        chain_times.append(runtime)
        print(f"Chain {chain + 1} runtime: {runtime / 60:.2f} min")
        all_samples.append(mcmc.get_samples()["z"].unsqueeze(0))

    total_runtime = float(sum(chain_times))
    print(f"\nTotal runtime: {total_runtime / 60:.2f} min")

    samples_grouped = torch.cat(all_samples, dim=0).detach().cpu()
    samples_flat = samples_grouped.reshape(-1, dim)

    # ----------------------------------------------------------
    # Save (no "true" key -- real data has no ground truth)
    # ----------------------------------------------------------
    map_out, field_scale, sigma = prob["map_out"], prob["field_scale"], prob["sigma"]
    stat_dict = {
        "samples": {"z": samples_flat, "z_grouped": samples_grouped},
        "diagnostics": {"chain_times": chain_times, "total_runtime": total_runtime},
        "meta": {
            "data_source": "real",
            "scenario": scenario,
            "prior_tag": prob["prior_tag"],
            "N": prob["N"],
            "dim": dim,
            "n_active": n_active,
            "support_R": support_R,
            "n_angles": prob["n_angles"],
            "n_phase": prob["n_phase"],
            "angle_idx": prob["angle_idx"].astype(int),
            "step_idx": prob["step_idx"].astype(int),
            "field_scale_mu": field_scale["mu"],
            "field_scale_delta": field_scale["delta"],
            "field_scale_eps": field_scale["eps"],
            "sigma_I": prob["sigma_I"],
            "sigma_T": sigma["mu"],
            "sigma_DPC": sigma["delta"],
            "sigma_D": sigma["eps"],
            "alpha_mu": prob["alpha_mu"],
            "alpha_delta": prob["alpha_delta"],
            "alpha_eps": prob["alpha_eps"],
            "alpha_joint": prob["alpha_joint"],
            "tv_beta": prob["tv_beta"],
            "lambda0": prob["lambda0"],
            "projector_interp": prob["projector_interp"],
            "init_pickle": init_pickle,
            "device": str(prob["device"]),
            "step_size": step_size,
            "adapt_step_size": adapt_step_size,
            "adapt_mass_matrix": adapt_mass_matrix,
            "max_tree_depth": max_tree_depth,
            "num_samples": num_samples,
            "warmup_steps": warmup_steps,
            "num_chains": num_chains,
        },
        "map": {ch: map_out[ch] for ch in CHANNELS},
    }

    saving_path.parent.mkdir(parents=True, exist_ok=True)
    with open(saving_path, "wb") as f:
        pickle.dump(stat_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved samples to: {saving_path}")

    # ----------------------------------------------------------
    # Cheap diagnostics
    # ----------------------------------------------------------
    if num_chains > 1:
        diag_dim = min(2000, dim)
        idx_diag = torch.randperm(dim)[:diag_dim]
        samples_diag = samples_grouped[:, :, idx_diag]
        ess = effective_sample_size(samples_diag)
        rhat = gelman_rubin(samples_diag)
        print("\nSubset diagnostics:")
        print(f"  diagnostic dims: {diag_dim}/{dim}")
        print(f"  min ESS:      {ess.min().item():.1f}")
        print(f"  median ESS:   {ess.median().item():.1f}")
        print(f"  max R-hat:    {rhat.max().item():.4f}")
        print(f"  median R-hat: {rhat.median().item():.4f}")
    else:
        print("\nSingle chain: skipping R-hat/ESS diagnostics.")

    return stat_dict


def none_or_float(value):
    if value.lower() in ("none", "null", "auto"):
        return None
    return float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="full", choices=SCENARIOS)
    parser.add_argument("--alpha_joint", type=none_or_float, default=None,
                         help="Override ALPHA_JOINT; 0 for TV-only, omit for the locked-in JTV value.")
    parser.add_argument("--alpha_mu", type=float, default=None,
                         help="Override run_scenarios.ALPHA_MU (MAP init AND sampled potential).")
    parser.add_argument("--alpha_delta", type=float, default=None)
    parser.add_argument("--alpha_eps", type=float, default=None)
    parser.add_argument("--tv_beta", type=float, default=None)
    parser.add_argument("--lambda0", type=float, default=None)
    parser.add_argument("--support_R", type=float, default=du.SUPPORT_R)
    parser.add_argument("--projector", type=str, default=None, choices=("bilinear", "nearest"),
                         help="Override data_utils.PROJECTOR_INTERP for this run.")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu",
                         help="Defaults to CUDA when available.")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--saving_path", type=str, default=None)
    parser.add_argument("--init_pickle", type=str, default=None,
                         help="Resume each chain from the corresponding chain's last sample in a "
                              "previous run's pickle, instead of from the MAP. Use to extend a run "
                              "that ended mid-relaxation.")

    parser.add_argument("--num_chains", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_tree_depth", type=int, default=6)
    parser.add_argument("--step_size", type=none_or_float, default=1e-2)
    parser.add_argument("--jitter_scale", type=float, default=1e-3)
    parser.add_argument("--adapt_step_size", dest="adapt_step_size", action="store_true", default=True)
    parser.add_argument("--no_adapt_step_size", dest="adapt_step_size", action="store_false")
    parser.add_argument("--adapt_mass_matrix", dest="adapt_mass_matrix", action="store_true", default=True)
    parser.add_argument("--no_adapt_mass_matrix", dest="adapt_mass_matrix", action="store_false")

    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    print(f"Using device: {args.device}")

    resolved_alpha_joint = ALPHA_JOINT if args.alpha_joint is None else args.alpha_joint
    prior_tag = "tv" if resolved_alpha_joint == 0 else "jtv"
    saving_path = (Path(args.saving_path) if args.saving_path
                   else BASE_DIR / "obs" / "uq" / args.scenario / prior_tag / "nuts_samples.pickle")

    run_nuts_real(
        saving_path=saving_path,
        scenario=args.scenario,
        alpha_joint=args.alpha_joint,
        support_R=args.support_R,
        tv_beta=args.tv_beta,
        alpha_mu=args.alpha_mu,
        alpha_delta=args.alpha_delta,
        alpha_eps=args.alpha_eps,
        lambda0=args.lambda0,
        projector_interp=args.projector,
        init_pickle=args.init_pickle,
        num_chains=args.num_chains,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        max_tree_depth=args.max_tree_depth,
        step_size=args.step_size,
        adapt_step_size=args.adapt_step_size,
        adapt_mass_matrix=args.adapt_mass_matrix,
        jitter_scale=args.jitter_scale,
        device=args.device,
        dtype=dtype,
    )
