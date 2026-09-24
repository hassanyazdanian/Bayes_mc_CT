"""
NUTS posterior sampling for synthetic (mu, delta, eps), on the same smoothed
joint-TV posterior used for the MAP pipeline (common/map_tv_jtv.py) and the
four limited-data scenarios (recon/run_scenarios.py).

Adapted from real_data/TV/Sampling_joint_TV_sparse_real_data.py. Differences
from that reference, and why:

  - No real-data preprocessing (background-column offset subtraction, T sign
    flip): synthetic sinograms from scenario_utils.simulate_and_retrieve are
    already in the forward model's native convention.

  - No extra real-data support crop (that script's support_R=0.50, a small
    disk sized to the physical sample holder). Sampling here uses the same
    R=0.98 circular FOV mask already used everywhere in this synthetic
    pipeline (radon_fanbeam.set_view_angles' own default, and
    map_tv_jtv.reconstruct_map's optimization domain) -- not a smaller,
    experiment-specific region. For "inclusion"/"multicontrast" the phantom
    content extends close to that circle, so a tighter crop would cut off
    real structure.

  - Likelihood is defined on T/DPC/D, as for the experimental data and in
    reconstruct_map / run_scenarios.py.

  - Data come from run_scenarios.simulate_data (512^2 phantom, bilinear
    projector) and the potential uses the 256^2 run_scenarios.RECON_INTERP
    (nearest) projector, exactly as in run_scenarios.py, so the sampled
    posterior is the one whose MAP is reported there (see the comment on
    SIM_INTERP/RECON_INTERP for why they differ).

  - The potential is evaluated on the *physical* fields mu = field_scale_mu *
    z_mu (etc.), not on z directly. TV is homogeneous of degree 1, so
    tv_prior(z) = tv_prior(mu) / field_scale_mu -- evaluating priors on z
    directly (as the reference script does) implicitly rescales alpha by
    1/field_scale and would need its own independent tuning. Evaluating on
    the physical fields keeps this script's potential mode exactly
    coincident with reconstruct_map's already-tuned MAP optimum, so the
    locked-in ALPHA_MU/ALPHA_DELTA/ALPHA_EPS/ALPHA_JOINT (recon/run_scenarios.py)
    carry over unchanged. z is only a reparameterization for HMC step-size
    conditioning.

  - MAP initialization and field scales come from running reconstruct_map
    in-process (common/map_tv_jtv.py) rather than loading a saved MAP .h5 --
    there is no separate saved MAP artifact per synthetic scenario to load.

  - --alpha_joint lets this script sample the TV-only posterior (alpha_joint=0)
    or the JTV posterior (omit, uses the locked-in ALPHA_JOINT) with the exact
    same procedure: a single zero-init reconstruct_map call at that prior's own
    alpha_joint gives both its MAP initialization and its field scales, so
    neither run is warm-started from or otherwise privileged by the other --
    a fair TV-vs-JTV comparison.

Usage:
    python nuts_synthetic.py --phantom multicontrast --scenario combined \
        --num_chains 2 --num_samples 200 --warmup_steps 200 --max_tree_depth 6
    python nuts_synthetic.py --phantom multicontrast --scenario combined --alpha_joint 0 \
        --num_chains 2 --num_samples 200 --warmup_steps 200 --max_tree_depth 6

Quick smoke test:
    python nuts_synthetic.py --phantom inclusion --scenario full \
        --num_chains 1 --num_samples 5 --warmup_steps 2 --max_tree_depth 3

Output: uq/obs/<phantom>/<scenario>/<tv|jtv>/nuts_samples.pickle
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from pyro.infer.mcmc import MCMC, NUTS
from pyro.ops.stats import effective_sample_size, gelman_rubin

BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
RECON_DIR = BASE_DIR / "recon"
for p in (BASE_DIR, COMMON_DIR, RECON_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from map_tv_jtv import joint_tv_prior, l2_amplitude_prior, reconstruct_map, tv_prior  # noqa: E402
from run_scenarios import (  # noqa: E402
    ALPHA_DELTA,
    ALPHA_EPS,
    ALPHA_JOINT,
    ALPHA_MU,
    LAMBDA0,
    N_ITER,
    NOISE_LEVEL,
    PHANTOM_SUPERSAMPLE,
    PHANTOM_TARGET_DPC_MAX,
    RECON_INTERP,
    SIM_INTERP,
    N_SPARSE_ANGLES,
    N_SPARSE_PHASE,
    TV_BETA,
    build_projector,
    det_spacing_of,
    load_dataset,
    override_phantom,
    simulate_data,
)
from scenario_utils import (  # noqa: E402
    img_to_vec,
    make_reconstruction_mask,
    make_sparse_angle_indices,
    vec_to_img,
)
from TLI_2D_forward import dpc_from_phase_sino  # noqa: E402

SCENARIOS = ("full", "sparse_angle", "sparse_step", "combined")


def scenario_sizes(d: Dict, scenario: str) -> Tuple[int, int]:
    if scenario == "full":
        return d["n_angles_full"], d["n_phase_full"]
    if scenario == "sparse_angle":
        return N_SPARSE_ANGLES, d["n_phase_full"]
    if scenario == "sparse_step":
        return d["n_angles_full"], N_SPARSE_PHASE
    if scenario == "combined":
        return N_SPARSE_ANGLES, N_SPARSE_PHASE
    raise ValueError(f"scenario={scenario!r} must be one of {SCENARIOS}")


# ============================================================
# Main NUTS routine
# ============================================================

def run_nuts_synthetic(
    saving_path: Path,
    phantom_name: str,
    scenario: str,
    alpha_joint: Optional[float] = None,
    tv_beta: Optional[float] = None,
    alpha_mu: Optional[float] = None,
    alpha_delta: Optional[float] = None,
    alpha_eps: Optional[float] = None,
    lambda0: Optional[float] = None,
    support_R: float = 0.98,
    num_chains: int = 2,
    num_samples: int = 200,
    warmup_steps: int = 200,
    max_tree_depth: int = 6,
    step_size: float = 1e-2,
    adapt_step_size: bool = True,
    adapt_mass_matrix: bool = True,
    jitter_scale: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict:
    device_t = torch.device(device)

    # ----------------------------------------------------------
    # 1. Phantom, scenario data, and retrieved noisy sinograms
    # ----------------------------------------------------------
    d = load_dataset(BASE_DIR / "obs" / "synthetic_data.h5")
    d = override_phantom(d, phantom_name, PHANTOM_SUPERSAMPLE, PHANTOM_TARGET_DPC_MAX, device_t, dtype)

    n_angles, n_phase = scenario_sizes(d, scenario)
    angle_idx = make_sparse_angle_indices(d["n_angles_full"], n_angles)
    det_spacing = det_spacing_of(d)
    noisy, sigma = simulate_data(d, angle_idx, n_phase, device_t, dtype)
    projector = build_projector(d, angle_idx, device_t, dtype, interp=RECON_INTERP)

    print(f"\nPhantom: {phantom_name}, scenario: {scenario} (n_angles={n_angles}, n_phase={n_phase})")
    print(f"  device={device_t}  simulate={d['N_sim']}^2 {SIM_INTERP}, reconstruct={d['N']}^2 {RECON_INTERP}, "
          f"likelihood T/DPC/D, noise={NOISE_LEVEL}")
    print(f"  sigma_T={sigma['T']:.4e}  sigma_DPC={sigma['DPC']:.4e}  sigma_D={sigma['D']:.4e}")

    T_obs_t = torch.as_tensor(noisy["T"], dtype=dtype, device=device_t)
    DPC_obs_t = torch.as_tensor(noisy["DPC"], dtype=dtype, device=device_t)
    D_obs_t = torch.as_tensor(noisy["D"], dtype=dtype, device=device_t)

    # ----------------------------------------------------------
    # 2. MAP estimate at this prior's own alpha_joint: NUTS
    #    initialization + field scales. Zero-init, not warm-started from
    #    any other prior's MAP -- TV (alpha_joint=0) and JTV get identical
    #    treatment, so neither is privileged in the comparison.
    # ----------------------------------------------------------
    # Unset overrides fall back to the locked-in run_scenarios.py values.
    resolved_alpha_joint = ALPHA_JOINT if alpha_joint is None else alpha_joint
    a_mu = ALPHA_MU if alpha_mu is None else alpha_mu
    a_delta = ALPHA_DELTA if alpha_delta is None else alpha_delta
    a_eps = ALPHA_EPS if alpha_eps is None else alpha_eps
    beta = TV_BETA if tv_beta is None else tv_beta
    lam0 = LAMBDA0 if lambda0 is None else lambda0
    prior_tag = "tv" if resolved_alpha_joint == 0 else "jtv"
    print(f"\nPrior: {prior_tag} (alpha_joint={resolved_alpha_joint})")
    print(f"  alpha_mu={a_mu:g} alpha_delta={a_delta:g} alpha_eps={a_eps:g} tv_beta={beta:g} lambda0={lam0:g}")
    print("Computing MAP (zero-init) for NUTS initialization...")
    map_out = reconstruct_map(
        noisy["T"], noisy["DPC"], noisy["D"], projector, d["N"],
        sigma["T"], sigma["DPC"], sigma["D"], det_spacing,
        alpha_mu=a_mu, alpha_delta=a_delta, alpha_eps=a_eps, alpha_joint=resolved_alpha_joint,
        tv_beta=beta, lambda0=lam0, n_iter=N_ITER, device=device_t, dtype=dtype,
    )

    field_scale = {
        ch: max(float(np.percentile(np.abs(map_out[ch]), 99)), 1e-12)
        for ch in ("mu", "delta", "eps")
    }
    print(f"Field scales (99th percentile |{prior_tag.upper()} MAP|):")
    for ch in ("mu", "delta", "eps"):
        print(f"  {ch}: {field_scale[ch]:.6e}")

    # ----------------------------------------------------------
    # 3. Support mask and masked/normalized MAP initialization
    # ----------------------------------------------------------
    N = d["N"]
    mask = make_reconstruction_mask(N, R=support_R, device=device_t)
    n_active = int(mask.sum().item())
    dim = 3 * n_active
    print(f"\nSupport mask: R={support_R}, active pixels={n_active}/{N*N}, sampling dim={dim}")

    def to_z(img: np.ndarray, scale: float) -> torch.Tensor:
        img_t = torch.as_tensor(img, dtype=dtype, device=device_t)
        return img_to_vec(img_t, mask) / scale

    z0 = torch.cat([to_z(map_out[ch], field_scale[ch]) for ch in ("mu", "delta", "eps")], dim=0)

    def unpack(z_flat: torch.Tensor):
        z_mu = vec_to_img(z_flat[:n_active], mask)
        z_delta = vec_to_img(z_flat[n_active:2 * n_active], mask)
        z_eps = vec_to_img(z_flat[2 * n_active:], mask)
        return z_mu, z_delta, z_eps

    # ----------------------------------------------------------
    # 4. Potential function: negative log posterior, on physical fields
    # ----------------------------------------------------------
    def potential(params):
        z_mu, z_delta, z_eps = unpack(params["z"])
        mu = field_scale["mu"] * z_mu
        delta = field_scale["delta"] * z_delta
        eps = field_scale["eps"] * z_eps

        T_pred = projector.make_sinogram(mu, return_physical=True)
        DPC_pred = -dpc_from_phase_sino(projector.make_sinogram(delta, return_physical=True), det_spacing=det_spacing)
        D_pred = projector.make_sinogram(eps, return_physical=True)

        loss_data = (
            0.5 / sigma["T"] ** 2 * torch.sum((T_pred - T_obs_t) ** 2)
            + 0.5 / sigma["DPC"] ** 2 * torch.sum((DPC_pred - DPC_obs_t) ** 2)
            + 0.5 / sigma["D"] ** 2 * torch.sum((D_pred - D_obs_t) ** 2)
        )
        prior_cw = (
            a_mu * tv_prior(mu, beta)
            + a_delta * tv_prior(delta, beta)
            + a_eps * tv_prior(eps, beta)
        )
        prior_jtv = resolved_alpha_joint * joint_tv_prior(mu, delta, eps, beta)
        prior_amp = l2_amplitude_prior(mu, delta, eps, lam0)

        return loss_data + prior_cw + prior_jtv + prior_amp

    with torch.no_grad():
        J0 = potential({"z": z0}).item()
    print(f"\nPotential at MAP initialization: {J0:.6e}")

    # ----------------------------------------------------------
    # 5. Run NUTS chains (independent jitter per chain)
    # ----------------------------------------------------------
    all_samples = []
    chain_times = []
    for chain in range(num_chains):
        print(f"\nRunning chain {chain + 1}/{num_chains}...")
        start = time.time()
        init = {"z": z0 + jitter_scale * torch.randn_like(z0)}

        nuts_kernel = NUTS(
            potential_fn=potential,
            max_tree_depth=max_tree_depth,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            step_size=step_size,
        )
        mcmc = MCMC(
            kernel=nuts_kernel,
            num_chains=1,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            initial_params=init,
        )
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
    # 6. Save
    # ----------------------------------------------------------
    stat_dict = {
        "samples": {"z": samples_flat, "z_grouped": samples_grouped},
        "diagnostics": {"chain_times": chain_times, "total_runtime": total_runtime},
        "meta": {
            "phantom_name": phantom_name,
            "scenario": scenario,
            "prior_tag": prior_tag,
            "N": N,
            "dim": dim,
            "n_active": n_active,
            "support_R": support_R,
            "n_angles": int(n_angles),
            "n_phase": int(n_phase),
            "angle_idx": angle_idx.astype(int),
            "field_scale_mu": field_scale["mu"],
            "field_scale_delta": field_scale["delta"],
            "field_scale_eps": field_scale["eps"],
            "sigma_T": sigma["T"],
            "sigma_DPC": sigma["DPC"],
            "sigma_D": sigma["D"],
            "likelihood": "T/DPC/D",
            "noise_level": dict(NOISE_LEVEL),
            "N_sim": d["N_sim"],
            "alpha_mu": a_mu,
            "alpha_delta": a_delta,
            "alpha_eps": a_eps,
            "alpha_joint": resolved_alpha_joint,
            "tv_beta": beta,
            "lambda0": lam0,
            "sim_interp": SIM_INTERP,
            "recon_interp": RECON_INTERP,
            "device": str(device_t),
            "step_size": step_size,
            "adapt_step_size": adapt_step_size,
            "adapt_mass_matrix": adapt_mass_matrix,
            "max_tree_depth": max_tree_depth,
            "num_samples": num_samples,
            "warmup_steps": warmup_steps,
            "num_chains": num_chains,
        },
        "map": {ch: map_out[ch] for ch in ("mu", "delta", "eps")},
        "true": {ch: d[ch] for ch in ("mu", "delta", "eps")},
    }

    saving_path.parent.mkdir(parents=True, exist_ok=True)
    with open(saving_path, "wb") as f:
        pickle.dump(stat_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved samples to: {saving_path}")

    # ----------------------------------------------------------
    # 7. Cheap diagnostics
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
    parser.add_argument("--phantom", type=str, default="inclusion")
    parser.add_argument("--scenario", type=str, default="full", choices=SCENARIOS)
    parser.add_argument("--alpha_joint", type=none_or_float, default=None,
                         help="Override ALPHA_JOINT; 0 for TV-only, omit for the locked-in JTV value.")
    parser.add_argument("--tv_beta", type=float, default=None, help="Override TV_BETA.")
    parser.add_argument("--alpha_mu", type=float, default=None, help="Override ALPHA_MU.")
    parser.add_argument("--alpha_delta", type=float, default=None, help="Override ALPHA_DELTA.")
    parser.add_argument("--alpha_eps", type=float, default=None, help="Override ALPHA_EPS.")
    parser.add_argument("--lambda0", type=float, default=None, help="Override LAMBDA0.")
    parser.add_argument("--support_R", type=float, default=0.98)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--saving_path", type=str, default=None)

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

    resolved_alpha_joint = ALPHA_JOINT if args.alpha_joint is None else args.alpha_joint
    prior_tag = "tv" if resolved_alpha_joint == 0 else "jtv"

    if args.saving_path is None:
        saving_path = BASE_DIR / "uq" / "obs" / args.phantom / args.scenario / prior_tag / "nuts_samples.pickle"
    else:
        saving_path = Path(args.saving_path)

    run_nuts_synthetic(
        saving_path=saving_path,
        phantom_name=args.phantom,
        scenario=args.scenario,
        alpha_joint=args.alpha_joint,
        tv_beta=args.tv_beta,
        alpha_mu=args.alpha_mu,
        alpha_delta=args.alpha_delta,
        alpha_eps=args.alpha_eps,
        lambda0=args.lambda0,
        support_R=args.support_R,
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
