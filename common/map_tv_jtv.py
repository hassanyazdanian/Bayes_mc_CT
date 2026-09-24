"""
MAP reconstruction of (mu, delta, eps) from retrieved sinograms, with either
independent channel-wise TV or joint-TV (JTV) priors -- matching the
posterior in the paper (Sec. II.B-II.C, eqs. 15-23).

TV(alpha_joint=0) and JTV(alpha_joint>0) share the same optimizer; only the
prior term differs.

Likelihood is on T/DPC/D, as for the experimental data (real_data/map_real.py):
delta enters through DPC = -d/du radon(delta), not through the integrated
phase P, which is only used for the FBP baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import LBFGS
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from scenario_utils import img_to_vec, make_reconstruction_mask, vec_to_img  # noqa: E402
from TLI_2D_forward import dpc_from_phase_sino, radon_fanbeam  # noqa: E402


# -----------------------------------------------------------------------------
# Priors
# -----------------------------------------------------------------------------

def tv_prior(x: torch.Tensor, beta: float = 1e-4) -> torch.Tensor:
    """Smoothed isotropic TV: sum sqrt(dx^2 + dy^2 + beta^2)."""
    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]
    dx_c = dx[:-1, :]
    dy_c = dy[:, :-1]
    return torch.sum(torch.sqrt(dx_c ** 2 + dy_c ** 2 + beta ** 2))


def joint_tv_prior(mu: torch.Tensor, delta: torch.Tensor, eps: torch.Tensor, beta: float = 1e-4) -> torch.Tensor:
    """Smoothed 3-channel joint TV: sum sqrt(|grad mu|^2 + |grad delta|^2 + |grad eps|^2 + beta^2)."""
    def grads(x):
        dx = (x[:, 1:] - x[:, :-1])[:-1, :]
        dy = (x[1:, :] - x[:-1, :])[:, :-1]
        return dx, dy

    dx_mu, dy_mu = grads(mu)
    dx_delta, dy_delta = grads(delta)
    dx_eps, dy_eps = grads(eps)

    return torch.sum(torch.sqrt(
        dx_mu ** 2 + dy_mu ** 2
        + dx_delta ** 2 + dy_delta ** 2
        + dx_eps ** 2 + dy_eps ** 2
        + beta ** 2
    ))


def l2_amplitude_prior(mu: torch.Tensor, delta: torch.Tensor, eps: torch.Tensor, lambda0: float = 1e-6) -> torch.Tensor:
    """Weak Tikhonov amplitude-control term (makes the TV/JTV prior proper on R^3N)."""
    return 0.5 * lambda0 * (torch.sum(mu ** 2) + torch.sum(delta ** 2) + torch.sum(eps ** 2))


# -----------------------------------------------------------------------------
# MAP optimizer
# -----------------------------------------------------------------------------

def reconstruct_map(
    T_obs: np.ndarray,
    DPC_obs: np.ndarray,
    D_obs: np.ndarray,
    projector: radon_fanbeam,
    N: int,
    sigma_T: float,
    sigma_DPC: float,
    sigma_D: float,
    det_spacing: float,
    alpha_mu: float,
    alpha_delta: float,
    alpha_eps: float,
    alpha_joint: float,
    tv_beta: float = 1e-4,
    lambda0: float = 1e-6,
    n_iter: int = 150,
    init_mu: Optional[np.ndarray] = None,
    init_delta: Optional[np.ndarray] = None,
    init_eps: Optional[np.ndarray] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, np.ndarray]:
    """
    MAP estimate of (mu, delta, eps) from retrieved T/DPC/D sinograms.

    alpha_joint=0 gives the channel-wise TV prior; alpha_joint>0 adds the
    joint-TV coupling term on top of it (matching pi_cw / pi_jtv in the
    paper). alpha_mu/alpha_delta/alpha_eps are independent per-channel
    weights, matching the paper's own model (eqs. 17, 19) rather than a
    single shared value.

    Note: with alpha_joint=0, the three channels' data-fit and TV terms are
    fully separable (T depends only on mu, P only on delta, D only on eps,
    and there is no cross term), so alpha_mu/alpha_delta/alpha_eps can be
    tuned independently of each other -- e.g. via three separate 1D sweeps
    -- without needing a joint 3D grid search.
    """
    device_t = torch.device(device)
    T_obs_t = torch.as_tensor(T_obs, device=device_t, dtype=dtype)
    DPC_obs_t = torch.as_tensor(DPC_obs, device=device_t, dtype=dtype)
    D_obs_t = torch.as_tensor(D_obs, device=device_t, dtype=dtype)

    mask = make_reconstruction_mask(N, R=0.98, device=device_t)

    def init_vec(init):
        img = torch.zeros((N, N), device=device_t, dtype=dtype) if init is None else torch.as_tensor(init, device=device_t, dtype=dtype)
        return img_to_vec(img, mask).clone()

    raw_mu = torch.nn.Parameter(init_vec(init_mu))
    raw_delta = torch.nn.Parameter(init_vec(init_delta))
    raw_eps = torch.nn.Parameter(init_vec(init_eps))

    optimizer = LBFGS([raw_mu, raw_delta, raw_eps], max_iter=n_iter, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        mu = vec_to_img(raw_mu, mask)
        delta = vec_to_img(raw_delta, mask)
        eps = vec_to_img(raw_eps, mask)

        T_pred = projector.make_sinogram(mu, return_physical=True)
        DPC_pred = -dpc_from_phase_sino(projector.make_sinogram(delta, return_physical=True), det_spacing=det_spacing)
        D_pred = projector.make_sinogram(eps, return_physical=True)

        loss_data = (
            0.5 / sigma_T ** 2 * torch.sum((T_pred - T_obs_t) ** 2)
            + 0.5 / sigma_DPC ** 2 * torch.sum((DPC_pred - DPC_obs_t) ** 2)
            + 0.5 / sigma_D ** 2 * torch.sum((D_pred - D_obs_t) ** 2)
        )

        prior_cw = (
            alpha_mu * tv_prior(mu, tv_beta)
            + alpha_delta * tv_prior(delta, tv_beta)
            + alpha_eps * tv_prior(eps, tv_beta)
        )
        prior_jtv = alpha_joint * joint_tv_prior(mu, delta, eps, tv_beta)
        prior_amp = l2_amplitude_prior(mu, delta, eps, lambda0)

        loss = loss_data + prior_cw + prior_jtv + prior_amp
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        mu = vec_to_img(raw_mu, mask)
        delta = vec_to_img(raw_delta, mask)
        eps = vec_to_img(raw_eps, mask)
        T_pred = projector.make_sinogram(mu, return_physical=True)
        P_pred = projector.make_sinogram(delta, return_physical=True)
        D_pred = projector.make_sinogram(eps, return_physical=True)
        DPC_pred = -dpc_from_phase_sino(P_pred, det_spacing=det_spacing)

    return {
        "mu": mu.cpu().numpy(),
        "delta": delta.cpu().numpy(),
        "eps": eps.cpu().numpy(),
        "T_pred": T_pred.cpu().numpy(),
        "P_pred": P_pred.cpu().numpy(),
        "D_pred": D_pred.cpu().numpy(),
        "DPC_pred": DPC_pred.cpu().numpy(),
        "history": history,
    }


# -----------------------------------------------------------------------------
# Single-channel MAP optimizer (for tuning alpha_mu/alpha_delta/alpha_eps)
# -----------------------------------------------------------------------------

def reconstruct_single_channel(
    obs: np.ndarray,
    projector: radon_fanbeam,
    N: int,
    sigma: float,
    alpha_tv: float,
    tv_beta: float = 1e-4,
    lambda0: float = 1e-6,
    n_iter: int = 150,
    init: Optional[np.ndarray] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    dpc_det_spacing: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Single-channel TV MAP reconstruction (mu, delta, or eps in isolation),
    with its own dedicated LBFGS optimizer. Pass dpc_det_spacing for delta:
    obs is then the DPC sinogram and the prediction is -d/du radon(img).

    At alpha_joint=0 the 3-channel objective in reconstruct_map() is
    mathematically separable (no cross terms between channels), but running
    all three through ONE LBFGS instance still couples their optimization
    *dynamics*: L-BFGS's quasi-Newton memory and strong-Wolfe line search
    operate on the full concatenated parameter vector, so a channel's
    trajectory is affected by what the other channels are doing even though
    the loss itself has no cross terms. Verified empirically: running
    mu/delta/eps jointly with mismatched per-channel alphas gives a
    genuinely different (usually worse) result for a given channel than
    running that channel alone at the same alpha. Use this function -- not
    reconstruct_map with alpha_joint=0 -- when independently tuning
    alpha_mu/alpha_delta/alpha_eps.
    """
    device_t = torch.device(device)
    obs_t = torch.as_tensor(obs, device=device_t, dtype=dtype)
    mask = make_reconstruction_mask(N, R=0.98, device=device_t)

    img0 = torch.zeros((N, N), device=device_t, dtype=dtype) if init is None else torch.as_tensor(init, device=device_t, dtype=dtype)
    raw = torch.nn.Parameter(img_to_vec(img0, mask).clone())

    optimizer = LBFGS([raw], max_iter=n_iter, line_search_fn="strong_wolfe")
    history = []

    def predict(img):
        sino = projector.make_sinogram(img, return_physical=True)
        return sino if dpc_det_spacing is None else -dpc_from_phase_sino(sino, det_spacing=dpc_det_spacing)

    def closure():
        optimizer.zero_grad()
        img = vec_to_img(raw, mask)
        pred = predict(img)
        loss_data = 0.5 / sigma ** 2 * torch.sum((pred - obs_t) ** 2)
        prior = alpha_tv * tv_prior(img, tv_beta) + 0.5 * lambda0 * torch.sum(img ** 2)
        loss = loss_data + prior
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        img = vec_to_img(raw, mask)
        pred = predict(img)

    return {"img": img.cpu().numpy(), "pred": pred.cpu().numpy(), "history": history}


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_metrics(x_true: np.ndarray, x_est: np.ndarray) -> Tuple[float, float, float]:
    """Relative error, SSIM, and PSNR, with both images normalized to the true image's range."""
    x_true = np.asarray(x_true, dtype=np.float64)
    x_est = np.asarray(x_est, dtype=np.float64)

    rel_err = np.linalg.norm(x_true - x_est) / (np.linalg.norm(x_true) + 1e-12)

    true_min = x_true.min()
    true_range = x_true.max() - true_min + 1e-12
    x_true_n = (x_true - true_min) / true_range
    x_est_n = np.clip((x_est - true_min) / true_range, 0.0, 1.0)

    return rel_err, ssim(x_true_n, x_est_n, data_range=1.0), psnr(x_true_n, x_est_n, data_range=1.0)
