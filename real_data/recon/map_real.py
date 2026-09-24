"""
MAP reconstruction (single-channel, for tuning; joint TV/JTV, for the main
run) on real Talbot-Lau data.

Differences from the synthetic common/map_tv_jtv.py, both deliberate:
  - Priors are evaluated on the *normalized* field z = x / field_scale, not
    the physical field directly. Synthetic map_tv_jtv.py evaluates on the
    physical field specifically to stay coincident with its own already-
    RE-tuned alpha values; there is no such established physical-space alpha
    to preserve here (no ground truth), so normalized-z evaluation --
    matching the real-data reference script -- is the more appropriate
    convention: it makes alpha comparable in scale to the reference script's
    own tuning and independent of each channel's raw physical magnitude.
  - Likelihood uses T/DPC/D (not T/P/D): the real-data reference script's
    validated choice for this dataset, kept as-is.

Reuses tv_prior/joint_tv_prior/l2_amplitude_prior from common/map_tv_jtv.py
and the mask/vec utilities from common/scenario_utils.py -- same math, just
called on normalized z here instead of the physical field.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.optim import LBFGS


BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR.parent / "common"
for _p in (COMMON_DIR, BASE_DIR, BASE_DIR / "recon", BASE_DIR / "uq"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from TLI_2D_forward import dpc_from_phase_sino, radon_fanbeam  # noqa: E402
from scenario_utils import img_to_vec, make_reconstruction_mask, vec_to_img  # noqa: E402
from map_tv_jtv import joint_tv_prior, l2_amplitude_prior, tv_prior  # noqa: E402


def make_dpc_predict(projector: radon_fanbeam, det_spacing: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def predict(img: torch.Tensor) -> torch.Tensor:
        P_pred = projector.make_sinogram(img, return_physical=True)
        return -dpc_from_phase_sino(P_pred, det_spacing=det_spacing)
    return predict


def make_direct_predict(projector: radon_fanbeam) -> Callable[[torch.Tensor], torch.Tensor]:
    def predict(img: torch.Tensor) -> torch.Tensor:
        return projector.make_sinogram(img, return_physical=True)
    return predict


# -----------------------------------------------------------------------------
# Single-channel optimizer (own dedicated LBFGS instance -- avoids the
# cross-channel L-BFGS coupling artifact documented in common/map_tv_jtv.py)
# -----------------------------------------------------------------------------

def reconstruct_single_channel_real(
    obs: np.ndarray,
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    N: int,
    sigma: float,
    alpha_tv: float,
    field_scale: float,
    tv_beta: float = 1e-4,
    lambda0: float = 1e-6,
    n_iter: int = 150,
    init: Optional[np.ndarray] = None,
    support_R: float = 0.5,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, np.ndarray]:
    device_t = torch.device(device)
    obs_t = torch.as_tensor(obs, device=device_t, dtype=dtype)
    mask = make_reconstruction_mask(N, R=support_R, device=device_t)

    z0_img = torch.zeros((N, N), device=device_t, dtype=dtype) if init is None \
        else torch.as_tensor(init / field_scale, device=device_t, dtype=dtype)
    raw = torch.nn.Parameter(img_to_vec(z0_img, mask).clone())

    optimizer = LBFGS([raw], max_iter=n_iter, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        z = vec_to_img(raw, mask)
        img = field_scale * z
        pred = predict_fn(img)
        loss_data = 0.5 / sigma ** 2 * torch.sum((pred - obs_t) ** 2)
        prior = alpha_tv * tv_prior(z, tv_beta) + 0.5 * lambda0 * torch.sum(z ** 2)
        loss = loss_data + prior
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        z = vec_to_img(raw, mask)
        img = field_scale * z
        pred = predict_fn(img)

    return {"img": img.cpu().numpy(), "pred": pred.cpu().numpy(), "history": history}


# -----------------------------------------------------------------------------
# Joint MAP optimizer (TV: alpha_joint=0, JTV: alpha_joint>0)
# -----------------------------------------------------------------------------

def reconstruct_map_real(
    T_obs: np.ndarray,
    DPC_obs: np.ndarray,
    D_obs: np.ndarray,
    projector: radon_fanbeam,
    det_spacing: float,
    N: int,
    sigma_T: float,
    sigma_DPC: float,
    sigma_D: float,
    alpha_mu: float,
    alpha_delta: float,
    alpha_eps: float,
    alpha_joint: float,
    field_scale_mu: float,
    field_scale_delta: float,
    field_scale_eps: float,
    tv_beta: float = 1e-4,
    lambda0: float = 1e-6,
    n_iter: int = 150,
    init_mu: Optional[np.ndarray] = None,
    init_delta: Optional[np.ndarray] = None,
    init_eps: Optional[np.ndarray] = None,
    support_R: float = 0.5,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, np.ndarray]:
    device_t = torch.device(device)
    T_obs_t = torch.as_tensor(T_obs, device=device_t, dtype=dtype)
    DPC_obs_t = torch.as_tensor(DPC_obs, device=device_t, dtype=dtype)
    D_obs_t = torch.as_tensor(D_obs, device=device_t, dtype=dtype)

    mask = make_reconstruction_mask(N, R=support_R, device=device_t)

    def init_vec(init, scale):
        img = torch.zeros((N, N), device=device_t, dtype=dtype) if init is None \
            else torch.as_tensor(init / scale, device=device_t, dtype=dtype)
        return img_to_vec(img, mask).clone()

    raw_mu = torch.nn.Parameter(init_vec(init_mu, field_scale_mu))
    raw_delta = torch.nn.Parameter(init_vec(init_delta, field_scale_delta))
    raw_eps = torch.nn.Parameter(init_vec(init_eps, field_scale_eps))

    optimizer = LBFGS([raw_mu, raw_delta, raw_eps], max_iter=n_iter, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        z_mu = vec_to_img(raw_mu, mask)
        z_delta = vec_to_img(raw_delta, mask)
        z_eps = vec_to_img(raw_eps, mask)

        mu = field_scale_mu * z_mu
        delta = field_scale_delta * z_delta
        eps = field_scale_eps * z_eps

        T_pred = projector.make_sinogram(mu, return_physical=True)
        P_pred = projector.make_sinogram(delta, return_physical=True)
        DPC_pred = -dpc_from_phase_sino(P_pred, det_spacing=det_spacing)
        D_pred = projector.make_sinogram(eps, return_physical=True)

        loss_data = (
            0.5 / sigma_T ** 2 * torch.sum((T_pred - T_obs_t) ** 2)
            + 0.5 / sigma_DPC ** 2 * torch.sum((DPC_pred - DPC_obs_t) ** 2)
            + 0.5 / sigma_D ** 2 * torch.sum((D_pred - D_obs_t) ** 2)
        )
        prior_cw = (
            alpha_mu * tv_prior(z_mu, tv_beta)
            + alpha_delta * tv_prior(z_delta, tv_beta)
            + alpha_eps * tv_prior(z_eps, tv_beta)
        )
        prior_jtv = alpha_joint * joint_tv_prior(z_mu, z_delta, z_eps, tv_beta)
        prior_amp = l2_amplitude_prior(z_mu, z_delta, z_eps, lambda0)

        loss = loss_data + prior_cw + prior_jtv + prior_amp
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        z_mu = vec_to_img(raw_mu, mask)
        z_delta = vec_to_img(raw_delta, mask)
        z_eps = vec_to_img(raw_eps, mask)
        mu = field_scale_mu * z_mu
        delta = field_scale_delta * z_delta
        eps = field_scale_eps * z_eps
        T_pred = projector.make_sinogram(mu, return_physical=True)
        P_pred = projector.make_sinogram(delta, return_physical=True)
        DPC_pred = -dpc_from_phase_sino(P_pred, det_spacing=det_spacing)
        D_pred = projector.make_sinogram(eps, return_physical=True)

    return {
        "mu": mu.cpu().numpy(), "delta": delta.cpu().numpy(), "eps": eps.cpu().numpy(),
        "T_pred": T_pred.cpu().numpy(), "DPC_pred": DPC_pred.cpu().numpy(), "D_pred": D_pred.cpu().numpy(),
        "history": history,
    }
