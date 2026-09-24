"""
Shared utilities for the limited-data scenario comparisons (full / sparse-angle
/ sparse-stepping / combined): angle subsetting, phase-stepping retrieval with
controlled post-retrieval noise, and the reconstruction-domain pixel mask used
by the MAP/JTV optimizer.

Noise model: independent Gaussian noise is added directly to the three
retrieved sinograms the likelihood is defined on (T, DPC, D), matching the
paper's Bayesian model (Sec. II.A: the likelihood is defined on the retrieved
sinograms, not on the raw intensities). The integrated phase P is then
obtained from the NOISY DPC by the same Fourier integration the retrieval
uses, exactly as for experimental data, so the FBP baseline for delta sees
integrated (correlated) DPC noise rather than independent noise of its own.
Retrieval itself runs on noiseless phase-stepping intensities resimulated at
exactly the scenario's n_phase (see simulate_and_retrieve), so it is
numerically exact; the injected noise level is scaled by
sqrt(n_phase_full / n_phase_used) to reflect the reduced averaging a real
reduced-stepping acquisition would suffer.

Phase steps are resimulated rather than subsetted from a denser stepping
curve: TLI_2D_forward.forward() samples phi_k = 2*pi*k/n_phase, and the
FFT-based harmonic extractor (ColumnPhaseSteppingProcessor.get_fft_coeffs)
assumes its input is evenly spaced over one full 2*pi period. A literal
subset of an n_phase_full=10 curve is only exactly evenly spaced when
n_sparse_phase divides 10 -- e.g. 4 of 10 has no exact subset (closest is
0/72/180/288 deg, not 0/90/180/270), and that spacing error leaks energy
between harmonics and silently corrupts T and D. A real reduced-stepping
scan would not subsample a denser curve either -- it takes fewer, evenly
spaced measurements from the start -- so resimulating is both correct and
physically faithful.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from phase_stepping import ColumnPhaseSteppingProcessor, PhaseSteppingConfig
from TLI_2D_forward import TLInterferometryForward2D, radon_fanbeam


# -----------------------------------------------------------------------------
# Angle subsetting
# -----------------------------------------------------------------------------

def make_sparse_angle_indices(n_angles: int, n_sparse_angles: int) -> np.ndarray:
    """Uniformly-spaced projection-angle indices out of `n_angles`."""
    if not (0 < n_sparse_angles <= n_angles):
        raise ValueError(f"n_sparse_angles={n_sparse_angles} must be in (0, {n_angles}].")
    idx = np.unique(np.linspace(0, n_angles - 1, n_sparse_angles, dtype=np.int64))
    return idx


# -----------------------------------------------------------------------------
# Retrieval with controlled post-retrieval noise
# -----------------------------------------------------------------------------

def simulate_and_retrieve(
    mu: np.ndarray,
    delta: np.ndarray,
    eps: np.ndarray,
    projector: radon_fanbeam,
    I0: float,
    vis: float,
    n_phase: int,
    n_phase_full: int,
    pixel_size: float,
    noise_level: Dict[str, float],
    bg: int = 5,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]:
    """
    Simulate noiseless phase-stepping intensities at exactly `n_phase` evenly
    spaced steps (for whatever view angles and grid `projector` is configured
    with -- mu/delta/eps must match its N_pix), retrieve T/DPC/D/P from them,
    then add independent Gaussian noise to T, DPC and D, each at its own level
    relative to the RMS of the noiseless sinogram (`noise_level` is a dict
    with keys "T","DPC","D"). The noisy P is the Fourier integral of the noisy
    DPC (see module docstring).

    Returns (noisy, sigma, clean): noisy/clean keyed by "T","DPC","D","P";
    sigma keyed by "T","DPC","D" (P has no white-noise sigma).
    """
    model = TLInterferometryForward2D(projector, I0=I0, vis=vis)
    out = model.forward(mu, delta, eps, n_phase=n_phase)
    # forward() returns (angle, detector, phase); retrieval wants (angle, phase, detector).
    I_meas = np.transpose(out.I_meas.detach().cpu().numpy(), (0, 2, 1)).astype(np.float32)
    I_ref = np.transpose(out.I_ref.detach().cpu().numpy(), (0, 2, 1)).astype(np.float32)

    proc = ColumnPhaseSteppingProcessor(
        PhaseSteppingConfig(
            pixel_size=pixel_size,
            harmonic=1,
            wrap_phase=True,
            norm_phase=True,
            bg=bg,
            negate_integrated_phase=True,
            apply_neg_log_T=True,
            apply_neg_log_D=True,
            neglog_eps=1e-6,
            left_trim=0,
            right_trim=0,
        )
    )
    clean = proc.build_sinograms(I_meas, I_ref)

    step_scale = np.sqrt(n_phase_full / n_phase)
    rng = np.random.default_rng(seed)
    noisy: Dict[str, np.ndarray] = {}
    sigma: Dict[str, float] = {}
    for name in ("T", "DPC", "D"):
        sig = clean[name]
        s = noise_level[name] * step_scale * np.linalg.norm(sig) / np.sqrt(sig.size)
        noisy[name] = (sig + s * rng.standard_normal(sig.shape)).astype(np.float32)
        sigma[name] = float(s)
    noisy["P"] = np.stack([proc.integrate_line_fourier(row) for row in noisy["DPC"]]).astype(np.float32)

    return noisy, sigma, clean


# -----------------------------------------------------------------------------
# Reconstruction-domain pixel mask (matches radon_fanbeam.set_view_angles' R)
# -----------------------------------------------------------------------------

def make_reconstruction_mask(N: int, R: float = 0.98, device="cpu") -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, N, device=device)
    x = torch.linspace(-1.0, 1.0, N, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return (X ** 2 + Y ** 2) <= (R ** 2)


def vec_to_img(v: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    img = torch.full(mask.shape, fill_value, dtype=v.dtype, device=v.device)
    img[mask] = v
    return img


def img_to_vec(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return img[mask]
