"""
Real-data loading, phase-stepping retrieval, geometry, and FBP baseline for
the Talbot-Lau experimental dataset (real_data/obs/: I_meas_central.npy,
I_ref_central.npy, angles.npy, steps.npy -- (361 angles, 10 phase steps, 843
raw detector pixels), a full 360-degree scan).

Adapted from noether/real_data/TV/MAP_joint_TV_real_data.py. Reuses
Bayes_mc_CT/common/ directly (TLI_2D_forward.py, phase_stepping.py,
fanbeam_astra.py are physics, identical between the two codebases except for
the bilinear-interpolation fix already in Bayes_mc_CT's copy).

Real-data-specific corrections not needed for synthetic data:
  - background-column offset subtraction (detector baseline drift)
  - T sign flip (attenuation sign convention)
  - tight R=0.5 support mask (physical sample doesn't fill the FOV, unlike
    synthetic phantoms which are built to nearly fill it)
  - sigma_T/DPC/D estimated from the flat-field reference measurement
    (I_ref), not from FBP-forward residuals (see estimate_sigma_I /
    propagate_sigma_montecarlo) -- FBP-residual-based sigma was found to be
    inflated by FBP's own reconstruction error, badly so under sparse
    angles (where FBP is heavily streaked). I_ref has no sample in the beam
    at all, so a noise estimate from it is completely independent of
    reconstruction quality.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
COMMON_DIR = BASE_DIR / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from TLI_2D_forward import radon_fanbeam  # noqa: E402
from phase_stepping import ColumnPhaseSteppingProcessor, PhaseSteppingConfig  # noqa: E402
from fanbeam_astra import FanbeamGeometry, ReconConfig, FanbeamReconstructor2D  # noqa: E402

OBS_DIR = Path(__file__).resolve().parent / "obs"

# ------------------------------------------------------------------------
# Fixed experimental configuration (from the validated reference script)
# ------------------------------------------------------------------------
I_MEAS_NAME = "I_meas_central.npy"
I_REF_NAME = "I_ref_central.npy"
STEPS_NAME = "steps.npy"
ANGLES_NAME = "angles.npy"

PIXEL_SIZE = 74.8e-6
HARMONIC = 1
WRAP_PHASE = True
NORM_PHASE = True
RETRIEVAL_BG = 50
NEGATE_INTEGRATED_PHASE = True
APPLY_NEG_LOG_T = True
APPLY_NEG_LOG_D = True
NEGLOG_EPS = 1e-6
LEFT_TRIM = 1
RIGHT_TRIM = 0

SAD = 1.35
OID = 0.28
DET_PITCH = 74.8e-6
DET_CENTER_OFFSET_PIX_ASTRA = -4.0
DET_CENTER_OFFSET_PIX_FORWARD = 4.0

N_RECON = 512
SUPPORT_R = 0.5

# Forward-projector interpolation kernel, shared by EVERY pipeline (MAP, CV,
# scenarios, UQ) so the MAP figures and the sampled posterior can never use
# different forward models. Change here, not per script.
#
# "nearest" is chosen deliberately. It is ~2.9x cheaper per gradient than
# bilinear (231 vs 671 ms for a potential+gradient), which is what makes full
# posterior UQ tractable across four scenarios and two priors -- the dominant
# practical constraint in multi-contrast CT, where every field is sampled
# jointly. The accuracy given up is well inside the noise floor:
#
#   forward-model difference nearest vs bilinear : 1.80% of ||A x||
#   measured noise sigma_T / RMS(T)              : 27.8%
#   -> the discretization error is ~15x SMALLER than the measurement noise
#
# (Object-like phantom, both sparse_angle and full geometry; bilinear is the
# more accurate quadrature in principle, O(h^2) vs O(h) interpolation error.)
#
# The usual argument for bilinear -- that nearest-neighbour lookup injects a
# pixel-grid staircase which a ramp filter amplifies -- describes the
# simulate-then-FBP path used for SYNTHETIC data. It does not apply here: on
# real data this projector only forward-projects the current estimate inside
# the MAP/posterior, and the FBP baselines come from ASTRA, not from it.
#
# Verified: the tv_beta/lambda0 sweep conclusions are identical under both
# kernels (kernel changes RE_delta by 0.010; tv_beta changes it by 0.529), so
# the prior-parameter choices do not depend on this setting.
PROJECTOR_INTERP = "nearest"

ASTRA_FBP_FILTER = "ram-lak"

BG_COLS = np.r_[0:50, -50:0]


def load_real_intensities() -> Dict[str, np.ndarray]:
    data = {
        "I_ref": np.load(OBS_DIR / I_REF_NAME).astype(np.float32),
        "I_meas": np.load(OBS_DIR / I_MEAS_NAME).astype(np.float32),
        "steps": np.load(OBS_DIR / STEPS_NAME).astype(np.int32),
        "angles": np.load(OBS_DIR / ANGLES_NAME).astype(np.float32),
    }
    print("Loaded experimental intensity data:")
    for k, v in data.items():
        print(f"  {k:7s}: {v.shape}")
    return data


def retrieve_and_correct_sinograms(I_meas: np.ndarray, I_ref: np.ndarray, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Phase-stepping retrieval (T, P, DPC, D), then the real-data-specific
    background-offset subtraction and attenuation sign correction."""
    proc = ColumnPhaseSteppingProcessor(PhaseSteppingConfig(
        pixel_size=PIXEL_SIZE, harmonic=HARMONIC, wrap_phase=WRAP_PHASE, norm_phase=NORM_PHASE,
        bg=RETRIEVAL_BG, negate_integrated_phase=NEGATE_INTEGRATED_PHASE,
        apply_neg_log_T=APPLY_NEG_LOG_T, apply_neg_log_D=APPLY_NEG_LOG_D, neglog_eps=NEGLOG_EPS,
        left_trim=LEFT_TRIM, right_trim=RIGHT_TRIM,
    ))
    sinos = proc.build_sinograms(I_meas=I_meas, I_ref=I_ref)

    T = np.asarray(sinos["T"], dtype=np.float32)
    P = np.asarray(sinos["P"], dtype=np.float32)
    DPC = np.asarray(sinos["DPC"], dtype=np.float32)
    D = np.asarray(sinos["D"], dtype=np.float32)

    T_offset = np.median(T[:, BG_COLS], axis=1, keepdims=True)
    D_offset = np.median(D[:, BG_COLS], axis=1, keepdims=True)
    DPC_offset = np.median(DPC[:, BG_COLS], axis=1, keepdims=True)
    T = T - T_offset
    D = D - D_offset
    DPC = DPC - DPC_offset
    T = -T  # attenuation sign convention

    if verbose:
        print("\nRetrieved + corrected sinograms:")
        for k, x in (("T", T), ("P", P), ("DPC", DPC), ("D", D)):
            print(f"  {k:3s}: shape={x.shape}, min={x.min(): .4e}, max={x.max(): .4e}, mean={x.mean(): .4e}")
    return {"T": T, "P": P, "DPC": DPC, "D": D}


def make_beta_deg(angles: np.ndarray, n_angles: int) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float32).ravel()
    if angles.size == n_angles:
        return (angles - angles[0]).astype(np.float32)
    return np.arange(n_angles, dtype=np.float32)


def build_projector(n_det_eff: int, L_phys: float, beta_deg: np.ndarray, device, dtype,
                    interp: Optional[str] = None) -> Tuple[radon_fanbeam, float]:
    SDD = SAD + OID
    DOD_phys = SDD - SAD
    det_width_phys = DET_PITCH * max(n_det_eff - 1, 1)
    det_center_offset_phys = DET_CENTER_OFFSET_PIX_FORWARD * DET_PITCH

    projector = radon_fanbeam(
        N_detect=int(n_det_eff), N_pix=int(N_RECON), L_phys=float(L_phys), N_quad=int(n_det_eff),
        DSO_phys=float(SAD), DOD_phys=float(DOD_phys), det_width_phys=float(det_width_phys),
        det_center_offset_phys=float(det_center_offset_phys), device=device, dtype=dtype,
        interp=PROJECTOR_INTERP if interp is None else interp,
    )
    beta_t = torch.as_tensor(np.deg2rad(beta_deg), dtype=dtype, device=device)
    projector.set_view_angles(beta_t)
    det_spacing = det_width_phys / max(n_det_eff - 1, 1)
    return projector, float(det_spacing)


def astra_fbp_real(sino: np.ndarray, beta_deg: np.ndarray, L_phys: float, fbp_filter: str = ASTRA_FBP_FILTER) -> np.ndarray:
    """FBP baseline via ASTRA's FBP_CUDA (requires a CUDA-enabled ASTRA build)."""
    geom = FanbeamGeometry(
        SAD=SAD, SDD=SAD + OID, det_pitch=DET_PITCH, L_phys=L_phys,
        det_center_offset_pix=DET_CENTER_OFFSET_PIX_ASTRA,
    )
    cfg = ReconConfig(N=N_RECON, method="FBP", fbp_filter=fbp_filter)
    R = FanbeamReconstructor2D(geom, cfg)
    R.set_angles_deg(np.asarray(beta_deg, dtype=np.float32))
    R.set_sinogram(np.asarray(sino, dtype=np.float32))
    rec = R.reconstruct()
    # NOT flipped, unlike the synthetic pipeline's astra_fbp. That flip was
    # validated against synthetic's known ground truth (ASTRA's row 0 = y-max
    # there); there is no ground truth here to verify the same claim for this
    # geometry, so the array is left as ASTRA returns it and displayed with
    # origin="lower" throughout.
    return np.asarray(rec, dtype=np.float32)


def robust_sigma_from_residual(res: np.ndarray, floor: float) -> float:
    """MAD-based robust sigma estimate with an RMS-based floor. Superseded by
    estimate_sigma_I/propagate_sigma_montecarlo below for sigma_T/DPC/D --
    kept only as a generic utility; do not use FBP-forward residuals for
    sigma estimation (see module docstring)."""
    r = np.asarray(res, dtype=np.float64).ravel()
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma_mad = 1.4826 * mad
    sigma_std = np.std(r)
    return float(max(sigma_mad, 0.25 * sigma_std, floor, 1e-12))


def estimate_sigma_I(I_ref: np.ndarray, harmonic: int = 1) -> float:
    """Robust sigma_I from I_ref's non-DC/non-harmonic FFT content: keep only
    the DC + `harmonic` component of each pixel's flat-field phase-stepping
    curve (exactly what phase_stepping.py's get_fft_coeffs treats as real
    signal -- every other FFT bin is implicitly noise to the retrieval), and
    take the residual from that smooth model. Reconstruction-independent:
    I_ref has no sample in the beam, so this never touches FBP or MAP."""
    n_angles, n_steps, n_pix = I_ref.shape
    F = np.fft.fft(I_ref, axis=1)

    keep = np.zeros(n_steps, dtype=bool)
    keep[0] = True
    keep[harmonic] = True
    keep[(-harmonic) % n_steps] = True  # conjugate bin, for a real-valued smooth model
    F_smooth = np.where(keep[None, :, None], F, 0.0)

    smooth = np.real(np.fft.ifft(F_smooth, axis=1))
    residual = I_ref - smooth

    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    sigma_mad = 1.4826 * mad
    sigma_std = np.std(residual)
    return float(max(sigma_mad, 0.25 * sigma_std, 1e-12))


def propagate_sigma_montecarlo(I_meas: np.ndarray, I_ref: np.ndarray, sigma_I: float,
                                n_trials: int = 50, seed: int = 0) -> Dict[str, float]:
    """Monte Carlo noise propagation: inject N(0, sigma_I^2) onto (I_meas,
    I_ref) many times, rerun the actual retrieve_and_correct_sinograms
    pipeline on each noisy realization, and take the empirical std across
    realizations for T/DPC/D -- captures retrieval's real nonlinear
    sensitivity rather than trusting an approximate analytic formula."""
    rng = np.random.default_rng(seed)
    T_samples, DPC_samples, D_samples = [], [], []

    print(f"  Monte Carlo sigma propagation: {n_trials} trials...", end="", flush=True)
    for i in range(n_trials):
        I_meas_k = I_meas + sigma_I * rng.standard_normal(I_meas.shape).astype(np.float32)
        I_ref_k = I_ref + sigma_I * rng.standard_normal(I_ref.shape).astype(np.float32)
        sinos_k = retrieve_and_correct_sinograms(I_meas_k, I_ref_k, verbose=False)
        T_samples.append(sinos_k["T"])
        DPC_samples.append(sinos_k["DPC"])
        D_samples.append(sinos_k["D"])
        if (i + 1) % 10 == 0:
            print(f" {i + 1}", end="", flush=True)
    print(" done")

    def robust_sigma(samples):
        arr = np.stack(samples, axis=0)
        per_pixel_std = np.std(arr, axis=0, ddof=1)
        return float(np.median(per_pixel_std))

    return {
        "sigma_T": robust_sigma(T_samples),
        "sigma_DPC": robust_sigma(DPC_samples),
        "sigma_D": robust_sigma(D_samples),
    }


_SIGMA_CACHE: Dict[Tuple[int, ...], Dict[str, float]] = {}


def sigma_for_stepping(I_meas: np.ndarray, I_ref: np.ndarray, step_idx: np.ndarray,
                       n_trials: int = 50, seed: int = 0) -> Dict[str, float]:
    """sigma_T/DPC/D for an acquisition that records only the phase steps in
    `step_idx`.

    sigma_I is a property of the detector and the per-step exposure, so it is
    estimated once from the full flat field; only the retrieval it is
    propagated through changes. Retrieving from half the steps averages half
    as much noise away and raises sigma by ~sqrt(2) (measured: 1.4133/1.4158/
    1.4156 for T/DPC/D at 5 of 10 steps), so the likelihood must be rebuilt
    per stepping -- reusing the full-scan sigma would make it twice as
    confident as the data warrant.

    sigma depends on the stepping only, not on the angles, so the Monte Carlo
    runs over all angles and is cached: scenarios sharing a stepping (full
    with sparse_angle, sparse_step with combined) estimate it once.
    """
    key = tuple(int(i) for i in step_idx)
    if key not in _SIGMA_CACHE:
        sigma_I = estimate_sigma_I(I_ref)
        _SIGMA_CACHE[key] = propagate_sigma_montecarlo(
            I_meas[:, step_idx, :], I_ref[:, step_idx, :], sigma_I,
            n_trials=n_trials, seed=seed)
    return _SIGMA_CACHE[key]


def robust_scale(x: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> float:
    """Robust amplitude scale (for field normalization) -- more stable than
    std when FBP has extreme streaking artifacts."""
    x = np.asarray(x, dtype=np.float64)
    q1, q2 = np.percentile(x, [q_low, q_high])
    return float(0.5 * (q2 - q1) + 1e-12)


def geometry_for(n_pix_raw: int) -> float:
    """L_phys (reconstruction FOV) from the RAW (pre-trim) detector pixel
    count -- matches the reference script's convention."""
    SDD = SAD + OID
    det_width_raw = DET_PITCH * n_pix_raw
    return det_width_raw * (SAD / SDD)
