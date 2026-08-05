# src/bquic2d/preprocess/phase_stepping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import h5py


@dataclass
class PhaseSteppingConfig:
    """
    Configuration for column-wise phase stepping processing.

    Expected input arrays:
      I_meas, I_ref : shape (n_angles, n_steps, n_pix)

    Outputs:
      T, DPC, P, D  : shape (n_angles, n_det_after_crop)
    """
    pixel_size: float = 74.8e-6     # physical sampling along the extracted line (meters); used for Fourier integration
    harmonic: int = 1               # which FFT harmonic to use
    eps: float = 1e-12

    # postprocessing switches
    wrap_phase: bool = True
    norm_phase: bool = True
    bg: int = 50                    # background length for phase mean-removal
    negate_integrated_phase: bool = True

    apply_neg_log_T: bool = True
    apply_neg_log_D: bool = True
    neglog_eps: float = 1e-6

    # detector trimming (remove edge outliers)
    left_trim: int = 165
    right_trim: int = 166


class ColumnPhaseSteppingProcessor:
    def __init__(self, cfg: PhaseSteppingConfig):
        self.cfg = cfg

    # --------------------------
    # helpers
    # --------------------------
    @staticmethod
    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        """Wrap to (-pi, pi]."""
        x = np.asarray(x, dtype=np.float32)
        return x - 2.0 * np.pi * np.floor((x + np.pi) / (2.0 * np.pi))

    def get_fft_coeffs(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        series: (n_steps, n_pix)
        returns:
          a0: mean intensity
          a1: 2*|FFT[h]|
          a2: angle(FFT[h])
        """
        series = np.asarray(series, dtype=np.float32)
        if series.ndim != 2:
            raise ValueError(f"series must be 2D (n_steps,n_pix). Got {series.shape}")

        n_steps = series.shape[0]
        h = int(self.cfg.harmonic)
        if not (0 <= h < n_steps):
            raise ValueError(f"harmonic={h} must satisfy 0 <= harmonic < n_steps={n_steps}")

        fft_result = np.fft.fft(series, axis=0) / float(n_steps)  # (n_steps, n_pix)

        a0 = np.real(fft_result[0, :])
        a1 = 2.0 * np.abs(fft_result[h, :])
        a2 = np.angle(fft_result[h, :])

        return a0.astype(np.float32), a1.astype(np.float32), a2.astype(np.float32)

    def extract_projection(
        self,
        a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
        r0: np.ndarray, r1: np.ndarray, r2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        All inputs shape (n_pix,)
        returns: T, DPC, D each (n_pix,)
        """
        eps = float(self.cfg.eps)

        a0 = np.asarray(a0, dtype=np.float32)
        a1 = np.asarray(a1, dtype=np.float32)
        a2 = np.asarray(a2, dtype=np.float32)
        r0 = np.asarray(r0, dtype=np.float32)
        r1 = np.asarray(r1, dtype=np.float32)
        r2 = np.asarray(r2, dtype=np.float32)

        T = a0 / np.clip(r0, eps, None)

        DPC = a2 - r2
        if self.cfg.wrap_phase:
            DPC = self._wrap_to_pi(DPC)

        # dark-field via visibility ratio
        vis  = a1 / np.clip(a0, eps, None)
        rvis = r1 / np.clip(r0, eps, None)
        D = vis / np.clip(rvis, eps, None)

        return T.astype(np.float32), DPC.astype(np.float32), D.astype(np.float32)

    def integrate_line_fourier(self, dpc_line: np.ndarray) -> np.ndarray:
        """
        Fourier integration of a 1D derivative signal.

        dpc_line: (n_pix,)
        returns P: (n_pix,)

        Uses: P(f) = DPC(f) / (2π i f), with f in cycles per meter (via fftfreq).
        """
        dpc_line = np.asarray(dpc_line, dtype=np.float32)
        n = dpc_line.size
        px = float(self.cfg.pixel_size)

        f = np.fft.fftfreq(n, d=px)          # cycles / meter
        F = np.fft.fft(dpc_line)

        filt = np.zeros_like(F, dtype=np.complex64)
        nz = f != 0
        filt[nz] = 1.0 / (2.0 * np.pi * 1j * f[nz])

        P = np.real(np.fft.ifft(F * filt)).astype(np.float32)

        if self.cfg.norm_phase:
            b = int(self.cfg.bg)
            if 2 * b < n:
                bg_ind = np.r_[0:b, n - b:n]
                P = P - np.mean(P[bg_ind]).astype(np.float32)

        if self.cfg.negate_integrated_phase:
            P = (-P).astype(np.float32)

        return P

    def _trim(self, x: np.ndarray) -> np.ndarray:
        lt = int(self.cfg.left_trim)
        rt = int(self.cfg.right_trim)
        if lt < 0 or rt < 0:
            raise ValueError("left_trim/right_trim must be >= 0")
        if lt + rt >= x.shape[-1]:
            raise ValueError(f"Trimming too large: left_trim+right_trim={lt+rt} >= n_pix={x.shape[-1]}")
        return x[..., lt:x.shape[-1]-rt]

    # --------------------------
    # main API
    # --------------------------
    def build_sinograms(self, I_meas: np.ndarray, I_ref: np.ndarray) -> Dict[str, np.ndarray]:
        """
        I_meas, I_ref: (n_angles, n_steps, n_pix)

        Returns dict with keys: 'T','DPC','P','D' each (n_angles, n_det_after_crop)
        """
        I_meas = np.asarray(I_meas, dtype=np.float32)
        I_ref  = np.asarray(I_ref, dtype=np.float32)

        if I_meas.shape != I_ref.shape:
            raise ValueError(f"I_meas and I_ref shapes differ: {I_meas.shape} vs {I_ref.shape}")
        if I_meas.ndim != 3:
            raise ValueError(f"Expected (n_angles,n_steps,n_pix), got {I_meas.shape}")

        n_angles, n_steps, n_pix = I_meas.shape

        T_sino   = np.zeros((n_angles, n_pix), dtype=np.float32)
        DPC_sino = np.zeros((n_angles, n_pix), dtype=np.float32)
        P_sino   = np.zeros((n_angles, n_pix), dtype=np.float32)
        D_sino   = np.zeros((n_angles, n_pix), dtype=np.float32)

        for i in range(n_angles):
            ser = I_meas[i, :, :]     # (n_steps, n_pix)
            ref = I_ref[i, :, :]

            a0, a1, a2 = self.get_fft_coeffs(ser)
            r0, r1, r2 = self.get_fft_coeffs(ref)

            T, DPC, D = self.extract_projection(a0, a1, a2, r0, r1, r2)
            P = self.integrate_line_fourier(DPC)

            if self.cfg.apply_neg_log_T:
                T = (-np.log(np.clip(T, self.cfg.neglog_eps, None))).astype(np.float32)
            if self.cfg.apply_neg_log_D:
                D = (-np.log(np.clip(D, self.cfg.neglog_eps, None))).astype(np.float32)

            T_sino[i, :] = T
            DPC_sino[i, :] = DPC
            P_sino[i, :] = P
            D_sino[i, :] = D

        # trim detector edges
        T_sino   = self._trim(T_sino)
        DPC_sino = self._trim(DPC_sino)
        P_sino   = self._trim(P_sino)
        D_sino   = self._trim(D_sino)

        return {"T": T_sino, "DPC": DPC_sino, "P": P_sino, "D": D_sino}

    @staticmethod
    def save_sinograms_npy(out_dir: str, sinos: Dict[str, np.ndarray]) -> None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        for k, v in sinos.items():
            np.save(os.path.join(out_dir, f"{k}_sino.npy"), np.asarray(v, dtype=np.float32))

    @staticmethod
    def save_fan_sinograms_h5(
        out_h5: str,
        sinos: Dict[str, np.ndarray],
        beta_deg: np.ndarray,
        SAD: float,
        SDD: float,
        det_pitch: float,
        overwrite: bool = True,
        keys: Tuple[str, ...] = ("T", "DPC", "P", "D"),
        note: str = "Fan-beam central-line sinograms from phase stepping.",
    ) -> str:
        """
        Saves datasets at root: T, DPC, P, D and beta_deg.
        Adds attrs: SAD, SDD, det_pitch.
        """
        import os
        os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)

        if overwrite and os.path.exists(out_h5):
            os.remove(out_h5)

        if "T" not in sinos:
            raise ValueError("sinos must contain at least 'T'")

        T = np.asarray(sinos["T"], dtype=np.float32)
        if T.ndim != 2:
            raise ValueError(f"sinos['T'] must be 2D, got {T.shape}")
        n_angles = T.shape[0]

        beta_deg = np.asarray(beta_deg, dtype=np.float32).ravel()
        if beta_deg.size != n_angles:
            raise ValueError(f"beta_deg length {beta_deg.size} != n_angles {n_angles}")

        with h5py.File(out_h5, "w") as f:
            for k in keys:
                if k in sinos:
                    arr = np.asarray(sinos[k], dtype=np.float32)
                    if arr.shape[0] != n_angles:
                        raise ValueError(f"sinos['{k}'] has wrong n_angles: {arr.shape} vs {n_angles}")
                    f.create_dataset(k, data=arr, compression="gzip", compression_opts=4)

            f.create_dataset("beta_deg", data=beta_deg)

            f.attrs["SAD"] = float(SAD)
            f.attrs["SDD"] = float(SDD)
            f.attrs["det_pitch"] = float(det_pitch)
            f.attrs["note"] = str(note)

        return out_h5


# ----------------------------
# tiny sanity tests (unit-ish)
# ----------------------------
def _test_fft_coeffs_basic():
    """
    A quick check: if series is constant over steps, harmonic amplitude should ~0.
    """
    cfg = PhaseSteppingConfig(harmonic=1)
    proc = ColumnPhaseSteppingProcessor(cfg)

    n_steps, n_pix = 10, 100
    series = np.ones((n_steps, n_pix), dtype=np.float32) * 5.0
    a0, a1, a2 = proc.get_fft_coeffs(series)

    assert np.allclose(a0, 5.0, atol=1e-5)
    assert np.all(a1 < 1e-3), f"Expected near-zero a1, got max={a1.max()}"
    print("[OK] _test_fft_coeffs_basic passed.")


def _test_integration_runs():
    cfg = PhaseSteppingConfig(pixel_size=1.0, norm_phase=True, bg=10)
    proc = ColumnPhaseSteppingProcessor(cfg)

    n = 256
    dpc = np.random.randn(n).astype(np.float32) * 0.01
    P = proc.integrate_line_fourier(dpc)
    assert P.shape == (n,)
    assert np.isfinite(P).all()
    print("[OK] _test_integration_runs passed.")


if __name__ == "__main__":
    _test_fft_coeffs_basic()
    _test_integration_runs()