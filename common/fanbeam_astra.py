# %%
# src/bquic2d/recon/fanbeam_astra.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import astra


@dataclass
class FanbeamGeometry:
    """
    Fan-beam geometry parameters.

    SAD: source-to-axis distance [m]
    SDD: source-to-detector distance [m]
    det_pitch: detector pixel size at detector [m]
    det_center_offset_pix: detector center shift (pixels), tangential direction
    """
    SAD: float = 1.35
    SDD: float = 1.63
    det_pitch: float = 74.8e-6
    L_phys: float = 2.0
    det_center_offset_pix: float = 0.0

    @property
    def ODD(self) -> float:
        return float(self.SDD - self.SAD)

    @property
    def det_center_offset_m(self) -> float:
        return float(self.det_center_offset_pix * self.det_pitch)


@dataclass
class ReconConfig:
    """
    Reconstruction config.

    method:
      - "SIRT", "SART", "CGLS" => CPU iterative (works everywhere)
      - "FBP" => CUDA fan-beam FBP (requires ASTRA CUDA)

    N: reconstructed image size N x N
    n_iters: iterations for iterative methods
    fbp_filter: ASTRA CUDA filter name
    """
    N: int = 512
    method: str = "SIRT"
    n_iters: int = 200
    fbp_filter: str = "ram-lak"


class FanbeamReconstructor2D:
    """
    Reconstruct a 2D slice from a FAN-BEAM sinogram using ASTRA.

    Important:
      - CPU iterative methods (SIRT/SART/CGLS) require proj_geom = 'fanflat' (NOT fanflat_vec).
      - If det_center_offset_pix != 0 and CPU method is used, we correct approximately by shifting
        the sinogram along detector axis before reconstruction.
      - CUDA FBP can be used if ASTRA CUDA is available (still easiest with 'fanflat').
    """

    def __init__(self, geom: FanbeamGeometry, cfg: ReconConfig):
        self.geom = geom
        self.cfg = cfg

        self.sino: Optional[np.ndarray] = None
        self.beta_rad: Optional[np.ndarray] = None
        self.rec: Optional[np.ndarray] = None
        self.fov: Optional[float] = None  # approximate isocenter FOV width [m]

    def set_angles_deg(self, beta_deg: np.ndarray):
        beta_deg = np.asarray(beta_deg, dtype=np.float32).ravel()
        self.beta_rad = np.deg2rad(beta_deg).astype(np.float32)

    def set_angles_rad(self, beta_rad: np.ndarray):
        self.beta_rad = np.asarray(beta_rad, dtype=np.float32).ravel()

    def set_sinogram(self, sino: np.ndarray):
        sino = np.asarray(sino, dtype=np.float32)
        if sino.ndim != 2:
            raise ValueError(f"sinogram must be 2D (n_angles,n_det). Got {sino.shape}")
        self.sino = sino

    def _make_fanflat_proj_geom(self, n_det: int):
        if self.beta_rad is None:
            raise RuntimeError("angles not set")
        # ASTRA fanflat geometry expects:
        # create_proj_geom('fanflat', det_pitch, n_det, angles, SAD, ODD)
        return astra.create_proj_geom(
            "fanflat",
            float(self.geom.det_pitch),
            int(n_det),
            self.beta_rad.astype(np.float32),
            float(self.geom.SAD),
            float(self.geom.ODD),
        )

    def _make_vol_geom(self, n_det: int):
        # fov = n_det * self.geom.det_pitch * (self.geom.SAD / self.geom.SDD)
        fov = self.geom.L_phys
        vol_geom = astra.create_vol_geom(self.cfg.N, self.cfg.N, -fov/2, fov/2, -fov/2, fov/2)
        return vol_geom, float(fov)

    def _apply_cpu_offset_correction(self, sino: np.ndarray) -> np.ndarray:
        """
        Approximate detector-center offset on CPU by shifting sinogram along detector axis.
        Positive det_center_offset_pix means detector center moved +tangential direction,
        which appears as a column shift. The sign convention can be verified in Script IV.
        """
        off = float(self.geom.det_center_offset_pix)
        if abs(off) < 1e-12:
            return sino

        # integer roll is safest; if you want subpixel later we can do Fourier shift.
        shift = int(np.round(off))
        if shift != 0:
            # axis=1 is detector index (columns)
            sino = np.roll(sino, shift=-shift, axis=1)
        return sino

    def reconstruct(self) -> np.ndarray:
        if self.sino is None:
            raise RuntimeError("sinogram not set. Call set_sinogram(...) first.")
        if self.beta_rad is None:
            raise RuntimeError("angles not set. Call set_angles_deg(...) or set_angles_rad(...) first.")
        if self.sino.shape[0] != self.beta_rad.size:
            raise ValueError(f"n_angles mismatch: sino {self.sino.shape[0]} vs angles {self.beta_rad.size}")

        method = self.cfg.method.upper()
        n_angles, n_det = self.sino.shape

        # For CPU iterative, must use fanflat geometry
        proj_geom = self._make_fanflat_proj_geom(n_det)
        vol_geom, fov = self._make_vol_geom(n_det)
        self.fov = fov

        sino = self.sino
        
        sino = self._apply_cpu_offset_correction(sino)

        sino_id = rec_id = proj_id = alg_id = None

        try:
            sino_id = astra.data2d.create("-sino", proj_geom, sino)
            rec_id = astra.data2d.create("-vol", vol_geom, 0.0)

            if method == "FBP":
                if not astra.astra.use_cuda():
                    raise RuntimeError(
                        "method='FBP' requires ASTRA CUDA (FBP_CUDA), but CUDA not available. "
                        "Use method='SIRT'/'SART'/'CGLS'."
                    )
                cfg = astra.astra_dict("FBP_CUDA")
                cfg["ProjectionDataId"] = sino_id
                cfg["ReconstructionDataId"] = rec_id
                cfg["option"] = {"FilterType": self.cfg.fbp_filter}
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)

            elif method in ("SIRT", "SART", "CGLS"):
                # CPU projector for fanflat
                proj_id = astra.create_projector("strip_fanflat", proj_geom, vol_geom)
                cfg = astra.astra_dict(method)
                cfg["ProjectionDataId"] = sino_id
                cfg["ReconstructionDataId"] = rec_id
                cfg["ProjectorId"] = proj_id
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id, int(self.cfg.n_iters))

            else:
                raise ValueError("method must be one of: 'FBP','SIRT','SART','CGLS'")

            rec = astra.data2d.get(rec_id)

        finally:
            if alg_id is not None:
                astra.algorithm.delete(alg_id)
            if proj_id is not None:
                astra.projector.delete(proj_id)
            if sino_id is not None or rec_id is not None:
                ids = []
                if sino_id is not None:
                    ids.append(sino_id)
                if rec_id is not None:
                    ids.append(rec_id)
                if ids:
                    astra.data2d.delete(ids)

        self.rec = np.asarray(rec, dtype=np.float32)
        return self.rec


# ----------------------------
# small sanity tests
# ----------------------------
def _test_basic_shapes():
    # quick "does it run" check (no truth)
    geom = FanbeamGeometry(SAD=1.35, SDD=1.63, det_pitch=1e-3, det_center_offset_pix=0.0)
    cfg = ReconConfig(N=64, method="SIRT", n_iters=2)
    R = FanbeamReconstructor2D(geom, cfg)

    n_angles, n_det = 10, 80
    sino = np.random.randn(n_angles, n_det).astype(np.float32)
    beta_deg = np.linspace(0, 180, n_angles, dtype=np.float32)

    R.set_sinogram(sino)
    R.set_angles_deg(beta_deg)
    rec = R.reconstruct()

    assert rec.shape == (64, 64)
    assert np.isfinite(rec).all()
    print("[OK] _test_basic_shapes passed.")


if __name__ == "__main__":
    _test_basic_shapes()