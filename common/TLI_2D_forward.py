# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# Fan-beam projector
# ============================================================

class radon_fanbeam:
    """
    Fan-beam projector on normalized object domain [-1,1]^2.

    Physical geometry is converted internally to normalized coordinates.
    """

    def __init__(
        self,
        N_detect: int,
        N_pix: int,
        L_phys: float,
        N_quad: int = 128,
        DSO_phys: float = 1.0,
        DOD_phys: float = 1.0,
        det_width_phys: float = 1.0,
        det_center_offset_phys: float = 0.0,
        device=None,
        dtype=torch.float32,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.N_detect = int(N_detect)
        self.N_quad = int(N_quad)
        self.N_pix = int(N_pix)

        self.L_norm = torch.tensor(2.0, dtype=dtype, device=self.device)
        self.L_phys = torch.tensor(float(L_phys), dtype=dtype, device=self.device)

        # x_phys = s * x_norm
        self.s = self.L_phys / 2.0

        self.DSO = torch.tensor(float(DSO_phys), dtype=dtype, device=self.device) / self.s
        self.DOD = torch.tensor(float(DOD_phys), dtype=dtype, device=self.device) / self.s
        self.det_width = torch.tensor(float(det_width_phys), dtype=dtype, device=self.device) / self.s
        self.det_center_offset = (
            torch.tensor(float(det_center_offset_phys), dtype=dtype, device=self.device) / self.s
        )

        u = torch.linspace(
            -self.det_width / 2,
            self.det_width / 2,
            self.N_detect,
            dtype=dtype,
            device=self.device,
        )
        u = u + self.det_center_offset

        self.det_x_local = u
        self.det_y_local = self.DOD * torch.ones_like(u)

        self.src_x_local = torch.zeros((), dtype=dtype, device=self.device)
        self.src_y_local = -self.DSO

        if self.N_detect > 1:
            self.det_spacing_norm = self.det_width / (self.N_detect - 1)
        else:
            self.det_spacing_norm = torch.tensor(1.0, dtype=dtype, device=self.device)

        self.det_spacing = self.det_spacing_norm * self.s  # physical spacing

        self.beta = None
        self.beta_deg = None

        self.DSO_phys = torch.tensor(float(DSO_phys), dtype=dtype, device=self.device)
        self.DOD_phys = torch.tensor(float(DOD_phys), dtype=dtype, device=self.device)
        self.SDD_phys = self.DSO_phys + self.DOD_phys

    def plot_setup(self):
        DSO = float(self.DSO.detach().cpu())
        DOD = float(self.DOD.detach().cpu())
        det_width = float(self.det_width.detach().cpu())
        
        # Source and detector (normalized coordinates)
        src = np.array([0.0, -DSO])
        det_x = np.array(self.det_x_local.detach().cpu())
        det_y = np.full_like(det_x, DOD)
        
        plt.figure(figsize=(6,6))
        
        # FoV square [-1,1]^2
        sq = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])
        plt.plot(sq[:,0], sq[:,1], linewidth=2, label="FoV square [-1,1]^2")
        
        # Optional FoV circle (R=1)
        t = np.linspace(0, 2*np.pi, 400)
        plt.plot(np.cos(t), np.sin(t), linestyle="--", label="FoV circle R=1")
        
        # Detector line and points
        plt.plot(det_x, det_y, label="detector line (bins)")
        plt.scatter([src[0]],[src[1]], s=80, marker="*", label="source")
        
        # Draw a few rays (pick 5 detector bins)
        idx = np.linspace(0, len(det_x)-1, 5).astype(int)
        for j in idx:
            plt.plot([src[0], det_x[j]], [src[1], det_y[j]], alpha=0.6)
        
        plt.axhline(0, linewidth=0.5)
        plt.axvline(0, linewidth=0.5)
        plt.gca().set_aspect("equal", "box")
        plt.xlim(-max(2, det_width/2 + 0.2), max(2, det_width/2 + 0.2))
        plt.ylim(-DSO-0.5, DOD+0.5)
        plt.title("radon_fanbeam geometry (normalized coords)")
        plt.legend()
        plt.show()

    def set_view_angles(self, beta: torch.Tensor, R: float = 0.98):
        beta = beta.to(self.device).to(self.dtype)

        beta = -beta  # to be consistent with astra

        self.beta = beta
        self.beta_deg = torch.rad2deg(beta).detach().cpu().numpy()

        M = beta.shape[0]

        src = torch.stack([self.src_x_local, self.src_y_local])         # [2]
        det = torch.stack([self.det_x_local, self.det_y_local], dim=0)  # [2, N_detect]

        v = det - src.view(2, 1)                                        # [2, N_detect]
        ray_len = torch.sqrt(v[0] ** 2 + v[1] ** 2)

        a = v[0] ** 2 + v[1] ** 2
        b = 2.0 * (src[0] * v[0] + src[1] * v[1])
        c = (src[0] ** 2 + src[1] ** 2) - (R ** 2)

        disc = b ** 2 - 4.0 * a * c
        valid = disc > 0

        sqrt_disc = torch.zeros_like(disc)
        sqrt_disc[valid] = torch.sqrt(disc[valid])

        t0 = torch.zeros_like(disc)
        t1 = torch.zeros_like(disc)
        t0[valid] = (-b[valid] - sqrt_disc[valid]) / (2.0 * a[valid])
        t1[valid] = (-b[valid] + sqrt_disc[valid]) / (2.0 * a[valid])

        t_enter = torch.minimum(t0, t1)
        t_exit = torch.maximum(t0, t1)

        t_enter = torch.clamp(t_enter, 0.0, 1.0)
        t_exit = torch.clamp(t_exit, 0.0, 1.0)

        seg_ok = valid & (t_exit > t_enter)

        k = torch.arange(self.N_quad, dtype=self.dtype, device=self.device)
        tau = (k + 0.5) / self.N_quad
        t = t_enter.view(-1, 1) + tau.view(1, -1) * (t_exit - t_enter).view(-1, 1)

        px_local = src[0] + t * v[0].view(-1, 1)
        py_local = src[1] + t * v[1].view(-1, 1)

        self.dx = (t_exit - t_enter) * ray_len / self.N_quad
        self.seg_ok = seg_ok

        cb = torch.cos(beta).view(-1, 1, 1)
        sb = torch.sin(beta).view(-1, 1, 1)

        px = px_local.unsqueeze(0).repeat(M, 1, 1)
        py = py_local.unsqueeze(0).repeat(M, 1, 1)

        self.QPX = cb * px - sb * py
        self.QPY = sb * px + cb * py

        # self.mask = seg_ok.view(1, -1, 1).repeat(M, 1, self.N_quad)
        # self.mask = (
        #     self.mask
        #     & (self.QPX >= -1.0) & (self.QPX <= 1.0)
        #     & (self.QPY >= -1.0) & (self.QPY <= 1.0)
        # )

        self.mask = seg_ok.view(1, -1, 1).repeat(M, 1, self.N_quad) & (self.QPX**2 + self.QPY**2 <= (R**2))
    

        QPX_IDX = ((self.QPX + 1.0) / 2.0) * (self.N_pix - 1)
        QPY_IDX = ((self.QPY + 1.0) / 2.0) * (self.N_pix - 1)

        self.QPX_IDX = QPX_IDX[self.mask].to(torch.int64)
        self.QPY_IDX = QPY_IDX[self.mask].to(torch.int64)

    def make_sinogram(self, im_norm: torch.Tensor, return_physical: bool = True):
        if self.beta is None:
            raise RuntimeError("Call set_view_angles(...) first.")

        im_norm = im_norm.to(self.device).to(self.dtype)

        temp = torch.zeros_like(self.QPX, dtype=self.dtype, device=self.device)
        temp[self.mask] = im_norm[self.QPY_IDX, self.QPX_IDX]
        sino_norm = torch.sum(temp, dim=-1) * self.dx.view(1, -1)

        if return_physical:
            return sino_norm * self.s
        return sino_norm


# def dpc_from_phase_sino(P_sino: torch.Tensor, det_spacing):
#     """
#     Detector-direction finite difference.
#     """
#     if torch.is_tensor(det_spacing):
#         h = det_spacing.to(P_sino.device).to(P_sino.dtype)
#     else:
#         h = torch.tensor(det_spacing, dtype=P_sino.dtype, device=P_sino.device)

#     DPC = torch.zeros_like(P_sino)
#     DPC[:, 1:-1] = (P_sino[:, 2:] - P_sino[:, :-2]) / (2.0 * h)
#     DPC[:, 0] = (P_sino[:, 1] - P_sino[:, 0]) / h
#     DPC[:, -1] = (P_sino[:, -1] - P_sino[:, -2]) / h
#     return DPC

def dpc_from_phase_sino(P_sino: torch.Tensor, det_spacing):
    """
    Detector-direction finite difference, 2nd-order central interior,
    4th-order one-sided boundaries.
    """
    if torch.is_tensor(det_spacing):
        h = det_spacing.to(P_sino.device).to(P_sino.dtype)
    else:
        h = torch.tensor(det_spacing, dtype=P_sino.dtype, device=P_sino.device)

    DPC = torch.zeros_like(P_sino)

    # Interior: 2nd-order central difference  O(h²)
    DPC[:, 1:-1] = (P_sino[:, 2:] - P_sino[:, :-2]) / (2.0 * h)

    # Left boundary: 4th-order one-sided forward  O(h⁴)
    # f'(x) ≈ (-25f₀ + 48f₁ - 36f₂ + 16f₃ - 3f₄) / (12h)
    DPC[:, 0] = (
        -25 * P_sino[:, 0]
        + 48 * P_sino[:, 1]
        - 36 * P_sino[:, 2]
        + 16 * P_sino[:, 3]
        -  3 * P_sino[:, 4]
    ) / (12.0 * h)

    # One cell in from left: 4th-order one-sided forward  O(h⁴)
    # f'(x) ≈ (-3f₀ - 10f₁ + 18f₂ - 6f₃ + f₄) / (12h)
    DPC[:, 1] = (
        -  3 * P_sino[:, 0]
        - 10 * P_sino[:, 1]
        + 18 * P_sino[:, 2]
        -  6 * P_sino[:, 3]
        +  1 * P_sino[:, 4]
    ) / (12.0 * h)

    # Right boundary: 4th-order one-sided backward  O(h⁴)
    # f'(x) ≈ (25f₋₁ - 48f₋₂ + 36f₋₃ - 16f₋₄ + 3f₋₅) / (12h)
    DPC[:, -1] = (
        + 25 * P_sino[:, -1]
        - 48 * P_sino[:, -2]
        + 36 * P_sino[:, -3]
        - 16 * P_sino[:, -4]
        +  3 * P_sino[:, -5]
    ) / (12.0 * h)

    # One cell in from right: 4th-order one-sided backward  O(h⁴)
    DPC[:, -2] = (
        +  3 * P_sino[:, -1]
        + 10 * P_sino[:, -2]
        - 18 * P_sino[:, -3]
        +  6 * P_sino[:, -4]
        -  1 * P_sino[:, -5]
    ) / (12.0 * h)

    return DPC

# def dpc_from_phase_sino(P_sino: torch.Tensor, det_spacing):
#     """
#     FFT-based detector-direction derivative.
#     det_spacing is the sample spacing h, not total detector width.
#     """
#     if not torch.is_tensor(P_sino):
#         P = torch.as_tensor(P_sino)
#     else:
#         P = P_sino

#     if torch.is_tensor(det_spacing):
#         h = det_spacing.to(P.device).to(P.dtype)
#     else:
#         h = torch.tensor(det_spacing, dtype=P.dtype, device=P.device)

#     n_angles, n_det = P.shape

#     freq = torch.fft.fftfreq(n_det, d=float(h.detach().cpu()))
#     freq = freq.to(P.device).to(P.dtype)

#     F = torch.fft.fft(P, dim=1)
#     dP = torch.fft.ifft(F * (2j * torch.pi * freq)[None, :], dim=1).real

#     return dP

# def dpc_from_phase_sino(
#     P_sino: torch.Tensor,
#     det_spacing,
#     pad_factor: int = 2,
#     taper_alpha: float = 0.0,
# ) -> torch.Tensor:
#     import torch
#     import torch.nn.functional as F

#     P = P_sino if torch.is_tensor(P_sino) else torch.as_tensor(P_sino)

#     if P.ndim != 2:
#         raise ValueError("P_sino must have shape (n_angles, n_det).")

#     device = P.device
#     dtype = P.dtype
#     n_angles, n_det = P.shape

#     h = float(det_spacing.detach().cpu()) if torch.is_tensor(det_spacing) else float(det_spacing)

#     n_pad = max(int(pad_factor * n_det), n_det)
#     start = (n_pad - n_det) // 2
#     stop = start + n_det

#     P_work = P

#     # Optional taper only for artifact suppression
#     if taper_alpha > 0.0:
#         width = int(taper_alpha * n_det / 2)
#         w = torch.ones(n_det, dtype=dtype, device=device)
#         if width > 0:
#             r = torch.arange(width, dtype=dtype, device=device)
#             taper = 0.5 * (1.0 - torch.cos(torch.pi * r / width))
#             w[:width] = taper
#             w[-width:] = taper.flip(0)
#         P_work = P_work * w[None, :]

#     P_pad = torch.zeros((n_angles, n_pad), dtype=dtype, device=device)
#     P_pad[:, start:stop] = P_work

#     freq = torch.fft.fftfreq(n_pad, d=h, dtype=torch.float64, device=device)
#     cdtype = torch.complex64 if dtype in (torch.float16, torch.float32) else torch.complex128
#     kernel = (2j * torch.pi * freq).to(cdtype)

#     F_hat = torch.fft.fft(P_pad, dim=1)
#     dP_pad = torch.fft.ifft(F_hat.to(cdtype) * kernel[None, :], dim=1).real

#     return dP_pad[:, start:stop]


# ============================================================
# DPC -> phase integrator
# ============================================================

class DPCPhaseIntegrator:
    @staticmethod
    def integrate_dpc_to_phase(
        DPC,
        det_width_phys,
        pad_factor=2,
        norm=True,
        bg_width=50,
    ):
        DPC = np.asarray(DPC, dtype=np.float64)
        n_angles, n_det = DPC.shape

        if n_det < 2:
            raise ValueError("Need at least 2 detector bins.")
        if pad_factor < 1:
            raise ValueError("pad_factor must be >= 1")

        du = det_width_phys / (n_det - 1)
        n_pad = max(int(pad_factor * n_det), n_det)

        start = (n_pad - n_det) // 2
        stop = start + n_det

        freq = np.fft.fftfreq(n_pad, d=du)

        filt = np.zeros(n_pad, dtype=np.complex128)
        nz = freq != 0
        filt[nz] = 1.0 / (2j * np.pi * freq[nz])
        filt[~nz] = 0.0

        g = DPC - DPC.mean(axis=1, keepdims=True)

        g_pad = np.zeros((n_angles, n_pad), dtype=np.float64)
        g_pad[:, start:stop] = g

        G = np.fft.fft(g_pad, axis=1)
        P_hat = G * filt[None, :]
        p_pad = np.fft.ifft(P_hat, axis=1).real
        p = p_pad[:, start:stop]

        if norm:
            bg_width = min(bg_width, max(1, n_det // 8))
            bg_ind = np.r_[0:bg_width, n_det - bg_width:n_det]
            p = p - p[:, bg_ind].mean(axis=1, keepdims=True)
        else:
            p = p - p.mean(axis=1, keepdims=True)

        return p.astype(np.float32)

# ============================================================
# TL interferometry forward model
# ============================================================

@dataclass
class ForwardOutput:
    I_meas: torch.Tensor
    I_ref: torch.Tensor
    T: torch.Tensor
    DPC: torch.Tensor
    D: torch.Tensor
    P: torch.Tensor
    phi: torch.Tensor


class TLInterferometryForward2D:
    """
    Full 2D TL-interferometry forward model.

    mu    -> attenuation sinogram T
    delta -> phase sinogram P and DPC = dP/du
    eps   -> dark-field sinogram D

    I_ref(phi)  = I0 * (1 + vis * cos(phi))
    I_meas(phi) = I0 * exp(-T) * (1 + vis * exp(-D) * cos(phi + DPC))
    """

    def __init__(self, projector: radon_fanbeam, I0=1.0, vis=0.3):
        if projector.beta is None:
            raise RuntimeError("Projector angles are not set. Call set_view_angles(...) first.")
        self.projector = projector
        self.I0 = float(I0)
        self.vis = float(vis)

    @property
    def device(self):
        return self.projector.device

    @property
    def dtype(self):
        return self.projector.dtype

    def _to_tensor_image(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        if x.ndim != 2:
            raise ValueError(f"Expected shape (N,N), got {tuple(x.shape)}")
        return x

    def project_fields(self, mu, delta, eps):
        mu = self._to_tensor_image(mu)
        delta = self._to_tensor_image(delta)
        eps = self._to_tensor_image(eps)

        T = self.projector.make_sinogram(mu, return_physical=True)
        P = self.projector.make_sinogram(delta, return_physical=True)
        D = self.projector.make_sinogram(eps, return_physical=True)
        return T, P, D

    
    def forward(self, mu, delta, eps, phi=None, n_phase=None, phi0=0.0):
        """
        Parameters
        ----------
        mu, delta, eps : array-like or torch.Tensor, shape (N, N)
    
        phi : array-like, optional
            Explicit phase-stepping positions in radians.
    
        n_phase : int, optional
            Number of equally spaced phase steps over [0, 2*pi).
    
        phi0 : float, optional
            Starting phase offset when n_phase is used.
    
        Returns
        -------
        ForwardOutput
        """
        if phi is None and n_phase is None:
            raise ValueError("Provide either phi or n_phase.")
    
        if phi is not None and n_phase is not None:
            raise ValueError("Provide only one of phi or n_phase, not both.")
    
        if n_phase is not None:
            phi = phi0 + 2.0 * np.pi * np.arange(n_phase) / n_phase
    
        T, P, D = self.project_fields(mu, delta, eps)
        
        h = self.projector.det_spacing * (self.projector.DSO_phys / self.projector.SDD_phys)
        DPC =  - dpc_from_phase_sino(P, h)
    
        A = torch.exp(-T)
        S = torch.exp(-D)
    
        phi_t = torch.as_tensor(phi, dtype=self.dtype, device=self.device)
        if phi_t.ndim != 1:
            raise ValueError("phi must be a 1D array-like")
    
        phi_grid = phi_t.view(1, 1, -1)
    
        I_ref = self.I0 * (1.0 + self.vis * torch.cos(phi_grid))
        I_meas = self.I0 * A.unsqueeze(-1) * (
            1.0 + self.vis * S.unsqueeze(-1) * torch.cos(phi_grid + DPC.unsqueeze(-1))
        )
    
        I_ref = torch.ones_like(I_meas) * I_ref
        
        return ForwardOutput(
            I_meas=I_meas,
            I_ref=I_ref,
            T=T,
            DPC=DPC,
            D=D,
            P=P,
            phi=phi_t
        )
    
    def forward_with_measured_ref(self, mu, delta, eps, I_ref_meas, phi=None, n_phase=None, phi0=0.0):
        """
        Forward model using measured reference coefficients in a coefficient-consistent way.

        Parameters
        ----------
        mu, delta, eps : array-like or torch.Tensor, shape (N, N)

        I_ref_meas : array-like or torch.Tensor
            Measured reference intensity stack.
            Expected shape: (n_angles, n_phase, n_det) or (n_angles, n_det, n_phase)

        phi : array-like, optional
            Explicit phase-stepping positions in radians.

        n_phase : int, optional
            Number of equally spaced phase steps over [0, 2*pi).

        phi0 : float, optional
            Starting phase offset when n_phase is used.

        Returns
        -------
        ForwardOutput
            I_ref is the measured reference reshaped to internal convention (angle, det, phase).
            I_meas is synthesized so that FFT-based retrieval recovers T, D, DPC consistently.
        """
        if phi is None and n_phase is None:
            raise ValueError("Provide either phi or n_phase.")
        if phi is not None and n_phase is not None:
            raise ValueError("Provide only one of phi or n_phase.")

        if n_phase is not None:
            phi = phi0 + 2.0 * np.pi * np.arange(n_phase) / n_phase

        # --------------------------------------------------
        # 1. Forward-project the three fields
        # --------------------------------------------------
        T, P, D = self.project_fields(mu, delta, eps)

        h = self.projector.det_spacing * (self.projector.DSO_phys / self.projector.SDD_phys)
        DPC = - dpc_from_phase_sino(P, h)   # or dpc_from_phase_sino(P, h) if that is your chosen convention

        # --------------------------------------------------
        # 2. Prepare phase grid
        # --------------------------------------------------
        phi_t = torch.as_tensor(phi, dtype=self.dtype, device=self.device)
        if phi_t.ndim != 1:
            raise ValueError("phi must be a 1D array-like")
        K = phi_t.numel()

        # internal convention: (angle, det, phase)
        I_ref_meas_t = torch.as_tensor(I_ref_meas, dtype=self.dtype, device=self.device)

        if I_ref_meas_t.ndim != 3:
            raise ValueError(f"I_ref_meas must be 3D, got shape {tuple(I_ref_meas_t.shape)}")

        # accept either (angle, step, pix) or (angle, pix, step)
        if I_ref_meas_t.shape[1] == K:
            I_ref_meas_t = I_ref_meas_t.permute(0, 2, 1)   # -> (angle, det, phase)
        elif I_ref_meas_t.shape[2] == K:
            pass
        else:
            raise ValueError(
                f"Neither axis 1 nor axis 2 matches n_phase={K}; got shape {tuple(I_ref_meas_t.shape)}"
            )

        if I_ref_meas_t.shape[0] != T.shape[0] or I_ref_meas_t.shape[1] != T.shape[1]:
            raise ValueError(
                f"I_ref_meas geometry mismatch: expected (n_angles, n_det, n_phase)=({T.shape[0]}, {T.shape[1]}, {K}) "
                f"but got {tuple(I_ref_meas_t.shape)}"
            )

        # --------------------------------------------------
        # 3. Extract measured reference first-harmonic coefficients
        #    Internal convention is (angle, det, phase)
        # --------------------------------------------------
        # FFT of measured reference
        F_ref = torch.fft.fft(I_ref_meas_t, dim=-1) / K

        # DC and first harmonic of measured reference
        c0_ref = F_ref[..., 0]
        c1_ref = F_ref[..., 1]

        A = torch.exp(-T)
        S = torch.exp(-D)

        # sign convention for phase shift
        phase_factor = torch.exp(1j * DPC)

        # build simulated spectrum
        F_sim = F_ref.clone()

        # DC term
        F_sim[..., 0] = c0_ref * A

        # first harmonic:
        # scale amplitude by exp(-T) exp(-D), rotate by DPC
        F_sim[..., 1] = c1_ref * A * S * phase_factor

        # enforce Hermitian symmetry for real-valued signal
        F_sim[..., -1] = torch.conj(F_sim[..., 1])

        # optional: scale higher harmonics by transmission only
        if K > 2:
            F_sim[..., 2:-1] = F_ref[..., 2:-1] * A.unsqueeze(-1)

        # reconstruct intensity stack
        I_meas = torch.real(torch.fft.ifft(F_sim * K, dim=-1))

        return ForwardOutput(
            I_meas=I_meas,
            I_ref=I_ref_meas_t,
            T=T,
            DPC=DPC,
            D=D,
            P=P,
            phi=phi_t,
        )

    
# ============================================================
# Phantom
# ============================================================

def make_three_field_phantom(N=256, kind="inclusion", dtype=np.float32):
    y = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing="ij")

    mu = np.zeros((N, N), dtype=np.float64)
    delta = np.zeros((N, N), dtype=np.float64)
    eps = np.zeros((N, N), dtype=np.float64)

    if kind == "inclusion":
        outer = X**2 + Y**2 < 0.42**2

        inc1 = (X + 0.10)**2 + (Y - 0.10)**2 < 0.10**2
        inc2 = (X - 0.13)**2 + (Y - 0.05)**2 < 0.08**2
        inc3 = (X + 0.02)**2 + (Y + 0.16)**2 < 0.07**2

        scale = 15.0

        mu[outer] = scale * 0.022
        mu[inc1] = scale * 0.050
        mu[inc2] = scale * 0.012
        mu[inc3] = scale * 0.030

        alpha = 0.45
        delta[:] = alpha * mu
        delta += 0.002 * scale * np.sin(3 * X) * np.cos(3 * Y)
        delta[inc1] *= 0.8
        delta[inc2] *= 1.5
        delta[inc3] *= 0.2
        delta[~outer] = 0.0
        delta = np.clip(delta, 0.0, None)

        eps[:] = 0.3 * mu

        texture = (
            np.sin(10 * X) * np.sin(10 * Y)
            + 0.5 * np.sin(20 * X + 0.3) * np.cos(15 * Y)
        )
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-12)

        eps += scale * 0.01 * texture
        eps[inc1] *= 1.5
        eps[inc2] *= 0.7
        eps[inc3] *= 2.0
        eps[~outer] = 0.0
        eps = np.clip(eps, 0.0, None)

    elif kind == "zero":
        pass

    else:
        raise ValueError(f"Unknown phantom kind: {kind}")

    return mu.astype(dtype), delta.astype(dtype), eps.astype(dtype)


# ============================================================
# Plotting
# ============================================================

def plot_phantoms_and_sinograms(mu, delta, eps, T, P, D, figsize=(14, 7), save_path=None):
    """
    2x3 figure:
      top row    = mu, delta, eps
      bottom row = T, P, D
    """
    if torch.is_tensor(T):
        T = T.detach().cpu().numpy()
    if torch.is_tensor(P):
        P = P.detach().cpu().numpy()
    if torch.is_tensor(D):
        D = D.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    top_data = [mu, delta, eps]
    top_titles = [r"True $\mu$", r"True $\delta$", r"True $\varepsilon$"]

    bot_data = [T, P, D]
    bot_titles = [r"Sinogram $T=A(\mu)$", r"Sinogram $P=A(\delta)$", r"Sinogram $D=A(\varepsilon)$"]

    for j in range(3):
        im = axes[0, j].imshow(top_data[j], origin="lower", extent=[-1, 1, -1, 1])
        axes[0, j].set_title(top_titles[j])
        axes[0, j].set_xlabel("x")
        axes[0, j].set_ylabel("y")
        axes[0, j].set_aspect("equal")
        fig.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)

    for j in range(3):
        im = axes[1, j].imshow(bot_data[j], origin="lower", aspect="auto")
        axes[1, j].set_title(bot_titles[j])
        axes[1, j].set_xlabel("detector bin")
        axes[1, j].set_ylabel("view index")
        fig.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.04)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def plot_intensity_diagnostics(
    I_meas,
    I_ref,
    angle_idx=None,
    det_idx=None,
    phase_idx=None,
    figsize=(15, 9),
    save_path=None,
):
    """
    Plot useful diagnostics for I_meas and I_ref.

    Parameters
    ----------
    I_meas, I_ref : torch.Tensor or np.ndarray
        Shape (n_angles, n_det, n_phase)

    angle_idx : int or None
        Angle index used for 1D detector profiles and phase-stepping curves.
        If None, use middle angle.

    det_idx : int or None
        Detector index used for phase-stepping curves.
        If None, use middle detector.

    phase_idx : int or None
        Phase-step index used for 2D sinogram-like intensity images.
        If None, use phase 0.

    figsize : tuple
        Figure size.

    save_path : str or None
        If given, save the figure.
    """
    if torch.is_tensor(I_meas):
        I_meas = I_meas.detach().cpu().numpy()
    if torch.is_tensor(I_ref):
        I_ref = I_ref.detach().cpu().numpy()

    if I_meas.shape != I_ref.shape:
        raise ValueError(f"I_meas and I_ref must have same shape, got {I_meas.shape} and {I_ref.shape}")

    if I_meas.ndim != 3:
        raise ValueError(f"Expected shape (n_angles, n_det, n_phase), got {I_meas.shape}")

    n_angles, n_det, n_phase = I_meas.shape

    if angle_idx is None:
        angle_idx = n_angles // 2
    if det_idx is None:
        det_idx = n_det // 2
    if phase_idx is None:
        phase_idx = 0

    angle_idx = int(np.clip(angle_idx, 0, n_angles - 1))
    det_idx = int(np.clip(det_idx, 0, n_det - 1))
    phase_idx = int(np.clip(phase_idx, 0, n_phase - 1))

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    # --------------------------------------------------
    # Top row: 2D intensity maps at one chosen phase step
    # --------------------------------------------------
    im0 = axes[0, 0].imshow(I_ref[:, :, phase_idx], origin="lower", aspect="auto")
    axes[0, 0].set_title(f"I_ref at phase_idx={phase_idx}")
    axes[0, 0].set_xlabel("detector bin")
    axes[0, 0].set_ylabel("view index")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(I_meas[:, :, phase_idx], origin="lower", aspect="auto")
    axes[0, 1].set_title(f"I_meas at phase_idx={phase_idx}")
    axes[0, 1].set_xlabel("detector bin")
    axes[0, 1].set_ylabel("view index")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(I_meas[:, :, phase_idx] - I_ref[:, :, phase_idx],
                            origin="lower", aspect="auto")
    axes[0, 2].set_title(f"I_meas - I_ref at phase_idx={phase_idx}")
    axes[0, 2].set_xlabel("detector bin")
    axes[0, 2].set_ylabel("view index")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --------------------------------------------------
    # Bottom row: 1D cuts and phase-stepping curves
    # --------------------------------------------------
    det_axis = np.arange(n_det)
    phase_axis = np.arange(n_phase)

    axes[1, 0].plot(det_axis, I_ref[angle_idx, :, phase_idx], label="I_ref")
    axes[1, 0].plot(det_axis, I_meas[angle_idx, :, phase_idx], label="I_meas")
    axes[1, 0].set_title(f"Detector profile at angle_idx={angle_idx}, phase_idx={phase_idx}")
    axes[1, 0].set_xlabel("detector bin")
    axes[1, 0].set_ylabel("intensity")
    axes[1, 0].legend()

    axes[1, 1].plot(phase_axis, I_ref[angle_idx, det_idx, :], "o-", label="I_ref")
    axes[1, 1].plot(phase_axis, I_meas[angle_idx, det_idx, :], "o-", label="I_meas")
    axes[1, 1].set_title(f"Phase-stepping curve at angle_idx={angle_idx}, det_idx={det_idx}")
    axes[1, 1].set_xlabel("phase-step index")
    axes[1, 1].set_ylabel("intensity")
    axes[1, 1].legend()

    axes[1, 2].plot(phase_axis, I_meas[angle_idx, det_idx, :] - I_ref[angle_idx, det_idx, :], "o-")
    axes[1, 2].set_title(f"Difference curve at angle_idx={angle_idx}, det_idx={det_idx}")
    axes[1, 2].set_xlabel("phase-step index")
    axes[1, 2].set_ylabel("I_meas - I_ref")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()  

# ============================================================
# Sanity checks
# ============================================================

def example_zero_phantom():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    N = 128
    N_det = 128
    N_quad = 128
    n_angles = 180

    beta = torch.linspace(0, 2 * torch.pi, n_angles + 1, dtype=dtype, device=device)[:-1]
    # phi = torch.linspace(0, 2 * torch.pi, 5, dtype=dtype, device=device)

    projector = radon_fanbeam(
        N_detect=N_det,
        N_pix=N,
        L_phys=0.06,
        N_quad=N_quad,
        DSO_phys=0.20,
        DOD_phys=0.20,
        det_width_phys=0.08,
        device=device,
        dtype=dtype,
    )
    projector.set_view_angles(beta)

    model = TLInterferometryForward2D(projector, I0=1.0, vis=0.3)

    mu, delta, eps = make_three_field_phantom(N=N, kind="zero")
    out = model.forward(mu, delta, eps, n_phase=10)

    print("=== zero phantom check ===")
    print("max |T|   =", float(out.T.abs().max()))
    print("max |P|   =", float(out.P.abs().max()))
    print("max |DPC| =", float(out.DPC.abs().max()))
    print("max |D|   =", float(out.D.abs().max()))
    print("max |I_meas - I_ref| =", float((out.I_meas - out.I_ref).abs().max()))


def example_inclusion_phantom(plot_result=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    N = 256
    N_det = 160
    N_quad = 160
    n_angles = 180

    beta = torch.linspace(0, 2 * torch.pi, n_angles + 1, dtype=dtype, device=device)[:-1]
    
    projector = radon_fanbeam(
        N_detect=N_det,
        N_pix=N,
        L_phys=0.06,
        N_quad=N_quad,
        DSO_phys=0.20,
        DOD_phys=0.20,
        det_width_phys=0.08,
        device=device,
        dtype=dtype,
    )
    projector.set_view_angles(beta)

    projector.plot_setup()
    
    model = TLInterferometryForward2D(projector, I0=1.0, vis=0.35)

    mu, delta, eps = make_three_field_phantom(N=N, kind="inclusion")
    out = model.forward(mu, delta, eps, n_phase= 10)
    

    # det_width_phys = float((projector.det_width * projector.s).detach().cpu().numpy())
    det_width_object = float(
        (projector.det_spacing * projector.DSO_phys / projector.SDD_phys).item()
    ) * (N_det - 1)
    P_rec = -DPCPhaseIntegrator.integrate_dpc_to_phase(
        out.DPC.detach().cpu().numpy(),
        det_width_phys=det_width_object,
        pad_factor=2,
        norm=True,
        bg_width=20,
    )

    P_true = out.P.detach().cpu().numpy()
    P_true0 = P_true - P_true.mean(axis=1, keepdims=True)
    P_rec0 = P_rec - P_rec.mean(axis=1, keepdims=True)

    rel_err = np.linalg.norm(P_rec0 - P_true0) / (np.linalg.norm(P_true0) + 1e-12)

    print("=== inclusion phantom check ===")
    print("I_meas shape:", tuple(out.I_meas.shape))
    print("I_ref  shape:", tuple(out.I_ref.shape))
    print("T shape:", tuple(out.T.shape))
    print("DPC shape:", tuple(out.DPC.shape))
    print("D shape:", tuple(out.D.shape))
    print("P shape:", tuple(out.P.shape))
    print("relative error in reconstructed phase (demeaned rows):", rel_err)
    print("I_meas min/max:", float(out.I_meas.min()), float(out.I_meas.max()))
    print("T min/max:", float(out.T.min()), float(out.T.max()))
    print("D min/max:", float(out.D.min()), float(out.D.max()))
    print("DPC min/max:", float(out.DPC.min()), float(out.DPC.max()))


    plot_intensity_diagnostics(out.I_meas, out.I_ref)
    
    
    plot_phantoms_and_sinograms(mu, delta, eps, out.T, out.P,out.D)
    # plot_phantoms_and_sinograms(mu, delta, eps, out.T, out.DPC,out.D)  # with the middle title changed to "DPC = dP/du"
    
    
    return mu, delta, eps, out


if __name__ == "__main__":
    example_zero_phantom()
    example_inclusion_phantom(plot_result=True)
# %%
