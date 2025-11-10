# This notebook provides a guide to running the 2D Navier-Stokes DNS solver.
# 
# 1.  **Part 1: Single Run Example** - We'll run one simulation, plot the results, and verify the incompressibility of the flow.
# 2.  **Part 2: Large-Scale Data Generation** - We'll provide the full script to generate a large dataset of 20 trajectories, complete with spin-up time, batching, and NaN checks.

import torch
import numpy as np
import math
from tqdm import tqdm
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

# =============================================================================
### 1. Analysis Helper Functions (NumPy-based)
# =============================================================================

def get_velocity_from_vorticity(w_field, domain_size):
    """
    Calculates the u and v velocity fields from a 2D vorticity field
    using the Biot-Savart law in Fourier space.
    """
    N = w_field.shape[0]
    w_h = np.fft.fft2(w_field)
    k_vec = (2 * np.pi / domain_size) * np.fft.fftfreq(N, d=1.0/N)
    kx, ky = np.meshgrid(k_vec, k_vec)
    k_sq = kx**2 + ky**2
    inv_k_sq = np.divide(1.0, k_sq, out=np.zeros_like(k_sq), where=k_sq!=0)
    psi_h = -w_h * inv_k_sq
    u_h = 1j * ky * psi_h
    v_h = -1j * kx * psi_h
    u = np.fft.ifft2(u_h).real
    v = np.fft.ifft2(v_h).real
    return u, v

def get_TKE_spectrum(u, v, domain_size):
    """
    Calculates the 1D radially-averaged Turbulent Kinetic Energy (TKE) 
    spectrum from 2D velocity fields.
    """
    N = u.shape[0]
    u_h = np.fft.fft2(u)
    v_h = np.fft.fft2(v)
    tke_2d = 0.5 * (np.abs(u_h)**2 + np.abs(v_h)**2) / (N**4)
    k_vec = (2 * np.pi / domain_size) * np.fft.fftfreq(N, d=1.0/N)
    kx, ky = np.meshgrid(k_vec, k_vec, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2)
    k_max_int = N // 2
    dk = 1.0
    k_bins = np.arange(0.5, k_max_int, dk)
    tke_1d_raw, bin_edges, _ = binned_statistic(
        k_mag.flatten(), 
        tke_2d.flatten(), 
        statistic='mean', 
        bins=k_bins
    )
    k_vals_raw = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid_indices = ~np.isnan(tke_1d_raw)
    k_vals_clean = k_vals_raw[valid_indices]
    tke_1d_clean = tke_1d_raw[valid_indices]
    return k_vals_clean, tke_1d_clean

# =============================================================================
### 2. Data Generation (PyTorch-based)
# =============================================================================

class GaussianRF:
    """ 
    Generates Gaussian Random Fields (GRF) with a specified power spectrum.
    """
    def __init__(self, size, domain_size=2*math.pi, alpha=2.5, tau=7.0, device=None):
        self.device = device
        self.size = (size, size)
        k_max = size // 2
        wavenumers_int = torch.cat(
            (torch.arange(0, k_max), torch.arange(-k_max, 0)), 0
        ).to(device).repeat(size, 1)
        k_x_int = wavenumers_int.transpose(0, 1)
        k_y_int = wavenumers_int
        scaling_factor = 2.0 * math.pi / domain_size
        k_x = k_x_int * scaling_factor
        k_y = k_y_int * scaling_factor
        sigma = tau**(0.5 * (2 * alpha - 2))
        k_sq = k_x**2 + k_y**2
        self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * (k_sq + tau**2)**(-alpha / 2.0)
        self.sqrt_eig[0, 0] = 0.0
            
    def sample(self, N_samples):
        coeff = torch.randn(N_samples, *self.size, 2, device=self.device)
        coeff[..., 0] = self.sqrt_eig * coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig * coeff[..., 1]
        u = torch.fft.ifftn(torch.view_as_complex(coeff), dim=[1, 2]).real
        return u

# =============================================================================
### 3. DNS Solver (PyTorch-based)
# =============================================================================

def solve_dns_vorticity(
    w0, f, visc, T, delta_t, record_steps, domain_size=2*math.pi
):
    """ 
    Solves the 2D Navies-Stokes equation in vorticity form.
    
    This function saves and returns the history for
    vorticity (w), u-velocity (u), and v-velocity (v).
    """
    L = domain_size
    N = w0.size()[-1]
    device = w0.device
    batch_size = w0.shape[0]
    
    if len(w0.shape) == 2: w0 = w0.unsqueeze(0)
    if len(f.shape) == 2: f = f.unsqueeze(0)
    
    # --- Wavenumber and Operator Setup ---
    k_max = N // 2
    k_y_int = torch.cat(
        (torch.arange(0, k_max), torch.arange(-k_max, 0)), 0
    ).to(device)
    k_x_int = k_y_int.view(N, 1)
    k_y, k_x = k_y_int * (2*math.pi/L), k_x_int * (2*math.pi/L)
    k_x_rfft = k_x[..., :k_max+1]
    k_y_rfft = k_y.view(1, N)[..., :k_max+1]
    lap = k_x_rfft**2 + k_y_rfft**2
    lap[0, 0] = 1.0
    dealias = (
        (torch.abs(k_x_rfft) <= (2/3)*k_max*(2*math.pi/L)) & 
        (torch.abs(k_y_rfft) <= (2/3)*k_max*(2*math.pi/L))
    ).unsqueeze(0)

    # --- Initialization ---
    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(f)
    
    total_steps = math.ceil(T / delta_t)
    record_interval = math.floor(total_steps / record_steps)
    if record_interval == 0: record_interval = 1 
    
    w_history = torch.zeros(batch_size, N, N, record_steps, device=device)
    u_history = torch.zeros(batch_size, N, N, record_steps, device=device)
    v_history = torch.zeros(batch_size, N, N, record_steps, device=device)
    
    c_record = 0
    desc = f"Running {N}x{N} DNS"
    
    for j in tqdm(range(total_steps), desc=desc):
        psi_h = w_h / lap
        u_phys = torch.fft.irfft2(1j * k_y_rfft * psi_h, s=(N, N))
        v_phys = torch.fft.irfft2(-1j * k_x_rfft * psi_h, s=(N, N))
        w_x_phys = torch.fft.irfft2(1j * k_x_rfft * w_h, s=(N, N))
        w_y_phys = torch.fft.irfft2(1j * k_y_rfft * w_h, s=(N, N))
        N_phys = u_phys * w_x_phys + v_phys * w_y_phys
        N_h_unaliased = dealias * torch.fft.rfft2(N_phys)
        
        denominator = 1.0 + 0.5 * delta_t * (visc * lap)
        numerator = (
            (1.0 - 0.5 * delta_t * (visc * lap)) * w_h 
            - delta_t * N_h_unaliased 
            + delta_t * f_h
        )
        w_h = numerator / denominator
        
        if torch.isnan(w_h).any():
            print(f"\nNaN detected in solution at step {j}. Ending simulation.")
            w_history[..., c_record:] = float('nan')
            u_history[..., c_record:] = float('nan')
            v_history[..., c_record:] = float('nan')
            break
            
        if j % record_interval == 0 and c_record < record_steps:
            w_history[..., c_record] = torch.fft.irfft2(w_h, s=(N, N))
            psi_h_save = w_h / lap
            u_history[..., c_record] = torch.fft.irfft2(
                1j * k_y_rfft * psi_h_save, s=(N, N)
            )
            v_history[..., c_record] = torch.fft.irfft2(
                -1j * k_x_rfft * psi_h_save, s=(N, N)
            )
            c_record += 1
            
    return (
        w_history.permute(0, 3, 1, 2), 
        u_history.permute(0, 3, 1, 2), 
        v_history.permute(0, 3, 1, 2)
    )

# =============================================================================
### 4.Helper Functions for Data Generation
# =============================================================================

def run_dns_spinup(w0, f, visc, T_spinup, delta_t, domain_size=2*math.pi):
    """
    Runs a DNS simulation for T_spinup seconds WITHOUT saving any snapshots.
    This is used to get the system to a statistically steady state.
    
    Returns the *final vorticity state* in Fourier space (w_h).
    """
    L = domain_size
    N = w0.size()[-1]
    device = w0.device
    
    if len(w0.shape) == 2: w0 = w0.unsqueeze(0)
    if len(f.shape) == 2: f = f.unsqueeze(0)
    
    # --- Wavenumber and Operator Setup (Identical to main solver) ---
    k_max = N // 2
    k_y_int = torch.cat(
        (torch.arange(0, k_max), torch.arange(-k_max, 0)), 0
    ).to(device)
    k_x_int = k_y_int.view(N, 1)
    k_y, k_x = k_y_int * (2*math.pi/L), k_x_int * (2*math.pi/L)
    k_x_rfft = k_x[..., :k_max+1]
    k_y_rfft = k_y.view(1, N)[..., :k_max+1]
    lap = k_x_rfft**2 + k_y_rfft**2
    lap[0, 0] = 1.0
    dealias = (
        (torch.abs(k_x_rfft) <= (2/3)*k_max*(2*math.pi/L)) & 
        (torch.abs(k_y_rfft) <= (2/3)*k_max*(2*math.pi/L))
    ).unsqueeze(0)

    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(f)
    
    total_steps = math.ceil(T_spinup / delta_t)
    
    desc = f"Spinning up {N}x{N} DNS"
    for j in tqdm(range(total_steps), desc=desc):
        psi_h = w_h / lap
        u_phys = torch.fft.irfft2(1j * k_y_rfft * psi_h, s=(N, N))
        v_phys = torch.fft.irfft2(-1j * k_x_rfft * psi_h, s=(N, N))
        w_x_phys = torch.fft.irfft2(1j * k_x_rfft * w_h, s=(N, N))
        w_y_phys = torch.fft.irfft2(1j * k_y_rfft * w_h, s=(N, N))
        N_phys = u_phys * w_x_phys + v_phys * w_y_phys
        N_h_unaliased = dealias * torch.fft.rfft2(N_phys)
        
        denominator = 1.0 + 0.5 * delta_t * (visc * lap)
        numerator = (
            (1.0 - 0.5 * delta_t * (visc * lap)) * w_h 
            - delta_t * N_h_unaliased 
            + delta_t * f_h
        )
        w_h = numerator / denominator
        
        if torch.isnan(w_h).any():
            print(f"\nNaN detected during spin-up at step {j}.")
            return None # Return None to signal failure
            
    return w_h # Return the final Fourier-space state

def calculate_divergence(u, v, domain_size):
    """
    Calculates the divergence div = du/dx + dv/dy in Fourier space
    for maximum accuracy.
    """
    N = u.shape[-1]
    device = u.device
    
    # --- Wavenumber Setup ---
    k_max = N // 2
    k_y_int = torch.cat(
        (torch.arange(0, k_max), torch.arange(-k_max, 0)), 0
    ).to(device)
    k_x_int = k_y_int.view(N, 1)
    k_y, k_x = k_y_int * (2*math.pi/domain_size), k_x_int * (2*math.pi/domain_size)
    k_x_rfft = k_x[..., :k_max+1]
    k_y_rfft = k_y.view(1, N)[..., :k_max+1]
    
    # --- FFT ---
    u_h = torch.fft.rfft2(u)
    v_h = torch.fft.rfft2(v)
    
    # --- Calculate derivatives ---
    du_dx_h = 1j * k_x_rfft * u_h
    dv_dy_h = 1j * k_y_rfft * v_h
    
    # --- Sum and Invert FFT ---
    div_h = du_dx_h + dv_dy_h
    div_phys = torch.fft.irfft2(div_h, s=(N, N))
    
    return div_phys