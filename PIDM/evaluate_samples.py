import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Single checkpoint or multiple checkpoints
checkpoint_steps = [2000, 4000, 6000, 8000, 10000]  # List of checkpoints to compare (0 = untrained)
# checkpoint_steps = [10000]  # Or just one checkpoint

# Plotting options
plot_separate = True  # Plot each checkpoint separately
plot_combined = True  # Plot all checkpoints together for comparison
save_figures = True
output_dir = 'evaluation_results'

# Physics parameters (must match training config)
DOMAIN_LENGTH = np.pi
VISC = 1e-3
PIXELS_PER_DIM = 64

# Which residuals to plot
plot_spatial_residuals = True  # Plot 2D spatial distribution of residuals
plot_residual_histograms = True  # Plot histograms of residual magnitudes

# ============================================================================
# Helper Functions for Physics Residual Computation
# ============================================================================

def load_sample_fields(sample_dir):
    """Load vorticity, u-velocity, v-velocity from CSV files."""
    w = np.loadtxt(os.path.join(sample_dir, 'vorticity.csv'), delimiter=',')
    u = np.loadtxt(os.path.join(sample_dir, 'u_velocity.csv'), delimiter=',')
    v = np.loadtxt(os.path.join(sample_dir, 'v_velocity.csv'), delimiter=',')
    return w, u, v

def compute_physics_residuals(w, u, v, domain_length=np.pi, pixels_per_dim=64):
    """
    Compute Navier-Stokes physics residuals:
    1. Vorticity-velocity relationship: ω - (∂v/∂x - ∂u/∂y) = 0
    2. Incompressibility: ∂u/∂x + ∂v/∂y = 0
    
    Uses spectral (FFT) derivatives for accuracy with periodic boundary conditions.
    """
    # Convert to torch tensors
    w_torch = torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    u_torch = torch.from_numpy(u).float().unsqueeze(0).unsqueeze(0)
    v_torch = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0)
    
    # Setup Fourier space grid
    N = pixels_per_dim
    k = (2 * np.pi / domain_length) * torch.fft.fftfreq(N)
    kx = k.view(-1, 1).repeat(1, N)
    ky = k.view(1, -1).repeat(N, 1)
    
    # Compute spatial derivatives using FFT
    u_fft = torch.fft.rfft2(u_torch.squeeze())
    v_fft = torch.fft.rfft2(v_torch.squeeze())
    
    u_x = torch.fft.irfft2(1j * kx[:, :u_fft.shape[-1]] * u_fft, s=(N, N))
    u_y = torch.fft.irfft2(1j * ky[:, :u_fft.shape[-1]] * u_fft, s=(N, N))
    v_x = torch.fft.irfft2(1j * kx[:, :v_fft.shape[-1]] * v_fft, s=(N, N))
    v_y = torch.fft.irfft2(1j * ky[:, :v_fft.shape[-1]] * v_fft, s=(N, N))
    
    # Compute residuals
    # Residual 1: Vorticity definition
    vorticity_from_velocity = v_x - u_y
    residual_vorticity = w_torch.squeeze() - vorticity_from_velocity
    
    # Residual 2: Incompressibility (divergence-free)
    residual_divergence = u_x + v_y
    
    res_vort = residual_vorticity.numpy()
    res_div = residual_divergence.numpy()
    
    res_vort_abs = np.abs(res_vort)
    res_div_abs = np.abs(res_div)
    
    return {
        'vorticity_residual': res_vort,
        'divergence_residual': res_div,
        'vorticity_residual_abs': res_vort_abs,
        'divergence_residual_abs': res_div_abs,
        'mean_vorticity_residual': np.mean(res_vort_abs),
        'mean_divergence_residual': np.mean(res_div_abs),
        'max_vorticity_residual': np.max(res_vort_abs),
        'max_divergence_residual': np.max(res_div_abs),
    }

# ============================================================================
# Create output directory
# ============================================================================
if save_figures and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================================================================
# Load residuals from sample_statistics.csv for each checkpoint
# ============================================================================

all_residuals = {}
all_stats = {}

for checkpoint_step in checkpoint_steps:
    load_path = f'./generated_samples_step_{checkpoint_step}'
    csv_path = os.path.join(load_path, 'sample_statistics.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping...")
        continue
    
    df = pd.read_csv(csv_path)
    
    # Extract residuals (exclude 'Mean' row if present)
    residuals = df[df['Sample Index'] != 'Mean']['Physics Residual'].astype(float).values
    
    all_residuals[checkpoint_step] = residuals
    
    all_stats[checkpoint_step] = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'median': np.median(residuals),
        'count': len(residuals)
    }
    
    print(f"\nCheckpoint {checkpoint_step}:")
    print(f"  Number of samples: {len(residuals)}")
    print(f"  Mean residual: {all_stats[checkpoint_step]['mean']:.6f}")
    print(f"  Std residual: {all_stats[checkpoint_step]['std']:.6f}")
    print(f"  Min residual: {all_stats[checkpoint_step]['min']:.6f}")
    print(f"  Max residual: {all_stats[checkpoint_step]['max']:.6f}")
    print(f"  Median residual: {all_stats[checkpoint_step]['median']:.6f}")

# ============================================================================
# Plot individual histograms for each checkpoint
# ============================================================================

# if plot_separate:
#     for checkpoint_step, residuals in all_residuals.items():
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         ax.hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
#         ax.axvline(all_stats[checkpoint_step]['mean'], color='red', linestyle='--', 
#                    linewidth=2, label=f"Mean: {all_stats[checkpoint_step]['mean']:.6f}")
#         ax.axvline(all_stats[checkpoint_step]['median'], color='orange', linestyle='--', 
#                    linewidth=2, label=f"Median: {all_stats[checkpoint_step]['median']:.6f}")
        
#         ax.set_xlabel('Physics Residual', fontsize=12)
#         ax.set_ylabel('Frequency', fontsize=12)
#         ax.set_title(f'Physics Residual Distribution - Checkpoint {checkpoint_step}', fontsize=14, fontweight='bold')
#         ax.legend(fontsize=10)
#         ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         if save_figures:
#             fig.savefig(os.path.join(output_dir, f'residual_histogram_step_{checkpoint_step}.png'), dpi=150)
#             print(f"Saved: {output_dir}/residual_histogram_step_{checkpoint_step}.png")
        
#         plt.close(fig)

# # ============================================================================
# # Plot combined histogram for comparison
# # ============================================================================

# if plot_combined and len(all_residuals) > 1:
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_residuals)))
    
#     for i, (checkpoint_step, residuals) in enumerate(all_residuals.items()):
#         ax.hist(residuals, bins=30, alpha=0.5, color=colors[i], 
#                 edgecolor='black', label=f'Step {checkpoint_step} (mean={all_stats[checkpoint_step]["mean"]:.6f})')
    
#     ax.set_xlabel('Physics Residual', fontsize=12)
#     ax.set_ylabel('Frequency', fontsize=12)
#     ax.set_title('Physics Residual Distribution - Checkpoint Comparison', fontsize=14, fontweight='bold')
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_figures:
#         fig.savefig(os.path.join(output_dir, 'residual_histogram_comparison.png'), dpi=150)
#         print(f"Saved: {output_dir}/residual_histogram_comparison.png")
    
#     plt.close(fig)

# ============================================================================
# Plot mean residuals vs checkpoint (if multiple checkpoints)
# ============================================================================

if len(all_residuals) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = list(all_stats.keys())
    means = [all_stats[step]['mean'] for step in steps]
    stds = [all_stats[step]['std'] for step in steps]
    
    ax.errorbar(steps, means, yerr=stds, fmt='o-', linewidth=2, markersize=8, 
                capsize=5, capthick=2, color='steelblue')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Mean Physics Residual', fontsize=12)
    ax.set_title('Mean Physics Residual vs Training Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figures:
        fig.savefig(os.path.join(output_dir, 'residual_vs_step.png'), dpi=150)
        print(f"Saved: {output_dir}/residual_vs_step.png")
    
    plt.close(fig)

# ============================================================================
# Compute and Plot Spatial Physics Residuals
# ============================================================================

print("\n" + "="*60)
print("Computing detailed physics residuals for each sample...")
print("="*60)

all_sample_residuals = {}

for checkpoint_step in checkpoint_steps:
    load_path = f'./generated_samples_step_{checkpoint_step}'
    
    if not os.path.exists(load_path):
        continue
    
    # Get list of sample directories
    sample_dirs = [d for d in os.listdir(load_path) 
                   if os.path.isdir(os.path.join(load_path, d)) and d.startswith('sample_')]
    sample_dirs.sort()
    
    checkpoint_residuals = {
        'vorticity': [],
        'divergence': [],
        'vorticity_spatial': [],
        'divergence_spatial': []
    }
    
    for sample_dir in sample_dirs:
        sample_path = os.path.join(load_path, sample_dir)
        
        w, u, v = load_sample_fields(sample_path)
        
        residuals = compute_physics_residuals(w, u, v, DOMAIN_LENGTH, PIXELS_PER_DIM)
        
        checkpoint_residuals['vorticity'].append(residuals['mean_vorticity_residual'])
        checkpoint_residuals['divergence'].append(residuals['mean_divergence_residual'])
        checkpoint_residuals['vorticity_spatial'].append(residuals['vorticity_residual'])
        checkpoint_residuals['divergence_spatial'].append(residuals['divergence_residual'])
        
        print(f"  {sample_dir}: Vorticity residual = {residuals['mean_vorticity_residual']:.6f}, "
              f"Divergence residual = {residuals['mean_divergence_residual']:.6f}")
        
        # Plot spatial distribution for first sample of each checkpoint
        if plot_spatial_residuals and sample_dir == 'sample_0':
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Vorticity residual
            im1 = axes[0, 0].imshow(residuals['vorticity_residual'], cmap='RdBu_r', origin='lower')
            axes[0, 0].set_title(f'Vorticity Residual\n(ω - (∂v/∂x - ∂u/∂y))', fontsize=12)
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(residuals['vorticity_residual_abs'], cmap='hot', origin='lower')
            axes[0, 1].set_title(f'|Vorticity Residual|\nMean: {residuals["mean_vorticity_residual"]:.6f}', fontsize=12)
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[0, 1])
            
            axes[0, 2].hist(residuals['vorticity_residual'].flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0, 2].set_xlabel('Vorticity Residual')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Histogram')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Row 2: Divergence residual
            im3 = axes[1, 0].imshow(residuals['divergence_residual'], cmap='RdBu_r', origin='lower')
            axes[1, 0].set_title(f'Divergence Residual\n(∂u/∂x + ∂v/∂y)', fontsize=12)
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            plt.colorbar(im3, ax=axes[1, 0])
            
            im4 = axes[1, 1].imshow(residuals['divergence_residual_abs'], cmap='hot', origin='lower')
            axes[1, 1].set_title(f'|Divergence Residual|\nMean: {residuals["mean_divergence_residual"]:.6f}', fontsize=12)
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('y')
            plt.colorbar(im4, ax=axes[1, 1])
            
            axes[1, 2].hist(residuals['divergence_residual'].flatten(), bins=50, alpha=0.7, color='coral', edgecolor='black')
            axes[1, 2].set_xlabel('Divergence Residual')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Histogram')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Physics Residuals - Checkpoint {checkpoint_step} - Sample 0', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            if save_figures:
                fig.savefig(os.path.join(output_dir, f'spatial_residuals_step_{checkpoint_step}_sample_0.png'), 
                           dpi=150, bbox_inches='tight')
                print(f"  Saved: {output_dir}/spatial_residuals_step_{checkpoint_step}_sample_0.png")
            
            plt.close(fig)
    
    all_sample_residuals[checkpoint_step] = checkpoint_residuals

# ============================================================================
# Plot residual distributions across all samples
# ============================================================================

if plot_residual_histograms and all_sample_residuals:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Vorticity residuals
    for checkpoint_step, residuals in all_sample_residuals.items():
        if len(residuals['vorticity']) > 0:
            axes[0].hist(residuals['vorticity'], bins=20, alpha=0.6, 
                        label=f'Step {checkpoint_step}', edgecolor='black')
    
    axes[0].set_xlabel('Mean Vorticity Residual |ω - (∂v/∂x - ∂u/∂y)|', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Vorticity Residual Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Divergence residuals
    for checkpoint_step, residuals in all_sample_residuals.items():
        if len(residuals['divergence']) > 0:
            axes[1].hist(residuals['divergence'], bins=20, alpha=0.6, 
                        label=f'Step {checkpoint_step}', edgecolor='black')
    
    axes[1].set_xlabel('Mean Divergence Residual |∂u/∂x + ∂v/∂y|', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Divergence Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_figures:
        fig.savefig(os.path.join(output_dir, 'residual_distributions_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/residual_distributions_comparison.png")
    
    plt.close(fig)

# ============================================================================
# Summary statistics
# ============================================================================

print("\n" + "="*60)
print("Summary of Physics Residuals:")
print("="*60)

for checkpoint_step, residuals in all_sample_residuals.items():
    if len(residuals['vorticity']) > 0:
        print(f"\nCheckpoint {checkpoint_step}:")
        print(f"  Vorticity residual (mean ± std): {np.mean(residuals['vorticity']):.6f} ± {np.std(residuals['vorticity']):.6f}")
        print(f"  Divergence residual (mean ± std): {np.mean(residuals['divergence']):.6f} ± {np.std(residuals['divergence']):.6f}")

print("\n" + "="*60)
print("Evaluation complete!")
print("="*60)