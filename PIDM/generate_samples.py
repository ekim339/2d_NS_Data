#!/usr/bin/env python3
"""
Generate samples with GIFs from a trained checkpoint without training
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals import Residuals
from src.data_utils import generalized_b_xy_c_to_image

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

checkpoint_name = 'run_1'         
checkpoint_step = 0                 # 0 = untrained model (before training)
num_samples = 8                     
create_gifs = True                 
output_dir = f'generated_samples_step_{checkpoint_step}' 

# ============================================================================
# Load Configuration
# ============================================================================

load_path = f'../trained_models/{checkpoint_name}'
config_path = Path(load_path, 'model', 'model.yaml')
checkpoint_path = Path(load_path, 'model', f'checkpoint_{checkpoint_step}.pt')

print("=" * 60)
print("Sample Generation from Trained Checkpoint")
print("=" * 60)

if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint not found: {checkpoint_path}")
    print(f"Available checkpoints in {load_path}/model/:")
    if os.path.exists(f'{load_path}/model/'):
        for f in os.listdir(f'{load_path}/model/'):
            if f.startswith('checkpoint_'):
                print(f"  - {f}")
    exit(1)

# Load config
config = yaml.safe_load(Path(config_path).read_text())
print(f"\nLoaded config from: {config_path}")
print(f"Loading checkpoint: {checkpoint_path}")

# ============================================================================
# Setup
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Parameters from config
diff_steps = config['diff_steps']
use_ddim_x0 = config['x0_estimation'] == 'sample'
ddim_steps = config['ddim_steps']
residual_grad_guidance = config['residual_grad_guidance']
M_correction = config['M_correction']
N_correction = config['N_correction']
correction_mode = config['correction_mode']

# Model parameters
output_dim = 3
pixels_per_dim = 64
domain_length = np.pi

print(f"\nModel Settings:")
print(f"  Diffusion steps: {diff_steps}")
print(f"  Output channels: {output_dim} (w, u, v)")
print(f"  Grid size: {pixels_per_dim}×{pixels_per_dim}")

# ============================================================================
# Create Model and Load Weights
# ============================================================================

model = Unet3D(dim=32, channels=output_dim, sigmoid_last_channel=False).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {num_params:,}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()
print(f"\nLoaded model weights from step {checkpoint_step}")

# ============================================================================
# Create Residuals (for physics evaluation)
# ============================================================================

residuals = Residuals(
    model=model,
    pixels_per_dim=pixels_per_dim,
    device=device,
    domain_length=domain_length,
    visc=1e-3,
    use_ddim_x0=use_ddim_x0,
    ddim_steps=ddim_steps
)

# ============================================================================
# Create Diffusion Utils
# ============================================================================

diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)

# ============================================================================
# Generate Samples
# ============================================================================

print(f"\n{'='*60}")
print(f"Generating {num_samples} samples...")
print(f"{'='*60}\n")

conditioning_input = None
sample_shape = (num_samples, output_dim, pixels_per_dim, pixels_per_dim)

output = diffusion_utils.p_sample_loop(
    conditioning_input, 
    sample_shape,
    save_output=True,
    surpress_noise=True,
    use_dynamic_threshold=False,
    residual_func=residuals,
    eval_residuals=True,
    return_optimizer=False,
    return_inequality=False,
    M_correction=M_correction,
    N_correction=N_correction,
    correction_mode=correction_mode
)

seqs = output[0]
residual = output[1]['residual']
residual = residual.abs().mean(dim=tuple(range(1, residual.ndim)))
if residual.ndim == 0:
    residual = residual.unsqueeze(0)

print(f"\nGeneration complete!")
print(f"  Mean physics residual: {residual.mean():.2e}")

# ============================================================================
# Save Outputs
# ============================================================================

os.makedirs(output_dir, exist_ok=True)
print(f"\nSaving to: {output_dir}/")

labels = ['sample', 'model_output']
channels_names = ['vorticity', 'u_velocity', 'v_velocity']

for seq_idx, seq in enumerate(seqs):
    if seq_idx == 1:  # Skip model_output
        continue
    
    seq = torch.stack(seq, dim=0)
    if len(seq.shape) == 6:
        seq = seq.squeeze(-3)
    
    last_preds = seq[-1].numpy()
    
    for sel_sample in range(num_samples):
        sample_dir = os.path.join(output_dir, f'sample_{sel_sample}')
        os.makedirs(sample_dir, exist_ok=True)
        
        for sel_channel in range(output_dim):
            # Get final prediction
            last_pred = last_preds[sel_sample, sel_channel]
            
            # Save CSV
            csv_filename = f'{channels_names[sel_channel]}.csv'
            np.savetxt(os.path.join(sample_dir, csv_filename), last_pred, delimiter=',')
            
            # Save PNG
            last_pred_normalized = (last_pred - last_pred.min()) / (last_pred.max() - last_pred.min())
            image = np.uint8(last_pred_normalized * 255)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image, cmap='bwr', vmin=0, vmax=255)
            ax.set_title(f'{channels_names[sel_channel]} - Sample {sel_sample}')
            ax.axis('off')
            
            res_idx = min(sel_sample, len(residual) - 1)
            plt.text(0.02, 0.98, f'Physics residual: {residual[res_idx]:.2e}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            png_filename = f'{channels_names[sel_channel]}.png'
            plt.savefig(os.path.join(sample_dir, png_filename), bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            if create_gifs:
                sel_seq = seq[:, sel_sample, sel_channel].detach().cpu().numpy()
                gif_filename = f'{channels_names[sel_channel]}.gif'
                gif_path = os.path.join(sample_dir, gif_filename)
                image_array_to_gif(sel_seq, gif_path)
                print(f"  Sample {sel_sample}, {channels_names[sel_channel]}: PNG + GIF")
            else:
                print(f"  Sample {sel_sample}, {channels_names[sel_channel]}: PNG")

# Save statistics
import pandas as pd
df_data = {
    'Sample Index': list(range(len(residual))) + ['Mean'],
    'Physics Residual': list(residual.detach().cpu().numpy()) + [residual.mean().item()]
}
df = pd.DataFrame(df_data)
stats_path = os.path.join(output_dir, 'sample_statistics.csv')
df.to_csv(stats_path, index=False)
print(f"\nStatistics saved to: {stats_path}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*60}")
print("Generation Complete!")
print(f"{'='*60}")
print(f"\nGenerated {num_samples} samples with {output_dim} channels each")
print(f"Output directory: {output_dir}/")
print(f"\nFiles per sample:")
print(f"  - {channels_names[0]}.png/csv/gif (vorticity)")
print(f"  - {channels_names[1]}.png/csv/gif (u-velocity)")
print(f"  - {channels_names[2]}.png/csv/gif (v-velocity)")
if create_gifs:
    print(f"\nAnimated GIFs created showing {diff_steps}-step denoising process")
print(f"\nPhysics residual (lower is better):")
print(f"  Mean: {residual.mean():.2e}")
print(f"  Min:  {residual.min():.2e}")
print(f"  Max:  {residual.max():.2e}")

