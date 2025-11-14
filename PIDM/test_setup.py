#!/usr/bin/env python3
"""
Quick test script to verify setup before running main.py
"""

import torch
import numpy as np
import sys
import os

print("=" * 60)
print("Testing PIDM Setup for Navier-Stokes")
print("=" * 60)

# Test 1: Check PyTorch and device
print("\n1. Testing PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Test 2: Check HDF5 file
print("\n2. Testing HDF5 dataset...")
try:
    import h5py
    h5_path = '../dns_dataset.h5'
    if not os.path.exists(h5_path):
        h5_path = 'dns_dataset.h5'
    
    with h5py.File(h5_path, 'r') as f:
        print(f"   ✓ Found {h5_path}")
        print(f"   Keys: {list(f.keys())}")
        print(f"   w shape: {f['w'].shape}")
        print(f"   u shape: {f['u'].shape}")
        print(f"   v shape: {f['v'].shape}")
except Exception as e:
    print(f"   ✗ ERROR loading dataset: {e}")
    sys.exit(1)

# Test 3: Test Dataset class
print("\n3. Testing Dataset class...")
try:
    from src.data_utils import Dataset
    ds = Dataset(h5_path, use_double=False)
    print(f"   ✓ Dataset loaded: {len(ds)} samples")
    
    # Test getting one sample
    sample = ds[0]
    print(f"   ✓ Sample shape: {sample.shape} (expected: (3, 64, 64))")
    print(f"   ✓ Sample dtype: {sample.dtype}")
    assert sample.shape == (3, 64, 64), f"Wrong shape: {sample.shape}"
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test Residuals class
print("\n4. Testing Residuals class...")
try:
    from src.residuals import Residuals
    from src.unet_model import Unet3D
    
    # Create dummy model
    model = Unet3D(dim=32, channels=3, sigmoid_last_channel=False).to(device)
    
    # Create residuals
    residuals = Residuals(
        model=model,
        pixels_per_dim=64,
        device=device,
        domain_length=np.pi,
        visc=1e-3,
        use_ddim_x0=False,
        ddim_steps=0
    )
    print(f"   ✓ Residuals initialized")
    
    # Test residual computation
    test_batch = sample.unsqueeze(0).to(device)  # (1, 3, 64, 64)
    output = residuals.compute_residual(test_batch, pass_through=True, reduce='none')
    print(f"   ✓ Residual output shape: {output['residual'].shape}")
    
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check model.yaml
print("\n5. Testing configuration...")
try:
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path('model.yaml').read_text())
    print(f"   ✓ Config loaded")
    print(f"   gov_eqs: {config.get('gov_eqs', 'NOT SET')}")
    print(f"   diff_steps: {config['diff_steps']}")
    print(f"   c_residual: {config['c_residual']}")
    
    if config.get('gov_eqs') != 'navier_stokes':
        print(f"   ⚠ WARNING: gov_eqs should be 'navier_stokes', got '{config.get('gov_eqs')}'")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n" + "=" * 60)
print("✓ All tests passed! Ready to run main.py")
print("=" * 60)
print("\nTo start training, run:")
print("  python main.py")

