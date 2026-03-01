#!/usr/bin/env python3
"""
Debug script: Diagnose model inference differences between systems.
Checks model configuration, normalization stats, and determinism.
"""

import json
import sys
import torch
import numpy as np
from pathlib import Path

# Add source to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_network.physdeepsif import build_physdeepsif

# ========================================================================
# Configuration
# ========================================================================
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / "checkpoint_best.pt"
NORM_STATS_PATH = PROJECT_ROOT / "outputs" / "models" / "normalization_stats.json"
LEADFIELD_PATH = PROJECT_ROOT / "data" / "leadfield_19x76.npy"
CONNECTIVITY_PATH = PROJECT_ROOT / "data" / "connectivity_76.npy"

# ========================================================================
# Main Debug
# ========================================================================

def main():
    print("\n" + "="*70)
    print("PhysDeepSIF Inference Debugging")
    print("="*70)
    
    # 1. Check paths
    print("\n[1] Checking file paths...")
    for name, path in [
        ("Checkpoint", CHECKPOINT_PATH),
        ("Normalization Stats", NORM_STATS_PATH),
        ("Leadfield", LEADFIELD_PATH),
        ("Connectivity", CONNECTIVITY_PATH),
    ]:
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}: {path}")
    
    if not CHECKPOINT_PATH.exists():
        print("\n[ERROR] Checkpoint not found!")
        return 1
    
    # 2. Load model
    print("\n[2] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = build_physdeepsif(
        leadfield_path=str(LEADFIELD_PATH),
        connectivity_path=str(CONNECTIVITY_PATH),
        lstm_dropout=0.0,
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)
    
    # Extract model state
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()  # CRITICAL: set to eval mode
    
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check model is in eval mode
    training_mode = any(m.training for m in model.modules())
    print(f"  Eval mode: {'✓' if not training_mode else '✗ (WRONG!)'}")
    
    # 3. Check normalization stats
    print("\n[3] Checking normalization statistics...")
    with open(NORM_STATS_PATH, 'r') as f:
        norm_stats = json.load(f)
    
    for key in ['eeg_mean', 'eeg_std', 'src_mean', 'src_std']:
        print(f"  {key}: {norm_stats[key]:.6f}")
    
    # 4. Test determinism (same input → same output?)
    print("\n[4] Testing model determinism...")
    test_eeg = torch.randn(1, 19, 400, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out1 = model(test_eeg).cpu().numpy()
        out2 = model(test_eeg).cpu().numpy()
    
    max_diff = np.abs(out1 - out2).max()
    print(f"  Max difference (same input, two runs): {max_diff:.2e}")
    if max_diff < 1e-6:
        print("  ✓ Model is deterministic")
    else:
        print(f"  ✗ Model produces different outputs (max diff: {max_diff})")
    
    # 5. Check for stochastic modules
    print("\n[5] Checking for stochastic modules...")
    stochastic_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            p = module.p
            if p > 0:
                stochastic_modules.append((name, f"Dropout(p={p})"))
    
    if stochastic_modules:
        print("  ✗ Found stochastic modules:")
        for name, desc in stochastic_modules:
            print(f"    - {name}: {desc}")
    else:
        print("  ✓ No active dropout layers")
    
    # 6. Check inference preprocessing consistency
    print("\n[6] Testing inference preprocessing...")
    
    # Simulate what the API does
    test_eeg_np = np.random.randn(19, 400).astype(np.float32)
    
    # Method 1: De-mean then normalize
    eeg_t1 = torch.from_numpy(test_eeg_np).to(device)
    eeg_t1 = eeg_t1 - eeg_t1.mean(dim=-1, keepdim=True)  # De-mean
    eeg_t1 = (eeg_t1 - norm_stats['eeg_mean']) / (norm_stats['eeg_std'] + 1e-7)
    
    # Method 2: Normalize then de-mean (wrong order)
    eeg_t2 = torch.from_numpy(test_eeg_np).to(device)
    eeg_t2 = (eeg_t2 - norm_stats['eeg_mean']) / (norm_stats['eeg_std'] + 1e-7)
    eeg_t2 = eeg_t2 - eeg_t2.mean(dim=-1, keepdim=True)  # De-mean
    
    diff = (eeg_t1 - eeg_t2).abs().max().item()
    print(f"  De-mean→normalize vs normalize→de-mean: diff = {diff:.6f}")
    if diff > 0.01:
        print("  ⚠ WARNING: Preprocessing order matters!")
    
    # 7. Run inference on realistic EEG
    print("\n[7] Running realistic inference...")
    
    # Create a "realistic" EEG: low freq (< 50 Hz)
    import scipy.signal as sig
    t = np.arange(400) / 200.0  # 2 seconds at 200 Hz
    realistic_eeg = np.zeros((19, 400), dtype=np.float32)
    for ch in range(19):
        # Mix of 10Hz and 25Hz sine waves
        alpha = 30.0 * np.sin(2 * np.pi * 10 * t)
        beta = 20.0 * np.sin(2 * np.pi * 25 * t)
        realistic_eeg[ch, :] = (alpha + beta) / 2.0
    
    # Run inference (matching API code exactly)
    eeg_t = torch.from_numpy(realistic_eeg[np.newaxis, ...]).to(device)
    eeg_t = eeg_t - eeg_t.mean(dim=-1, keepdim=True)  # De-mean
    eeg_t = (eeg_t - norm_stats['eeg_mean']) / (norm_stats['eeg_std'] + 1e-7)
    
    with torch.no_grad():
        source_pred = model(eeg_t)  # (1, 76, 400)
    
    source_np = source_pred.cpu().numpy()[0]  # (76, 400)
    
    print(f"  Source shape: {source_np.shape}")
    print(f"  Source range: [{source_np.min():.4f}, {source_np.max():.4f}]")
    print(f"  Source mean: {source_np.mean():.6f}")
    print(f"  Source std: {source_np.std():.6f}")
    
    # Compute epileptogenicity (same as API)
    region_range = np.ptp(source_np, axis=1)  # (76,)
    inverted_range = -region_range
    
    z_scores = (inverted_range - inverted_range.mean()) / (inverted_range.std() + 1e-10)
    ei_raw = 1.0 / (1.0 + np.exp(-z_scores))  # Sigmoid
    
    top_5_idx = np.argsort(ei_raw)[::-1][:5]
    print(f"\n  Top 5 epileptogenic regions (by EI):")
    for i, idx in enumerate(top_5_idx, 1):
        print(f"    {i}. Region {idx}: EI = {ei_raw[idx]:.4f}, Range = {region_range[idx]:.4f}")
    
    print("\n" + "="*70)
    print("Debug complete. Check output above for issues.")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
