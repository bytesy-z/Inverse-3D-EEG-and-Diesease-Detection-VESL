---
license: mit
library_name: pytorch
tags:
- eeg
- source-localization
- epilepsy
- brain
- physdeepsif
- tvb
---

# PhysDeepSIF — Physics-Informed Deep Source Imaging Framework

## Model Description

PhysDeepSIF is a physics-constrained deep learning model for **EEG source localization** and **epileptogenicity mapping**. It takes 19-channel scalp EEG (10-20 montage, linked-ear reference) and produces 76-region source activity estimates using a biophysically grounded approach.

The model combines:
- **Spatial Module**: MLP `19→128→256→256→128→76` with ReLU, BatchNorm, and skip connections
- **Temporal Module**: 2-layer BiLSTM (hidden=76, dropout=0.1)
- **Physics-informed loss**: Amplitude + temporal smoothness constraints

Pre-trained on ~23,000 synthetic TVB Epileptor simulations.

## Architecture

```
Input:  19-channel EEG (400 time samples)
  ↓
Spatial MLP (19→128→256→256→128→76)
  ↓
BiLSTM (2 layers, hidden=76)
  ↓
Output: 76-region source activity (× 400 time samples)
```

## Input / Output

| | Shape | Description |
|---|---|---|
| **Input** | `(19, 400)` | 19 EEG channels (10-20 montage), 2 seconds at 200 Hz, linked-ear reference |
| **Output** | `(76, 400)` | Source activity for 76 Desikan-Killiany brain regions over time |

## Performance

- **Distance Localization Error (DLE)**: 31 mm
- **Area Under Curve (AUC)**: 0.923
- **Regions**: 76 (Desikan-Killiany parcellation)

## Training Data

Synthetic EEG generated using:
- **Brain model**: The Virtual Brain (TVB) Epileptor neural mass model
- **Forward model**: MNE BEM (Boundary Element Method) on fsaverage template head
- **Leadfield**: 19×76 projection matrix
- **Augmentation**: White noise (SNR 5-30 dB), colored 1/f^α noise, skull attenuation, spectral shaping

## Usage

```python
import torch
import numpy as np

# Load model
model = torch.load("physdeepsif_best_checkpoint.pt")
model.eval()

# Load your 19-channel EEG data (400 samples, 200 Hz)
eeg = np.load("eeg_data.npy")  # shape: (19, 400)

# Run inference
with torch.no_grad():
    source_activity = model(torch.from_numpy(eeg).float().unsqueeze(0))
    source_activity = source_activity.squeeze(0).numpy()

print(source_activity.shape)  # (76, 400)
```

## Dependencies

- PyTorch ≥ 2.1
- NumPy

## Citation

Academic final year project. Architecture inspired by Sun et al. (2022, PNAS) — DeepSIF. Epileptor model from Jirsa et al. (2014).
