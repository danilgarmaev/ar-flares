# Physics-Informed Attention Mechanism

## Overview
This module implements a novel "Physics-Informed Attention" mechanism for Solar Flare prediction.
Instead of relying solely on static texture (CNN) or generic temporal dynamics (LSTM/Optical Flow), we explicitly leverage the **Flux Emergence** property of active regions.

## Hypothesis
Solar flares are driven by the rapid emergence and cancellation of magnetic flux. This dynamic activity is most visible in the **difference images** ($I_t - I_{t-1}$).
By using the difference image to generate a spatial attention map, we can force the model to focus its feature extraction on regions of high magnetic activity.

## Implementation
### 1. Data Loading (`datasets.py`)
- When `use_diff_attention=True`, the dataloader fetches pairs of frames: $[t-1, t]$.
- It computes the difference: $D = I_t - I_{t-1}$.
- It yields a tuple: `(Image_t, Diff_t)`.

### 2. Model Architecture (`models.py`)
- **Class:** `DiffAttentionModel`
- **Branch 1 (Static):** Standard Backbone (e.g., ConvNeXt) extracts features from $I_t$.
- **Branch 2 (Dynamic):** A lightweight CNN processes $D$ to produce a "Gate" or "Attention Map".
- **Fusion:** The static features are modulated by the dynamic gate: $F' = F \cdot (1 + Gate)$.
- This acts as a residual attention mechanism: if the dynamics are uninformative, the gate can be 0, reverting to the static model.

## Usage
To run the experiment:
```bash
python run_physics_attention.py
```

## Configuration
In `config.py` or via overrides:
```python
CFG["use_diff_attention"] = True
CFG["backbone"] = "convnext_tiny" # or any other supported backbone
```
