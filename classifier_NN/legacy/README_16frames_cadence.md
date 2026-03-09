# 16-Frame Cadence Experiments

This directory contains experiments testing video models with 16-frame sequences at different temporal cadences, matching the methodology from the macros project.

## Overview

The macros project (from another paper using the same dataset) creates video sequences with:
- **16 frames per video**
- **72 minutes between consecutive frames**
- **18 hours total temporal context**

We're testing how different temporal cadences affect model performance.

## Temporal Configurations

| Cadence | Stride | Offsets | Total Context | Notes |
|---------|--------|---------|---------------|-------|
| 36 min  | 3 steps | [-45, -42, ..., -3, 0] | 9 hours | Shorter context, finer temporal detail |
| 72 min  | 6 steps | [-90, -84, ..., -6, 0] | **18 hours** | **Matches macros project** |
| 108 min | 9 steps | [-135, -126, ..., -9, 0] | 27 hours | Extended context, coarser sampling |

*Note: Base temporal resolution is 12 minutes (native HMI cadence)*

## Your Previous Experiments

Your previous attempts used:
- **8 frames** (T=8)
- **48 minutes between frames** (k=4)
- **~5.6 hours total context** (28 steps × 12 min)
- **Best TSS: ~0.40**

This is **3.2× shorter temporal context** than the macros project, which may explain the suboptimal performance.

## Models

### Available Models
- **r2plus1d_18**: R(2+1)D from torchvision (READY ✅)
- **r3d_18**: 3D ResNet from torchvision (READY ✅)
- **timesformer**: TimeSformer from HuggingFace (READY ✅)
- **slowfast**: SlowFast from pytorchvideo (READY ✅, but requires T≥32)

### VideoMAE Status
**VideoMAE is NOT YET IMPLEMENTED** ❌

To add VideoMAE support, you would need to:

1. **Install transformers with VideoMAE support:**
   ```bash
   pip install transformers>=4.30.0
   ```

2. **Add VideoMAE model class in `classifier_NN/models.py`:**
   ```python
   class VideoMAEWrapper(nn.Module):
       """Wrapper around HuggingFace VideoMAE for sequence magnetograms."""
       
       def __init__(
           self,
           num_frames: int = 16,
           image_size: int = 224,
           num_classes: int = 2,
           pretrained: bool = True,
           model_id: str = "MCG-NJU/videomae-base",
       ):
           super().__init__()
           from transformers import VideoMAEForVideoClassification
           
           # Convert grayscale to RGB
           self.input_proj = nn.Conv3d(1, 3, kernel_size=1)
           
           # Load VideoMAE
           if pretrained:
               self.model = VideoMAEForVideoClassification.from_pretrained(
                   model_id,
                   num_labels=num_classes,
                   ignore_mismatched_sizes=True,
               )
           else:
               from transformers import VideoMAEConfig
               config = VideoMAEConfig(
                   image_size=image_size,
                   num_frames=num_frames,
                   num_channels=3,
                   num_labels=num_classes,
               )
               self.model = VideoMAEForVideoClassification(config)
       
       def forward(self, x):
           # x: (B, T, 1, H, W) -> (B, 1, T, H, W)
           x = x.permute(0, 2, 1, 3, 4)
           x = self.input_proj(x)  # (B, 3, T, H, W)
           # VideoMAE expects (B, T, 3, H, W) or (B, 3, T, H, W) depending on version
           # Check the transformers docs for the exact format
           return self.model(x).logits
   ```

3. **Add case in `build_model()` function:**
   ```python
   if backbone.lower() in ["videomae", "video_mae"]:
       model = VideoMAEWrapper(
           num_frames=cfg.get("seq_T", 16),
           image_size=cfg.get("image_size", 224),
           num_classes=num_classes,
           pretrained=cfg.get("pretrained_videomae", True),
       )
       return model
   ```

4. **Prefetch weights (on login node with internet):**
   ```bash
   python -c "from transformers import VideoMAEForVideoClassification; \
              VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base')"
   ```

## Quick Start

### Test Single Configuration (Quick Sanity Check)
```bash
bash scripts/test_16frames_cadence.sh
```
This runs R(2+1)D with 72-min cadence for 5 epochs as a quick test.

### Full Sweep (All 3 Cadences, 3 Seeds Each)
```bash
bash scripts/run_16frames_cadence_sweep.sh
```
This runs 9 experiments total (3 cadences × 3 seeds).

### Custom Run
```python
# Single experiment
python -m classifier_NN.legacy.run_experiments_16frames_cadence \
    --backbone r2plus1d_18 \
    --interval-min 72 \
    --seed 0 \
    --single \
    --pretrained-3d

# Multiple cadences, single seed
python -m classifier_NN.legacy.run_experiments_16frames_cadence \
    --backbone r2plus1d_18 \
    --interval-min 36 \
    --interval-min 72 \
    --interval-min 108 \
    --seed 0 \
    --pretrained-3d

# Full sweep with custom parameters
python -m classifier_NN.legacy.run_experiments_16frames_cadence \
    --backbone r2plus1d_18 \
    --interval-min 36 --interval-min 72 --interval-min 108 \
    --n-seeds 3 \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --pretrained-3d
```

## Configuration Details

### Fixed Parameters
- **Frames per sequence (T)**: 16
- **Image size**: 112×112
- **Batch size**: 8 (adjust based on GPU memory)
- **Optimizer**: AdamW with weight_decay=0.01
- **Learning rate**: 1e-4
- **Scheduler**: Cosine annealing
- **Epochs**: 50 (with early stopping patience=5)
- **Model selection**: Best TSS on validation set

### Variable Parameters
- **Cadence** (interval_min): 36, 72, or 108 minutes
- **Temporal stride**: Automatically computed from cadence (cadence ÷ 12)
- **Offsets**: Automatically generated based on cadence and T=16

## Expected Outcomes

Based on similar experiments:
- **Longer temporal context (72-108 min)** should capture magnetic field evolution better
- **18-hour window (72 min)** is validated by the macros project
- **R(2+1)D** should perform better than your previous 8-frame attempt (TSS ~0.40)

## Results Location

Results are saved in:
- Individual experiments: `results/16frames_cadence{X}min_r2plus1d_18_seed{Y}/`
- Aggregated summary: `results/summary_16frames_cadence_sweep.json`

## Data Consistency

The experiment script uses the same WebDataset shards as your previous experiments, so the data is consistent. The only differences are:
1. More frames per sequence (16 vs 8)
2. Different temporal sampling (varied cadences)
3. Longer temporal context (9-27 hours vs 5.6 hours)

## References

- Original dataset: [Dryad Repository](https://doi.org/10.5061/dryad.jq2bvq898)
- Macros project configuration: `macros/make_hmi_video_metadata.py`
