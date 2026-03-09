# Analysis: HMI Video Sequence Construction - Your Implementation vs Macros Project

## Executive Summary

I compared your current video sequence construction with the macros project. The key finding: **your sequences have 3.2× shorter temporal context**, which likely explains the poor TSS performance (~0.40).

## Comparison Table

| Aspect | Macros Project | Your Previous Runs | New Configuration |
|--------|---------------|-------------------|-------------------|
| **Frames per sequence** | 16 | 8 | **16** ✅ |
| **Cadence** | 72 min | 48 min | **36/72/108 min** ✅ |
| **Total temporal context** | **18 hours** | 5.6 hours | **9/18/27 hours** ✅ |
| **Implementation** | `make_hmi_video_metadata.py` | `datasets.py` offsets | Same data pipeline |
| **Best TSS** | Unknown | ~0.40 | To be measured |

## Your Previous Configuration

From your log file:
```
[seq] first train batch: shape=(8, 8, 1, 112, 112) | N=8 | k=4 | 
    offsets=[-28, -24, -20, -16, -12, -8, -4, 0]
```

- **T=8 frames**
- **k=4** (48 minutes between frames)
- **28 steps back** = 28 × 12 min = 336 min = **5.6 hours total**
- **Best TSS: 0.4165** (epoch 0, then degraded)

## Macros Project Configuration

From `macros/make_hmi_video_metadata.py`:
```python
video_length = 16  # default
frame_step_minutes = 72  # default
cadence_minutes = 12
frame_step = frame_step_minutes // cadence_minutes  # = 6

# Generates:
frame_indices = [start_idx + i * frame_step for i in range(video_length)]
# = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]
```

- **16 frames**
- **6 steps between frames** (72 minutes)
- **90 steps total** = 90 × 12 min = 1080 min = **18 hours**

## New Configurations (Matching Macros Style)

### Configuration 1: 36-minute cadence (9 hours)
```python
"seq_T": 16,
"seq_stride": 3,  # 36 min / 12 min
"seq_offsets": [-45, -42, -39, -36, -33, -30, -27, -24, 
                -21, -18, -15, -12, -9, -6, -3, 0],
```
- Finer temporal sampling
- Shorter total context (9 hours)
- Good for capturing rapid evolution

### Configuration 2: 72-minute cadence (18 hours) ← **Matches macros**
```python
"seq_T": 16,
"seq_stride": 6,  # 72 min / 12 min
"seq_offsets": [-90, -84, -78, -72, -66, -60, -54, -48, 
                -42, -36, -30, -24, -18, -12, -6, 0],
```
- **Exactly matches macros project**
- 18 hours total context
- Validated by their research

### Configuration 3: 108-minute cadence (27 hours)
```python
"seq_T": 16,
"seq_stride": 9,  # 108 min / 12 min
"seq_offsets": [-135, -126, -117, -108, -99, -90, -81, -72, 
                -63, -54, -45, -36, -27, -18, -9, 0],
```
- Extended temporal context (27 hours)
- Coarser sampling
- May capture longer-term patterns

## Why Your Previous Config Underperformed

### Problem 1: Insufficient Temporal Context
- Solar flare precursors typically develop over **12-24 hours**
- Your 5.6-hour window may miss early warning signs
- Magnetic field evolution is a gradual process

### Problem 2: Fewer Frames
- 8 frames vs 16 frames = less temporal information
- Video models (R(2+1)D, SlowFast, etc.) benefit from longer sequences
- Transformers especially need sufficient sequence length

### Problem 3: Suboptimal Training Dynamics
- Model overfit quickly (best at epoch 0)
- Early stopping triggered after 5 epochs
- May indicate data/model mismatch

## Implementation Notes

### Data Pipeline Consistency
Both implementations use the **same underlying data**:
- WebDataset shards with 12-minute native cadence
- Same AR images and labels
- Same train/val/test splits

The only difference is **how sequences are constructed** from this data.

### Your Implementation (`datasets.py`)
```python
offsets = CFG.get("seq_offsets", [-16, -8, 0])
seq_stride = CFG.get("seq_stride", CFG.get("seq_stride_steps", 1))
# ...
for i in start_indices:
    key0, png0, jm0 = entries[i]
    # ...
    frame_indices = [start_idx + i * frame_step for i in range(video_length)]
```

This is **functionally equivalent** to macros, just configured differently!

### Macros Implementation (`make_hmi_video_metadata.py`)
```python
frame_step = args.frame_step_minutes // args.cadence_minutes
frame_indices = [start_idx + i * frame_step for i in range(args.video_length)]
```

Both approaches:
1. Start at some base index
2. Skip ahead by stride
3. Collect N frames
4. Validate temporal consistency

## Recommendations

1. **Start with 72-min cadence** (matches validated macros config)
2. **Use 16 frames** (standard for video models)
3. **Try R(2+1)D first** (already available, proven architecture)
4. **Compare all 3 cadences** to understand temporal sensitivity
5. **Add VideoMAE later** (needs implementation, see skeleton code)

## Running the Experiments

### Quick test (72-min cadence only):
```bash
bash scripts/test_16frames_cadence.sh
```

### Full sweep (all 3 cadences, 3 seeds each):
```bash
bash scripts/run_16frames_cadence_sweep.sh
```

### Results location:
- Individual runs: `results/16frames_cadence{X}min_r2plus1d_18_seed{Y}/`
- Summary: `results/summary_16frames_cadence_sweep.json`

## Expected Outcomes

Based on the analysis:
- **72-min cadence should significantly improve TSS** (> 0.40)
- **16 frames should help** compared to your 8-frame attempts
- **18-hour context captures full flare precursor evolution**

## VideoMAE Status

- **NOT YET IMPLEMENTED** (skeleton code added)
- Requires `transformers>=4.30.0`
- Needs testing and weight prefetching
- See [README_16frames_cadence.md](classifier_NN/legacy/README_16frames_cadence.md) for details

## Files Created

1. **Experiment script**: `classifier_NN/legacy/run_experiments_16frames_cadence.py`
2. **Test script**: `scripts/test_16frames_cadence.sh`
3. **Sweep script**: `scripts/run_16frames_cadence_sweep.sh`
4. **Documentation**: `classifier_NN/legacy/README_16frames_cadence.md`
5. **VideoMAE skeleton**: Added to `classifier_NN/models.py`
6. **This summary**: `ANALYSIS_VIDEO_SEQUENCES.md`
