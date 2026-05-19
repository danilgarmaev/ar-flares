# 16-Frame Video Cadence Experiments - Quick Start Guide

## Summary

You're testing **2 models** with **3 temporal cadences** = **6 GPU runs (1 seed each)**

### Models
1. **R(2+1)D** (`r2plus1d_18`) - Ready ✅
2. **VideoMAE** (`videomae`) - Needs prefetching ⚠️

### Cadences  
- **36 min** → 9 hours total context
- **72 min** → 18 hours total context (matches macros project)
- **108 min** → 27 hours total context

## Augmentation Comparison

### ✅ Your Current Implementation
From `classifier_NN/datasets.py`:
```python
angle = random.uniform(-30.0, 30.0)  # ±30° rotation
img_t = transforms.functional.rotate(img_t, angle)
```

**This is CORRECT!** Your sequences already use ±30° rotation, which is appropriate for:
- Preserving magnetic field orientation
- Avoiding 90° rotations that would artificially change physics
- Consistent with best practices for magnetogram augmentation

### Macros Project
The macros project **does not appear to use augmentation** (no augmentation code found in their scripts). They focus on data construction only.

**Your augmentation strategy is sound - no changes needed!**

## Step 1: Prefetch VideoMAE Weights (One-Time Setup)

Run this on the **login node** (with internet):

```bash
bash scripts/prefetch_videomae.sh
```

This downloads VideoMAE weights to your cache directory for offline compute node use.

## Step 2: Run Smoke Test

Quick validation (R(2+1)D, 72-min cadence, 5 epochs):

```bash
sbatch scripts/smoke_test_16frames_a100.slurm
```

Check results:
```bash
# Wait for completion
squeue -u $USER

# Check output
tail -f logs/arfl-16f-smoke_*.err
ls -lht results/ | head -5
```

Expected: ~1 hour runtime, should complete without errors.

## Step 3: Run Full Sweep

All 6 experiments (2 models × 3 cadences):

```bash
sbatch scripts/run_16frames_cadence_sweep_a100.slurm
```

Or use the automated submission script:
```bash
bash scripts/submit_16frames_experiments.sh
```

This will:
1. Run smoke test first
2. Wait for it to complete
3. If successful, submit the full sweep
4. If failed, ask for confirmation

## Monitor Progress

```bash
# Check job status
watch -n 5 'squeue -u $USER'

# Check logs (replace JOBID with actual job array ID)
tail -f logs/arfl-16f-cadence_JOBID_*.err

# Check results
ls -lht results/ | head -20

# Quick summary
grep "TSS:" results/*/log.txt | tail -10
```

## Expected Runtime

Per experiment (50 epochs):
- **R(2+1)D**: ~3-4 hours on A100
- **VideoMAE**: ~4-6 hours on A100 (transformer-based, slower)

Total for 6 runs (with array parallelization): **~4-6 hours**

## Results Location

Individual experiments:
```
results/16frames_cadence{36,72,108}min_{r2plus1d_18,videomae}_seed0/
```

Each contains:
- `log.txt` - Full training log
- `metrics.json` - Final test metrics
- `best_tss.pt` - Best model checkpoint
- `plots/` - Training curves

Summary file (when all complete):
```
results/summary_16frames_cadence_sweep.json
```

## Troubleshooting

### VideoMAE Import Error
If you see "cannot import VideoMAE":
```bash
source /home/dgarmaev/envs/ar-flares/bin/activate
pip install --upgrade transformers>=4.30.0
bash scripts/prefetch_videomae.sh
```

### Out of Memory
Reduce batch size in SLURM script:
```bash
export BATCH_SIZE=4
sbatch scripts/run_16frames_cadence_sweep_a100.slurm
```

### Job Failed Silently
Check error logs:
```bash
cat logs/arfl-16f-cadence_JOBID_TASKID.err
```

Common issues:
- Missing weights (run prefetch script)
- Data path issues (check AR_FLARES_* environment variables)
- CUDA out of memory (reduce batch size)

## Configuration Details

All runs use:
- **Image size**: 112×112
- **Frames**: 16
- **Batch size**: 8 (default, adjustable)
- **Learning rate**: 1e-4
- **Optimizer**: AdamW with weight_decay=0.01
- **Scheduler**: Cosine annealing
- **Epochs**: 50 (with early stopping patience=5)
- **Pretrained weights**: Yes (for both models)
- **Augmentation**: ±30° rotation, flips, polarity inversion, noise

## Quick Commands Reference

```bash
# Prefetch VideoMAE (login node, one-time)
bash scripts/prefetch_videomae.sh

# Smoke test (1 GPU, ~1 hour)
sbatch scripts/smoke_test_16frames_a100.slurm

# Full sweep (6 GPUs, ~4-6 hours with parallelization)
sbatch scripts/run_16frames_cadence_sweep_a100.slurm

# Auto-submit (smoke test → full sweep)
bash scripts/submit_16frames_experiments.sh

# Monitor
squeue -u $USER
tail -f logs/arfl-16f-cadence_*.err

# Results
ls -lht results/ | head -20
```

## What to Expect

### Your Previous Results
- 8 frames, 48-min cadence (5.6 hours)
- **TSS: ~0.40**

### Expected Improvement
With 16 frames and longer context (especially 72-min = 18 hours):
- **TSS: > 0.50** (hopefully!)
- Better capture of magnetic field evolution
- More temporal information for the video model

The 72-min cadence should perform best since it matches the validated macros configuration.

## Next Steps After Results

1. Check `results/summary_16frames_cadence_sweep.json`
2. Compare TSS across cadences and models
3. Identify best configuration
4. Optionally run additional seeds for statistical significance
5. Document findings for your paper
