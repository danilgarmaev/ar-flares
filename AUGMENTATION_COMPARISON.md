# Video Data Augmentation Comparison

## Context

The macros project (using the same dataset) **does not use augmentation** for their video sequences. They focus purely on temporal data construction with 16 frames at 72-minute cadence.

Your current single-frame implementation uses:
- ±30° rotation
- Horizontal/vertical flips
- Polarity inversion
- Integer noise (±5)

For video sequences, augmentation is more complex because it must be **consistent across all frames** to preserve temporal coherence.

## Current Implementation

Your sequence augmentation ([datasets.py#L414-L456](classifier_NN/datasets.py#L414-L456)) already handles this correctly:

```python
# Sample geometric decisions ONCE per sequence
hflip = random.random() < 0.5
vflip = random.random() < vflip_prob
angle = random.uniform(-30.0, 30.0)  # ±30° rotation
polarity = True

# Apply SAME transforms to ALL frames
for img in pil_imgs:
    if hflip:
        img_t = transforms.functional.hflip(img_t)
    if vflip:
        img_t = transforms.functional.vflip(img_t)
    img_t = transforms.functional.rotate(img_t, angle)  # Same angle for all
    # ... polarity & noise
```

**This is the correct approach!** All frames get the same geometric transformation.

## Experiment Options

### Option 1: No Augmentation (Default, matches macros)
```bash
sbatch scripts/run_16frames_cadence_sweep_a100.slurm
```
- 6 runs: 2 models × 3 cadences
- No augmentation (like macros project)
- Fastest baseline

### Option 2: With Augmentation
```bash
# Modify SLURM script to add --use-aug flag
# Or run manually for specific configs
```

### Option 3: Full Comparison (12 runs)
```bash
sbatch scripts/run_16frames_with_aug_comparison_a100.slurm
```
- 12 runs: 2 models × 3 cadences × 2 aug settings
- Tests both with and without augmentation
- Takes 2× as long but provides complete comparison

## Pros & Cons

### Without Augmentation (Default)
**Pros:**
- ✅ Matches validated macros approach
- ✅ Faster training (no augmentation overhead)
- ✅ Simpler to interpret
- ✅ Less risk of unrealistic transformations

**Cons:**
- ❌ Less data diversity
- ❌ Potential for overfitting
- ❌ May not benefit from your existing augmentation pipeline

### With Augmentation
**Pros:**
- ✅ More training diversity
- ✅ Better generalization (potentially)
- ✅ Uses your well-tested augmentation strategy
- ✅ Consistent with your single-frame experiments

**Cons:**
- ❌ Slower training (~10-15% overhead)
- ❌ More hyperparameters to tune
- ❌ Diverges from macros methodology

## Recommendation

**Start without augmentation** to match the macros baseline:
1. Run the basic 6-experiment sweep (no aug)
2. See if you get better results than your previous TSS ~0.40
3. If results are promising, optionally test with augmentation

If you want to be thorough, run the full 12-experiment comparison, but it will take twice as long and use twice as many GPU hours.

## Current Job Status

Your smoke test is pending due to unavailable nodes:
```
JOBID         NAME              ST  TIME_LEFT  REASON
57577093      arfl-16f-smoke    PD  1:00:00    (ReqNodeNotAvail, UnavailableNodes:...)
```

Narval nodes `ng[11105-11106,20104,30708,31202]` are currently unavailable. Your job will start when A100 GPUs become available.

**Typical wait times on Narval:**
- A100 GPUs during business hours: 30 min - 4 hours
- A100 GPUs overnight/weekends: Usually < 1 hour
- Could be longer if many jobs ahead of you

## While You Wait

### Check queue position:
```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.6D %R %Q"
```

### Estimate wait time:
```bash
squeue -j 57577093 --start
```

### Cancel if needed:
```bash
scancel 57577093  # Cancel smoke test
scancel 57576109  # Cancel other jobs
```

### Monitor node availability:
```bash
sinfo -p narval-gpu --format="%20P %5D %6t %10C %10G"
```

## Next Steps

Once your smoke test completes:

1. **Check results:**
   ```bash
   cat logs/arfl-16f-smoke_57577093.err
   grep "TSS:" results/*/log.txt
   ```

2. **If successful, choose your sweep:**
   - Basic (6 runs, no aug): Already submitted or use `run_16frames_cadence_sweep_a100.slurm`
   - Full comparison (12 runs): `sbatch scripts/run_16frames_with_aug_comparison_a100.slurm`

3. **Monitor full sweep:**
   ```bash
   watch -n 10 'squeue -u $USER'
   ```

## Modified Files

Updated [run_experiments_16frames_cadence.py](classifier_NN/legacy/run_experiments_16frames_cadence.py) to support:
- `--use-aug` flag: Enable augmentation
- `--no-aug` flag: Disable augmentation (default)

New SLURM script for comparison:
- [run_16frames_with_aug_comparison_a100.slurm](scripts/run_16frames_with_aug_comparison_a100.slurm)
