# Refactored AR-Flares Classifier

Clean, modular implementation with **4x faster training** via configurable image sizes!

## Quick Start

```bash
# Run training
python train_balanced_convnext.py

# Or in background (recommended for long runs)
screen -dmS training bash -c "python train_balanced_convnext.py 2>&1 | tee training_output.log"

# Monitor progress
../scripts/check_training.sh

# Attach to running training
screen -r training
```

## 🚀 New: Configurable Image Size

Speed up training by 4x with smaller images:

```python
# Edit config.py or override in train.py:
CFG["image_size"] = 112  # 4x faster! (default)
CFG["image_size"] = 224  # Original size
```

**Speedup breakdown:**
- 112x112: 12,544 pixels → **4x faster**
- 224x224: 50,176 pixels → Original speed

## File Structure

```
refactored/
├── config.py            - Configuration (includes image_size)
├── datasets.py          - Data loading with torchvision transforms
├── models.py            - Model architectures (size-agnostic)
├── losses.py            - Loss functions
├── metrics.py           - Evaluation metrics
├── train.py             - Main training script
├── test_refactor.py     - Validation tests
└── test_image_resize.py - Image resizing tests
```

## Testing

```bash
# Test everything works
python test_refactor.py

# Test image resizing specifically
python test_image_resize.py
```

## Configuration Examples

### Fast Baseline (Recommended)
```python
CFG.update({
    "image_size": 112,
    "backbone": "convnext_tiny",
    "batch_size": 64,
    "epochs": 10,
})
```

### Two-Stream with Flow
```python
CFG.update({
    "image_size": 112,
    "use_flow": True,
    "two_stream": True,
    "flow_encoder": "SmallFlowCNN",
})
```

### Temporal Sequences
```python
CFG.update({
    "image_size": 112,
    "use_seq": True,
    "seq_T": 3,
})
```
