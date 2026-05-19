# AR-Flares Classifier

## Quick Start

### Refactored Implementation (Recommended - 4x Faster! 🚀)
```bash
cd refactored/
python train.py  # Uses 112x112 images by default for 4x speedup
```

### Original Implementation
```bash
python model_training.py  # Uses 224x224 images
```

## 🚀 New Feature: Configurable Image Size

The refactored version now supports different image sizes for faster training:

```python
# In refactored/train.py or refactored/config.py:
CFG["image_size"] = 112  # 4x faster training! (75% fewer pixels)
CFG["image_size"] = 224  # Original size (more detail)
```

**Performance Impact:**
- `112x112`: ~4x faster training, 75% fewer pixels
- `224x224`: Original speed, maximum detail

Most models handle different sizes well via adaptive pooling!

## Structure

```
classifier_NN/
├── model_training.py          # Original monolithic script (1,498 lines)
├── model_training_backup.py   # Backup of original
│
└── refactored/                 # ✨ Clean modular implementation
    ├── config.py               # Configuration (includes image_size)
    ├── datasets.py             # Data loading (with resize transform)
    ├── models.py               # Model architectures
    ├── losses.py               # Loss functions
    ├── metrics.py              # Evaluation
    ├── train.py                # Main training script
    ├── test_refactor.py        # Tests
    └── test_image_resize.py    # Image resize tests
```

## Why Use Refactored?

✅ **4x faster training** with 112x112 images  
✅ Easier to navigate (6 focused files vs 1 huge file)  
✅ Easier to modify (change one module at a time)  
✅ Easier to test (individual components)  
✅ Better organized and documented  
✅ Configurable image sizes

## Configuration Examples

### Fast Training (112x112)
```python
CFG.update({
    "image_size": 112,
    "backbone": "convnext_tiny",
    "batch_size": 64,
    "epochs": 10,
})
```

### Full Quality (224x224)
```python
CFG.update({
    "image_size": 224,
    "backbone": "vit_base_patch16_224",
    "batch_size": 32,  # May need smaller batch
    "epochs": 10,
})
```
