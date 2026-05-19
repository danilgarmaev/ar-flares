"""
Training script for balanced ConvNeXt-Small experiment.

This configuration:
- Uses ConvNeXt-Small backbone
- Balances classes to 50/50 by subsampling negatives
- Removes class weights from loss (balanced data doesn't need them)
- Uses 112x112 images for faster training
- Fine-tunes the full model (not just the head)
"""
import sys
sys.path.insert(0, '.')

from config import CFG
from train import main

# Configuration for balanced ConvNeXt-Small
CFG.update({
    # Model
    "backbone": "convnext_small",     # ConvNeXt-Small for solar dynamics
    "pretrained": True,                # Start from ImageNet weights
    "freeze_backbone": False,          # Fine-tune entire model
    
    # Data
    "image_size": 112,                 # 112x112 for 4x speedup
    "balance_classes": True,           # 🔥 50/50 class balance
    "batch_size": 64,                  # Good batch size for T4
    "num_workers": 0,                  # Single process (debugging DataLoader hang)
    
    # Training
    "lr": 3e-5,                        # Conservative LR for full fine-tuning
    "epochs": 15,                      # More epochs for balanced training
    "use_focal": False,                # No focal loss needed with balanced data
    "use_flow": False,                 # Single-stream baseline
    "two_stream": False,
    "use_seq": False,
    
    # Experiment name
    "model_name": "convnext_small_balanced_112x112",
})

print("="*80)
print("BALANCED CONVNEXT-SMALL TRAINING")
print("="*80)
print(f"Backbone: {CFG['backbone']}")
print(f"Image size: {CFG['image_size']}x{CFG['image_size']}")
print(f"Class balancing: {CFG['balance_classes']} (50/50 split)")
print(f"Batch size: {CFG['batch_size']}")
print(f"Epochs: {CFG['epochs']}")
print(f"Learning rate: {CFG['lr']}")
print(f"Freeze backbone: {CFG['freeze_backbone']}")
print("="*80)
print()

# Run training
main()
