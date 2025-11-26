import os
import sys
from .config import get_default_cfg, apply_physics_attention_overrides
from .train import main

# Physics-Informed Attention Experiment
# This script runs the training with the Difference Image Attention mechanism.

def run_physics_experiment():
    print(f"\n{'='*40}")
    print(f"🚀 Launching Physics-Informed Attention Experiment")
    print(f"{'='*40}\n")
    
    # Override Config
    overrides = {
        "model_name": "Physics_Attn_ConvNextTiny",
        "backbone": "convnext_tiny",
        "image_size": 112,           # Faster training
        "pretrained": True,
        "freeze_backbone": False,
        
        # Training Hyperparams
        "lr": 1e-4,
        "epochs": 15,
        "batch_size": 64,
        "use_aug": True,
        "use_focal": True,
        "focal_gamma": 2.0,
        
        # Regularization
        "drop_rate": 0.2,
        "drop_path_rate": 0.1,
        
        # Disable Mixup for now (tuple input compatibility)
        "mixup": 0.0,
        "cutmix": 0.0,
    }

    base_cfg = get_default_cfg()
    cfg = apply_physics_attention_overrides(base_cfg)
    cfg.update(overrides)

    # Run training
    try:
        main(cfg)
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_physics_experiment()
