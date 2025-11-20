import os
import sys
from config import CFG
from train import main

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
        
        # Enable Physics-Informed Attention
        "use_diff_attention": True,
        
        # Disable conflicting options
        "use_flow": False,
        "two_stream": False,
        "use_seq": False,
        
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
    
    CFG.update(overrides)
    
    # Run training
    try:
        main()
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_physics_experiment()
