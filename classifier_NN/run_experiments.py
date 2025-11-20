import os
import sys
from config import CFG
from train import main

# Common settings for all experiments
COMMON_OVERRIDES = {
    "image_size": 112,
    "backbone": "convnext_tiny",
    "pretrained": True,
    "freeze_backbone": False,
    "lr": 3e-5,
    "epochs": 20,
    "batch_size": 64,
    "balance_classes": True,
    "balance_mode": "prob",
    "neg_keep_prob": 0.25,
    "redirect_log": True,  # Redirect stdout to log.txt for background runs
}

def run_experiment(name, overrides):
    print(f"\n{'='*40}")
    print(f"🚀 Launching Experiment: {name}")
    print(f"{'='*40}\n")
    
    # Reset CFG to defaults + common overrides
    CFG.update(COMMON_OVERRIDES)
    
    # Apply specific overrides
    CFG.update(overrides)
    CFG["model_name"] = name
    
    # Run training
    try:
        main()
    except Exception as e:
        print(f"❌ Experiment {name} failed: {e}")

if __name__ == "__main__":
    # Experiment 1: Basic + Heavy Regularization
    run_experiment("Exp1_Basic_Reg", {
        "use_flow": False,
        "two_stream": False,
        "use_aug": False,
        "drop_rate": 0.3,
        "drop_path_rate": 0.2,
        "weight_decay": 0.05,  # Stronger L2
        "label_smoothing": 0.1
    })

    # Experiment 2: Basic + Augmentation (Safe: H-Flip, Rotation, Scale)
    run_experiment("Exp2_Basic_Aug", {
        "use_flow": False,
        "two_stream": False,
        "use_aug": True,       # Enable augmentation
        "drop_rate": 0.2,      # Standard dropout
        "drop_path_rate": 0.1,
        "weight_decay": 0.05,
        "label_smoothing": 0.0
    })

    # Experiment 3: Flow (Two-Stream) + Heavy Regularization
    run_experiment("Exp3_Flow_Reg", {
        "use_flow": True,
        "two_stream": True,
        "use_aug": False,
        "drop_rate": 0.3,
        "drop_path_rate": 0.2,
        "weight_decay": 0.05,
        "label_smoothing": 0.1
    })

    # Experiment 4: Flow (Two-Stream) + Augmentation
    run_experiment("Exp4_Flow_Aug", {
        "use_flow": True,
        "two_stream": True,
        "use_aug": True,       # Enable augmentation
        "drop_rate": 0.2,
        "drop_path_rate": 0.1,
        "weight_decay": 0.05,
        "label_smoothing": 0.0
    })
    
    print("\n✅ All experiments completed!")
