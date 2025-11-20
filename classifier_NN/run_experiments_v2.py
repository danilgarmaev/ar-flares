import os
import sys
from config import CFG
from train import main

# Common settings for all experiments
COMMON_OVERRIDES = {
    "image_size": 112,
    "pretrained": True,
    "freeze_backbone": False,
    "lr": 5e-4,  # Higher LR for smaller model + mixup
    "epochs": 20,
    "batch_size": 128, # Can fit more with smaller model
    "balance_classes": True,
    "balance_mode": "prob",
    "neg_keep_prob": 0.25,
    "redirect_log": True,
    "use_focal": False, # Use BCE/CrossEntropy
    "use_aug": True,    # Always use safe augmentation
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
    # Experiment 1: ConvNeXt-Atto (Tiny model) + Mixup
    run_experiment("Exp1_Atto_Mixup", {
        "backbone": "convnext_atto",
        "mixup": 0.8,
        "cutmix": 1.0,
        "mixup_prob": 1.0,
        "label_smoothing": 0.1,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1,
    })

    # Experiment 2: ConvNeXt-Atto + No Mixup (Baseline for small model)
    run_experiment("Exp2_Atto_Baseline", {
        "backbone": "convnext_atto",
        "mixup": 0.0,
        "cutmix": 0.0,
        "label_smoothing": 0.0,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1,
    })

    # Experiment 3: ConvNeXt-Tiny + Mixup (To see if big model can be tamed)
    run_experiment("Exp3_Tiny_Mixup", {
        "backbone": "convnext_tiny",
        "mixup": 0.8,
        "cutmix": 1.0,
        "mixup_prob": 1.0,
        "label_smoothing": 0.1,
        "drop_rate": 0.3,
        "drop_path_rate": 0.2,
    })
    
    print("\n✅ All experiments completed!")
