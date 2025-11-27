import os
import sys
from .config import get_default_cfg
from .train import main


def build_experiment_cfg(name: str, overrides: dict, common_overrides: dict | None = None):
    """Utility to build a fresh cfg for a single experiment.

    This centralizes how we construct configs so other runners (v2, physics)
    can share behavior.
    """
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    return cfg

# Common settings for all experiments
COMMON_OVERRIDES = {
    "image_size": 112,
    "backbone": "convnext_tiny",
    "pretrained": True,
    "freeze_backbone": False,
    "lr": 1e-4,
    "epochs": 10,
    "batch_size": 64,
    "balance_classes": True,
    "balance_mode": "prob",
    "neg_keep_prob": 0.25,
    "redirect_log": True,  # Redirect stdout to log.txt for background runs
    "num_workers": 4,      # Increase workers to feed GPU faster
    "persistent_workers": True, # Keep workers alive to reduce overhead
}

def run_experiment(name, overrides):
    print(f"\n{'='*40}")
    print(f"🚀 Launching Experiment: {name}")
    print(f"{'='*40}\n")

    # Build a fresh config for this experiment
    cfg = build_experiment_cfg(name, overrides, COMMON_OVERRIDES)

    # Run training
    try:
        main(cfg)
    except Exception as e:
        print(f"❌ Experiment {name} failed: {e}")

if __name__ == "__main__":
    # # Experiment: ResNet-18 + Augmentation (No Mixup)
    # run_experiment("ResNet18_Aug_NoMixup", {
    #     "backbone": "resnet18",      # Smaller backbone
    #     "use_flow": False,
    #     "two_stream": False,
    #     "use_aug": True,
    #     "drop_rate": 0.2,            # ResNet head dropout
    #     "drop_path_rate": 0.0,       # ResNet doesn't use drop-path
    #     "weight_decay": 0.0001,      # Classic ResNet-style wd
    #     "label_smoothing": 0.0,
    #     "mixup": 0.0,
    #     "cutmix": 0.0,
    # })

    # Experiment: EfficientNet-B0 + Augmentation (No Mixup)
    run_experiment("EffNetB0_Aug_NoMixup", {
        "backbone": "efficientnet_b0",
        "use_flow": False,
        "two_stream": False,
        "use_aug": True,
        "drop_rate": 0.2,
        "drop_path_rate": 0.1,
        "weight_decay": 0.0001,
        "label_smoothing": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
    })
    
    print("\n✅ All experiments completed!")
