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
    # Baseline: ResNet-18 + Augmentation (No Mixup), standard CE loss
    # run_experiment("ResNet18_Aug_NoMixup", {
    #     "backbone": "resnet18",      # Smaller backbone
    #     "use_flow": False,
    #     "two_stream": False,
    #     "use_aug": True,
    #     "epochs": 20,
    #     "drop_rate": 0.2,            # ResNet head dropout
    #     "drop_path_rate": 0.0,       # ResNet doesn't use drop-path
    #     "weight_decay": 0.0001,      # Classic ResNet-style wd
    #     "label_smoothing": 0.0,
    #     "mixup": 0.0,
    #     "cutmix": 0.0,
    #     "loss_type": "ce",
    # })

    # # Skill-oriented: ResNet-18 + Augmentation with TSS-oriented loss
    # run_experiment("ResNet18_Aug_SkillTSS", {
    #     "backbone": "resnet18",
    #     "use_flow": False,
    #     "two_stream": False,
    #     "use_aug": True,
    #     "epochs": 20,
    #     "drop_rate": 0.2,
    #     "drop_path_rate": 0.0,
    #     "weight_decay": 0.0001,
    #     "label_smoothing": 0.0,
    #     "mixup": 0.0,
    #     "cutmix": 0.0,
    #     "loss_type": "skill_tss",
    #     "tss_loss_weight": 0.1,
    # })

    run_experiment("ResNet34_5to1", {
    "backbone": "resnet34",
    "image_size": 224,
    "batch_size": 48,
    "epochs": 10,
    "lr": 0.001,
    "weight_decay": 0.01,
    "balance_classes": False,
    "loss_type": "ce_weighted",
    "class_weights": [1.0, 5.0],
    "optimizer": "adam_paper",
    "redirect_log": True,
    })

    run_experiment("MobileNet_5to1", {
        "backbone": "mobilenetv2_100",
        "image_size": 224,
        "batch_size": 48,
        "epochs": 10,
        "lr": 0.001,
        "weight_decay": 0.01,
        "balance_classes": False,
        "loss_type": "ce_weighted",
        "class_weights": [1.0, 5.0],
        "optimizer": "adam_paper",
        "redirect_log": True,
    })

    run_experiment("MobileViT_5to1", {
        "backbone": "mobilevit_s",
        "image_size": 224,
        "batch_size": 48,
        "epochs": 10,
        "lr": 0.001,
        "weight_decay": 0.001,
        "balance_classes": False,
        "loss_type": "ce_weighted",
        "class_weights": [1.0, 5.0],
        "optimizer": "adam_paper",
        "redirect_log": True,
    })

    # # VGG-style PNG experiment mimicking transfer_learning notebook (PNG, no undersampling)
    # run_experiment("VGG_Paper_PNG", {
    #     "backbone": "vgg16_bn",  # use a strong 2D backbone; no VGG in current timm setup
    #     "use_flow": False,
    #     "two_stream": False,
    #     "use_seq": False,
    #     "image_size": 224,
    #     "batch_size": 64,
    #     "epochs": 10,
    #     "lr": 1e-3,
    #     # no negative undersampling: use full distribution
    #     "balance_classes": False,
    #     # CE with explicit class weights (computed from train labels in training script)
    #     "loss_type": "ce_weighted",
    #     "use_focal": False,
    #     # Optimizer: plain Adam with paper hyperparameters
    #     "optimizer": "adam_paper",
    #     "beta1": 0.9,
    #     "beta2": 0.999,
    #     "adam_eps": 1e-7,
    #     "adam_amsgrad": False,
    #     # Freeze all backbone layers and train only final classifier head
    #     "freeze_all_but_head": True,
    #     "redirect_log": True,
    # })
    
    print("\n✅ All experiments completed!")
