"""Experiment A1: Spatial resolution ablation with VGG16 transfer learning.

Runs three configurations:
  - A1-224: original resolution (224x224)
  - A1-112: downsample to 112 then upsample to 224
  - A1-56:  downsample to 56 then upsample to 224

This isolates the effect of spatial resolution while keeping VGG16 unchanged.
"""

from .config import get_default_cfg
from .train import main


def build_experiment_cfg(name: str, overrides: dict, common_overrides: dict | None = None):
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    return cfg


COMMON_A1_OVERRIDES = {
    # single-frame only
    "use_seq": False,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,

    # model
    "backbone": "vgg16_tv",          # torchvision VGG16
    "pretrained": True,
    "freeze_backbone": True,         # transfer learning: train classifier head

    # input resolution (always feed 224x224 into VGG16)
    "image_size": 224,

    # training
    "batch_size": 64,
    "optimizer": "adam_paper",
    "lr": 1e-3,
    "epochs": 10,

    # imbalance handling
    "balance_classes": False,
    "loss_type": "ce_weighted",

    # no augmentation
    "use_aug": False,

    # model selection
    "model_selection": "val_loss",

    # logging
    "redirect_log": True,
}


def run_experiment(name: str, overrides: dict):
    print(f"\n{'='*40}")
    print(f"🚀 Launching Experiment: {name}")
    print(f"{'='*40}\n")

    cfg = build_experiment_cfg(name, overrides, COMMON_A1_OVERRIDES)
    try:
        main(cfg)
    except Exception as e:
        print(f"❌ Experiment {name} failed: {e}")


if __name__ == "__main__":
    # A1-224: original resolution
    run_experiment("A1-224_vgg16", {
        "spatial_downsample_factor": 1,
    })

    # A1-112: downsample (nearest) to 112 then upsample (bilinear) to 224
    run_experiment("A1-112_vgg16", {
        "spatial_downsample_factor": 2,
    })

    # A1-56: downsample (nearest) to 56 then upsample (bilinear) to 224
    run_experiment("A1-56_vgg16", {
        "spatial_downsample_factor": 4,
    })

    print("\n✅ Experiment A1 completed!")
