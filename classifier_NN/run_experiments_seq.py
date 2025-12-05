import os
import sys

from .config import get_default_cfg
from .train import main


def build_experiment_cfg(name: str, overrides: dict, common_overrides: dict | None = None):
    """Utility to build a fresh cfg for a single 3D-sequence experiment.

    This mirrors run_experiments.py but is tailored to sequence models
    (3D CNN / video transformer). We always start from get_default_cfg()
    so changes in config.py propagate here automatically.
    """
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    return cfg


# Common settings for sequence experiments (3D CNN)
COMMON_SEQ_OVERRIDES = {
    # use sequences of frames
    "use_seq": True,
    "seq_T": 3,                # number of frames per sequence (e.g. last 3 hours)
    # image / training defaults
    "image_size": 112,
    "pretrained": False,       # Simple3DCNN does not use timm pretraining
    "freeze_backbone": False,
    "lr": 1e-4,
    "epochs": 20,
    "batch_size": 32,          # slightly smaller for 3D memory footprint
    "balance_classes": True,
    "balance_mode": "prob",
    "neg_keep_prob": 0.25,
    "redirect_log": True,
    "num_workers": 4,
    "persistent_workers": True,
    # regularization
    "drop_rate": 0.2,
    "drop_path_rate": 0.0,
}


def run_experiment(name: str, overrides: dict):
    print(f"\n{'='*40}")
    print(f"🚀 Launching Sequence Experiment: {name}")
    print(f"{'='*40}\n")

    cfg = build_experiment_cfg(name, overrides, COMMON_SEQ_OVERRIDES)

    try:
        main(cfg)
    except Exception as e:
        print(f"❌ Experiment {name} failed: {e}")


if __name__ == "__main__":
    # Baseline 3D CNN over sequences of magnetograms
    run_experiment("simple3dcnn_seq_T3", {
        "backbone": "simple3dcnn",   # selects Simple3DCNN in models.build_model
        "use_flow": False,
        "two_stream": False,
        "use_diff_attention": False,
        "use_aug": True,
        # loss setup – start from standard CE; can swap to skill_tss if desired
        "loss_type": "skill_tss",
        "use_focal": False,
    })

    # ResNet3D (r3d_18) sequence model for comparison
    run_experiment("resnet3d_simple_seq_T3", {
        "backbone": "resnet3d_simple",  # selects ResNet3DSimple in models.build_model
        "use_flow": False,
        "two_stream": False,
        "use_diff_attention": False,
        "use_aug": True,
        # Optional: enable pretrained 3D weights if available
        "pretrained_3d": False,
        "loss_type": "skill_tss",
        "use_focal": False,
    })

    print("\n✅ All sequence experiments completed!")
