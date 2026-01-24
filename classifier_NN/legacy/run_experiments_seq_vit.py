import os
import sys

from ..config import get_default_cfg
from ..train import main


def build_experiment_cfg(name: str, overrides: dict, common_overrides: dict | None = None):
    """Build config for a single sequence ViT experiment.

    Mirrors run_experiments.py / run_experiments_seq.py but targets
    sequence ViT-style models (TemporalWrapper, VideoTransformer).
    """
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    return cfg


# Common settings for sequence ViT experiments
COMMON_SEQ_VIT_OVERRIDES = {
    # sequence settings
    "use_seq": True,
    "seq_T": 3,
    "seq_offsets": [-16, -8, 0],   # same temporal pattern as 3D CNN run
    # image / training defaults
    "image_size": 224,
    "pretrained": True,
    "freeze_backbone": False,
    "lr": 1e-4,
    "epochs": 10,          # start with 10; can bump to 30 once stable
    "batch_size": 64,
    "balance_classes": True,
    "balance_mode": "prob",
    "neg_keep_prob": 0.25,
    "redirect_log": True,
    "num_workers": 4,
    "persistent_workers": True,
    # regularization
    "drop_rate": 0.2,
    "drop_path_rate": 0.1,
}


def run_experiment(name: str, overrides: dict):
    print(f"\n{'='*40}")
    print(f"🚀 Launching Sequence ViT Experiment: {name}")
    print(f"{'='*40}\n")

    cfg = build_experiment_cfg(name, overrides, COMMON_SEQ_VIT_OVERRIDES)

    try:
        main(cfg)
    except Exception as e:
        print(f"❌ Experiment {name} failed: {e}")


if __name__ == "__main__":
    # V1: TemporalWrapper with ViT-B/16, mean over time
    run_experiment("seq_vit_base_mean_T3", {
        "backbone": "vit_base_patch16_224",
        "use_flow": False,
        "two_stream": False,
        "use_diff_attention": False,
        "use_aug": True,
        "seq_aggregate": "mean",     # TemporalWrapper setting
        "loss_type": "ce",
        "use_focal": False,
    })

    # V2: TemporalWrapper with ViT-B/16, temporal attention
    run_experiment("seq_vit_base_attn_T3", {
        "backbone": "vit_base_patch16_224",
        "use_flow": False,
        "two_stream": False,
        "use_diff_attention": False,
        "use_aug": True,
        "seq_aggregate": "attn",
        "loss_type": "ce",
        "use_focal": False,
    })

    # V3: VideoTransformer with ViT-B/16 backbone + temporal encoder
    run_experiment("video_vit_base_T3", {
        "backbone": "video_transformer",       # dispatch to VideoTransformer in build_model
        "video_backbone": "vit_base_patch16_224",
        "video_heads": 4,
        "video_layers": 2,
        "use_flow": False,
        "two_stream": False,
        "use_diff_attention": False,
        "use_aug": True,
        "loss_type": "ce",
        "use_focal": False,
    })

    print("\n✅ All sequence ViT experiments completed!")
