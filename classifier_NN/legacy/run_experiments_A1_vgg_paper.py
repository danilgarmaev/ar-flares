"""Experiment A1 (paper-style VGG): spatial resolution ablation.

This mirrors the saved "VGG_Paper_PNG" configuration:
- backbone: timm "vgg16_bn"
- optimizer: Adam (paper-style)
- loss: weighted CE (no balancing)
- freeze all but classifier head

Resolution ablation matches our A1 definition: we always feed 224x224 to the
network, but optionally downsample spatially (nearest) to 112 or 56 and then
upsample back to 224 before the model.

Runs:
- A1paper-224: spatial_downsample_factor=1
- A1paper-112: spatial_downsample_factor=2
- A1paper-56:  spatial_downsample_factor=4

Usage:
    python -m classifier_NN.legacy.run_experiments_A1_vgg_paper
    python -m classifier_NN.legacy.run_experiments_A1_vgg_paper --smoke
    python -m classifier_NN.legacy.run_experiments_A1_vgg_paper --epochs 20
"""

import argparse

from ..config import get_default_cfg
from ..train import main


def _build_cfg(name: str, overrides: dict, common_overrides: dict | None = None) -> dict:
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    cfg["run_id"] = name
    return cfg


PAPER_VGG_COMMON = {
    # single-frame only
    "use_seq": False,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,

    # model (paper)
    "backbone": "vgg16_bn",  # timm VGG16-BN
    "pretrained": True,
    "freeze_all_but_head": True,

    # input size (A1 ablation is via spatial_downsample_factor)
    "image_size": 224,

    # training (paper)
    "batch_size": 256,
    "optimizer": "adam_paper",
    "lr": 1e-3,
    "epochs": 20,

    # loss / imbalance (paper)
    "balance_classes": False,
    "loss_type": "ce_weighted",

    # augmentation: paper-style (Resize + HFlip + RandomAffine)
    "use_aug": True,
    "aug_preset": "paper_vgg",

    # dataloader (paper)
    "num_workers": 8,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "pin_memory": True,

    # selection
    "model_selection": "tss",

    # logging
    "redirect_log": True,
}


EXPS = [
    ("A1paper-224_vgg16bn", 1),
    ("A1paper-112_vgg16bn", 2),
    ("A1paper-56_vgg16bn", 4),
]


def run_all(*, smoke: bool, epochs: int | None):
    common = dict(PAPER_VGG_COMMON)
    if epochs is not None:
        common["epochs"] = int(epochs)

    if smoke:
        common.update(
            {
                "epochs": 1,
                "batch_size": 16,
                "num_workers": 0,
                "persistent_workers": False,
                "redirect_log": False,
                "val_max_batches": 5,
                "max_train_shards": 2,
                "max_val_shards": 1,
                "max_test_shards": 1,
            }
        )

    for name, factor in EXPS:
        print("\n" + "=" * 80)
        print(f"Launching: {name} | spatial_downsample_factor={factor}")
        print("=" * 80 + "\n")

        cfg = _build_cfg(
            name,
            {
                "spatial_downsample_factor": factor,
                "notes": (
                    "A1 resolution ablation using paper-style VGG config. "
                    "Downsample->upsample before VGG; no augmentation."
                ),
            },
            common,
        )
        main(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run a tiny 1-epoch shard-limited smoke test")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs for all runs")
    args = ap.parse_args()

    run_all(smoke=bool(args.smoke), epochs=args.epochs)
