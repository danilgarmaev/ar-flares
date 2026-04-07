from __future__ import annotations

import argparse
import os
import re

from ..config import get_default_cfg
from ..train import main as train_main
from .run_experiments_16frames_cadence import COMMON_CONFIG, _interval_to_stride


def _sanitize(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _parse_offsets(text: str) -> list[int]:
    offsets = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not offsets:
        raise ValueError("seq-offsets must contain at least one integer")
    if offsets[-1] != 0:
        raise ValueError(f"seq-offsets must end at 0, got {offsets}")
    if offsets != sorted(offsets):
        raise ValueError(f"seq-offsets must be non-decreasing, got {offsets}")
    return offsets


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 frame-count study with explicit temporal subsets")
    ap.add_argument("--backbone", type=str, default="mvit")
    ap.add_argument("--cadence-min", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frame-count", type=int, required=True)
    ap.add_argument("--seq-offsets", type=str, required=True, help="Comma-separated offsets in 12-minute steps")
    ap.add_argument("--min-flare-class", choices=["C", "M"], default="C")
    ap.add_argument("--run-tag", type=str, default=None)

    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--scheduler", type=str, default=None)
    ap.add_argument("--backbone-lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--grad-clip-norm", type=float, default=None)
    ap.add_argument("--early-stopping-min-epoch", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--loss-type", type=str, default=None)
    ap.add_argument("--threshold-min-recall", type=float, default=None)
    ap.add_argument("--threshold-min-precision", type=float, default=None)
    ap.add_argument("--pretrained-3d", action="store_true", default=None)
    ap.add_argument("--pretrained-mvit", action="store_true", default=None)
    ap.add_argument("--no-pretrained-mvit", dest="pretrained_mvit", action="store_false", default=None)
    ap.add_argument("--mvit-model-id", type=str, default=None)
    ap.add_argument("--use-multi-gpu", action="store_true", default=None)
    ap.add_argument("--multi-gpu-max-devices", type=int, default=None)
    ap.add_argument("--balance-classes", action="store_true", default=None)
    ap.add_argument("--auto-set-neg-keep-probs", action="store_true", default=None)
    ap.add_argument("--target-neg-none", type=int, default=None)
    ap.add_argument("--neg-keep-prob-c", type=float, default=None)
    ap.add_argument("--use-aug", action="store_true", default=None)
    ap.add_argument("--no-balanced-batch-sampling", dest="balanced_batch_sampling", action="store_false", default=None)
    args = ap.parse_args()

    offsets = _parse_offsets(args.seq_offsets)
    if len(offsets) != int(args.frame_count):
        raise ValueError(f"frame-count={args.frame_count} but got {len(offsets)} offsets")

    cfg = get_default_cfg()
    cfg.update(COMMON_CONFIG)

    overrides = {
        "backbone": args.backbone,
        "seed": int(args.seed),
        "seq_T": int(args.frame_count),
        "seq_offsets": offsets,
        "seq_stride": _interval_to_stride(int(args.cadence_min)),
        "seq_stride_steps": _interval_to_stride(int(args.cadence_min)),
        "min_flare_class": str(args.min_flare_class),
    }

    for key in [
        "batch_size",
        "epochs",
        "lr",
        "scheduler",
        "backbone_lr",
        "head_lr",
        "warmup_epochs",
        "grad_clip_norm",
        "early_stopping_min_epoch",
        "image_size",
        "num_workers",
        "loss_type",
        "threshold_min_recall",
        "threshold_min_precision",
        "pretrained_3d",
        "pretrained_mvit",
        "mvit_model_id",
        "use_multi_gpu",
        "multi_gpu_max_devices",
        "balance_classes",
        "auto_set_neg_keep_probs",
        "target_neg_none",
        "neg_keep_prob_c",
        "use_aug",
        "balanced_batch_sampling",
    ]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    min_class = str(overrides.get("min_flare_class", "C") or "C").upper()
    min_class_tag = f"{min_class.lower()}plus"
    aug_tag = "aug" if bool(overrides.get("use_aug", False)) else "noaug"
    image_size = int(overrides.get("image_size", cfg.get("image_size", 224)))
    target_neg_none = overrides.get("target_neg_none")
    if isinstance(target_neg_none, int):
        data_tag = f"dsnone{target_neg_none}"
    elif bool(overrides.get("balance_classes", False)):
        data_tag = "balanced"
    else:
        data_tag = "full"

    exp_name = (
        f"A3-T{args.frame_count}-cadence-{args.cadence_min}min_{args.backbone}_seed{args.seed}_"
        f"{min_class_tag}_{aug_tag}_{data_tag}_{image_size}"
    )
    if args.run_tag:
        exp_name = f"{exp_name}_{_sanitize(args.run_tag)}"

    overrides["model_name"] = exp_name
    overrides["run_id"] = exp_name
    overrides["notes"] = (
        f"A3 frame-count study: cadence={args.cadence_min}min, T={args.frame_count}, "
        f"offsets={offsets}, min_class={min_class}, seed={args.seed}."
    )

    cfg.update(overrides)
    print("=" * 80)
    print(f"Running: {exp_name}")
    print(f"  cadence={args.cadence_min}min stride={cfg['seq_stride']} T={cfg['seq_T']}")
    print(f"  offsets={offsets}")
    print("=" * 80)
    train_main(cfg)


if __name__ == "__main__":
    main()
