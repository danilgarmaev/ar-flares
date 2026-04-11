from __future__ import annotations

import argparse
import re

from ..config import get_default_cfg
from ..train import main as train_main
from .run_experiments_16frames_cadence import COMMON_CONFIG


def _sanitize(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _parse_window_min(window_hours: float) -> int:
    window_min = int(round(float(window_hours) * 60.0))
    if window_min <= 0:
        raise ValueError(f"window-hours must be positive, got {window_hours}")
    if window_min % 12 != 0:
        raise ValueError(
            f"window-hours must resolve to a multiple of 12 minutes, got {window_min} min"
        )
    return window_min


def _cadence_steps_for_target_window(*, window_min: int, T: int) -> int:
    if T <= 0:
        raise ValueError(f"frame-count must be positive, got {T}")
    if T == 1:
        return 1

    window_steps = window_min // 12
    denom = T - 1
    return max(1, int((window_steps + denom // 2) // denom))


def _offsets_for_window_cadence(*, window_min: int, T: int) -> tuple[list[int], int]:
    stride_steps = _cadence_steps_for_target_window(window_min=window_min, T=T)
    offsets = list(range(-(T - 1) * stride_steps, 1, stride_steps))
    return offsets, stride_steps


def _window_tag(window_hours: float) -> str:
    value = float(window_hours)
    if value.is_integer():
        return f"{int(value)}h"
    return f"{str(value).replace('.', 'p')}h"


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 fixed-frame temporal window study")
    ap.add_argument("--backbone", type=str, default="mvit")
    ap.add_argument("--window-hours", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frame-count", type=int, default=8)
    ap.add_argument(
        "--seq-stride-steps",
        type=int,
        default=1,
        help="Endpoint/window stride in native 12-minute steps. Default is 1, i.e. slide windows every 12 minutes.",
    )
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

    window_min = _parse_window_min(args.window_hours)
    frame_count = int(args.frame_count)
    offsets, cadence_steps = _offsets_for_window_cadence(window_min=window_min, T=frame_count)
    seq_stride_steps = int(args.seq_stride_steps)
    if seq_stride_steps <= 0:
        raise ValueError(f"seq-stride-steps must be positive, got {seq_stride_steps}")

    cfg = get_default_cfg()
    cfg.update(COMMON_CONFIG)

    overrides = {
        "backbone": args.backbone,
        "seed": int(args.seed),
        "seq_T": frame_count,
        "seq_offsets": offsets,
        "seq_stride": seq_stride_steps,
        "seq_stride_steps": seq_stride_steps,
        "seq_target_window_min": window_min,
        "seq_target_window_hours": window_min / 60.0,
        "seq_actual_window_min": (max(offsets) - min(offsets)) * 12,
        "seq_actual_window_hours": (max(offsets) - min(offsets)) * 12 / 60.0,
        "seq_interval_min": cadence_steps * 12,
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

    window_tag = _window_tag(args.window_hours)
    exp_name = (
        f"A3-T{frame_count}-window-{window_tag}_{args.backbone}_seed{args.seed}_"
        f"{min_class_tag}_{aug_tag}_{data_tag}_{image_size}"
    )
    if args.run_tag:
        exp_name = f"{exp_name}_{_sanitize(args.run_tag)}"

    overrides["model_name"] = exp_name
    overrides["run_id"] = exp_name
    overrides["notes"] = (
        f"A3 temporal window study: target_window={window_min / 60.0:g}h, "
        f"actual_window={(max(offsets) - min(offsets)) * 12 / 60.0:g}h, "
        f"T={frame_count}, cadence={cadence_steps * 12}min, "
        f"window_stride={seq_stride_steps * 12}min, offsets={offsets}, "
        f"min_class={min_class}, seed={args.seed}."
    )

    cfg.update(overrides)
    print("=" * 80)
    print(f"Running: {exp_name}")
    print(f"  target_window={window_min / 60.0:g}h actual_window={cfg['seq_actual_window_hours']:g}h T={cfg['seq_T']}")
    print(f"  cadence={cfg['seq_interval_min']} min")
    print(f"  window_stride={cfg['seq_stride']} steps ({cfg['seq_stride'] * 12} min)")
    print(f"  offsets={offsets}")
    print("=" * 80)
    train_main(cfg)


if __name__ == "__main__":
    main()
