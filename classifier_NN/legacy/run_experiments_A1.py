"""Experiment A1: Spatial resolution sweep for single-frame models.

Backbones: ResNet-50, EfficientNet-B0
Resolutions: 224, 112, 56, 28 (downsample->upsample into 224x224)

Runs each configuration for N seeds and reports mean±std TSS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from statistics import mean, stdev

from ..config import get_default_cfg
from ..train import main


RESOLUTIONS = [224, 128, 112, 56, 28]
BACKBONES = [
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
]
N_RUNS_DEFAULT = 5


def _resolution_downsample_size(resolution: int) -> int:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    return int(resolution)


def _build_cfg(name: str, overrides: dict, common_overrides: dict | None = None) -> dict:
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    cfg["run_id"] = name
    return cfg


def _sanitize_name_suffix(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


COMMON_A1_OVERRIDES = {
    # single-frame only
    "use_seq": False,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,

    # input resolution (always feed 224x224 into model)
    "image_size": 224,

    # training
    "batch_size": 64,
    "optimizer": "adam_paper",
    "lr": 1e-3,
    "epochs": 50,

    # model
    # Default to pretrained transfer learning; override with --no-pretrained or --smoke.
    "pretrained": True,
    "freeze_backbone": True,

    # imbalance handling
    "balance_classes": False,
    "loss_type": "ce_weighted",

    # no augmentation
    "use_aug": False,

    # model selection / logging
    "model_selection": "tss",
    "redirect_log": True,

    # early stopping
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,
}


def run_single(
    *,
    backbone: str,
    resolution: int,
    seed: int,
    epochs: int | None = None,
    name_suffix: str | None = None,
):
    ds = _resolution_downsample_size(resolution)
    name = f"A1-{resolution}_{backbone}_seed{seed}"
    if name_suffix:
        name = f"{name}_{_sanitize_name_suffix(name_suffix)}"
    overrides = {
        "backbone": backbone,
        "spatial_downsample_size": ds,
        "seed": seed,
        "notes": (
            "A1 resolution sweep (single-frame): downsample->upsample before backbone. "
            f"Backbone={backbone}, resolution={resolution}, seed={seed}."
        ),
    }
    if epochs is not None:
        overrides["epochs"] = int(epochs)

    print("\n" + "=" * 80)
    print(f"Launching: {name} | spatial_downsample_size={ds}")
    print("=" * 80 + "\n")

    cfg = _build_cfg(name, overrides, COMMON_A1_OVERRIDES)
    # If we're not using pretrained weights, freezing the backbone would make
    # training ineffective (random frozen features). Flip to full finetune.
    if not bool(cfg.get("pretrained", True)):
        cfg["freeze_backbone"] = False
    exp_dir, results = main(cfg)
    return exp_dir, results


def run_sweep(
    *,
    backbones: list[str] | None = None,
    resolutions: list[int] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    common_overrides: dict | None = None,
    name_suffix: str | None = None,
    write_summary: bool = True,
):
    backbones = backbones or list(BACKBONES)
    resolutions = resolutions or list(RESOLUTIONS)
    seeds = seeds or list(range(N_RUNS_DEFAULT))

    summary = {
        "metric": "TSS",
        "resolutions": resolutions,
        "backbones": {},
        "n_runs": len(seeds),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for backbone in backbones:
        summary["backbones"][backbone] = {}
        for resolution in resolutions:
            tss_values = []
            for seed in seeds:
                try:
                    if common_overrides:
                        old_common = dict(COMMON_A1_OVERRIDES)
                        try:
                            COMMON_A1_OVERRIDES.update(common_overrides)
                            _, results = run_single(
                                backbone=backbone,
                                resolution=resolution,
                                seed=seed,
                                epochs=epochs,
                                name_suffix=name_suffix,
                            )
                        finally:
                            COMMON_A1_OVERRIDES.clear()
                            COMMON_A1_OVERRIDES.update(old_common)
                    else:
                        _, results = run_single(
                            backbone=backbone,
                            resolution=resolution,
                            seed=seed,
                            epochs=epochs,
                            name_suffix=name_suffix,
                        )
                    tss_values.append(float(results["TSS"]))
                except Exception as e:
                    print(f"❌ Run failed: backbone={backbone}, res={resolution}, seed={seed} -> {e}")
            if not tss_values:
                continue
            tss_mean = mean(tss_values)
            tss_std = stdev(tss_values) if len(tss_values) > 1 else 0.0
            summary["backbones"][backbone][str(resolution)] = {
                "tss_values": tss_values,
                "mean_tss": tss_mean,
                "std_tss": tss_std,
                "n": len(tss_values),
            }
            print(
                f"\n[{backbone} | {resolution}] TSS = {tss_mean:.4f} ± {tss_std:.4f} (n={len(tss_values)})\n"
            )

    if write_summary:
        # Save summary to results directory
        cfg = get_default_cfg()
        out_dir = cfg.get("results_base", ".")
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary_a1_resolution_sweep.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")
        return summary_path
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="Backbone(s) to run (timm model names). Repeat for multiple backbones.",
    )
    ap.add_argument(
        "--resolution",
        action="append",
        type=int,
        default=None,
        help="Resolution(s) to run (224, 128, 112, 56, 28). Repeat for multiple.",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=N_RUNS_DEFAULT,
        help="Number of runs per config (used to generate seeds 0..N-1).",
    )
    ap.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Explicit seed(s) to use. Overrides --runs if provided.",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs for all runs")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate for all runs")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size for all runs")
    ap.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Optimizer name (e.g. 'adamw', 'adam_paper')",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for AdamW/AdamW-like optimizers",
    )
    ap.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="LR scheduler ('onecycle', 'cosine', 'none')",
    )
    ap.add_argument(
        "--loss-type",
        type=str,
        default=None,
        help="Loss function name (e.g. 'weighted_bce', 'ce_weighted')",
    )
    ap.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Optional lower learning rate for backbone parameters",
    )
    ap.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="Optional higher learning rate for classification head parameters",
    )
    ap.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Number of initial head-only warmup epochs before full fine-tuning",
    )
    ap.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Gradient clipping max norm",
    )
    ap.add_argument(
        "--use-aug",
        dest="use_aug",
        action="store_true",
        default=None,
        help="Enable training augmentation",
    )
    ap.add_argument(
        "--no-aug",
        dest="use_aug",
        action="store_false",
        default=None,
        help="Disable training augmentation",
    )
    ap.add_argument(
        "--aug-preset",
        type=str,
        default=None,
        help="Augmentation preset (e.g. 'robust', 'paper_vgg')",
    )
    ap.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=None,
        help="Freeze backbone (train linear head only)",
    )
    ap.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        default=None,
        help="Unfreeze backbone (fine-tune all weights)",
    )
    ap.add_argument("--max-train-shards", type=int, default=None, help="Limit number of train shards")
    ap.add_argument("--max-val-shards", type=int, default=None, help="Limit number of val shards")
    ap.add_argument("--max-test-shards", type=int, default=None, help="Limit number of test shards")
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="Override steps/epoch (sanity runs)")
    ap.add_argument("--smoke", action="store_true", help="Tiny smoke run (1 epoch, shard/batch limits, no pretrained)")
    ap.add_argument(
        "--single",
        action="store_true",
        help="Run exactly one (backbone,resolution,seed) config and do not write the sweep summary JSON",
    )
    ap.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Use pretrained weights (requires cached weights or network access)",
    )
    ap.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        default=None,
        help="Do not use pretrained weights (safe default on compute nodes)",
    )
    ap.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to run_id/model_name (helps avoid collisions in parallel runs)",
    )
    ap.add_argument(
        "--min-flare-class",
        type=str,
        default=None,
        choices=["C", "M", "c", "m"],
        help=(
            "Label threshold: 'C' uses shard JSON labels (C+ baseline). "
            "'M' remaps labels from the intensity label file (M+/X+ positives)."
        ),
    )
    ap.add_argument(
        "--mvit-model-id",
        type=str,
        default=None,
        help="Torchvision MViT variant for MViT backbones (e.g. mvit_v1_b, mvit_v2_s)",
    )
    ap.add_argument(
        "--mvit-remove-temporal-conv",
        dest="mvit_remove_temporal_conv",
        action="store_true",
        default=None,
        help="Replace the MViT temporal patch embedding with a spatial-only patch embedding",
    )
    ap.add_argument(
        "--no-mvit-remove-temporal-conv",
        dest="mvit_remove_temporal_conv",
        action="store_false",
        default=None,
        help="Keep the original MViT temporal patch embedding",
    )

    # Shard-based negative downsampling controls (no external JSON required)
    ap.add_argument(
        "--balance-classes",
        dest="balance_classes",
        action="store_true",
        default=None,
        help="Enable negative downsampling in Train split",
    )
    ap.add_argument(
        "--no-balance-classes",
        dest="balance_classes",
        action="store_false",
        default=None,
        help="Disable negative downsampling in Train split",
    )
    ap.add_argument(
        "--balance-mode",
        type=str,
        default=None,
        choices=["prob", "fixed"],
        help="Balancing mode: 'prob' random per epoch, 'fixed' deterministic subset",
    )
    ap.add_argument(
        "--neg-keep-prob",
        type=float,
        default=None,
        help="Default keep probability for negatives (used when subtype keep probs are not set)",
    )
    ap.add_argument(
        "--neg-keep-prob-none",
        type=float,
        default=None,
        help="Keep probability for NONE negatives (meta.reg_label == '0')",
    )
    ap.add_argument(
        "--neg-keep-prob-c",
        type=float,
        default=None,
        help="Keep probability for C-class negatives (meta.reg_label startswith 'C')",
    )
    ap.add_argument(
        "--auto-set-neg-keep-probs",
        dest="auto_set_neg_keep_probs",
        action="store_true",
        default=None,
        help="Scan Train shards at startup and auto-set keep probabilities from targets",
    )
    ap.add_argument(
        "--no-auto-set-neg-keep-probs",
        dest="auto_set_neg_keep_probs",
        action="store_false",
        default=None,
        help="Disable auto-setting keep probabilities from shards",
    )
    ap.add_argument(
        "--target-neg-total",
        type=int,
        default=None,
        help="Target total negatives to keep (auto-set uses this to compute neg_keep_prob)",
    )
    ap.add_argument(
        "--target-neg-none",
        type=int,
        default=None,
        help="Target NONE negatives to keep (auto-set computes neg_keep_prob_none)",
    )
    ap.add_argument(
        "--target-neg-c",
        type=int,
        default=None,
        help="Target C-class negatives to keep (auto-set computes neg_keep_prob_c)",
    )
    args = ap.parse_args()

    resolutions = args.resolution if args.resolution else None
    backbones = args.backbone if args.backbone else None
    seeds = args.seed if args.seed is not None else list(range(int(args.runs)))

    common_overrides = {}

    if args.max_train_shards is not None:
        common_overrides["max_train_shards"] = int(args.max_train_shards)
    if args.max_val_shards is not None:
        common_overrides["max_val_shards"] = int(args.max_val_shards)
    if args.max_test_shards is not None:
        common_overrides["max_test_shards"] = int(args.max_test_shards)
    if args.steps_per_epoch is not None:
        common_overrides["steps_per_epoch"] = int(args.steps_per_epoch)

    if args.lr is not None:
        common_overrides["lr"] = float(args.lr)
    if args.batch_size is not None:
        common_overrides["batch_size"] = int(args.batch_size)

    if args.optimizer is not None:
        common_overrides["optimizer"] = str(args.optimizer)

    if args.weight_decay is not None:
        common_overrides["weight_decay"] = float(args.weight_decay)

    if args.scheduler is not None:
        common_overrides["scheduler"] = str(args.scheduler)

    if args.loss_type is not None:
        common_overrides["loss_type"] = str(args.loss_type)

    if args.backbone_lr is not None:
        common_overrides["backbone_lr"] = float(args.backbone_lr)

    if args.head_lr is not None:
        common_overrides["head_lr"] = float(args.head_lr)

    if args.warmup_epochs is not None:
        common_overrides["warmup_epochs"] = int(args.warmup_epochs)

    if args.grad_clip_norm is not None:
        common_overrides["grad_clip_norm"] = float(args.grad_clip_norm)

    if args.use_aug is not None:
        common_overrides["use_aug"] = bool(args.use_aug)

    if args.aug_preset is not None:
        common_overrides["aug_preset"] = str(args.aug_preset)

    if args.freeze_backbone is not None:
        common_overrides["freeze_backbone"] = bool(args.freeze_backbone)

    if args.min_flare_class is not None:
        common_overrides["min_flare_class"] = str(args.min_flare_class).upper()

    if args.mvit_model_id is not None:
        common_overrides["mvit_model_id"] = str(args.mvit_model_id)

    if args.mvit_remove_temporal_conv is not None:
        common_overrides["mvit_remove_temporal_conv"] = bool(args.mvit_remove_temporal_conv)

    if args.balance_classes is not None:
        common_overrides["balance_classes"] = bool(args.balance_classes)

    if args.balance_mode is not None:
        common_overrides["balance_mode"] = str(args.balance_mode)

    if args.neg_keep_prob is not None:
        common_overrides["neg_keep_prob"] = float(args.neg_keep_prob)

    if args.neg_keep_prob_none is not None:
        common_overrides["neg_keep_prob_none"] = float(args.neg_keep_prob_none)

    if args.neg_keep_prob_c is not None:
        common_overrides["neg_keep_prob_c"] = float(args.neg_keep_prob_c)

    if args.auto_set_neg_keep_probs is not None:
        common_overrides["auto_set_neg_keep_probs"] = bool(args.auto_set_neg_keep_probs)

    if args.target_neg_total is not None:
        common_overrides["target_neg_total"] = int(args.target_neg_total)

    if args.target_neg_none is not None:
        common_overrides["target_neg_none"] = int(args.target_neg_none)

    if args.target_neg_c is not None:
        common_overrides["target_neg_c"] = int(args.target_neg_c)

    if args.smoke:
        common_overrides.update({
            "epochs": 1,
            "batch_size": 16,
            "num_workers": 2,
            "persistent_workers": False,
            "redirect_log": False,
            "val_max_batches": 5,
            "max_train_shards": 2,
            "max_val_shards": 1,
            "max_test_shards": 1,
            "early_stopping_patience": 2,
            "pretrained": False,
            # Critical: dataset shard limiting does NOT affect our global
            # sample-count estimate, so cap steps explicitly for sanity.
            "steps_per_epoch": 50,
        })
    if args.pretrained is not None:
        common_overrides["pretrained"] = bool(args.pretrained)

    if not common_overrides:
        common_overrides = None

    if args.single:
        if not backbones or len(backbones) != 1:
            raise SystemExit("--single requires exactly one --backbone")
        if not resolutions or len(resolutions) != 1:
            raise SystemExit("--single requires exactly one --resolution")
        if args.seed is None or len(args.seed) != 1:
            raise SystemExit("--single requires exactly one --seed")
        if common_overrides:
            # Apply common overrides for single-run mode (sanity runs, etc.)
            old_common = dict(COMMON_A1_OVERRIDES)
            try:
                COMMON_A1_OVERRIDES.update(common_overrides)
                run_single(
                    backbone=backbones[0],
                    resolution=int(resolutions[0]),
                    seed=int(args.seed[0]),
                    epochs=args.epochs,
                    name_suffix=args.name_suffix,
                )
            finally:
                COMMON_A1_OVERRIDES.clear()
                COMMON_A1_OVERRIDES.update(old_common)
        else:
            run_single(
                backbone=backbones[0],
                resolution=int(resolutions[0]),
                seed=int(args.seed[0]),
                epochs=args.epochs,
                name_suffix=args.name_suffix,
            )
    else:
        run_sweep(
            backbones=backbones,
            resolutions=resolutions,
            seeds=seeds,
            epochs=args.epochs,
            common_overrides=common_overrides,
            name_suffix=args.name_suffix,
            write_summary=True,
        )
