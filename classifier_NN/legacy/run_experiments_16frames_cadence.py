"""Experiment: 16-frame sequences with varying cadences (36, 72, 108 minutes).

This script tests video models with 16 frames at different temporal sampling rates:
- 36 min cadence: 9 hours total temporal context
- 72 min cadence: 18 hours total temporal context (matches macros project)
- 108 min cadence: 27 hours total temporal context

Models to test:
- r2plus1d_18 (R(2+1)D from torchvision)
- VideoMAE (when available, needs to be implemented)
- MViT (torchvision, when available)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from statistics import mean, stdev

from ..config import get_default_cfg
from ..train import main as train_main


def _interval_to_stride(interval_min: int) -> int:
    """Convert cadence interval in minutes to stride in 12-minute steps."""
    if interval_min % 12 != 0:
        raise ValueError(f"interval_min must be a multiple of 12, got {interval_min}")
    return int(interval_min // 12)


def _generate_offsets(interval_min: int, T: int = 16) -> list[int]:
    """Generate offsets for T frames with given cadence interval.
    
    Args:
        interval_min: Cadence in minutes (e.g. 36, 72, 108)
        T: Number of frames (default 16)
    
    Returns:
        List of offsets in 12-minute steps, e.g. [-90, -84, ..., -6, 0]
    
    Examples:
        36 min (stride=3): [-45, -42, -39, ..., -3, 0] = 9 hours
        72 min (stride=6): [-90, -84, -78, ..., -6, 0] = 18 hours  
        108 min (stride=9): [-135, -126, -117, ..., -9, 0] = 27 hours
    """
    stride = _interval_to_stride(interval_min)
    offsets = list(range(-(T - 1) * stride, 1, stride))
    return offsets


COMMON_CONFIG = {
    # Sequence mode settings
    "use_seq": True,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,

    # Fixed for this experiment
    "seq_T": 16,
    "image_size": 112,  # default for R(2+1)D; VideoMAE often needs 224

    # Training defaults
    "batch_size": 8,  # Reduce for memory-intensive video models
    "num_workers": 2,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "lr": 1e-4,
    "backbone_lr": None,
    "head_lr": None,
    "epochs": 50,
    "scheduler": "cosine",
    "warmup_epochs": 0,
    "grad_clip_norm": None,
    "pretrained_mvit": True,
    "mvit_model_id": "mvit_v2_s",

    # Imbalance handling
    "balance_classes": False,
    "loss_type": "ce_weighted",
    "class_weight_mode": "fixed",

    # Augmentation (configurable via --use-aug / --no-aug)
    "use_aug": False,  # Default: no augmentation for video sequences

    # Model selection
    "model_selection": "tss",
    "redirect_log": True,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,
    "early_stopping_min_epoch": 3,

    # Robust threshold tuning (validation)
    "threshold_min_recall": 0.5,
    "threshold_min_precision": 0.15,
    "threshold_min": 0.0,
    "threshold_max": 1.0,
    "threshold_min_pos_rate": None,
    "threshold_max_pos_rate": None,
    "threshold_fallback_mode": "tss",

    # Checkpointing
    "save_last_checkpoint": True,
    "save_last_full_checkpoint": True,
    "save_best_f1": False,
    "run_tag": None,
}


def run_experiment(
    backbone: str,
    interval_min: int,
    seed: int = 0,
    common_overrides: dict | None = None,
) -> tuple[str, dict]:
    """Run a single experiment with specified backbone, cadence, and seed.
    
    Args:
        backbone: Model name (e.g., "r2plus1d_18", "videomae")
        interval_min: Cadence in minutes (36, 72, or 108)
        seed: Random seed
        common_overrides: Optional dict to override common config values
    
    Returns:
        Tuple of (experiment_directory, results_dict)
    """
    offsets = _generate_offsets(interval_min, T=16)
    stride = _interval_to_stride(interval_min)
    total_hours = (len(offsets) - 1) * interval_min / 60
    
    # Build config
    cfg = get_default_cfg()
    cfg.update(COMMON_CONFIG)
    if common_overrides:
        cfg.update(common_overrides)

    # Build experiment name after overrides so metadata is encoded in run_id.
    min_class = str(cfg.get("min_flare_class", "C") or "C").upper()
    min_class_tag = f"{min_class.lower()}plus"
    use_aug = bool(cfg.get("use_aug", False))
    aug_tag = "aug" if use_aug else "noaug"
    image_size = int(cfg.get("image_size", 112))
    run_tag = str(cfg.get("run_tag") or "").strip()
    target_neg_none = cfg.get("target_neg_none")
    target_neg_total = cfg.get("target_neg_total")
    if isinstance(target_neg_none, int):
        data_tag = f"dsnone{target_neg_none}"
    elif isinstance(target_neg_total, int):
        data_tag = f"dstotal{target_neg_total}"
    elif bool(cfg.get("balance_classes", False)):
        data_tag = "balanced"
    else:
        data_tag = "full"
    exp_name = (
        f"A3-16-cadence-{interval_min}min_{backbone}_seed{seed}_"
        f"{min_class_tag}_{aug_tag}_{data_tag}_{image_size}"
    )
    if run_tag:
        exp_name = f"{exp_name}_{run_tag}"
    
    # Set experiment-specific parameters
    cfg.update({
        "backbone": backbone,
        "seed": seed,
        "seq_offsets": offsets,
        "seq_stride": stride,
        "seq_stride_steps": stride,
        "model_name": exp_name,
        "run_id": exp_name,
        "notes": (
            f"16-frame sequence with {interval_min}min cadence "
            f"({total_hours:.1f}h total context). "
            f"Offsets: {offsets[:3]}...{offsets[-3:]} "
            f"Seed: {seed}"
        ),
    })
    
    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"  Cadence: {interval_min} min (stride={stride})")
    print(f"  Temporal context: {total_hours:.1f} hours")
    print(f"  Offsets: {offsets}")
    print(f"{'='*80}\n")
    
    exp_dir, results = train_main(cfg)
    return exp_dir, results


def run_sweep(
    backbones: list[str],
    intervals_min: list[int],
    seeds: list[int],
    common_overrides: dict | None = None,
) -> str:
    """Run experiments for all combinations of backbones, intervals, and seeds.
    
    Args:
        backbones: List of model names (e.g., ["r2plus1d_18"])
        intervals_min: List of cadences in minutes (e.g., [36, 72, 108])
        seeds: List of random seeds (e.g., [0, 1, 2])
        common_overrides: Optional dict to override common config values
    
    Returns:
        Path to summary JSON file
    """
    summary = {
        "experiment": "16-frame cadence sweep",
        "T": 16,
        "image_size": 112,
        "intervals_min": intervals_min,
        "metric": "TSS",
        "backbones": {},
        "n_seeds": len(seeds),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    for bb in backbones:
        summary["backbones"][bb] = {}
        
        for interval in intervals_min:
            tss_values = []
            exp_dirs = []
            
            for seed in seeds:
                try:
                    exp_dir, results = run_experiment(
                        backbone=bb,
                        interval_min=interval,
                        seed=seed,
                        common_overrides=common_overrides,
                    )
                    exp_dirs.append(exp_dir)
                    
                    # Extract TSS from results
                    tss = float(results.get("test_tss_at_val_threshold", results.get("TSS", 0.0)))
                    tss_values.append(tss)
                    
                    print(f"✅ {bb} | {interval}min | seed{seed} | TSS={tss:.4f}")
                    
                except Exception as e:
                    print(f"❌ Failed: {bb} | {interval}min | seed{seed}")
                    print(f"   Error: {e}")
                    continue
            
            if not tss_values:
                print(f"⚠️ No successful runs for {bb} | {interval}min")
                continue
            
            # Calculate statistics
            mean_tss = mean(tss_values)
            std_tss = stdev(tss_values) if len(tss_values) > 1 else 0.0
            
            summary["backbones"][bb][str(interval)] = {
                "tss_values": tss_values,
                "mean_tss": mean_tss,
                "std_tss": std_tss,
                "n": len(tss_values),
                "exp_dirs": exp_dirs,
            }
            
            print(f"\n📊 Summary [{bb} | {interval}min]: TSS = {mean_tss:.4f} ± {std_tss:.4f} (n={len(tss_values)})\n")
    
    # Save summary
    cfg0 = get_default_cfg()
    out_dir = cfg0.get("results_base", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary_16frames_cadence_sweep.json")
    
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Sweep complete! Summary saved to:")
    print(f"   {out_path}")
    print(f"{'='*80}\n")
    
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test 16-frame sequences with different temporal cadences"
    )
    
    # Model selection
    ap.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="Backbone(s) to test: r2plus1d_18, r3d_18, timesformer, slowfast, etc. Can be specified multiple times.",
    )
    
    # Cadence selection
    ap.add_argument(
        "--interval-min",
        action="append",
        type=int,
        default=None,
        help="Cadence interval(s) in minutes: 36, 72, 108. Can be specified multiple times.",
    )
    
    # Seeds
    ap.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Random seed(s). Can be specified multiple times. If not provided, uses [0].",
    )
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds to run (0 to n-seeds-1). Ignored if --seed is provided.",
    )
    
    # Single run mode
    ap.add_argument(
        "--single",
        action="store_true",
        help="Run a single experiment (requires exactly one backbone, interval, and seed).",
    )
    
    # Training parameters
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--scheduler", type=str, default=None, help="LR scheduler name (e.g. cosine)")
    ap.add_argument("--backbone-lr", type=float, default=None, help="Backbone LR for differential fine-tuning")
    ap.add_argument("--head-lr", type=float, default=None, help="Head LR for differential fine-tuning")
    ap.add_argument("--warmup-epochs", type=int, default=None, help="Number of head-only warmup epochs")
    ap.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient clipping norm")
    ap.add_argument("--early-stopping-min-epoch", type=int, default=None, help="Minimum 1-based epoch before best_tss checkpointing / early stopping")
    ap.add_argument("--image-size", type=int, default=None, help="Input image size (e.g., 112 or 224)")
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="Limit steps per epoch for quick testing")
    ap.add_argument("--val-max-batches", type=int, default=None, help="Limit validation batches")
    ap.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count")
    ap.add_argument("--pretrained-3d", action="store_true", default=False, help="Use pretrained 3D weights")
    ap.add_argument("--pretrained-mvit", action="store_true", default=None, help="Use pretrained MViT weights")
    ap.add_argument("--no-pretrained-mvit", dest="pretrained_mvit", action="store_false", default=None, help="Disable pretrained MViT weights")
    ap.add_argument("--mvit-model-id", type=str, default=None, help="Torchvision MViT variant (e.g. mvit_v2_s)")
    ap.add_argument("--use-multi-gpu", action="store_true", default=None, help="Enable DataParallel when multiple GPUs are visible")
    ap.add_argument("--use-ddp", action="store_true", default=None, help="Enable DistributedDataParallel when launched under torchrun")
    ap.add_argument("--ddp-backend", type=str, default=None, help="Distributed backend for DDP (default: nccl)")
    ap.add_argument("--multi-gpu-max-devices", type=int, default=None, help="Cap number of GPUs for DataParallel/DDP")
    ap.add_argument("--strict-complete-history", action="store_true", default=None, help="Require complete past context (no padding)")
    ap.add_argument("--allow-padded-history", dest="strict_complete_history", action="store_false", default=None, help="Allow legacy padded past context")
    ap.add_argument("--video-backbone", type=str, default=None, help="Per-frame timm backbone for video_transformer (e.g. swin_tiny_patch4_window7_224)")
    ap.add_argument("--video-heads", type=int, default=None, help="Number of temporal attention heads for video_transformer")
    ap.add_argument("--video-layers", type=int, default=None, help="Number of temporal transformer layers for video_transformer")
    ap.add_argument(
        "--loss-type",
        choices=["ce", "ce_weighted", "bce", "weighted_bce", "focal", "binary_focal", "skill_tss"],
        default=None,
        help="Loss function type",
    )
    ap.add_argument(
        "--class-weight-mode",
        choices=["fixed", "dynamic"],
        default=None,
        help="Class-weight source for imbalanced runs (ignored when balanced sampling is enabled)",
    )
    ap.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Alpha (positive-class balance factor) for focal / binary_focal loss (default 0.25)",
    )
    ap.add_argument(
        "--balanced-batch-sampling",
        action="store_true",
        default=None,
        help="Enable per-batch balancing: interleave positives and negatives in the train stream",
    )
    ap.add_argument(
        "--no-balanced-batch-sampling",
        dest="balanced_batch_sampling",
        action="store_false",
        default=None,
        help="Disable per-batch balancing (default)",
    )
    ap.add_argument(
        "--balanced-batch-neg-buffer",
        type=int,
        default=None,
        help="Max negatives to buffer for balanced-batch sampling (default 8192)",
    )
    ap.add_argument("--threshold-min-recall", type=float, default=None, help="Minimum recall constraint during threshold tuning")
    ap.add_argument("--threshold-min-precision", type=float, default=None, help="Minimum precision constraint during threshold tuning")
    ap.add_argument("--threshold-min", type=float, default=None, help="Lower bound for threshold search")
    ap.add_argument("--threshold-max", type=float, default=None, help="Upper bound for threshold search")
    ap.add_argument("--threshold-min-pos-rate", type=float, default=None, help="Optional lower bound on predicted-positive rate")
    ap.add_argument("--threshold-max-pos-rate", type=float, default=None, help="Optional upper bound on predicted-positive rate")
    ap.add_argument(
        "--threshold-fallback-mode",
        choices=["tss", "pr_sum"],
        default=None,
        help="Fallback when threshold constraints are infeasible",
    )

    # Labeling / class threshold
    ap.add_argument(
        "--min-flare-class",
        choices=["C", "M", "c", "m"],
        default=None,
        help="Positive-class threshold: C+ or M+",
    )

    # Train downsampling / balancing controls
    ap.add_argument("--balance-classes", action="store_true", default=None, help="Enable train-time negative subsampling")
    ap.add_argument("--no-balance-classes", dest="balance_classes", action="store_false", default=None, help="Disable train-time negative subsampling")
    ap.add_argument("--neg-keep-prob", type=float, default=None, help="Base negative keep probability")
    ap.add_argument("--neg-keep-prob-none", type=float, default=None, help="Keep probability for NONE negatives")
    ap.add_argument("--neg-keep-prob-c", type=float, default=None, help="Keep probability for C negatives")
    ap.add_argument("--auto-set-neg-keep-probs", action="store_true", default=None, help="Auto-compute keep probabilities from targets")
    ap.add_argument("--no-auto-set-neg-keep-probs", dest="auto_set_neg_keep_probs", action="store_false", default=None, help="Disable auto-compute keep probabilities")
    ap.add_argument("--target-neg-total", type=int, default=None, help="Target total negatives to keep")
    ap.add_argument("--target-neg-none", type=int, default=None, help="Target NONE negatives to keep")
    ap.add_argument("--target-neg-c", type=int, default=None, help="Target C negatives to keep")
    
    # Augmentation
    ap.add_argument("--use-aug", action="store_true", default=None, help="Enable augmentation (±30° rotation, flips)")
    ap.add_argument("--no-aug", dest="use_aug", action="store_false", default=None, help="Disable augmentation (default)")
    ap.add_argument("--run-tag", type=str, default=None, help="Optional suffix appended to the run_id")
    
    args = ap.parse_args()
    
    # Set defaults
    backbones = args.backbone or ["r2plus1d_18"]
    intervals = args.interval_min or [36, 72, 108]
    seeds = args.seed if args.seed else list(range(args.n_seeds))
    
    # Validate single mode
    if args.single:
        if len(backbones) != 1 or len(intervals) != 1 or len(seeds) != 1:
            raise ValueError(
                "--single requires exactly one backbone, interval, and seed. "
                f"Got {len(backbones)} backbone(s), {len(intervals)} interval(s), {len(seeds)} seed(s)."
            )
    
    # Build overrides from command-line args
    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.scheduler is not None:
        overrides["scheduler"] = str(args.scheduler)
    if args.backbone_lr is not None:
        overrides["backbone_lr"] = float(args.backbone_lr)
    if args.head_lr is not None:
        overrides["head_lr"] = float(args.head_lr)
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = int(args.warmup_epochs)
    if args.grad_clip_norm is not None:
        overrides["grad_clip_norm"] = float(args.grad_clip_norm)
    if args.early_stopping_min_epoch is not None:
        overrides["early_stopping_min_epoch"] = int(args.early_stopping_min_epoch)
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if args.steps_per_epoch is not None:
        overrides["steps_per_epoch"] = args.steps_per_epoch
    if args.use_aug is not None:
        overrides["use_aug"] = args.use_aug
    if args.val_max_batches is not None:
        overrides["val_max_batches"] = args.val_max_batches
    if args.num_workers is not None:
        overrides["num_workers"] = int(args.num_workers)
    if args.pretrained_3d:
        overrides["pretrained_3d"] = True
    if args.pretrained_mvit is not None:
        overrides["pretrained_mvit"] = bool(args.pretrained_mvit)
    if args.mvit_model_id is not None:
        overrides["mvit_model_id"] = str(args.mvit_model_id)
    if args.use_multi_gpu is not None:
        overrides["use_multi_gpu"] = args.use_multi_gpu
    if args.use_ddp is not None:
        overrides["use_ddp"] = args.use_ddp
    if args.ddp_backend is not None:
        overrides["ddp_backend"] = str(args.ddp_backend)
    if args.multi_gpu_max_devices is not None:
        overrides["multi_gpu_max_devices"] = int(args.multi_gpu_max_devices)
    if args.strict_complete_history is not None:
        overrides["seq_require_complete_history"] = bool(args.strict_complete_history)
    if args.video_backbone is not None:
        overrides["video_backbone"] = str(args.video_backbone)
    if args.video_heads is not None:
        overrides["video_heads"] = int(args.video_heads)
    if args.video_layers is not None:
        overrides["video_layers"] = int(args.video_layers)
    if args.loss_type is not None:
        overrides["loss_type"] = args.loss_type
    if args.class_weight_mode is not None:
        overrides["class_weight_mode"] = str(args.class_weight_mode)
    if args.threshold_min_recall is not None:
        overrides["threshold_min_recall"] = float(args.threshold_min_recall)
    if args.threshold_min_precision is not None:
        overrides["threshold_min_precision"] = float(args.threshold_min_precision)
    if args.threshold_min is not None:
        overrides["threshold_min"] = float(args.threshold_min)
    if args.threshold_max is not None:
        overrides["threshold_max"] = float(args.threshold_max)
    if args.threshold_min_pos_rate is not None:
        overrides["threshold_min_pos_rate"] = float(args.threshold_min_pos_rate)
    if args.threshold_max_pos_rate is not None:
        overrides["threshold_max_pos_rate"] = float(args.threshold_max_pos_rate)
    if args.threshold_fallback_mode is not None:
        overrides["threshold_fallback_mode"] = str(args.threshold_fallback_mode)
    if args.min_flare_class is not None:
        overrides["min_flare_class"] = str(args.min_flare_class).upper()
    if args.balance_classes is not None:
        overrides["balance_classes"] = args.balance_classes
    if args.neg_keep_prob is not None:
        overrides["neg_keep_prob"] = float(args.neg_keep_prob)
    if args.neg_keep_prob_none is not None:
        overrides["neg_keep_prob_none"] = float(args.neg_keep_prob_none)
    if args.neg_keep_prob_c is not None:
        overrides["neg_keep_prob_c"] = float(args.neg_keep_prob_c)
    if args.auto_set_neg_keep_probs is not None:
        overrides["auto_set_neg_keep_probs"] = args.auto_set_neg_keep_probs
    if args.target_neg_total is not None:
        overrides["target_neg_total"] = int(args.target_neg_total)
    if args.target_neg_none is not None:
        overrides["target_neg_none"] = int(args.target_neg_none)
    if args.target_neg_c is not None:
        overrides["target_neg_c"] = int(args.target_neg_c)
    if args.run_tag is not None:
        overrides["run_tag"] = str(args.run_tag)

    if getattr(args, "focal_alpha", None) is not None:
        overrides["focal_alpha"] = float(args.focal_alpha)
    if getattr(args, "balanced_batch_sampling", None) is not None:
        overrides["balanced_batch_sampling"] = bool(args.balanced_batch_sampling)
    if getattr(args, "balanced_batch_neg_buffer", None) is not None:
        overrides["balanced_batch_neg_buffer"] = int(args.balanced_batch_neg_buffer)

    # If any target is requested, default to enabling auto probability computation
    # and balanced sampling unless explicitly overridden by flags above.
    if any(v is not None for v in [args.target_neg_total, args.target_neg_none, args.target_neg_c]):
        overrides.setdefault("auto_set_neg_keep_probs", True)
        overrides.setdefault("balance_classes", True)
    
    # Print configuration
    print("\n" + "="*80)
    print("16-Frame Cadence Sweep Configuration")
    print("="*80)
    print(f"Backbones: {backbones}")
    print(f"Cadences: {intervals} minutes")
    print(f"Seeds: {seeds}")
    image_size_to_print = overrides.get("image_size", COMMON_CONFIG.get("image_size", 112))
    print(f"Image size: {image_size_to_print}x{image_size_to_print}")
    print(f"Frames per sequence: 16")
    if overrides:
        print(f"Overrides: {overrides}")
    print("="*80 + "\n")
    
    # Run experiments
    if args.single:
        exp_dir, results = run_experiment(
            backbone=backbones[0],
            interval_min=intervals[0],
            seed=seeds[0],
            common_overrides=overrides if overrides else None,
        )
        print(f"\n✅ Single experiment complete!")
        print(f"   Directory: {exp_dir}")
        print(f"   TSS: {results.get('test_tss_at_val_threshold', results.get('TSS', 'N/A'))}")
    else:
        summary_path = run_sweep(
            backbones=backbones,
            intervals_min=intervals,
            seeds=seeds,
            common_overrides=overrides if overrides else None,
        )
        print(f"Summary available at: {summary_path}")


if __name__ == "__main__":
    main()
