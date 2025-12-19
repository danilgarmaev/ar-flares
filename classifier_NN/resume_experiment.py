"""Resume (warm-start) training for an existing experiment directory.

Important: existing checkpoints in this repo are *model weights only* (state_dict).
That means we cannot perfectly resume optimizer/scheduler/scaler states.
This script warm-starts from a saved model checkpoint and continues training.

It is designed to avoid overwriting existing artifacts by writing new outputs
with a user-provided suffix.

Example (extend to 20 epochs total using best_val_loss as warm-start):

python -m classifier_NN.resume_experiment \
  "/path/to/exp_dir" \
  --warm-start best_val_loss.pt \
    --max-epochs 20 \
  --suffix best_tss_20_epochs

Outputs:
- <exp_dir>/<suffix>.pt                         (best-by-val-TSS across 1..(old+extra))
- <exp_dir>/epoch_metrics_<suffix>.jsonl
- <exp_dir>/plots_<suffix>/
- <exp_dir>/metrics_<suffix>.json
- <exp_dir>/<model_name>_<suffix>_summary.md
- <exp_dir>/log_<suffix>.txt (when redirect enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Any

import torch
from timm.optim import create_optimizer_v2

from .config import CFG, SPLIT_DIRS
from .datasets import count_samples_all_shards, create_dataloaders
from .losses import get_loss_function
from .metrics import evaluate_model
from .models import build_model
from .train import get_class_counts, train_epoch, validate_epoch


def _append_jsonl_safely(path: str, record: dict, *, fallback_dir: str | None = None) -> str:
    """Append a record to a jsonl file.

    If we hit an I/O error (common on flaky network FS/inode corruption), try:
    1) remove the target file and recreate
    2) fall back to a new filename (same dir by default, or fallback_dir if provided)

    Returns the path actually used.
    """
    payload = json.dumps(record) + "\n"
    try:
        with open(path, "a", buffering=1) as f:
            f.write(payload)
        return path
    except OSError as e:
        # Attempt recovery for EIO
        if getattr(e, "errno", None) == 5:
            try:
                os.remove(path)
            except Exception:
                pass
            try:
                with open(path, "a", buffering=1) as f:
                    f.write(payload)
                return path
            except Exception:
                pass

        # Fallback to a new filename
        base_dir = fallback_dir or os.path.dirname(path)
        os.makedirs(base_dir, exist_ok=True)
        fallback_path = os.path.join(
            base_dir,
            os.path.basename(path).replace(".jsonl", "") + f"_fallback_{int(datetime.now().timestamp())}.jsonl",
        )
        with open(fallback_path, "a", buffering=1) as f:
            f.write(payload)
        print(f"Warning: failed to append to {path} ({e}); wrote to fallback {fallback_path} instead")
        return fallback_path


def _coerce_opt_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    # When loading an optimizer state dict saved on a different device,
    # tensor states need to be moved.
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _load_checkpoint(path: str, device: torch.device) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
    """Load checkpoint.

    Supports either:
    - raw model `state_dict` (Tensor mapping)
    - dict checkpoint with keys like: model_state_dict/state_dict, optimizer_state_dict, epoch

    Returns: (model_state_dict, optimizer_state_dict_or_None, epoch_or_None)
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and (
        "model_state_dict" in ckpt or "state_dict" in ckpt or "model" in ckpt
    ):
        model_sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
        if not isinstance(model_sd, dict):
            raise ValueError(f"Checkpoint {path} has unexpected model state type: {type(model_sd)}")
        opt_sd = ckpt.get("optimizer_state_dict")
        epoch = ckpt.get("epoch")
        if epoch is not None:
            epoch = int(epoch)
        return model_sd, opt_sd, epoch
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint {path} is not a state_dict or supported dict checkpoint")
    # Raw state_dict path
    return ckpt, None, None


def _load_cfg(exp_dir: str) -> dict:
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {exp_dir}")
    with open(config_path, "r") as f:
        return json.load(f)


def _safe_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def _open_log(exp_dir: str, suffix: str) -> None:
    log_path = os.path.join(exp_dir, f"log_{suffix}.txt")
    # Redirect stdout for a clean per-resume log without clobbering the original
    import sys

    sys.stdout = open(log_path, "w", buffering=1)
    print(f"[LOG REDIRECTED] resume_experiment -> {log_path}")


def _write_summary_md(path: str, exp_dir: str, cfg: dict, suffix: str, results: dict) -> None:
    lines = []
    lines.append("# AR-Flares Resume Training Summary\n\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Experiment dir:** `{exp_dir}`\n")
    lines.append(f"**Suffix:** `{suffix}`\n")
    lines.append(f"**Backbone:** `{cfg.get('backbone')}`\n")
    lines.append(f"**Use Sequences:** `{cfg.get('use_seq', False)}`\n")
    lines.append(f"**Image size:** `{cfg.get('image_size', None)}`\n")
    if cfg.get("spatial_downsample_factor", 1) not in (None, 1):
        lines.append(f"**Spatial downsample factor:** `{cfg.get('spatial_downsample_factor')}`\n")
    lines.append("\n## Results (Test)\n")
    lines.append(f"- ROC AUC: {results['AUC']:.4f}\n")
    lines.append(f"- PR-AUC: {results['PR_AUC']:.4f}\n")
    lines.append(f"- Precision/Recall/F1: {results['Precision']:.4f} / {results['Recall']:.4f} / {results['F1']:.4f}\n")
    lines.append(f"- TSS/HSS: {results['TSS']:.4f} / {results['HSS']:.4f}\n")
    lines.append(f"- Best TSS: {results['Best_TSS']:.4f} @ threshold={results['Best_threshold']:.3f}\n")
    lines.append("\n## Confusion Matrix (best-TSS threshold)\n")
    lines.append(f"- TP/TN/FP/FN: {results['TP']} / {results['TN']} / {results['FP']} / {results['FN']}\n")

    with open(path, "w") as f:
        f.writelines(lines)


def resume_experiment(
    exp_dir: str,
    warm_start_ckpt: str,
    baseline_best_tss_ckpt: str,
    max_epochs: int,
    suffix: str,
    device_str: str | None = None,
    redirect_log: bool = True,
    append_epoch_metrics: bool = False,
) -> dict:
    cfg = _load_cfg(exp_dir)
    CFG.update(cfg)

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if redirect_log:
        _open_log(exp_dir, suffix)
    print(f"Using device: {device}")

    warm_path = os.path.join(exp_dir, warm_start_ckpt)
    if not os.path.exists(warm_path):
        raise FileNotFoundError(f"Missing warm-start checkpoint: {warm_path}")

    baseline_path = os.path.join(exp_dir, baseline_best_tss_ckpt)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Missing baseline best-TSS checkpoint: {baseline_path}")

    out_ckpt_path = os.path.join(exp_dir, f"{suffix}.pt")
    out_best_val_tss_path = os.path.join(exp_dir, f"best_val_tss_{suffix}.pt")
    out_plots_dir = os.path.join(exp_dir, f"plots_{suffix}")
    os.makedirs(out_plots_dir, exist_ok=True)

    # Initialize the best-by-TSS checkpoint for the 20-epoch run as the previous run's best_tss.
    # If resumed training beats it, we'll overwrite <suffix>.pt.
    _safe_copy(baseline_path, out_ckpt_path)
    _safe_copy(baseline_path, out_best_val_tss_path)

    # Compute steps_per_epoch (matches train.py logic)
    total_train_raw = count_samples_all_shards(SPLIT_DIRS["Train"])
    if cfg.get("balance_classes", False):
        full_counts = get_class_counts()
        Nn, Nf = full_counts.get(0, 1), full_counts.get(1, 1)
        neg_keep_prob = cfg.get("neg_keep_prob", 0.25)
        effective_train = int(Nf + Nn * neg_keep_prob)
    else:
        if cfg.get("use_seq", False):
            T = max(1, cfg.get("seq_T", 1))
            effective_train = max(1, total_train_raw // T)
        else:
            effective_train = total_train_raw
    steps_per_epoch = max(1, effective_train // cfg["batch_size"])
    print(
        f"Train samples (raw): {total_train_raw:,} | effective (est): {effective_train:,} | steps/epoch: {steps_per_epoch:,}"
    )

    print("Creating dataloaders...")
    dls = create_dataloaders()

    print("Building model...")
    model = build_model(cfg=cfg, num_classes=2).to(device)
    warm_model_sd, warm_opt_sd, warm_epoch = _load_checkpoint(warm_path, device)
    model.load_state_dict(warm_model_sd)

    # Loss function setup (matches train.py)
    full_counts = get_class_counts()
    Nn, Nf = full_counts.get(0, 1), full_counts.get(1, 1)
    if cfg.get("balance_classes", False):
        class_weights = None
        print("Using balanced sampling - no class weights")
    else:
        class_weights = torch.tensor([1.0, Nn / Nf], dtype=torch.float32, device=device)
        print(f"Using class weights: {class_weights.tolist()}")

    criterion = get_loss_function(
        cfg=cfg,
        use_focal=cfg.get("use_focal", False),
        gamma=cfg.get("focal_gamma", 2.0),
        class_weights=class_weights,
        use_mixup=False,
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )

    # Optimizer + scheduler (try to restore optimizer if present)
    if cfg.get("optimizer", "adamw") == "adam_paper":
        print("Using Adam optimizer (paper-style)")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
            eps=cfg.get("adam_eps", 1e-7),
            amsgrad=cfg.get("adam_amsgrad", False),
            weight_decay=0.0,
        )
    else:
        optimizer = create_optimizer_v2(
            model,
            opt="adamw",
            lr=cfg["lr"],
            weight_decay=0.05,
            layer_decay=0.75,
        )

    scaler = torch.GradScaler("cuda", enabled=torch.cuda.is_available() and device.type == "cuda")

    if warm_opt_sd is not None:
        try:
            optimizer.load_state_dict(warm_opt_sd)
            _coerce_opt_state_to_device(optimizer, device)
            print("Restored optimizer_state_dict from checkpoint.")
        except Exception as e:
            print(f"Warning: failed to load optimizer_state_dict; continuing with fresh optimizer. Error: {e}")

    start_epoch_cfg = int(cfg.get("epochs", 0))
    # If the checkpoint contains an epoch, use it; otherwise fall back to cfg["epochs"]
    start_epoch = int(warm_epoch) if warm_epoch is not None else start_epoch_cfg
    target_epochs = int(max_epochs)
    if target_epochs < start_epoch:
        raise ValueError(f"max_epochs={target_epochs} is less than start_epoch={start_epoch}")
    extra_epochs = target_epochs - start_epoch

    # Fresh scheduler for the continuation horizon (we cannot perfectly resume OneCycle without its state)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=max(1, int(extra_epochs)) if extra_epochs > 0 else 1,
        steps_per_epoch=steps_per_epoch,
    )

    # Baseline best-TSS: evaluate baseline checkpoint on Validation to seed comparison.
    print("Computing baseline Val TSS from existing best_tss...")
    baseline_model = build_model(cfg=cfg, num_classes=2).to(device)
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    _, baseline_val_metrics = validate_epoch(
        baseline_model,
        dls["Validation"],
        criterion,
        device,
        max_batches=cfg.get("val_max_batches", None),
    )
    best_val_tss = float(baseline_val_metrics["val_best_tss"])
    print(f"Baseline Val best-TSS: {best_val_tss:.4f} (from {baseline_best_tss_ckpt})")

    print("\n" + "=" * 80)
    print(f"Warm-start from {warm_start_ckpt} and continue to max_epochs={target_epochs} (start_epoch={start_epoch})")
    print("=" * 80 + "\n")

    # By default, keep resume history separate to avoid mutating the original run.
    # Optionally, append to the original epoch_metrics.jsonl for simpler plotting.
    if append_epoch_metrics:
        epoch_metrics_path = os.path.join(exp_dir, "epoch_metrics.jsonl")
    else:
        epoch_metrics_path = os.path.join(exp_dir, f"epoch_metrics_{suffix}.jsonl")

    if extra_epochs <= 0:
        print("No additional epochs to run (already at or beyond max_epochs).")
    else:
        for i in range(int(extra_epochs)):
            global_epoch = start_epoch + i + 1
            print(f"\nEpoch {global_epoch}/{target_epochs}")
            print("-" * 40)

            avg_train_loss, train_metrics = train_epoch(
                model,
                dls["Train"],
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                steps_per_epoch,
                mixup_fn=None,
            )

            avg_val_loss, val_metrics = validate_epoch(
                model,
                dls["Validation"],
                criterion,
                device,
                max_batches=cfg.get("val_max_batches", None),
            )

            print(
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | Val Acc: {val_metrics['val_acc']:.2f}%"
            )
            print(
                f"Train P/R/F1: {train_metrics['train_precision']:.3f}/{train_metrics['train_recall']:.3f}/{train_metrics['train_f1']:.3f} | "
                f"Val P/R/F1: {val_metrics['val_precision']:.3f}/{val_metrics['val_recall']:.3f}/{val_metrics['val_f1']:.3f}"
            )
            print(f"Train AUC: {train_metrics['train_auc']:.3f} | Val AUC: {val_metrics['val_auc']:.3f}")
            print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
            print(f"Val Best TSS: {val_metrics['val_best_tss']:.4f} @ threshold={val_metrics['val_best_threshold']:.3f}")

            epoch_record = {
                "epoch": global_epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **train_metrics,
                **val_metrics,
            }
            epoch_metrics_path = _append_jsonl_safely(epoch_metrics_path, epoch_record)

            if float(val_metrics["val_best_tss"]) > best_val_tss:
                best_val_tss = float(val_metrics["val_best_tss"])
                torch.save(model.state_dict(), out_ckpt_path)
                torch.save(model.state_dict(), out_best_val_tss_path)
                print(
                    f"🌟 New Best Val TSS across resumed run: {best_val_tss:.4f} -> "
                    f"Saved to {out_best_val_tss_path} and {out_ckpt_path}"
                )

    print("\n" + "=" * 80)
    print(f"Evaluating on test set using {out_ckpt_path}")
    print("=" * 80 + "\n")

    model.load_state_dict(torch.load(out_ckpt_path, map_location=device))
    results = evaluate_model(
        model,
        dls["Test"],
        device,
        out_plots_dir,
        cfg.get("backbone", "model"),
        save_pr_curve=bool(cfg.get("save_pr_curve", True)),
    )

    results.update(
        {
            "resume": True,
            "warm_start": warm_start_ckpt,
            "baseline_best_tss": baseline_best_tss_ckpt,
            "max_epochs": int(target_epochs),
            "start_epoch": int(start_epoch),
            "extra_epochs": int(extra_epochs),
            "suffix": suffix,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    metrics_path = os.path.join(exp_dir, f"metrics_{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    summary_path = os.path.join(exp_dir, f"{cfg.get('model_name', 'experiment')}_{suffix}_summary.md")
    _write_summary_md(summary_path, exp_dir, cfg, suffix, results)

    print(f"Wrote: {out_ckpt_path}")
    print(f"Wrote: {epoch_metrics_path}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {out_plots_dir}/")

    return {
        "checkpoint": out_ckpt_path,
        "epoch_metrics": epoch_metrics_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "plots": out_plots_dir,
        **results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", help="One or more experiment directories")
    parser.add_argument("--warm-start", default="best_val_loss.pt", help="Warm-start checkpoint filename inside exp_dir")
    parser.add_argument("--baseline-best-tss", default="best_tss.pt", help="Baseline best-TSS checkpoint filename inside exp_dir")
    parser.add_argument("--max-epochs", type=int, default=20, help="Train until this epoch count (default: 20)")
    parser.add_argument(
        "--suffix",
        default="best_tss_20_epochs",
        help="Suffix used for output files (e.g. <suffix>.pt, metrics_<suffix>.json, plots_<suffix>/)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override: cuda, cuda:0, or cpu. If omitted, auto-detect.",
    )
    parser.add_argument(
        "--no-redirect-log",
        action="store_true",
        help="Print progress to terminal instead of writing to log_<suffix>.txt",
    )
    parser.add_argument(
        "--append-epoch-metrics",
        action="store_true",
        help="Append resumed epochs into exp_dir/epoch_metrics.jsonl (mutates original history)",
    )
    args = parser.parse_args()

    for exp_dir in args.exp_dirs:
        out = resume_experiment(
            exp_dir=exp_dir,
            warm_start_ckpt=args.warm_start,
            baseline_best_tss_ckpt=args.baseline_best_tss,
            max_epochs=args.max_epochs,
            suffix=args.suffix,
            device_str=args.device,
            redirect_log=not args.no_redirect_log,
            append_epoch_metrics=args.append_epoch_metrics,
        )
        print(f"\nDone: {exp_dir}")
        print(f"  ckpt:    {out['checkpoint']}")
        print(f"  metrics: {out['metrics']}")
        print(f"  plots:   {out['plots']}")
        print(f"  TSS: {out['TSS']:.4f} | AUC: {out['AUC']:.4f} | PR-AUC: {out['PR_AUC']:.4f}")


if __name__ == "__main__":
    main()
