"""
Clean training script for AR-flares classification.
Modular design with separate files for config, models, data, losses, and metrics.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import pytz
from timm.optim import create_optimizer_v2
from timm.data import Mixup
from sklearn.metrics import roc_auc_score

# Import our modules
from .config import CFG, SPLIT_DIRS, get_default_cfg
from .datasets import create_dataloaders, count_labels_all_shards, count_samples_all_shards
from .models import build_model
from .losses import get_loss_function
from .metrics import evaluate_model, find_best_threshold_tss


def _save_full_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    scaler: torch.GradScaler | None,
    epoch: int,
    cfg: dict,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "cfg": dict(cfg),
    }
    torch.save(ckpt, path)

def setup_experiment(cfg=None):
    """Create experiment directory. Optionally redirect stdout if cfg['redirect_log'] is True.

    If cfg is None, falls back to global CFG for backwards compatibility.
    """
    if cfg is None:
        cfg = CFG
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    # Prefer explicit run_id if provided, otherwise derive from timestamp+model name
    suffix = cfg.get("run_id") or cfg["model_name"]
    exp_id = f"{now_et.strftime('%Y-%m-%d %H:%M:%S')}_{suffix}"
    exp_dir = os.path.join(cfg["results_base"], exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if cfg.get("redirect_log", True):
        log_path = os.path.join(exp_dir, "log.txt")
        sys.stdout = open(log_path, "w", buffering=1)
        print(f"[LOG REDIRECTED] Starting experiment {exp_id}")
    else:
        print(f"Starting experiment {exp_id} (stdout in terminal)")
    print(f"Image size: {cfg.get('image_size', 224)}x{cfg.get('image_size', 224)}")
    return exp_dir, plot_dir


def get_class_counts():
    """Get class distribution for loss weighting."""
    # Using hardcoded counts for speed (you can uncomment the scanning code if needed)
    full_counts = {0: 610108, 1: 149249}
    
    # Uncomment to scan shards dynamically:
    # from datasets import LABEL_MAP
    # if LABEL_MAP is not None:
    #     full_counts = count_labels_all_shards(SPLIT_DIRS["Train"])
    # else:
    #     full_counts = {0: 1, 1: 1}
    
    print(f"Class counts (Train): {full_counts}")
    return full_counts


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    steps_per_epoch,
    mixup_fn=None,
    cfg: dict | None = None,
):
    """Train for one epoch. Returns (avg_loss, metrics_dict)."""
    model.train()
    logged_first_batch = False
    running_loss = 0.0
    seen = 0
    all_probs = []
    all_labels = []
    pbar = tqdm(total=steps_per_epoch, desc=f"Training", leave=True)
    for inputs, labels, _ in dataloader:
        if (not logged_first_batch) and (cfg is not None) and bool(cfg.get("use_seq", False)):
            if not (isinstance(inputs, torch.Tensor) and inputs.ndim == 5):
                raise AssertionError(
                    f"Sequence inputs must be a 5D Tensor (B,T,1,H,W). Got type={type(inputs)} shape={getattr(inputs, 'shape', None)}"
                )
            b, t, c, h, w = inputs.shape
            if c != 1:
                raise AssertionError(f"Sequence channel must be 1. Got shape={tuple(inputs.shape)}")
            expected_t = cfg.get("seq_T", None)
            if expected_t is not None and t != expected_t:
                raise AssertionError(f"Sequence length mismatch: cfg seq_T={expected_t} but batch T={t}")
            seq_stride_eff = cfg.get("seq_stride", cfg.get("seq_stride_steps", None))
            print(
                f"[seq] first train batch: shape={(b, t, c, h, w)} | N={cfg.get('seq_T')} | k={seq_stride_eff} | offsets={cfg.get('seq_offsets')}"
            )
            logged_first_batch = True

        if isinstance(inputs, (list, tuple)):
            inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
        else:
            inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Keep original labels for metrics
        orig_labels = labels.clone()
        
        # DEBUG (disabled): print shapes before Mixup on first batch
        # if seen == 0:
        #     print("[DEBUG] pre-Mixup inputs type:", type(inputs))
        #     if isinstance(inputs, torch.Tensor):
        #         print("[DEBUG] pre-Mixup inputs.shape:", inputs.shape)
        #     else:
        #         print("[DEBUG] pre-Mixup inputs is tuple, shapes:", [x.shape for x in inputs])
        #     print("[DEBUG] pre-Mixup labels.shape:", labels.shape, "dtype:", labels.dtype)
        #     print("[DEBUG] pre-Mixup labels[0:5]:", labels[:5])
        
        # Apply Mixup
        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
        
        # DEBUG (disabled): after Mixup on first batch
        # if seen == 0:
        #     if isinstance(inputs, torch.Tensor):
        #         print("[DEBUG] post-Mixup inputs.shape:", inputs.shape)
        #     else:
        #         print("[DEBUG] post-Mixup inputs is tuple, shapes:", [x.shape for x in inputs])
        #     print("[DEBUG] post-Mixup labels.shape:", labels.shape, "dtype:", labels.dtype)
        #     print("[DEBUG] post-Mixup labels[0]:", labels[0])
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            # DEBUG (disabled): log first-batch output shape
            # if seen == 0:
            #     print("[DEBUG] outputs.shape:", outputs.shape)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Per-batch LR update for OneCycleLR
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        seen += 1
        # collect probabilities (on original labels for metrics)
        batch_probs = torch.softmax(outputs.detach(), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(batch_probs)
        all_labels.extend(orig_labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
        if seen >= steps_per_epoch:
            break
    pbar.close()
    
    # Compute metrics (always on hard labels)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)
    # If labels accidentally became one-hot, collapse back to class indices
    if all_labels_np.ndim == 2 and all_labels_np.shape[1] == 2:
        all_labels_np = all_labels_np.argmax(axis=1)
    try:
        auc = roc_auc_score(all_labels_np, all_probs_np)
    except ValueError:
        auc = float("nan")
    preds_np = (all_probs_np >= 0.5).astype(int)
    acc = (preds_np == all_labels_np).mean() * 100.0
    
    # Calculate P/R/F1 for training
    TP = ((all_labels_np == 1) & (preds_np == 1)).sum()
    FP = ((all_labels_np == 0) & (preds_np == 1)).sum()
    FN = ((all_labels_np == 1) & (preds_np == 0)).sum()
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    metrics = {
        "train_loss": running_loss / max(1, seen),
        "train_acc": acc,
        "train_auc": auc,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1,
    }
    return running_loss / max(1, seen), metrics


def validate_epoch(model, dataloader, criterion, device, max_batches: int | None = None):
    """Validate for one epoch.

    Args:
        model: torch model
        dataloader: validation dataloader
        criterion: loss
        device: torch device
        max_batches: if set, only evaluate this many batches (useful for faster
            validation during large sequence experiments).

    Returns:
        (avg_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    n_samples = 0

    with torch.no_grad():
        # pbar = tqdm(total=max_batches, desc="Validation", leave=True) if max_batches else tqdm(desc="Validation", leave=True)
        # for bidx, (inputs, labels, _) in enumerate(dataloader, start=1):
        
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        # try to infer number of batches from the dataloader if available
        try:
            total_batches = len(dataloader)
        except TypeError:
            # DataLoader.__len__ can raise when wrapping an IterableDataset
            # that doesn't implement __len__ (e.g., our tar-shard datasets).
            total_batches = None
        if max_batches is None:
            total = total_batches
        else:
            total = max_batches if total_batches is None else min(max_batches, total_batches)
        pbar = tqdm(total=total, desc="Validation", leave=True, disable=not is_tty, dynamic_ncols=True)
        for bidx, (inputs, labels, _) in enumerate(dataloader, start=1):
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)
            if max_batches is not None and bidx >= int(max_batches):
                break
        pbar.close()

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)

    # Metrics at fixed 0.5 threshold (legacy view)
    preds_05 = (probs_np >= 0.5).astype(int)
    val_acc_05 = 100.0 * (preds_05 == labels_np).mean()

    TP_05 = ((labels_np == 1) & (preds_05 == 1)).sum()
    FP_05 = ((labels_np == 0) & (preds_05 == 1)).sum()
    FN_05 = ((labels_np == 1) & (preds_05 == 0)).sum()
    precision_05 = TP_05 / (TP_05 + FP_05 + 1e-7)
    recall_05 = TP_05 / (TP_05 + FN_05 + 1e-7)
    f1_05 = 2 * precision_05 * recall_05 / (precision_05 + recall_05 + 1e-7)

    # AUC
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = float("nan")

    # Best TSS threshold sweep
    best_val_tss, best_val_threshold = find_best_threshold_tss(labels_np, probs_np)

    # Metrics at best-TSS threshold (this is what we report as validation P/R/F1)
    preds_best = (probs_np >= best_val_threshold).astype(int)
    val_acc_best = 100.0 * (preds_best == labels_np).mean()

    TP_b = ((labels_np == 1) & (preds_best == 1)).sum()
    FP_b = ((labels_np == 0) & (preds_best == 1)).sum()
    FN_b = ((labels_np == 1) & (preds_best == 0)).sum()
    precision_best = TP_b / (TP_b + FP_b + 1e-7)
    recall_best = TP_b / (TP_b + FN_b + 1e-7)
    f1_best = 2 * precision_best * recall_best / (precision_best + recall_best + 1e-7)

    val_loss = running_loss / max(1, n_samples)

    metrics = {
        # Best-threshold metrics (primary)
        "val_acc": val_acc_best,
        "val_precision": precision_best,
        "val_recall": recall_best,
        "val_f1": f1_best,
        "val_auc": auc,
        "val_best_tss": best_val_tss,
        "val_best_threshold": best_val_threshold,
        # Also expose 0.5-based metrics for reference
        "val_acc_0.5": val_acc_05,
        "val_precision_0.5": precision_05,
        "val_recall_0.5": recall_05,
        "val_f1_0.5": f1_05,
    }

    return val_loss, metrics


def save_summary(exp_dir, tag, results, cfg=None):
    """Save experiment summary in Markdown format (Obsidian-friendly)."""
    summary_path = os.path.join(exp_dir, f"{tag}_summary.md")
    
    if cfg is None:
        cfg = CFG

    with open(summary_path, "w") as f:
        f.write(f"# AR-Flares Experiment Summary\n\n")
        f.write(f"**Experiment tag:** `{tag}`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Backbone:** {cfg['backbone']}\n")
        f.write(f"**Use Flow:** {cfg['use_flow']}\n")
        f.write(f"**Use Sequences:** {cfg.get('use_seq', False)}\n")
        f.write(f"**Freeze Backbone:** {cfg['freeze_backbone']}\n")
        f.write(f"**Learning Rate:** {cfg['lr']}\n")
        f.write(f"**Epochs:** {cfg['epochs']}\n")
        f.write(f"**Batch Size:** {cfg['batch_size']}\n")
        f.write(f"**Seed:** `{cfg.get('seed', 'N/A')}`\n")
        if cfg.get("run_id"):
            f.write(f"**Run ID:** `{cfg['run_id']}`\n")
        if cfg.get("notes"):
            f.write(f"**Notes:** {cfg['notes']}\n")
        f.write("\n")

        f.write("### Results\n")
        f.write(f"- **AUC:** {results['AUC']:.4f}\n")
        f.write(f"- **PR-AUC:** {results['PR_AUC']:.4f}\n")
        f.write(f"- **TSS:** {results['TSS']:.4f}\n")
        f.write(f"- **HSS:** {results['HSS']:.4f}\n")
        f.write(f"- **Best threshold (TSS):** {results['Best_threshold']:.3f}\n")
        f.write(f"- **Best TSS:** {results['Best_TSS']:.4f}\n\n")

        f.write("### Confusion Matrix (threshold=0.5)\n")
        f.write(f"- TP: {results['TP']}\n")
        f.write(f"- TN: {results['TN']}\n")
        f.write(f"- FP: {results['FP']}\n")
        f.write(f"- FN: {results['FN']}\n\n")

        f.write("### Notes\n")
        f.write("- Model trained on: Train split\n")
        f.write("- Validated on: Validation split\n")
        f.write("- Tested on: Test split\n")
        f.write("- Next steps: _fill this in manually in Obsidian_\n\n")

        f.write("### File Paths\n")
        f.write(f"- Model: `{os.path.join(exp_dir, f'{tag}.pt')}`\n")
        f.write(f"- Metrics: `{os.path.join(exp_dir, 'metrics.json')}`\n")
        f.write(f"- Log: `{os.path.join(exp_dir, 'log.txt')}`\n")
    
    print(f"Summary written to {summary_path}")


def main(cfg=None):
    """Main training pipeline.

    If cfg is None, uses global CFG for backwards compatibility. New callers
    should pass in a fresh cfg instance from get_default_cfg() and apply
    overrides locally.
    """
    if cfg is None:
        # Backwards-compatible path: use global CFG directly
        cfg = CFG
    else:
        # When a per-run cfg is provided (from run_experiments_*.py),
        # push overrides into the global CFG so helpers like
        # create_dataloaders() that read CFG see the correct settings
        # (e.g., image_size, use_seq, etc.).
        CFG.update(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup experiment directory
    exp_dir, plot_dir = setup_experiment(cfg)
    
    # Get class counts for loss weighting
    full_counts = get_class_counts()
    Nn, Nf = full_counts.get(0, 1), full_counts.get(1, 1)
    
    # Calculate training steps
    total_train_raw = count_samples_all_shards(SPLIT_DIRS["Train"])
    if cfg.get("balance_classes", False):
        neg_keep_prob = cfg.get("neg_keep_prob", 0.25)
        effective_train = int(Nf + Nn * neg_keep_prob)
    else:
        # For sequence models, each training "sample" is a multi-frame
        # clip (T frames). Approximate the number of clips as
        # total_frames / T so steps/epoch stays in a reasonable range
        # even when using long sequences like T=16 or T=32.
        if cfg.get("use_seq", False):
            T = max(1, cfg.get("seq_T", 1))
            effective_train = max(1, total_train_raw // T)
        else:
            effective_train = total_train_raw
    steps_per_epoch = max(1, effective_train // cfg["batch_size"])
    print(f"Train samples (raw): {total_train_raw:,} | effective (est): {effective_train:,} | steps/epoch: {steps_per_epoch:,}")
    
    # Build model using this run's config
    model = build_model(cfg=cfg, num_classes=2)

    # For VGG-style experiments, optionally freeze all backbone layers and
    # train only the final classifier, matching the original Keras notebook
    # behavior (all layers except the last are non-trainable).
    if cfg.get("freeze_all_but_head", False):
        # Heuristic: treat common classifier attributes as "head" and freeze
        # everything else. This covers timm VGG, ResNet, ConvNeXt, etc.
        head_modules = []
        for attr in ["head", "classifier", "fc"]:
            if hasattr(model, attr):
                m = getattr(model, attr)
                if m is not None:
                    head_modules.append(m)

        head_params = set()
        for hm in head_modules:
            for p in hm.parameters():
                head_params.add(p)

        for p in model.parameters():
            p.requires_grad = (p in head_params)

        print("Freezing all backbone layers; training only final classifier head.")

    model = model.to(device)
    
    # Create dataloaders
    print("Creating dataloaders...")
    dls = create_dataloaders()
    
    # Setup Mixup
    mixup_fn = None
    mixup_active = cfg.get("mixup", 0.0) > 0 or cfg.get("cutmix", 0.0) > 0
    if mixup_active:
        print("Using Mixup/Cutmix")
        mixup_fn = Mixup(
            mixup_alpha=cfg.get("mixup", 0.0), 
            cutmix_alpha=cfg.get("cutmix", 0.0), 
            prob=cfg.get("mixup_prob", 1.0), 
            switch_prob=cfg.get("mixup_switch_prob", 0.5), 
            mode=cfg.get("mixup_mode", 'batch'),
            label_smoothing=cfg.get("label_smoothing", 0.1), 
            num_classes=2
        )

    # --- FIX START: Conflict Resolution ---
    # If Mixup is on, we MUST disable Focal Loss and Class Weights
    # because standard Focal Loss expects integer labels, but Mixup creates floats.
    if mixup_active and cfg["use_focal"]:
        print("⚠️ WARNING: Focal Loss is incompatible with Mixup in this setup.")
        print(">> Automatically disabling Focal Loss and Class Weights to prevent shape crash.")
        cfg["use_focal"] = False
        class_weights = None 
    # --- FIX END ---

    # Setup loss function
    if cfg.get("balance_classes", False):
        # Balanced data - no class weights needed
        class_weights = None
        print("Using balanced sampling - no class weights")
    else:
        # Imbalanced data - use class weights (Nn: negatives, Nf: positives)
        class_weights = torch.tensor([1.0, Nn / Nf], dtype=torch.float32, device=device)
        print(f"Using class weights: {class_weights.tolist()}")
    
    criterion = get_loss_function(
        cfg=cfg,
        use_focal=cfg.get("use_focal", False),
        gamma=cfg.get("focal_gamma", 2.0),
        class_weights=class_weights,
        use_mixup=mixup_active,
        label_smoothing=cfg.get("label_smoothing", 0.0)
    )
    
    # Setup optimizer
    # For VGG-style paper replication, allow opting into plain Adam with
    # the original hyperparameters (lr=1e-3, betas=(0.9,0.999), eps=1e-7, amsgrad=False).
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
            opt='adamw',
            lr=cfg["lr"],
            weight_decay=0.05,
            layer_decay=0.75
        )
    
    # Setup gradient scaler for mixed precision
    scaler = torch.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    # Restore OneCycleLR scheduler (per-batch scheduling)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
    max_lr=cfg["lr"],
    epochs=cfg["epochs"],
        steps_per_epoch=steps_per_epoch,
    )

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    history = []
    best_val_tss = -1.0  # track best validation TSS
    best_val_loss = float("inf")  # track best (lowest) validation loss
    save_best_f1 = bool(cfg.get("save_best_f1", False))
    if save_best_f1:
        best_val_f1 = -1.0  # track best validation F1
    epoch_metrics_path = os.path.join(exp_dir, "epoch_metrics.jsonl")
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        print("-" * 40)
        
        # Train
        avg_train_loss, train_metrics = train_epoch(
            model,
            dls["Train"],
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            steps_per_epoch,
            mixup_fn=mixup_fn,
            cfg=cfg,
        )
        
        # Validate
        avg_val_loss, val_metrics = validate_epoch(
            model,
            dls["Validation"],
            criterion,
            device,
            max_batches=cfg.get("val_max_batches", None),
        )
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_metrics['train_acc']:.2f}% | Val Acc: {val_metrics['val_acc']:.2f}%")
        print(f"Train P/R/F1: {train_metrics['train_precision']:.3f}/{train_metrics['train_recall']:.3f}/{train_metrics['train_f1']:.3f} | Val P/R/F1: {val_metrics['val_precision']:.3f}/{val_metrics['val_recall']:.3f}/{val_metrics['val_f1']:.3f}")
        print(f"Train AUC: {train_metrics['train_auc']:.3f} | Val AUC: {val_metrics['val_auc']:.3f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"Val Best TSS: {val_metrics['val_best_tss']:.4f} @ threshold={val_metrics['val_best_threshold']:.3f}")
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_record)
        with open(epoch_metrics_path, "a") as ef:
            ef.write(json.dumps(epoch_record) + "\n")

        # Save resumable last-epoch checkpoints (overwritten each epoch)
        if cfg.get("save_last_checkpoint", True):
            torch.save(model.state_dict(), os.path.join(exp_dir, "last.pt"))
        if cfg.get("save_last_full_checkpoint", True):
            _save_full_checkpoint(
                os.path.join(exp_dir, "last_full.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                cfg=cfg,
            )

        # Optional: best checkpoint by validation loss (useful for ablations)
        selection = str(cfg.get("model_selection", "tss")).lower()
        if selection in ["val_loss", "loss"]:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_loss_epoch = epoch
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_val_loss.pt"))
                print(f"🌟 New Best Val Loss: {best_val_loss:.4f} @ epoch={epoch} -> Saved to best_val_loss.pt")
        
        # Update best checkpoints every epoch (no minimum epoch constraint)
        # Best TSS checkpoint
        if val_metrics["val_best_tss"] > best_val_tss:
            best_val_tss = val_metrics["val_best_tss"]
            best_tss_epoch = epoch
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_tss.pt"))
            print(f"🌟 New Best TSS (Val): {best_val_tss:.4f} @ epoch={epoch} -> Saved to best_tss.pt")
        # Best F1 checkpoint (optional; disabled by default to save space)
        if save_best_f1:
            if val_metrics["val_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                best_f1_epoch = epoch
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_f1.pt"))
                print(f"🌟 New Best F1 (Val): {best_val_f1:.4f} @ epoch={epoch} -> Saved to best_f1.pt")
    if str(cfg.get("model_selection", "tss")).lower() in ["val_loss", "loss"] and "best_loss_epoch" in locals():
        print(f"Best Val Loss checkpoint: epoch {best_loss_epoch}, loss={best_val_loss:.4f}")
    print(f"Best TSS checkpoint: epoch {best_tss_epoch}, TSS={best_val_tss:.4f}")
    if save_best_f1 and "best_f1_epoch" in locals():
        print(f"Best F1 checkpoint: epoch {best_f1_epoch}, F1={best_val_f1:.4f}")
    # Evaluate on test set using best-TSS checkpoint
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")

    selection = str(cfg.get("model_selection", "tss")).lower()
    ckpt_name = {
        "tss": "best_tss.pt",
        "f1": "best_f1.pt",
        "val_loss": "best_val_loss.pt",
        "loss": "best_val_loss.pt",
    }.get(selection, "best_tss.pt")
    ckpt_path = os.path.join(exp_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} (model_selection={selection}) for test evaluation...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"Warning: No {ckpt_name} found; evaluating with current model weights (last epoch).")
    
    results = evaluate_model(
        model,
        dls["Test"],
        device,
    plot_dir,
    cfg["backbone"],
    save_pr_curve=cfg["save_pr_curve"]
    )
    
    # Add metadata to results
    results.update({
    "seed": cfg.get("seed", None),
    "backbone": cfg["backbone"],
    "use_flow": cfg["use_flow"],
    "use_seq": cfg.get("use_seq", False),
    "epochs": cfg["epochs"],
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # Save metrics
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"ROC AUC: {results['AUC']:.4f}")
    print(f"PR-AUC: {results['PR_AUC']:.4f}")
    print(f"Precision: {results['Precision']:.4f} | Recall: {results['Recall']:.4f} | F1: {results['F1']:.4f}")
    print(f"TSS: {results['TSS']:.4f} | HSS: {results['HSS']:.4f}")
    print(f"Best TSS: {results['Best_TSS']:.4f} @ threshold={results['Best_threshold']:.3f}")
    print("="*80 + "\n")
    
    # Save summary markdown
    # Plot curves
    try:
        import matplotlib.pyplot as plt
        epochs = [r["epoch"] for r in history]
        tr_loss = [r["train_loss"] for r in history]
        vl_loss = [r["val_loss"] for r in history]
        tr_acc = [r["train_acc"] for r in history]
        vl_acc = [r["val_acc"] for r in history]
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, vl_loss, label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curves")
        plt.subplot(1,2,2)
        plt.plot(epochs, tr_acc, label="Train Acc")
        plt.plot(epochs, vl_acc, label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.title("Accuracy Curves")
        curves_path = os.path.join(plot_dir, "training_curves.png")
        plt.tight_layout(); plt.savefig(curves_path, dpi=200); plt.close()
        print(f"Training curves saved to {curves_path}")
    except Exception as e:
        print(f"Curve plotting failed: {e}")

    # Derive an experiment tag for the summary from cfg/model_name,
    # falling back to the experiment directory name if needed.
    tag = cfg.get("model_name", os.path.basename(exp_dir))
    save_summary(exp_dir, tag, results, cfg)
    
    print(f"\n✅ Training complete! Results saved to: {exp_dir}\n")


if __name__ == "__main__":
    # Example: Override config for specific experiments
    CFG.update({
        "image_size": 112,           # 🚀 Use 112x112 for 4x faster training! (vs 224x224)
        "backbone": "convnext_tiny",
        "pretrained": True,
        "freeze_backbone": False,
        "lr": 3e-5,
        "epochs": 10,
        "batch_size": 64,
        "use_flow": False,           # Set to True for optical flow
        "two_stream": False,
        "use_seq": False,            # Set to True for temporal sequences
        "seq_T": 3,
        "seq_stride_steps": 8,
        "seq_offsets": [-16, -8, 0],
        "seq_aggregate": "mean",
        "use_focal": True,
        "focal_gamma": 2.0,
        "model_name": "convnext_tiny_112x112",
        # Regularization overrides
        "drop_rate": 0.3,            # Stronger dropout for tiny model
        "drop_path_rate": 0.2,       # Stronger stochastic depth
    })
    
    main()
