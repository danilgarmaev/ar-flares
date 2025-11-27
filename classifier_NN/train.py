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


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, steps_per_epoch, mixup_fn=None):
    """Train for one epoch. Returns (avg_loss, metrics_dict)."""
    model.train()
    running_loss = 0.0
    seen = 0
    all_probs = []
    all_labels = []
    pbar = tqdm(total=steps_per_epoch, desc=f"Training", leave=True)
    for inputs, labels, _ in dataloader:
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


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch. Returns (avg_loss, metrics_dict, best_val_tss)."""
    model.eval()
    val_loss = 0.0
    vbatches = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            vbatches += 1
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / max(1, vbatches)
    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    # Ensure validation labels are 1D ints
    if labels_np.ndim == 2 and labels_np.shape[1] == 2:
        labels_np = labels_np.argmax(axis=1)
    preds = (probs_np >= 0.5).astype(int)
    val_acc = 100.0 * (preds == labels_np).mean()
    TP = ((labels_np == 1) & (preds == 1)).sum()
    FP = ((labels_np == 0) & (preds == 1)).sum()
    FN = ((labels_np == 1) & (preds == 0)).sum()
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except Exception:
        auc = float('nan')

    # Compute best validation TSS over thresholds (like evaluate_model)
    best_val_tss, best_val_threshold = find_best_threshold_tss(labels_np, probs_np)

    metrics = {
        "val_acc": val_acc,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_auc": auc,
        "val_best_tss": best_val_tss,
        "val_best_threshold": best_val_threshold,
    }
    return avg_val_loss, metrics


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
        cfg = CFG
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
        effective_train = total_train_raw
    steps_per_epoch = max(1, effective_train // cfg["batch_size"])
    print(f"Train samples (raw): {total_train_raw:,} | effective (est): {effective_train:,} | steps/epoch: {steps_per_epoch:,}")
    
    # Build model using this run's config
    model = build_model(cfg=cfg, num_classes=2)
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
        # Imbalanced data - use class weights
        class_weights = torch.tensor([1.0, Nn/Nf], dtype=torch.float32, device=device)
        print(f"Using class weights: {class_weights.tolist()}")
    
    criterion = get_loss_function(
        use_focal=cfg["use_focal"],
        gamma=cfg["focal_gamma"],
        class_weights=class_weights,
        use_mixup=mixup_active,
        label_smoothing=cfg.get("label_smoothing", 0.0)
    )
    
    # Setup optimizer
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
    epoch_metrics_path = os.path.join(exp_dir, "epoch_metrics.jsonl")
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        print("-" * 40)
        
        # Train
        avg_train_loss, train_metrics = train_epoch(
            model, dls["Train"], criterion, optimizer, scheduler, scaler, device, steps_per_epoch, mixup_fn=mixup_fn
        )
        
        # Validate
        avg_val_loss, val_metrics = validate_epoch(model, dls["Validation"], criterion, device)
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
        
        # Only save best-TSS checkpoint based on validation TSS to save space
        if val_metrics['val_best_tss'] > best_val_tss:
            best_val_tss = val_metrics['val_best_tss']
            best_tss_path = os.path.join(exp_dir, "best_tss.pt")
            torch.save(model.state_dict(), best_tss_path)
            print(f"🌟 New Best TSS (Val): {best_val_tss:.4f} -> Saved to {best_tss_path}")

    # Save final model tag only for logging, not as an extra large checkpoint
    tag = f'{cfg["model_name"]}_lr{cfg["lr"]}_ep{cfg["epochs"]}{"_focal" if cfg["use_focal"] else ""}'
    print(f"\nTraining finished for {tag}")
    
    # Evaluate on test set using best-TSS checkpoint
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")

    best_tss_path = os.path.join(exp_dir, "best_tss.pt")
    if os.path.exists(best_tss_path):
        print(f"Loading best-TSS checkpoint from {best_tss_path} for test evaluation...")
        model.load_state_dict(torch.load(best_tss_path, map_location=device))
    else:
        print("Warning: No best_tss.pt found; evaluating with current model weights (last epoch).")
    
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
    save_summary(exp_dir, tag, results)
    
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
