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
from sklearn.metrics import roc_auc_score

# Import our modules
from config import CFG, SPLIT_DIRS
from datasets import create_dataloaders, count_labels_all_shards, count_samples_all_shards
from models import build_model
from losses import get_loss_function
from metrics import evaluate_model


def setup_experiment():
    """Create experiment directory. Optionally redirect stdout if CFG['redirect_log'] is True."""
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    exp_id = f"{now_et.strftime('%Y-%m-%d %H:%M:%S')}_{CFG['model_name']}"
    exp_dir = os.path.join(CFG["results_base"], exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)
    if CFG.get("redirect_log", True):
        log_path = os.path.join(exp_dir, "log.txt")
        sys.stdout = open(log_path, "w", buffering=1)
        print(f"[LOG REDIRECTED] Starting experiment {exp_id}")
    else:
        print(f"Starting experiment {exp_id} (stdout in terminal)")
    print(f"Image size: {CFG.get('image_size', 224)}x{CFG.get('image_size', 224)}")
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


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, steps_per_epoch):
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
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        seen += 1
        # collect probabilities
        batch_probs = torch.softmax(outputs.detach(), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(batch_probs)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
        if seen >= steps_per_epoch:
            break
    pbar.close()
    avg_loss = running_loss / max(1, seen)
    # metrics
    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    preds = (probs_np >= 0.5).astype(int)
    train_acc = 100.0 * (preds == labels_np).mean()
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
    metrics = {"train_acc": train_acc, "train_precision": precision, "train_recall": recall, "train_f1": f1, "train_auc": auc}
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch. Returns (avg_loss, metrics_dict)."""
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
    metrics = {"val_acc": val_acc, "val_precision": precision, "val_recall": recall, "val_f1": f1, "val_auc": auc}
    return avg_val_loss, metrics


def save_summary(exp_dir, tag, results):
    """Save experiment summary in Markdown format (Obsidian-friendly)."""
    summary_path = os.path.join(exp_dir, f"{tag}_summary.md")
    
    with open(summary_path, "w") as f:
        f.write(f"# AR-Flares Experiment Summary\n\n")
        f.write(f"**Experiment tag:** `{tag}`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Backbone:** {CFG['backbone']}\n")
        f.write(f"**Use Flow:** {CFG['use_flow']}\n")
        f.write(f"**Use Sequences:** {CFG.get('use_seq', False)}\n")
        f.write(f"**Freeze Backbone:** {CFG['freeze_backbone']}\n")
        f.write(f"**Learning Rate:** {CFG['lr']}\n")
        f.write(f"**Epochs:** {CFG['epochs']}\n")
        f.write(f"**Batch Size:** {CFG['batch_size']}\n")
        f.write(f"**Seed:** `{CFG.get('seed', 'N/A')}`\n\n")
        
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


def main():
    """Main training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup experiment directory
    exp_dir, plot_dir = setup_experiment()
    
    # Get class counts for loss weighting
    full_counts = get_class_counts()
    Nn, Nf = full_counts.get(0, 1), full_counts.get(1, 1)
    
    # Calculate training steps
    total_train_raw = count_samples_all_shards(SPLIT_DIRS["Train"])
    if CFG.get("balance_classes", False):
        neg_keep_prob = CFG.get("neg_keep_prob", 0.25)
        effective_train = int(Nf + Nn * neg_keep_prob)
    else:
        effective_train = total_train_raw
    steps_per_epoch = max(1, effective_train // CFG["batch_size"])
    print(f"Train samples (raw): {total_train_raw:,} | effective (est): {effective_train:,} | steps/epoch: {steps_per_epoch:,}")
    
    # Build model
    model = build_model(num_classes=2)
    model = model.to(device)
    
    # Create dataloaders
    print("Creating dataloaders...")
    dls = create_dataloaders()
    
    # Setup loss function
    if CFG.get("balance_classes", False):
        # Balanced data - no class weights needed
        class_weights = None
        print("Using balanced sampling - no class weights")
    else:
        # Imbalanced data - use class weights
        class_weights = torch.tensor([1.0, Nn/Nf], dtype=torch.float32, device=device)
        print(f"Using class weights: {class_weights.tolist()}")
    
    criterion = get_loss_function(
        use_focal=CFG["use_focal"],
        gamma=CFG["focal_gamma"],
        class_weights=class_weights
    )
    
    # Setup optimizer
    optimizer = create_optimizer_v2(
        model,
        opt='adamw',
        lr=CFG["lr"],
        weight_decay=0.05,
        layer_decay=0.75
    )
    
    # Setup gradient scaler for mixed precision
    scaler = torch.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG["lr"],
        epochs=CFG["epochs"],
        steps_per_epoch=steps_per_epoch
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    history = []
    epoch_metrics_path = os.path.join(exp_dir, "epoch_metrics.jsonl")
    for epoch in range(CFG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        print("-" * 40)
        
        # Train
        avg_train_loss, train_metrics = train_epoch(
            model, dls["Train"], criterion, optimizer, scheduler, scaler, device, steps_per_epoch
        )
        
        # Validate
        avg_val_loss, val_metrics = validate_epoch(model, dls["Validation"], criterion, device)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_metrics['train_acc']:.2f}% | Val Acc: {val_metrics['val_acc']:.2f}%")
        print(f"Train P/R/F1: {train_metrics['train_precision']:.3f}/{train_metrics['train_recall']:.3f}/{train_metrics['train_f1']:.3f} | Val P/R/F1: {val_metrics['val_precision']:.3f}/{val_metrics['val_recall']:.3f}/{val_metrics['val_f1']:.3f}")
        print(f"Train AUC: {train_metrics['train_auc']:.3f} | Val AUC: {val_metrics['val_auc']:.3f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
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
    
    # Save model
    tag = f'{CFG["model_name"]}_lr{CFG["lr"]}_ep{CFG["epochs"]}{"_focal" if CFG["use_focal"] else ""}'
    model_path = os.path.join(exp_dir, f"{tag}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")
    
    results = evaluate_model(
        model,
        dls["Test"],
        device,
        plot_dir,
        CFG["backbone"],
        save_pr_curve=CFG["save_pr_curve"]
    )
    
    # Add metadata to results
    results.update({
        "seed": CFG.get("seed", None),
        "backbone": CFG["backbone"],
        "use_flow": CFG["use_flow"],
        "use_seq": CFG.get("use_seq", False),
        "epochs": CFG["epochs"],
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
