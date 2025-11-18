import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

from config import CFG
from metrics import MetricsCalculator, find_best_threshold


def evaluate_model(model, dataloader, device, exp_dir):
    """Evaluate model on test set and generate plots."""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_preds_class = []
    metrics_calc = MetricsCalculator()
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds_class = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_preds_class.extend(preds_class)
            metrics_calc.update(labels, preds_class)
    
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds_class)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {CFG['backbone']}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(exp_dir, "plots", "roc.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # PR curve
    if CFG.get("save_pr_curve", True):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        
        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, label=f'PR-AUC = {pr_auc:.3f}', linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {CFG['backbone']}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(exp_dir, "plots", "pr.png"), dpi=200, bbox_inches='tight')
        plt.close()
    else:
        pr_auc = float('nan')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format='d')
    plt.title(f"Confusion Matrix (threshold=0.5) - {CFG['backbone']}")
    plt.savefig(os.path.join(exp_dir, "plots", "confusion_matrix.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Find best threshold
    best_thresh, best_tss = find_best_threshold(y_true, y_prob, metric='tss')
    
    # Compute final metrics
    final_metrics = metrics_calc.compute()
    
    # Save results
    results = {
        "AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "TPR": float(final_metrics["TPR"]),
        "TNR": float(final_metrics["TNR"]),
        "TSS": float(final_metrics["TSS"]),
        "HSS": float(final_metrics["HSS"]),
        "Best_TSS": float(best_tss),
        "Best_threshold": float(best_thresh),
        "TP": int(final_metrics["TP"]),
        "TN": int(final_metrics["TN"]),
        "FP": int(final_metrics["FP"]),
        "FN": int(final_metrics["FN"]),
        "backbone": CFG["backbone"],
        "use_flow": CFG["use_flow"],
        "use_diff": CFG.get("use_diff", False),
        "two_stream": CFG.get("two_stream", False),
        "epochs": CFG["epochs"],
        "lr": CFG["lr"],
        "batch_size": CFG["batch_size"],
    }
    
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"TSS: {best_tss:.4f} @ threshold={best_thresh:.3f}")
    print(f"HSS: {final_metrics['HSS']:.4f}")
    
    # Generate summary markdown
    summary_path = os.path.join(exp_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# AR-Flares Experiment Summary\n\n")
        f.write(f"**Model:** `{CFG['model_name']}`\n")
        f.write(f"**Backbone:** {CFG['backbone']}\n")
        f.write(f"**Date:** {results.get('date', 'N/A')}\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Use Flow: {CFG['use_flow']}\n")
        f.write(f"- Use Diff: {CFG.get('use_diff', False)}\n")
        f.write(f"- Two-Stream: {CFG.get('two_stream', False)}\n")
        f.write(f"- Freeze Backbone: {CFG['freeze_backbone']}\n")
        f.write(f"- Learning Rate: {CFG['lr']}\n")
        f.write(f"- Epochs: {CFG['epochs']}\n")
        f.write(f"- Batch Size: {CFG['batch_size']}\n\n")
        
        f.write("## Results\n")
        f.write(f"- **AUC:** {roc_auc:.4f}\n")
        f.write(f"- **PR-AUC:** {pr_auc:.4f}\n")
        f.write(f"- **TSS:** {best_tss:.4f} (threshold={best_thresh:.3f})\n")
        f.write(f"- **HSS:** {final_metrics['HSS']:.4f}\n")
        f.write(f"- **TPR:** {final_metrics['TPR']:.4f}\n")
        f.write(f"- **TNR:** {final_metrics['TNR']:.4f}\n\n")
        
        f.write("## Confusion Matrix (threshold=0.5)\n")
        f.write(f"- TP: {final_metrics['TP']}\n")
        f.write(f"- TN: {final_metrics['TN']}\n")
        f.write(f"- FP: {final_metrics['FP']}\n")
        f.write(f"- FN: {final_metrics['FN']}\n\n")
        
        f.write("## Notes\n")
        f.write("_Add your observations here_\n")
    
    print(f"Summary written to {summary_path}")
    
    return results