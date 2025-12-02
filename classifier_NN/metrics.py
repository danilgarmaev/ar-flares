"""Metrics and evaluation utilities for AR-flares classification."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve


class MetricsCalculator:
    """
    Calculate classification metrics including TPR, TNR, TSS, and HSS.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.TP = self.TN = self.FP = self.FN = 0
    
    def update(self, y_true, y_pred):
        """
        Update metrics with new predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted binary labels (not probabilities)
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        neg_t, neg_p = 1 - y_true, 1 - y_pred
        self.TP += int(np.sum(y_true * y_pred))
        self.TN += int(np.sum(neg_t * neg_p))
        self.FP += int(np.sum(neg_t * y_pred))
        self.FN += int(np.sum(y_true * neg_p))
    
    def compute(self):
        """
        Compute final metrics.
        
        Returns:
            Dictionary with TPR, TNR, TSS, HSS, and confusion matrix values
        """
        TPR = self.TP / (self.TP + self.FN + 1e-7)
        TNR = self.TN / (self.TN + self.FP + 1e-7)
        TSS = TPR + TNR - 1
        HSS = 2 * (self.TP * self.TN - self.FN * self.FP) / (
              (self.TP + self.FN) * (self.FN + self.TN) + 
              (self.TP + self.FP) * (self.FP + self.TN) + 1e-7)
        
        precision = self.TP / (self.TP + self.FP + 1e-7)
        recall = TPR
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        total = self.TP + self.TN + self.FP + self.FN
        accuracy = (self.TP + self.TN) / (total + 1e-7)

        return {
            "TPR": TPR,
            "TNR": TNR,
            "TSS": TSS,
            "HSS": HSS,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
            "TP": self.TP,
            "TN": self.TN,
            "FP": self.FP,
            "FN": self.FN
        }


def find_best_threshold_tss(y_true, y_probs):
    """
    Find optimal threshold by maximizing TSS (True Skill Statistic).
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities for positive class
        
    Returns:
        Tuple of (best_tss, best_threshold)
    """
    best_tss, best_t = -1.0, 0.5
    
    for t in np.linspace(0.05, 0.95, 181):
        pred = (y_probs >= t).astype(int)
        TP = int(((y_true == 1) & (pred == 1)).sum())
        TN = int(((y_true == 0) & (pred == 0)).sum())
        FP = int(((y_true == 0) & (pred == 1)).sum())
        FN = int(((y_true == 1) & (pred == 0)).sum())
        
        TPR = TP / (TP + FN + 1e-7)
        TNR = TN / (TN + FP + 1e-7)
        TSS = TPR + TNR - 1
        
        if TSS > best_tss:
            best_tss, best_t = TSS, t
    
    return best_tss, best_t


def plot_roc_curve(y_true, y_probs, save_path, model_name="Model"):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities for positive class
        save_path: Path to save the plot
        model_name: Name of the model for title
        
    Returns:
        AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_pr_curve(y_true, y_probs, save_path, model_name="Model"):
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities for positive class
        save_path: Path to save the plot
        model_name: Name of the model for title
        
    Returns:
        PR-AUC score
    """
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(rec, prec)
    
    plt.figure()
    plt.plot(rec, prec, label=f'PR-AUC = {pr_auc:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, save_path, model_name="Model", threshold=0.5):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        save_path: Path to save the plot
        model_name: Name of the model for title
        threshold: Decision threshold used
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format='d')
    plt.title(f"Confusion Matrix (threshold={threshold}) - {model_name}")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def evaluate_model(model, dataloader, device, save_dir, model_name, save_pr_curve=True):
    """Comprehensive model evaluation with all metrics and plots.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run evaluation on
        save_dir: Directory to save plots
        model_name: Name of the model for plots and filenames
        save_pr_curve: Whether to save PR curve

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    all_probs, all_labels, all_preds_class = [], [], []
    metrics_calc = MetricsCalculator()
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            # Keep 0.5 predictions for compatibility / quick view
            preds_class_05 = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_preds_class.extend(preds_class_05)
            metrics_calc.update(labels, preds_class_05)
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds_class_05 = np.array(all_preds_class)
    
    # ROC curve
    roc_path = f"{save_dir}/roc.png"
    roc_auc = plot_roc_curve(all_labels, all_probs, roc_path, model_name)
    
    # PR curve
    pr_auc = float('nan')
    if save_pr_curve:
        pr_path = f"{save_dir}/pr.png"
        pr_auc = plot_pr_curve(all_labels, all_probs, pr_path, model_name)
    
    # Find best threshold by TSS
    best_tss, best_t = find_best_threshold_tss(all_labels, all_probs)
    
    # Confusion matrix at 0.5 threshold (legacy view)
    cm05_path = f"{save_dir}/confusion_matrix_0.5.png"
    plot_confusion_matrix(all_labels, all_preds_class_05, cm05_path, model_name, threshold=0.5)
    
    # Confusion matrix at best-TSS threshold
    preds_best = (all_probs >= best_t).astype(int)
    cmbest_path = f"{save_dir}/confusion_matrix_bestTSS.png"
    plot_confusion_matrix(all_labels, preds_best, cmbest_path, model_name, threshold=best_t)
    
    # Compute final metrics using the **best-TSS threshold** for classification
    metrics_calc_best = MetricsCalculator()
    metrics_calc_best.update(all_labels, preds_best)
    final_metrics = metrics_calc_best.compute()
    
    results = {
        "AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "TPR": float(final_metrics["TPR"]),
        "TNR": float(final_metrics["TNR"]),
        "TSS": float(final_metrics["TSS"]),
        "HSS": float(final_metrics["HSS"]),
        "Precision": float(final_metrics["Precision"]),
        "Recall": float(final_metrics["Recall"]),
        "F1": float(final_metrics["F1"]),
        "Accuracy": float(final_metrics["Accuracy"]),
        "Best_TSS": float(best_tss),
        "Best_threshold": float(best_t),
        "TP": int(final_metrics["TP"]),
        "TN": int(final_metrics["TN"]),
        "FP": int(final_metrics["FP"]),
        "FN": int(final_metrics["FN"]),
    }
    
    return results
