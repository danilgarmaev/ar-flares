"""Metrics and evaluation utilities for AR-flares classification."""
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from tqdm import tqdm


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


def find_best_threshold_tss(
    y_true,
    y_probs,
    *,
    min_recall: float = 0.5,
    min_precision: float = 0.15,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    min_pos_rate: float | None = None,
    max_pos_rate: float | None = None,
    fallback_mode: str = "tss",
):
    """Select an operating threshold using Riggi et al. (2026)-style constraints.

    Search for the threshold $\tau^*$ that maximizes TSS subject to:
      - recall >= min_recall
      - precision >= min_precision

        If no threshold satisfies the constraints, use a robust fallback:
            - fallback_mode="tss" (default): maximize unconstrained TSS
            - fallback_mode="pr_sum": maximize (precision + recall)

    Args:
        y_true: Ground truth binary labels (0/1)
        y_probs: Predicted probabilities for positive class
        min_recall: Minimum allowed recall for constrained search
        min_precision: Minimum allowed precision for constrained search
        min_threshold: Lower bound of threshold search window
        max_threshold: Upper bound of threshold search window
        min_pos_rate: Optional minimum predicted-positive fraction
        max_pos_rate: Optional maximum predicted-positive fraction
        fallback_mode: Fallback strategy when constraints are infeasible

    Returns:
        (best_tss, best_threshold)
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)

    # Important: do not artificially clip the search range.
    # Some models (esp. with class weighting) can produce probabilities that
    # live almost entirely below 0.05; coarse grids over [0,1] can miss the
    # narrow support, so include a dense grid over the observed range too.
    lo = float(np.nanmin(y_probs)) if y_probs.size else 0.0
    hi = float(np.nanmax(y_probs)) if y_probs.size else 1.0
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))

    min_threshold = float(np.clip(min_threshold, 0.0, 1.0))
    max_threshold = float(np.clip(max_threshold, 0.0, 1.0))
    if max_threshold < min_threshold:
        min_threshold, max_threshold = max_threshold, min_threshold

    mode = str(fallback_mode).lower().strip()
    if mode not in {"tss", "pr_sum"}:
        mode = "tss"

    grid_full = np.linspace(0.0, 1.0, 1001)
    grid_obs = np.linspace(lo, hi, 1001) if hi > lo else np.array([lo])
    thresholds = np.unique(np.concatenate((grid_full, grid_obs, [0.0, 1.0])))
    thresholds = thresholds[(thresholds >= min_threshold) & (thresholds <= max_threshold)]
    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=float)

    best_constrained = (-1.0, 0.5, -1.0)  # (tss, threshold, pr_sum)
    found_constrained = False
    best_tss_fallback = (-1.0, 0.5, -1.0)  # (tss, threshold, pr_sum)
    best_prsum_fallback = (-1.0, 0.5, -1.0)  # (pr_sum, threshold, tss)

    def _within_pos_rate(pred_pos_rate: float) -> bool:
        if min_pos_rate is not None and pred_pos_rate < float(min_pos_rate):
            return False
        if max_pos_rate is not None and pred_pos_rate > float(max_pos_rate):
            return False
        return True

    for t in thresholds:
        pred = (y_probs >= t).astype(int)
        TP = int(((y_true == 1) & (pred == 1)).sum())
        TN = int(((y_true == 0) & (pred == 0)).sum())
        FP = int(((y_true == 0) & (pred == 1)).sum())
        FN = int(((y_true == 1) & (pred == 0)).sum())

        recall = TP / (TP + FN + 1e-7)
        precision = TP / (TP + FP + 1e-7)
        tnr = TN / (TN + FP + 1e-7)
        tss = recall + tnr - 1.0
        pr_sum = float(precision + recall)
        pred_pos_rate = float(pred.mean())
        within_rate = _within_pos_rate(pred_pos_rate)

        # Robust fallbacks are tracked only inside optional predicted-positive
        # rate bounds to avoid trivial all-positive/all-negative operating points.
        if within_rate:
            if tss > best_tss_fallback[0] or (
                tss == best_tss_fallback[0] and (pr_sum > best_tss_fallback[2] or (pr_sum == best_tss_fallback[2] and t > best_tss_fallback[1]))
            ):
                best_tss_fallback = (float(tss), float(t), pr_sum)
            if pr_sum > best_prsum_fallback[0] or (
                pr_sum == best_prsum_fallback[0] and (tss > best_prsum_fallback[2] or (tss == best_prsum_fallback[2] and t > best_prsum_fallback[1]))
            ):
                best_prsum_fallback = (pr_sum, float(t), float(tss))

        # Constrained optimum: maximize TSS subject to constraints
        if within_rate and recall >= float(min_recall) and precision >= float(min_precision):
            found_constrained = True
            if tss > best_constrained[0] or (
                tss == best_constrained[0] and (pr_sum > best_constrained[2] or (pr_sum == best_constrained[2] and t > best_constrained[1]))
            ):
                best_constrained = (float(tss), float(t), pr_sum)

    if found_constrained:
        return best_constrained[0], best_constrained[1]

    # No feasible threshold met (recall, precision) constraints.
    # Fall back robustly, preferring non-degenerate operating points.
    if mode == "pr_sum" and best_prsum_fallback[0] >= 0.0:
        return best_prsum_fallback[2], best_prsum_fallback[1]
    if mode == "tss" and best_tss_fallback[0] >= 0.0:
        return best_tss_fallback[0], best_tss_fallback[1]

    # If rate-bounded fallback had no candidate (unlikely), do an unconstrained
    # full sweep fallback.
    unconstrained_best_tss = (-1.0, 0.5, -1.0)
    unconstrained_best_prsum = (-1.0, 0.5, -1.0)
    for t in thresholds:
        pred = (y_probs >= t).astype(int)
        TP = int(((y_true == 1) & (pred == 1)).sum())
        TN = int(((y_true == 0) & (pred == 0)).sum())
        FP = int(((y_true == 0) & (pred == 1)).sum())
        FN = int(((y_true == 1) & (pred == 0)).sum())
        recall = TP / (TP + FN + 1e-7)
        precision = TP / (TP + FP + 1e-7)
        tnr = TN / (TN + FP + 1e-7)
        tss = recall + tnr - 1.0
        pr_sum = float(precision + recall)
        if tss > unconstrained_best_tss[0] or (
            tss == unconstrained_best_tss[0] and (pr_sum > unconstrained_best_tss[2] or (pr_sum == unconstrained_best_tss[2] and t > unconstrained_best_tss[1]))
        ):
            unconstrained_best_tss = (float(tss), float(t), pr_sum)
        if pr_sum > unconstrained_best_prsum[0] or (
            pr_sum == unconstrained_best_prsum[0] and (tss > unconstrained_best_prsum[2] or (tss == unconstrained_best_prsum[2] and t > unconstrained_best_prsum[1]))
        ):
            unconstrained_best_prsum = (pr_sum, float(t), float(tss))

    if mode == "pr_sum":
        return unconstrained_best_prsum[2], unconstrained_best_prsum[1]
    return unconstrained_best_tss[0], unconstrained_best_tss[1]


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


def evaluate_model(
    model,
    dataloader,
    device,
    save_dir,
    model_name,
    save_pr_curve=True,
    *,
    fixed_threshold: float | None = None,
):
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

    # IterableDatasets may not define __len__
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    
    with torch.no_grad():
        pbar = tqdm(total=total_batches, desc="Test", leave=True, disable=not is_tty, dynamic_ncols=True)
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

            pbar.update(1)
        pbar.close()
    
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
    
    # Find best threshold by TSS (NOTE: this is best-on-test; optimistic)
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

    fixed_metrics = None
    if fixed_threshold is not None:
        fixed_t = float(fixed_threshold)
        preds_fixed = (all_probs >= fixed_t).astype(int)
        metrics_calc_fixed = MetricsCalculator()
        metrics_calc_fixed.update(all_labels, preds_fixed)
        fixed_metrics = metrics_calc_fixed.compute()
        cmfixed_path = f"{save_dir}/confusion_matrix_fixed.png"
        plot_confusion_matrix(all_labels, preds_fixed, cmfixed_path, model_name, threshold=fixed_t)
    
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

    if fixed_metrics is not None:
        results.update(
            {
                "fixed_threshold": float(fixed_threshold),
                "fixed_TSS": float(fixed_metrics["TSS"]),
                "fixed_TPR": float(fixed_metrics["TPR"]),
                "fixed_TNR": float(fixed_metrics["TNR"]),
                "fixed_HSS": float(fixed_metrics["HSS"]),
                "fixed_Precision": float(fixed_metrics["Precision"]),
                "fixed_Recall": float(fixed_metrics["Recall"]),
                "fixed_F1": float(fixed_metrics["F1"]),
                "fixed_Accuracy": float(fixed_metrics["Accuracy"]),
            }
        )
    
    return results
