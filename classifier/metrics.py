import numpy as np
import torch


class MetricsCalculator:
    """Calculate solar flare prediction metrics (TSS, HSS, TPR, TNR)."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.TP = self.TN = self.FP = self.FN = 0
    
    def update(self, y_true, y_pred):
        """Update confusion matrix counts."""
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
        """Compute final metrics."""
        TPR = self.TP / (self.TP + self.FN + 1e-7)
        TNR = self.TN / (self.TN + self.FP + 1e-7)
        TSS = TPR + TNR - 1
        HSS = 2 * (self.TP * self.TN - self.FN * self.FP) / (
            (self.TP + self.FN) * (self.FN + self.TN) + 
            (self.TP + self.FP) * (self.FP + self.TN) + 1e-7
        )
        
        return {
            "TPR": TPR,
            "TNR": TNR,
            "TSS": TSS,
            "HSS": HSS,
            "TP": self.TP,
            "TN": self.TN,
            "FP": self.FP,
            "FN": self.FN,
        }


def find_best_threshold(y_true, y_probs, metric='tss', n_steps=181):
    """Find optimal threshold by maximizing TSS."""
    best_score = -1.0
    best_thresh = 0.5
    
    for t in np.linspace(0.05, 0.95, n_steps):
        y_pred = (y_probs >= t).astype(int)
        
        TP = int(((y_true == 1) & (y_pred == 1)).sum())
        TN = int(((y_true == 0) & (y_pred == 0)).sum())
        FP = int(((y_true == 0) & (y_pred == 1)).sum())
        FN = int(((y_true == 1) & (y_pred == 0)).sum())
        
        TPR = TP / (TP + FN + 1e-7)
        TNR = TN / (TN + FP + 1e-7)
        TSS = TPR + TNR - 1
        
        if TSS > best_score:
            best_score = TSS
            best_thresh = t
    
    return best_thresh, best_score