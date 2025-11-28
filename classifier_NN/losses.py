"""Loss functions for AR-flares classification."""
import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples by down-weighting easy samples.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        weight: Class weights tensor
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma * logpt)
        return loss.mean()


class SkillOrientedTSSLoss(nn.Module):
    """Score-oriented loss that nudges training toward higher TSS.

    This implements a soft confusion-matrix based approximation of TSS:
        TP = sum(y * p)
        TN = sum((1 - y) * (1 - p))
        FP = sum((1 - y) * p)
        FN = sum(y * (1 - p))

    where p = sigmoid(logits). From these we compute
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        TSS = TPR + TNR - 1.

    The final loss is a combination of standard BCE-with-logits and the
    negative soft TSS:

        loss = bce_weight * BCE(logits, y) - tss_weight * TSS_soft

    Using a small tss_weight keeps training stable while biasing the model
    toward configurations with better TSS.
    """

    def __init__(self, bce_weight: float = 1.0, tss_weight: float = 0.1):
        super().__init__()
        self.bce_weight = bce_weight
        self.tss_weight = tss_weight
        # BCEWithLogits over scalar probabilities
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, num_classes) or (N,) for binary; targets: class indices
        if logits.dim() == 2 and logits.size(1) > 1:
            # Multi-class: reduce to positive-vs-rest using the positive class index 1.
            # This keeps the loss focused on the flare/no-flare decision.
            logits = logits[:, 1]
            targets = (targets == 1).float()
        else:
            # Binary case (logits shape [N] or [N,1])
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            if targets.dim() > 1:
                targets = targets.squeeze(1)
            targets = targets.float()

        probs = torch.sigmoid(logits)

        # Standard BCE term
        bce_loss = self.bce(logits, targets)

        # Soft confusion-matrix over the batch
        tp = (targets * probs).sum()
        fp = ((1.0 - targets) * probs).sum()
        fn = (targets * (1.0 - probs)).sum()
        tn = ((1.0 - targets) * (1.0 - probs)).sum()

        eps = 1e-7
        tpr_soft = tp / (tp + fn + eps)
        tnr_soft = tn / (tn + fp + eps)
        tss_soft = tpr_soft + tnr_soft - 1.0

        # Maximize TSS by subtracting it from the loss
        loss = self.bce_weight * bce_loss - self.tss_weight * tss_soft
        return loss


def get_loss_function(cfg=None, use_focal=False, gamma=2.0, class_weights=None, use_mixup=False, label_smoothing=0.0):
    """Factory function to get the appropriate loss function.
    
    Args:
        cfg: Optional experiment config dict. If provided, can specify
            loss_type ("ce", "focal", "skill_tss") and weights.
        use_focal: Whether to use Focal Loss (legacy flag, ignored if cfg specifies loss_type).
        gamma: Focal loss gamma parameter
        class_weights: Tensor of class weights for handling imbalance
        use_mixup: Whether Mixup/Cutmix is enabled (requires SoftTargetCrossEntropy)
        label_smoothing: Amount of label smoothing (0.0 to 1.0)
        
    Returns:
        Loss function (nn.Module)
    """
    # Mixup always uses soft-target cross entropy regardless of other settings
    if use_mixup:
        return SoftTargetCrossEntropy()

    # If a cfg dict is provided, prefer it as the source of truth
    if cfg is not None:
        loss_type = cfg.get("loss_type", "ce")
        if loss_type == "skill_tss":
            return SkillOrientedTSSLoss(
                bce_weight=cfg.get("bce_weight", 1.0),
                tss_weight=cfg.get("tss_loss_weight", 0.1),
            )
        elif loss_type == "focal":
            gamma = cfg.get("focal_gamma", gamma)
            return FocalLoss(gamma=gamma, weight=class_weights)
        # fall through to standard cross entropy for "ce" or unknown types

    # Legacy behaviour without cfg
    if use_focal:
        return FocalLoss(gamma=gamma, weight=class_weights)
    
    if label_smoothing > 0:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
    return nn.CrossEntropyLoss(weight=class_weights)
