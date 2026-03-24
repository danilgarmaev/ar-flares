"""Loss functions for AR-flares classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples by down-weighting easy samples.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        weight: Class weights tensor
    """
    def __init__(self, gamma=2.0, alpha=None, weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            # alpha weighting per class index
            alpha_t = torch.full_like(pt, 1.0 - self.alpha)
            alpha_t[targets == 1] = self.alpha
            focal_weight = alpha_t * focal_weight
        loss = -(focal_weight * logpt)
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


class WeightedCrossEntropy(nn.Module):
    """Variant of CrossEntropyLoss that computes class weights from counts.

    This is a thin wrapper so experiments can select loss_type="ce_weighted"
    and pass in precomputed class_weights=[w0,w1,...].
    """
    def __init__(self, class_weights=None, label_smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.loss(logits, targets)


class BinaryCrossEntropyLogits(nn.Module):
    """Binary cross-entropy from 2-class logits.

    For 2-class logits [z0, z1], BCE over (z1 - z0) is equivalent to
    2-class cross-entropy. This lets us use explicit BCE semantics while
    keeping the existing 2-logit model heads unchanged.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        if logits.dim() == 2 and logits.size(1) >= 2:
            # Convert [z0, z1] -> scalar logit for positive class.
            logits_pos = logits[:, 1] - logits[:, 0]
        elif logits.dim() == 2 and logits.size(1) == 1:
            logits_pos = logits.squeeze(1)
        else:
            logits_pos = logits
        targets_f = targets.float()
        return self.loss(logits_pos, targets_f)

class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss operating on raw 2-class logits.

    Extracts the positive-class logit from 2-class output and applies the
    standard binary focal loss formula:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(z_pos) for positives (y=1),
          p_t = 1 - sigmoid(z_pos) for negatives (y=0),
          and z_pos is z1 (or z1 - z0) from the 2-class logit head.

    This is numerically stable: the cross-entropy term is computed via
    ``F.binary_cross_entropy_with_logits``.

    Args:
        gamma: Focusing parameter (default 2.0). Higher = harder-example focus.
        alpha: Positive-class balancing factor (default 0.25).
               ``(1 - alpha)`` is applied to the negative class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Reduce 2-class logits to a single positive-class logit
        if logits.dim() == 2 and logits.size(1) >= 2:
            logits_pos = logits[:, 1] - logits[:, 0]
        elif logits.dim() == 2 and logits.size(1) == 1:
            logits_pos = logits.squeeze(1)
        else:
            logits_pos = logits

        targets_f = targets.float()

        # p_t = sigmoid(z) for y=1; 1-sigmoid(z) for y=0
        probs = torch.sigmoid(logits_pos)
        p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)

        # alpha_t: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets_f + (1.0 - self.alpha) * (1.0 - targets_f)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** self.gamma

        # Numerically stable BCE per sample
        bce = F.binary_cross_entropy_with_logits(logits_pos, targets_f, reduction="none")

        loss = alpha_t * focal_weight * bce
        return loss.mean()


def get_loss_function(cfg=None, use_focal=False, gamma=2.0, class_weights=None, use_mixup=False, label_smoothing=0.0):
    """Factory function to get the appropriate loss function.
    
    Args:
        cfg: Optional experiment config dict. If provided, can specify
            loss_type ("ce", "ce_weighted", "focal", "skill_tss") and weights.
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
        elif loss_type in {"bce", "weighted_bce"}:
            # Optional positive-class weighting for BCE can be passed via cfg.
            pos_w = cfg.get("bce_pos_weight", None)
            if pos_w is None and loss_type == "weighted_bce" and class_weights is not None:
                try:
                    # Reuse the positive-vs-negative ratio already computed in train.py.
                    pos_w = float(class_weights[1].detach().cpu().item())
                except Exception:
                    pos_w = None
            pos_w_t = None
            if pos_w is not None:
                pos_w_t = torch.tensor(float(pos_w), dtype=torch.float32)
            return BinaryCrossEntropyLogits(pos_weight=pos_w_t)
        elif loss_type == "focal":
            gamma = cfg.get("focal_gamma", gamma)
            alpha = cfg.get("focal_alpha", None)
            return FocalLoss(gamma=gamma, alpha=alpha, weight=class_weights)
        elif loss_type == "binary_focal":
            gamma = cfg.get("focal_gamma", gamma)
            alpha = float(cfg.get("focal_alpha", 0.25))
            return BinaryFocalLoss(gamma=gamma, alpha=alpha)
        elif loss_type == "ce_weighted":
            # Explicit weighted CE variant (expects class_weights passed from training setup)
            return WeightedCrossEntropy(class_weights=class_weights, label_smoothing=label_smoothing)
        # fall through to standard cross entropy for "ce" or unknown types

    # Legacy behaviour without cfg
    if use_focal:
        return FocalLoss(gamma=gamma, weight=class_weights)
    
    if label_smoothing > 0:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
    return nn.CrossEntropyLoss(weight=class_weights)
