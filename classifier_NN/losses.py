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


def get_loss_function(use_focal=False, gamma=2.0, class_weights=None, use_mixup=False, label_smoothing=0.0):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        use_focal: Whether to use Focal Loss
        gamma: Focal loss gamma parameter
        class_weights: Tensor of class weights for handling imbalance
        use_mixup: Whether Mixup/Cutmix is enabled (requires SoftTargetCrossEntropy)
        label_smoothing: Amount of label smoothing (0.0 to 1.0)
        
    Returns:
        Loss function (nn.Module)
    """
    if use_mixup:
        # Mixup generates soft targets, so we need a loss that handles them
        return SoftTargetCrossEntropy()
    
    if use_focal:
        return FocalLoss(gamma=gamma, weight=class_weights)
    
    if label_smoothing > 0:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
    return nn.CrossEntropyLoss(weight=class_weights)
