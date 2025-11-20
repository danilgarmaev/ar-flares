"""Loss functions for AR-flares classification."""
import torch
import torch.nn as nn


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
        self.ce = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        return -((1 - pt) ** self.gamma * logpt)


def get_loss_function(use_focal=False, gamma=2.0, class_weights=None):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        use_focal: Whether to use Focal Loss
        gamma: Focal loss gamma parameter
        class_weights: Tensor of class weights for handling imbalance
        
    Returns:
        Loss function (nn.Module)
    """
    if use_focal:
        return FocalLoss(gamma=gamma, weight=class_weights)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)
