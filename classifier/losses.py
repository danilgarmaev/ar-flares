import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        return -((1 - pt) ** self.gamma * logpt)


def get_loss_function(class_weights, use_focal=False, gamma=2.0):
    """Factory for loss functions."""
    if use_focal:
        return FocalLoss(gamma=gamma, weight=class_weights)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)