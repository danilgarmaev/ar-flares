"""LearnableScaler normalization replacement.

Adapted from Anh Tuan Nghiem's LearnableScaler repository:
https://github.com/nghiemtuan/LearnableScaler
"""

from __future__ import annotations

import torch
from torch import nn


class LearnableScaler(nn.Module):
    """Channel-last learnable affine scaler for Transformer features."""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        del eps, affine
        self.a = nn.Parameter(torch.ones(int(num_channels)))
        self.b = nn.Parameter(torch.zeros(int(num_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * max(0, x.dim() - 1) + [-1]
        return self.a.view(*shape) * x + self.b.view(*shape)


class LearnableScaler2d(nn.Module):
    """NCHW learnable affine scaler for CNN feature maps."""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        del eps, affine
        self.a = nn.Parameter(torch.ones(int(num_channels)))
        self.b = nn.Parameter(torch.zeros(int(num_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a[None, :, None, None] * x + self.b[None, :, None, None]


def _num_features(module: nn.Module) -> int:
    if hasattr(module, "weight") and module.weight is not None:
        return int(len(module.weight))
    normalized_shape = getattr(module, "normalized_shape", None)
    if normalized_shape is not None:
        if isinstance(normalized_shape, int):
            return int(normalized_shape)
        return int(normalized_shape[-1])
    num_features = getattr(module, "num_features", None)
    if num_features is not None:
        return int(num_features)
    raise ValueError(f"Cannot infer feature count for {module.__class__.__name__}")


def replace_layers(model: nn.Module, old: type[nn.Module], new: type[nn.Module]) -> nn.Module:
    """Recursively replace modules of type ``old`` with ``new(num_features)``."""
    for name, module in list(model.named_children()):
        if len(list(module.children())) > 0:
            replace_layers(module, old, new)
        if isinstance(module, old):
            replacement = new(_num_features(module))
            try:
                model[int(name)] = replacement
            except (ValueError, TypeError):
                setattr(model, name, replacement)
    return model


def replace_norm(model: nn.Module, norm_cls: type[nn.Module] = LearnableScaler) -> nn.Module:
    """Replace LayerNorm with ``norm_cls``; this is the email-compatible API."""
    return replace_layers(model, nn.LayerNorm, norm_cls)
