import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def sobel_grad_mag(x):
    """Compute gradient magnitude using Sobel operator."""
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 4.0
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 4.0
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def topk_crops_from_grad(x, k=8, crop_size=64, stride=32):
    """Select top-K crops based on gradient magnitude."""
    B, _, H, W = x.shape
    g = sobel_grad_mag(x)
    
    unfold = F.unfold(g, kernel_size=crop_size, stride=stride)
    scores = unfold.mean(dim=1)
    
    k = min(k, scores.shape[1])
    _, topk_idx = scores.topk(k, dim=1)
    
    x_unf = F.unfold(x, kernel_size=crop_size, stride=stride)
    crops_list = []
    for b in range(B):
        cols = x_unf[b, :, topk_idx[b]]
        crops = cols.t().contiguous().view(k, 1, crop_size, crop_size)
        crops_list.append(crops)
    
    return crops_list


class LocalCropCNN(nn.Module):
    """CNN encoder for 64x64 crops → embedding."""
    def __init__(self, in_ch=1, embed_dim=192, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, crops_b):
        h = self.net(crops_b)
        return self.proj(h)


class CrossAttn(nn.Module):
    """Cross-attention: global queries local tokens."""
    def __init__(self, dim_q, dim_kv, num_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(dim_q, dim_q)
        self.k = nn.Linear(dim_kv, dim_q)
        self.v = nn.Linear(dim_kv, dim_q)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=dropout
        )
        self.proj = nn.Linear(dim_q, dim_q)

    def forward(self, q_tok, kv_tok):
        Q = self.q(q_tok)
        K = self.k(kv_tok)
        V = self.v(kv_tok)
        out, _ = self.attn(Q, K, V)
        return self.proj(out)


class MultiScaleFusionModel(nn.Module):
    """
    Multi-scale fusion model:
    - Global: ViT over full 224x224 image
    - Local: top-K gradient-based 64x64 crops
    - Fusion: cross-attention + classifier
    """
    def __init__(self, num_classes=2, k_crops=8, crop_size=64, stride=32,
                 vit_name="vit_base_patch16_224", local_dim=192, num_heads=4,
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        self.k_crops = k_crops
        self.crop_size = crop_size
        self.stride = stride

        # Global ViT
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.global_dim = self.vit.num_features

        # Local crop encoder
        self.local = LocalCropCNN(in_ch=1, embed_dim=local_dim)

        # Cross-attention
        self.cross = CrossAttn(
            dim_q=self.global_dim, 
            dim_kv=local_dim, 
            num_heads=num_heads, 
            dropout=0.1
        )

        # Classifier
        self.head = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            nn.Linear(self.global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # (B, 1, 224, 224)
        B = x.size(0)

        # Global branch
        x3 = x.repeat(1, 3, 1, 1)
        g = self.vit(x3)
        g_cls = g.unsqueeze(1)

        # Local crops
        crops_list = topk_crops_from_grad(x, k=self.k_crops, 
                                          crop_size=self.crop_size, 
                                          stride=self.stride)
        
        # Encode local crops
        local_tokens = []
        for b in range(B):
            tok_b = self.local(crops_list[b])
            local_tokens.append(tok_b)
        
        # Pad to fixed K
        K = max(t.shape[0] for t in local_tokens)
        Dl = local_tokens[0].shape[1]
        kv = x.new_zeros((B, K, Dl), dtype=local_tokens[0].dtype)
        for b in range(B):
            kb = local_tokens[b].shape[0]
            kv[b, :kb, :] = local_tokens[b]

        # Cross-attention fusion
        fused = self.cross(g_cls, kv).squeeze(1)

        # Classify
        return self.head(fused)