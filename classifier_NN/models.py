"""Model architectures for AR-flares classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.video import r3d_18
from peft import LoraConfig, get_peft_model, TaskType

from .config import CFG

# Get image size from config (used as a fallback; prefer cfg passed into build_model)
IMG_SIZE = CFG.get("image_size", 224)


# ===================== UTILITY FUNCTIONS =====================
def _enable_head_grads(model: nn.Module):
    """Turn grads back on for the classifier head, regardless of attribute name."""
    enabled = 0
    # Prefer timm's get_classifier() if available
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module):
            for p in head.parameters(): 
                p.requires_grad = True
                enabled += p.numel()

    # Also try common names
    for name in ["head", "fc", "classifier", "cls", "last_linear"]:
        m = getattr(model, name, None)
        if isinstance(m, nn.Module):
            for p in m.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    enabled += p.numel()
    if enabled == 0:
        # Fallback: enable the last Linear modules we can find
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            for p in last_linear.parameters(): 
                p.requires_grad = True


# ===================== LORA INTEGRATION =====================
class PeftTimmWrapper(nn.Module):
    """Wrapper to make timm models compatible with PEFT/LoRA"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        # PEFT passes input_ids, but timm models expect raw tensors
        x = input_ids if input_ids is not None else pixel_values
        return self.model(x)


def apply_lora_to_timm(model, r=8, alpha=16, dropout=0.1):
    """Apply LoRA fine-tuning to a timm model."""
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv", "fc1", "fc2"],  # ViT layers
        bias="none"
    )
    model = PeftTimmWrapper(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ===================== DOMAIN-SPECIFIC CNN =====================
class CANSmall(nn.Module):
    """
    Convolutional Attention Network for magnetograms (domain-specific CNN+attention).
    Designed for single-channel grayscale input (1×224×224).
    """
    def __init__(self, in_chans=1, num_classes=2, embed_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
        )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # (B,1,H,W)
        h = self.conv(x)                      # (B,embed_dim,H',W')
        h = h.flatten(2).transpose(1, 2)      # (B,N,embed_dim)
        h, _ = self.attn(h, h, h)
        h = h.mean(1)                         # global attention pooling
        return self.head(h)


# ===================== FLOW ENCODERS =====================
class SmallFlowCNN(nn.Module):
    """Lightweight CNN for optical flow encoding."""
    def __init__(self, out_dim=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class MediumFlowCNN(nn.Module):
    """Medium-capacity CNN for optical flow encoding."""
    def __init__(self, out_dim=256, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class FlowResNetTiny(nn.Module):
    """A mini ResNet block for optical flow."""
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=False, in_chans=2, num_classes=out_dim)
    
    def forward(self, x):  # (B,2,H,W)
        return self.backbone(x)


# ===================== TWO-STREAM MODEL =====================
class TwoStreamModel(nn.Module):
    """Two-stream architecture combining image and optical flow encoders."""
    def __init__(self, img_backbone="deit_tiny_patch16_224", flow_encoder="SmallFlowCNN",
                 num_classes=2, pretrained=True, freeze_backbone=True, flow_dim=128):
        super().__init__()
        # Image stream
        self.img_model = timm.create_model(img_backbone, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.img_model.parameters():
                p.requires_grad = False
        img_feat_dim = self.img_model.num_features

        # Flow stream
        if flow_encoder == "SmallFlowCNN":
            self.flow_model = SmallFlowCNN(out_dim=flow_dim)
        elif flow_encoder == "MediumFlowCNN":
            self.flow_model = MediumFlowCNN(out_dim=flow_dim)
        elif flow_encoder == "FlowResNetTiny":
            self.flow_model = FlowResNetTiny(out_dim=flow_dim)
        else:
            raise ValueError(f"Unknown flow encoder '{flow_encoder}'")

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(img_feat_dim + flow_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Handle two-stream input (tuple) or concatenated tensor
        if isinstance(x, (list, tuple)):
            img, flow = x  # unpack the two inputs
        else:
            # fallback: derive img and flow from single tensor
            img = x[:, :1].repeat(1, 3, 1, 1)
            flow = x[:, 1:, :, :]
        
        img_emb = self.img_model(img)
        flow_emb = self.flow_model(flow)
        z = torch.cat([img_emb, flow_emb], dim=1)
        return self.head(z)


# ===================== TEMPORAL WRAPPER =====================
class TemporalWrapper(nn.Module):
    """
    Shared 2D backbone across T frames. Input: (B,T,1,224,224).
    ViT/ConvNeXt expects 3ch so we repeat channel.
    Aggregate temporally via mean or attention, then classify.
    """
    def __init__(self, backbone_name="vit_base_patch16_224", num_classes=2,
                 pretrained=True, freeze_backbone=False, aggregate="mean"):
        super().__init__()
        self.aggregate = aggregate
        # feature extractor
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.backbone.parameters(): 
                p.requires_grad = False
        self.feat_dim = self.backbone.num_features

        if aggregate == "attn":
            self.temporal_attn = nn.MultiheadAttention(self.feat_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(self.feat_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B,T,1,H,W)
        B, T, _, H, W = x.shape
        x = x.reshape(B*T, 1, H, W).repeat(1, 3, 1, 1)  # -> (B*T,3,H,W)
        feats = self.backbone(x)                        # (B*T, D)
        feats = feats.view(B, T, self.feat_dim)         # (B,T,D)

        if self.aggregate == "mean":
            z = feats.mean(dim=1)                       # (B,D)
        else:
            # attention over time (CLS-free): self-attend and pool
            z, _ = self.temporal_attn(feats, feats, feats)  # (B,T,D)
            z = self.norm(z).mean(dim=1)

        return self.head(z)


# ===================== 3D CNN & VIDEO TRANSFORMERS =====================
class Simple3DCNN(nn.Module):
    """Lightweight 3D CNN for sequences of grayscale magnetograms.

    Expects input of shape (B, T, 1, H, W). Internally we permute to
    (B, 1, T, H, W) for Conv3d. The network produces a per-sequence
    representation which is then classified.
    """

    def __init__(self, in_chans: int = 1, num_frames: int = 3, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        # Input layout for conv3d: (B, C, T, H, W)
        self.num_frames = num_frames
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(in_chans, c, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # pool spatial only

            nn.Conv3d(c, 2 * c, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(2 * c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # pool time and space

            nn.Conv3d(2 * c, 4 * c, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(4 * c),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * c, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W) -> (B, 1, T, H, W)
        if x.dim() == 5:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Simple3DCNN expects input of shape (B,T,1,H,W), got {x.shape}")
        h = self.features(x)
        return self.classifier(h)


class ResNet3DSimple(nn.Module):
    """Wrapper around torchvision r3d_18 for sequence magnetograms.

    Expects input of shape (B, T, 1, H, W). Internally permutes to
    (B, 1, T, H, W) and maps single-channel input to 3 channels via a
    learnable 1x1x1 Conv before feeding into r3d_18.
    """

    def __init__(self, num_frames: int = 3, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.num_frames = num_frames

        # r3d_18 expects (B, 3, T, H, W). We keep a lightweight
        # 1x1x1 conv to map 1ch -> 3ch instead of repeating.
        self.input_proj = nn.Conv3d(1, 3, kernel_size=1)

        self.backbone = r3d_18(pretrained=pretrained)
        in_feats = self.backbone.fc.in_features
        # Replace classification head
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W) -> (B, 1, T, H, W)
        if x.dim() != 5:
            raise ValueError(f"ResNet3DSimple expects (B,T,1,H,W), got {x.shape}")
        B, T, C, H, W = x.shape
        assert C == 1, "ResNet3DSimple currently assumes 1-channel input"

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,1,T,H,W)
        x = self.input_proj(x)                     # (B,3,T,H,W)
        return self.backbone(x)


class VideoTransformer(nn.Module):
    """Video Transformer using a 2D image backbone + temporal attention.

    This reuses any timm image backbone (ConvNeXt, ViT, etc.) as a
    per-frame feature extractor and applies a transformer encoder over
    the temporal dimension.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        num_classes: int = 2,
        num_frames: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        # Use timm backbone as per-frame encoder (no classifier)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.feat_dim = self.backbone.num_features

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.feat_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W) grayscale -> repeat to 3 channels per frame
        if x.dim() != 5:
            raise ValueError(f"VideoTransformer expects (B,T,1,H,W), got {x.shape}")
        B, T, C, H, W = x.shape
        assert C == 1, "VideoTransformer currently assumes 1-channel input"

        x = x.view(B * T, C, H, W).repeat(1, 3, 1, 1)  # (B*T,3,H,W)
        feats = self.backbone(x)  # (B*T, D)
        feats = feats.view(B, T, self.feat_dim)  # (B,T,D)

        # Add simple temporal encoding (learned position embeddings over T)
        # This keeps things minimal; could be extended with more explicit encodings.
        feats = self.temporal_encoder(feats)  # (B,T,D)
        z = feats.mean(dim=1)  # global temporal pooling
        return self.cls_head(z)


# ===================== MULTI-SCALE FUSION =====================
def sobel_grad_mag(x1):
    """Compute gradient magnitude using Sobel operator. x1: (B,1,H,W), returns (B,1,H,W)"""
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x1.dtype, device=x1.device).view(1,1,3,3)/4.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x1.dtype, device=x1.device).view(1,1,3,3)/4.0
    gx = F.conv2d(x1, kx, padding=1)
    gy = F.conv2d(x1, ky, padding=1)
    g = torch.sqrt(gx*gx + gy*gy + 1e-8)
    return g


def topk_crops_from_grad(x1, k=8, crop_size=64, stride=32):
    """
    Extract top-K crops by gradient magnitude.
    x1: (B,1,224,224), returns list of length B with tensors (k,1,crop_size,crop_size).
    """
    B, _, H, W = x1.shape
    g = sobel_grad_mag(x1)  # (B,1,H,W)

    # Unfold gradient to windows and score by mean
    unfold = F.unfold(g, kernel_size=crop_size, stride=stride)  # (B, crop_size*crop_size, L)
    scores = unfold.mean(dim=1)                                 # (B, L)

    # Get top-K indices per batch
    k = min(k, scores.shape[1])
    topk_vals, topk_idx = scores.topk(k, dim=1)                 # (B,k)

    # Unfold original image and gather crops
    x_unf = F.unfold(x1, kernel_size=crop_size, stride=stride)  # (B, crop_size*crop_size, L)
    crops_list = []
    for b in range(B):
        cols = x_unf[b, :, topk_idx[b]]                         # (crop_area, k)
        crops = cols.t().contiguous().view(k, 1, crop_size, crop_size)
        crops_list.append(crops)
    return crops_list  # length B


class LocalCropCNN(nn.Module):
    """Local CNN for 64x64 crops -> token embedding."""
    def __init__(self, in_ch=1, embed_dim=192, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 32x32
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),      # 16x16
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),    # 8x8
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, crops_b):  # (k,1,64,64) -> (k, embed_dim)
        h = self.net(crops_b)
        z = self.proj(h)
        return z


class CrossAttn(nn.Module):
    """Cross-Attention block: query=global CLS, keys/values=local tokens"""
    def __init__(self, dim_q, dim_kv, num_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(dim_q, dim_q)
        self.k = nn.Linear(dim_kv, dim_q)
        self.v = nn.Linear(dim_kv, dim_q)
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(dim_q, dim_q)

    def forward(self, q_tok, kv_tok):
        """
        q_tok: (B,1,Dq)  [global CLS token]
        kv_tok: (B,K,Dv) [local crop tokens]
        returns: (B,1,Dq)
        """
        Q = self.q(q_tok)
        K = self.k(kv_tok)
        V = self.v(kv_tok)
        out, _ = self.attn(Q, K, V)   # (B,1,Dq)
        return self.proj(out)


class MultiScaleFusionModel(nn.Module):
    """
    Multi-scale fusion model combining global ViT features with local high-gradient crops.
    Global branch: ViT-Base over 224x224 (grayscale -> repeated to 3ch internally).
    Local branch: top-K high-gradient 64x64 crops -> LocalCropCNN -> tokens.
    Fusion: Cross-attention (CLS queries locals), then MLP head.
    """
    def __init__(self, num_classes=2, k_crops=8, crop_size=64, stride=32,
                 vit_name="vit_base_patch16_224", local_dim=192, num_heads=4,
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        self.k_crops = k_crops
        self.crop_size = crop_size
        self.stride = stride

        # Global ViT encoder without classifier (features only)
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.vit.parameters(): 
                p.requires_grad = False
        self.global_dim = self.vit.num_features  # 768 for ViT-B/16

        # Local crop encoder
        self.local = LocalCropCNN(in_ch=1, embed_dim=local_dim)

        # Cross-attention: map local tokens to global dim inside attention block
        self.cross = CrossAttn(dim_q=self.global_dim, dim_kv=local_dim, num_heads=num_heads, dropout=0.1)

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            nn.Linear(self.global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # x: (B,1,224,224) grayscale
        B = x.size(0)

        # Global branch (repeat to 3ch for ViT)
        x3 = x.repeat(1, 3, 1, 1)                            # (B,3,224,224)
        g = self.vit(x3)                                     # (B, Dg) pooled features
        g_cls = g.unsqueeze(1)                               # (B,1,Dg) as a "CLS" token

        # Local crops
        crops_list = topk_crops_from_grad(x, k=self.k_crops, crop_size=self.crop_size, stride=self.stride)
        # Pack variable list into a single tensor of tokens
        local_tokens = []
        for b in range(B):
            tok_b = self.local(crops_list[b])                # (K, Dl)
            local_tokens.append(tok_b)
        # Pad to fixed K if needed
        K = max(t.shape[0] for t in local_tokens)
        Dl = local_tokens[0].shape[1]
        kv = x.new_zeros((B, K, Dl), dtype=local_tokens[0].dtype)  # (B,K,Dl)
        for b in range(B):
            kb = local_tokens[b].shape[0]
            kv[b, :kb, :] = local_tokens[b]

        # Cross-attention: CLS queries locals
        fused = self.cross(g_cls, kv)                        # (B,1,Dg)
        fused = fused.squeeze(1)                             # (B,Dg)

        # Classify
        logits = self.head(fused)
        return logits


# ===================== PHYSICS-INFORMED ATTENTION MODEL =====================
class DiffAttentionModel(nn.Module):
    """
    Physics-Informed Attention Model.
    Uses the Difference Image (t - t-1) to generate a spatial attention map.
    
    Branch 1: Main Image -> Backbone -> Features
    Branch 2: Diff Image -> Shallow CNN -> Attention Map (1, H/32, W/32)
    
    Combination: Features * (1 + Attention)
    """
    def __init__(self, backbone_name="convnext_tiny", num_classes=2, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Main Backbone (Image)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=3)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.num_features = self.backbone.num_features
        
        # Attention Branch (Diff Image)
        # Simple 3-layer CNN to extract "Activity Map" from difference image
        # Input: (B, 1, H, W) -> Output: (B, 1, 1, 1) or (B, 1, H_feat, W_feat)
        # For simplicity with global pooling backbones, we'll compute a Global Attention Weight
        # or we can try to inject it earlier.
        # Let's do "Feature Modulation": Diff branch predicts a scalar weight per channel?
        # Or better: Diff branch predicts a spatial map.
        # Since ConvNeXt pools to (B, C), spatial attention must happen BEFORE pooling.
        # But timm models encapsulate the pooling.
        # Strategy: Use Diff to predict a "Gate" vector (B, C) that reweights channels.
        # "Channel Attention" guided by Dynamics.
        
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # H/2
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # H/4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # H/8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # (B, 64, 1, 1)
            nn.Flatten(),            # (B, 64)
            nn.Linear(64, self.num_features), # (B, C)
            nn.Sigmoid() # Gate [0, 1]
        )
        
        # Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
        )
        
    def forward(self, x):
        # x is tuple (img, diff)
        img, diff = x
        
        # Extract static features
        feats = self.backbone(img) # (B, C)
        
        # Extract dynamic attention gate
        gate = self.diff_encoder(diff) # (B, C)
        
        # Modulate features: "Pay attention to channels that align with dynamics"
        # We use residual connection: F' = F * (1 + Gate)
        # This allows the model to ignore dynamics if irrelevant (Gate=0)
        modulated_feats = feats * (1 + gate)
        
        # Classify
        logits = self.head(modulated_feats)
        return logits


# ===================== MODEL BUILDER =====================
def build_model(cfg=None, num_classes=2):
    """Factory function to build the appropriate model based on a config dict.

    Args:
        cfg: configuration dictionary. If None, falls back to global CFG for
             backwards compatibility.
        num_classes: number of output classes.
    """

    if cfg is None:
        cfg = CFG

    backbone = cfg.get("backbone", CFG.get("backbone", "resnet18"))
    img_size = cfg.get("image_size", IMG_SIZE)

    # Explicit 3D CNN video backbone
    if backbone.lower() in ["simple3dcnn", "3d_cnn", "resnet3d_simple"]:
        if backbone.lower() in ["simple3dcnn", "3d_cnn"]:
            model = Simple3DCNN(
                in_chans=1,
                num_frames=cfg.get("seq_T", 3),
                num_classes=num_classes,
                base_channels=cfg.get("video_base_channels", 32),
            )
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Built Simple3DCNN with T={cfg.get('seq_T', 3)} | trainable={n_train:,}")
            return model
        else:  # "resnet3d_simple"
            model = ResNet3DSimple(
                num_frames=cfg.get("seq_T", 3),
                num_classes=num_classes,
                pretrained=cfg.get("pretrained_3d", False),
            )
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Built ResNet3DSimple (r3d_18) with T={cfg.get('seq_T', 3)} | trainable={n_train:,}")
            return model

    # Explicit Video Transformer backbone (multi-scale temporal transformer)
    if backbone.lower() in ["video_transformer", "timeformer", "video_vit"]:
        model = VideoTransformer(
            backbone_name=cfg.get("video_backbone", cfg.get("backbone", "convnext_tiny")),
            num_classes=num_classes,
            num_frames=cfg.get("seq_T", 3),
            pretrained=cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
            num_heads=cfg.get("video_heads", 4),
            num_layers=cfg.get("video_layers", 2),
            dropout=cfg.get("drop_rate", 0.1),
        )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Built VideoTransformer over {cfg.get('video_backbone', cfg.get('backbone', 'convnext_tiny'))} "
            f"with T={cfg.get('seq_T', 3)} | trainable={n_train:,}"
        )
        return model

    # Physics-Informed Attention Model
    if cfg.get("use_diff_attention", False):
        model = DiffAttentionModel(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
        )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Built Physics-Informed DiffAttentionModel over {backbone} | trainable={n_train:,}")
        return model

    # Two-stream model
    if cfg.get("use_flow") and cfg.get("two_stream", False):
        model = TwoStreamModel(
            img_backbone=backbone,
            flow_encoder=cfg.get("flow_encoder", "SmallFlowCNN"),
            num_classes=num_classes,
            pretrained=cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
        )
        if cfg.get("use_lora", False):
            print("Using LoRA fine-tuning")
            model = apply_lora_to_timm(model)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Built Two-Stream model over {backbone} | trainable={n_train:,}")
        return model

    # Temporal sequence model
    if cfg.get("use_seq"):
        model = TemporalWrapper(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
            aggregate=cfg.get("seq_aggregate", "mean"),
        )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Built TemporalWrapper over {backbone} | trainable={n_train:,}")
        return model

    # Multi-scale fusion model
    if backbone.lower() in ["ms_fusion", "multiscale_fusion"]:
        model = MultiScaleFusionModel(
            num_classes=num_classes,
            k_crops=cfg.get("k_crops", 8),
            crop_size=cfg.get("crop_size", 64),
            stride=cfg.get("crop_stride", 32),
            vit_name=cfg.get("vit_name", "vit_base_patch16_224"),
            local_dim=cfg.get("local_dim", 192),
            num_heads=cfg.get("cross_heads", 4),
            pretrained=cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
        )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Built MultiScaleFusionModel over {backbone} | trainable={n_train:,}")
        return model

    # Domain-specific CAN model
    if backbone.lower() in ["can_small", "can"]:
        model = CANSmall(in_chans=1, num_classes=num_classes)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Built CANSmall (Convolutional Attention Network) | trainable={n_train:,}")
        return model

    # Standard single-stream model (ConvNeXt/ViT/ResNet/VGG/etc.)
    # NOTE: Some families (e.g. VGG) do not support ConvNeXt/ViT-style kwargs such as
    # ``drop_path_rate``. Keep the dispatcher family-aware to avoid unexpected kwarg
    # errors like ``VGG.__init__() got an unexpected keyword argument 'drop_path_rate'``.
    in_chans = 3 if cfg.get("use_flow") else (2 if cfg.get("use_diff") else 3)

    backbone_lower = backbone.lower()

    # VGG family: do NOT pass ConvNeXt/ViT-specific arguments such as drop_path_rate.
    if backbone_lower.startswith("vgg"):
        model = timm.create_model(
            backbone,
            pretrained=cfg.get("pretrained", True),
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=cfg.get("drop_rate", 0.0),
        )
    else:
        model = timm.create_model(
            backbone,
            pretrained=cfg.get("pretrained", True),
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=cfg.get("drop_rate", 0.0),
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
        )

    # Reinitialize the classification head for safety
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        nn.init.xavier_uniform_(model.head.weight)
        model.head.bias.data.zero_()
    elif hasattr(model, "head") and isinstance(model.head, nn.Sequential):
        for m in model.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    if cfg.get("freeze_backbone", False):
        for p in model.parameters():
            p.requires_grad = False
        _enable_head_grads(model)

    if cfg.get("use_lora", False):
        print("Using LoRA fine-tuning")
        model = apply_lora_to_timm(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built single-stream model with backbone={backbone}, in_chans={in_chans}, img_size={img_size}, trainable={n_train:,}")
    return model
