import torch
import torch.nn as nn
import timm


class SmallFlowCNN(nn.Module):
    """Lightweight CNN for optical flow (2-channel input)."""
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
            nn.AdaptiveAvgPool2d((1, 1))
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
    """Medium CNN for optical flow."""
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
            nn.AdaptiveAvgPool2d((1, 1))
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
    """ResNet-based flow encoder."""
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18", 
            pretrained=False, 
            in_chans=2, 
            num_classes=out_dim
        )

    def forward(self, x):
        return self.backbone(x)


class TwoStreamModel(nn.Module):
    """
    Two-stream architecture for image + optical flow.
    - Image stream: pretrained timm backbone
    - Flow stream: lightweight CNN
    - Fusion: concatenate embeddings → MLP classifier
    """
    def __init__(self, img_backbone="deit_tiny_patch16_224", 
                 flow_encoder="SmallFlowCNN",
                 num_classes=2, pretrained=True, 
                 freeze_backbone=True, flow_dim=128):
        super().__init__()
        
        # Image stream
        self.img_model = timm.create_model(
            img_backbone, 
            pretrained=pretrained, 
            num_classes=0
        )
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
            raise ValueError(f"Unknown flow encoder: {flow_encoder}")

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(img_feat_dim + flow_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Handle tuple input (img, flow) or concatenated tensor
        if isinstance(x, (list, tuple)):
            img, flow = x
        else:
            # Fallback: split channels
            img = x[:, :3]
            flow = x[:, 3:]
        
        img_emb = self.img_model(img)
        flow_emb = self.flow_model(flow)
        z = torch.cat([img_emb, flow_emb], dim=1)
        return self.head(z)