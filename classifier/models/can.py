import torch.nn as nn


class CANSmall(nn.Module):
    """
    Convolutional Attention Network for magnetograms.
    Designed for single-channel grayscale input (1×224×224).
    """
    def __init__(self, in_chans=1, num_classes=2, embed_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim), 
            nn.ReLU(inplace=True),
        )
        
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads=num_heads,
            dropout=dropout, 
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # (B, 1, H, W)
        h = self.conv(x)                      # (B, embed_dim, H', W')
        h = h.flatten(2).transpose(1, 2)      # (B, N, embed_dim)
        h, _ = self.attn(h, h, h)             # Self-attention
        h = h.mean(1)                         # Global attention pooling
        return self.head(h)