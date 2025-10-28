from torchview import draw_graph
import torch
from model_training import TwoStreamModel

# --- create model ---
model = TwoStreamModel(
    img_backbone="deit_tiny_patch16_224",
    flow_encoder="SmallFlowCNN",
    num_classes=2,
    pretrained=False,
    freeze_backbone=True
)

# --- wrapper for torchview ---
class TwoStreamWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def forward(self, img, flow):
        # pack into tuple so it matches your internal forward
        return self.base((img, flow))

wrapped_model = TwoStreamWrapper(model)

# --- dummy data ---
img = torch.randn(1, 3, 224, 224)
flow = torch.randn(1, 2, 224, 224)

# --- draw graph ---
graph = draw_graph(
    wrapped_model,
    input_data=(img, flow),
    expand_nested=True,
    save_graph=True,
    filename="two_stream_model")
