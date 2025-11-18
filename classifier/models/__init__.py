import timm
import torch.nn as nn

from config import CFG
from .can import CANSmall
from .two_stream import TwoStreamModel
from .multiscale import MultiScaleFusionModel
from .utils import enable_head_grads, apply_lora_to_timm


def build_model(num_classes=2):
    """Main model factory."""
    
    # Two-stream architecture
    if CFG.get("use_flow") and CFG.get("two_stream", False):
        model = TwoStreamModel(
            img_backbone=CFG["backbone"],
            flow_encoder=CFG.get("flow_encoder", "SmallFlowCNN"),
            num_classes=num_classes,
            pretrained=CFG["pretrained"],
            freeze_backbone=CFG["freeze_backbone"]
        )
        if CFG.get("use_lora", False):
            print("Applying LoRA to two-stream model...")
            model = apply_lora_to_timm(model)
        print("Built TwoStreamModel")
        return model

    # Multi-scale fusion
    if CFG["backbone"].lower() in ["ms_fusion", "multiscale_fusion"]:
        model = MultiScaleFusionModel(
            num_classes=num_classes,
            k_crops=CFG.get("k_crops", 8),
            crop_size=CFG.get("crop_size", 64),
            stride=CFG.get("crop_stride", 32),
            vit_name=CFG.get("vit_name", "vit_base_patch16_224"),
            local_dim=CFG.get("local_dim", 192),
            num_heads=CFG.get("cross_heads", 4),
            pretrained=CFG.get("pretrained", True),
            freeze_backbone=CFG.get("freeze_backbone", False),
        )
        print("Built MultiScaleFusionModel")
        return model

    # Custom CAN
    if CFG["backbone"].lower() in ["can_small", "can"]:
        model = CANSmall(in_chans=1, num_classes=num_classes)
        print("Built CANSmall")
        return model

    # Standard single-stream (timm backbone)
    in_chans = 3 if CFG.get("use_flow") else (2 if CFG.get("use_diff") else 3)
    model = timm.create_model(
        CFG["backbone"],
        pretrained=CFG["pretrained"],
        num_classes=num_classes,
        in_chans=in_chans
    )

    if CFG["freeze_backbone"]:
        for p in model.parameters():
            p.requires_grad = False
        enable_head_grads(model)

    if CFG.get("use_lora", False):
        print("Applying LoRA...")
        model = apply_lora_to_timm(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built single-stream with in_chans={in_chans}, trainable={n_train:,}")
    return model