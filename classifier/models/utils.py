import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


def enable_head_grads(model: nn.Module):
    """Enable gradients for classifier head."""
    enabled = 0
    
    # Try timm's get_classifier()
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module):
            for p in head.parameters():
                p.requires_grad = True
                enabled += p.numel()

    # Common head names
    for name in ["head", "fc", "classifier", "cls", "last_linear"]:
        m = getattr(model, name, None)
        if isinstance(m, nn.Module):
            for p in m.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    enabled += p.numel()
    
    if enabled == 0:
        # Fallback: last Linear layer
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            for p in last_linear.parameters():
                p.requires_grad = True


class PeftTimmWrapper(nn.Module):
    """Wrapper to make timm models compatible with PEFT/LoRA."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        x = input_ids if input_ids is not None else pixel_values
        return self.model(x)


def apply_lora_to_timm(model, r=8, alpha=16, dropout=0.1):
    """Apply LoRA fine-tuning to a timm model."""
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv", "fc1", "fc2"],
        bias="none"
    )
    model = PeftTimmWrapper(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model