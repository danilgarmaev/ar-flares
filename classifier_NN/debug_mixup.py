import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

BATCH_SIZE = 128
NUM_CLASSES = 2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fake batch similar to your real setup
    inputs = torch.randn(BATCH_SIZE, 3, 112, 112, device=device)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

    print("=== ORIGINAL ===")
    print("inputs:", inputs.shape)
    print("labels:", labels.shape, "dtype:", labels.dtype)
    print("labels[0:5]:", labels[:5])

    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=NUM_CLASSES,
    )

    inputs_mix, labels_mix = mixup_fn(inputs, labels)

    print("\n=== AFTER MIXUP ===")
    print("inputs_mix:", inputs_mix.shape)
    print("labels_mix:", labels_mix.shape, "dtype:", labels_mix.dtype)
    print("labels_mix[0]:", labels_mix[0])

    # Tiny linear model to generate logits
    model = nn.Linear(3 * 112 * 112, NUM_CLASSES).to(device)
    logits = model(inputs_mix.view(BATCH_SIZE, -1))

    print("\n=== LOGITS ===")
    print("logits:", logits.shape)

    criterion = SoftTargetCrossEntropy().to(device)

    # Try loss
    loss = criterion(logits, labels_mix)
    print("\nLoss computed successfully:", loss.item())


if __name__ == "__main__":
    main()
