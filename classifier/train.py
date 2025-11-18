import os, sys, json
import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from config import CFG, SPLIT_DIRS
from datasets import make_dataloaders, count_samples_all_shards
from models import build_model
from losses import get_loss_function
from evaluate import evaluate_model


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, steps_per_epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    seen = 0
    
    pbar = tqdm(total=steps_per_epoch, desc="Training", leave=True)
    
    for inputs, labels, _ in dataloader:
        if isinstance(inputs, (list, tuple)):
            inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
        else:
            inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        seen += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
        
        if seen >= steps_per_epoch:
            break
    
    pbar.close()
    return running_loss / max(1, seen)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            batches += 1
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = val_loss / max(1, batches)
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment directory
    exp_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{CFG['model_name']}"
    exp_dir = os.path.join(CFG["results_base"], exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)

    # Redirect stdout to log
    log_path = os.path.join(exp_dir, "log.txt")
    sys.stdout = open(log_path, "w", buffering=1)
    print(f"Starting experiment {exp_id}")

    # Class weights
    full_counts = {1: 149249, 0: 610108}  # hardcoded for speed
    Nn, Nf = full_counts.get(0, 1), full_counts.get(1, 1)
    print(f"Class counts (Train): {full_counts}")
    
    class_weights = torch.tensor([1.0, Nn / max(Nf, 1)], dtype=torch.float32, device=device)

    # Data
    total_train = count_samples_all_shards(SPLIT_DIRS["Train"])
    steps_per_epoch = max(1, total_train // CFG["batch_size"])
    print(f"Train samples: {total_train:,} | steps/epoch: {steps_per_epoch:,}")

    dataloaders = make_dataloaders()

    # Model
    model = build_model(num_classes=2).to(device)
    
    # Loss
    criterion = get_loss_function(
        class_weights, 
        use_focal=CFG["use_focal"], 
        gamma=CFG["focal_gamma"]
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters!"
    optimizer = optim.Adam(trainable_params, lr=CFG["lr"])
    scaler = torch.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Training loop
    print("Starting training...")
    for epoch in range(CFG["epochs"]):
        train_loss = train_one_epoch(
            model, dataloaders["Train"], criterion, optimizer, scaler, device, steps_per_epoch
        )
        
        val_loss, val_acc = validate(model, dataloaders["Validation"], criterion, device)
        
        print(f"Epoch {epoch+1}/{CFG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

    # Save model
    model_path = os.path.join(exp_dir, f"{CFG['model_name']}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_model(model, dataloaders["Test"], device, exp_dir)

    print(f"\nExperiment complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()