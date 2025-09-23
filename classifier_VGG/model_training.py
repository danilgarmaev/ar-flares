import os, io, glob, json, tarfile, random
from typing import Iterable, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

import timm
from timm.data import resolve_data_config, create_transform

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from collections import Counter

# ===================== CONFIG =====================
IS_COMPUTE_CANADA = os.path.exists('/home/dgarmaev/scratch')
CFG = {
    # paths
    "wds_base": '/home/dgarmaev/scratch/ar-flares/data/wds_out' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/wds_out',
    "results_base": '/home/dgarmaev/scratch/dgarmaev/ar-flares/results' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/results',

    # data loader
    "batch_size": 64,
    "num_workers": 4,          # keep modest to avoid worker deaths
    "pin_memory": True,
    "persistent_workers": False,
    "prefetch_factor": 2,

    # model / training
    "backbone": "vgg16",       # try: "vgg16", "resnet50", "efficientnet_b0", "deit_tiny_patch16_224"
    "pretrained": True,
    "freeze_backbone": True,   # head-only fine-tune
    "lr": 1e-4,
    "epochs": 5,

    # loss
    "use_focal": False,        # set True to try focal loss
    "focal_gamma": 2.0,

    # plots
    "save_pr_curve": True,
}
SPLIT_DIRS = {
    "Train": os.path.join(CFG["wds_base"], "train"),
    "Validation": os.path.join(CFG["wds_base"], "val"),
    "Test": os.path.join(CFG["wds_base"], "test"),
}
# ===================================================

# ---------------- WebDataset Iterable ----------------
class TarShardDataset(IterableDataset):
    def __init__(self, shard_paths: List[str], transform=None,
                 shuffle_shards=True, shuffle_samples=False, shuffle_buffer_size=2048, seed=42):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.transform = transform
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def _sample_iter_from_tar(self, tar_path: str):
        with tarfile.open(tar_path, "r") as tf:
            by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}
            for m in tf.getmembers():
                if not m.isfile(): continue
                if m.name.endswith(".png"):
                    key = m.name[:-4]; by_key.setdefault(key, {})["png"] = m
                elif m.name.endswith(".json"):
                    key = m.name[:-5]; by_key.setdefault(key, {})["json"] = m
            keys = list(by_key.keys())
            if self.shuffle_samples:
                rng = random.Random(self.seed ^ hash(tar_path))
                rng.shuffle(keys)
            for k in keys:
                parts = by_key[k]
                if "png" not in parts or "json" not in parts: continue
                try:
                    png_bytes = tf.extractfile(parts["png"]).read()
                    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    if self.transform: img = self.transform(img)
                    meta = json.loads(tf.extractfile(parts["json"]).read().decode("utf-8"))
                    label = int(meta["label"])
                    yield img, label, meta
                except Exception:
                    continue

    def _roundrobin(self, iters: List[Iterable]):
        active = [iter(x) for x in iters]
        while active:
            new_active = []
            for it in active:
                try:
                    yield next(it); new_active.append(it)
                except StopIteration:
                    pass
            active = new_active

    def _buffered_shuffle(self, iterable):
        rng = random.Random(self.seed)
        buf = []
        for x in iterable:
            buf.append(x)
            if len(buf) >= self.shuffle_buffer_size:
                rng.shuffle(buf)
                while buf: yield buf.pop()
        rng.shuffle(buf)
        while buf: yield buf.pop()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shards = self.shard_paths; wid = 0
        else:
            wid = worker_info.id
            shards = self.shard_paths[wid::worker_info.num_workers]
        shards = list(shards)
        if self.shuffle_shards:
            rng = random.Random(self.seed + wid); rng.shuffle(shards)
        gens = [self._sample_iter_from_tar(sp) for sp in shards]
        stream = self._roundrobin(gens)
        if self.shuffle_samples:
            stream = self._buffered_shuffle(stream)
        for sample in stream:
            yield sample

def list_shards(split_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(split_dir, "*.tar")))

def make_dataset(split_name: str, transform, shuffle_shards=True, shuffle_samples=False, seed=42):
    shard_dir = SPLIT_DIRS[split_name]
    shards = list_shards(shard_dir)
    if not shards: raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
    return TarShardDataset(shards, transform=transform,
                           shuffle_shards=shuffle_shards,
                           shuffle_samples=shuffle_samples, seed=seed)

# --------- shard scanners (fast, no images) ---------
def count_labels_all_shards(split_dir: str) -> Dict[int, int]:
    ctr = Counter()
    for tar_path in glob.glob(os.path.join(split_dir, "*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".json"):
                    try:
                        meta = json.loads(tf.extractfile(m).read().decode("utf-8"))
                        ctr.update([int(meta["label"])])
                    except Exception:
                        continue
    return dict(ctr)

def count_samples_all_shards(split_dir: str) -> int:
    total = 0
    for tar_path in glob.glob(os.path.join(split_dir, "*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            total += sum(1 for m in tf.getmembers() if m.isfile() and m.name.endswith(".json"))
    return total

# ---------------- losses ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.reduction = reduction
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)         # [batch]
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------- model builder ----------------
def _enable_head_grads(model: nn.Module):
    """Turn grads back on for the classifier head, regardless of attribute name."""
    enabled = 0
    # Prefer timm's get_classifier() if available
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module):
            for p in head.parameters(): p.requires_grad = True; enabled += p.numel()

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
            for p in last_linear.parameters(): p.requires_grad = True

def build_model(backbone: str, num_classes=2, pretrained=True, freeze_backbone=True):
    """
    Robust head reset across timm backbones; returns (model, eval_transform).
    """
    # Create model WITHOUT forcing num_classes here (so we can reset cleanly)
    model = timm.create_model(backbone, pretrained=pretrained)

    # Replace classifier using timm API if available
    if hasattr(model, "reset_classifier"):
        # most timm models support this
        model.reset_classifier(num_classes=num_classes)
    else:
        # manual heads for rare cases
        replaced = False
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes); replaced = True
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                # find last Linear
                li = None
                for i in reversed(range(len(model.classifier))):
                    if isinstance(model.classifier[i], nn.Linear):
                        li = i; break
                if li is None: raise RuntimeError("Couldn't locate classifier Linear layer.")
                in_features = model.classifier[li].in_features
                model.classifier[li] = nn.Linear(in_features, num_classes); replaced = True
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes); replaced = True
        elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes); replaced = True
        if not replaced:
            raise RuntimeError(f"Don't know how to replace head for backbone='{backbone}'")

    # Build transform for inference/eval (same for train unless you add aug)
    data_cfg = resolve_data_config({}, model=model)
    eval_transform = create_transform(**data_cfg, is_training=False)

    # Freeze backbone if requested, then re-enable head gradients
    if freeze_backbone:
        for p in model.parameters(): p.requires_grad = False
        _enable_head_grads(model)

    # Sanity check
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_train:,} / {n_total:,}")
    if n_train == 0:
        raise RuntimeError("No trainable parameters found. Check head replacement / freezing logic.")

    return model, eval_transform

# ---------------- metrics helper ----------------
class MetricsCalculator:
    def __init__(self): self.reset()
    def reset(self): self.TP=self.TN=self.FP=self.FN=0
    def update(self, y_true, y_pred):
        if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()
        neg_t, neg_p = 1 - y_true, 1 - y_pred
        self.TP += int(np.sum(y_true * y_pred))
        self.TN += int(np.sum(neg_t * neg_p))
        self.FP += int(np.sum(neg_t * y_pred))
        self.FN += int(np.sum(y_true * neg_p))
    def compute(self):
        TPR = self.TP / (self.TP + self.FN + 1e-7)
        TNR = self.TN / (self.TN + self.FP + 1e-7)
        TSS = TPR + TNR - 1
        HSS = 2 * (self.TP * self.TN - self.FN * self.FP) / (
              (self.TP + self.FN)*(self.FN + self.TN) + (self.TP + self.FP)*(self.FP + self.TN) + 1e-7)
        return {"TPR":TPR,"TNR":TNR,"TSS":TSS,"HSS":HSS,
                "TP":self.TP,"TN":self.TN,"FP":self.FP,"FN":self.FN}

# ============================ MAIN ============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # exact counts & steps
    full_counts = count_labels_all_shards(SPLIT_DIRS["Train"])
    Nn, Nf = full_counts.get(0, 1), full_counts.get(1, 1)
    print(f"Class counts (Train FULL): {full_counts}")

    total_train = count_samples_all_shards(SPLIT_DIRS["Train"])
    steps_per_epoch = max(1, total_train // CFG["batch_size"])
    print(f"Train samples: {total_train:,} | steps/epoch: {steps_per_epoch:,}")

    # model & transform
    model, infer_transform = build_model(
        CFG["backbone"],
        num_classes=2,
        pretrained=CFG["pretrained"],
        freeze_backbone=CFG["freeze_backbone"],
    )
    model = model.to(device)

    # dataloaders
    train_ds = make_dataset("Train", infer_transform, shuffle_shards=True,  shuffle_samples=True)
    val_ds   = make_dataset("Validation", infer_transform, shuffle_shards=False, shuffle_samples=False)
    test_ds  = make_dataset("Test", infer_transform, shuffle_shards=False,   shuffle_samples=False)

    common = dict(
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
        persistent_workers=CFG["persistent_workers"],
        prefetch_factor=CFG["prefetch_factor"],
    )
    dls = {
        "Train": DataLoader(train_ds, **common),
        "Validation": DataLoader(val_ds, **common),
        "Test": DataLoader(test_ds, **common),
    }

    # loss
    class_weights = torch.tensor([1.0, Nn / max(Nf,1)], dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=CFG["focal_gamma"], weight=class_weights) if CFG["use_focal"] \
                else nn.CrossEntropyLoss(weight=class_weights)

    # optimizer on TRAINABLE params only
    head_params = [p for p in model.parameters() if p.requires_grad]
    assert len(head_params) > 0, "No trainable parameters found (head may still be frozen)."
    optimizer = optim.Adam(head_params, lr=CFG["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # save dirs
    tag = f'{CFG["backbone"]}_lr{CFG["lr"]}_ep{CFG["epochs"]}{"_focal" if CFG["use_focal"] else ""}'
    model_path   = os.path.join(CFG["results_base"], "models", f"{tag}.pt")
    plot_dir     = os.path.join(CFG["results_base"], "plots")
    metrics_dir  = os.path.join(CFG["results_base"], "metrics")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True); os.makedirs(metrics_dir, exist_ok=True)
    roc_path = os.path.join(plot_dir, f"{tag}_roc.png")
    pr_path  = os.path.join(plot_dir, f"{tag}_pr.png")
    cm_path  = os.path.join(plot_dir, f"{tag}_cm.png")
    metrics_path = os.path.join(metrics_dir, f"{tag}_metrics.txt")

    # ------------------ Training ------------------
    print("Starting training...")
    for epoch in range(CFG["epochs"]):
        model.train(); running_loss = 0.0; seen = 0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{CFG['epochs']} - Train", leave=True)
        for inputs, labels, _ in dls["Train"]:
            inputs = inputs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item(); seen += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}"); pbar.update(1)
            if seen >= steps_per_epoch: break
        pbar.close()
        avg_train_loss = running_loss / max(1, seen)

        # ---- Validation ----
        model.eval(); val_loss=0.0; correct=0; total=0; vbatches=0
        with torch.no_grad():
            for inputs, labels, _ in dls["Validation"]:
                inputs = inputs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item(); vbatches += 1
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / max(1, vbatches)
        val_acc = 100.0 * correct / max(1, total)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
   
    # ------------------ Save model ------------------
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
 
    # ------------------ Test eval ------------------
    model.eval()
    all_probs, all_labels, all_preds_class = [], [], []
    metrics_calc = MetricsCalculator()
    with torch.no_grad():
        for inputs, labels, _ in dls["Test"]:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
            preds_class = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_preds_class.extend(preds_class)
            metrics_calc.update(labels, preds_class)
    al = np.array(all_labels); ap = np.array(all_probs)

    # ROC
    fpr, tpr, _ = roc_curve(al, ap); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}'); plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {CFG['backbone']}"); plt.legend(); plt.savefig(roc_path, dpi=200, bbox_inches='tight'); plt.close()

    # PR-AUC
    if CFG["save_pr_curve"]:
        prec, rec, _ = precision_recall_curve(al, ap)
        pr_auc = auc(rec, prec)
        plt.figure(); plt.plot(rec, prec, label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve - {CFG['backbone']}"); plt.legend()
        plt.savefig(pr_path, dpi=200, bbox_inches='tight'); plt.close()
    else:
        pr_auc = float('nan')

    # threshold by TSS
    best_tss, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (ap >= t).astype(int)
        TP = int(((al==1) & (pred==1)).sum())
        TN = int(((al==0) & (pred==0)).sum())
        FP = int(((al==0) & (pred==1)).sum())
        FN = int(((al==1) & (pred==0)).sum())
        TPR = TP / (TP + FN + 1e-7); TNR = TN / (TN + FP + 1e-7)
        TSS = TPR + TNR - 1
        if TSS > best_tss: best_tss, best_t = TSS, t

    # CM at 0.5
    cm = confusion_matrix(al, all_preds_class)
    plt.figure(figsize=(8,6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format='d')
    plt.title(f"Confusion Matrix (thr=0.5) - {CFG['backbone']}")
    cm_path = os.path.join(os.path.dirname(roc_path), f"{os.path.splitext(os.path.basename(roc_path))[0].replace('_roc','')}_cm.png")
    plt.savefig(cm_path, dpi=200, bbox_inches='tight'); plt.close()

    final = metrics_calc.compute()
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"PR_AUC: {pr_auc:.4f}\n")
        f.write(f"TPR: {final['TPR']:.4f}\n")
        f.write(f"TNR: {final['TNR']:.4f}\n")
        f.write(f"TSS: {final['TSS']:.4f}\n")
        f.write(f"HSS: {final['HSS']:.4f}\n")
        f.write(f"TP: {final['TP']}, TN: {final['TN']}, FP: {final['FP']}, FN: {final['FN']}\n")
        f.write(f"Best_TSS: {best_tss:.4f} @ threshold={best_t:.3f}\n")
    print(f"Metrics saved to {metrics_path}")
    print(f"ROC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | TSS*: {best_tss:.4f} @ thr={best_t:.3f}")

if __name__ == "__main__":
    main()
  