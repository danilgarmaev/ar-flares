import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torchvision import models
from test_indexing import main as load_data  # ✅ Import your dataloader setup

# Load real dataloaders
all_datasets, dataloaders = load_data()
print("✅ Loaded dataloaders from test_indexing.py")

# Define the model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
learning_rate = 0.001
epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# print("🚀 Starting training...")
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in dataloaders["Train"]:
#         inputs, labels = inputs.to("cuda"), labels.to("cuda")
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloaders['Train']):.4f}")
# print("✅ Training complete")

# Training loop with validation
print("🚀 Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloaders["Train"]:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(dataloaders["Train"])

    # ✅ Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_inputs, val_labels in dataloaders["Validation"]:
            val_inputs, val_labels = val_inputs.to("cuda"), val_labels.to("cuda")
            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()

            _, predicted = torch.max(val_outputs, 1)
            correct += (predicted == val_labels).sum().item()
            total += val_labels.size(0)

    avg_val_loss = val_loss / len(dataloaders["Validation"])
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] — "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")


# Save model
model_name = f"resnet18_lr{learning_rate}_ep{epochs}.pt"
torch.save(model.state_dict(), model_name)
print(f"💾 Model saved as {model_name}")

# Evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in dataloaders["Test"]:
        inputs = inputs.to("cuda")
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
roc_path = model_name.replace(".pt", "_roc.png")
plt.savefig(roc_path)
print(f"📈 ROC curve saved as {roc_path}")

# Confusion Matrix
preds_class = [1 if p >= 0.5 else 0 for p in all_preds]
cm = confusion_matrix(all_labels, preds_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
cm_path = model_name.replace(".pt", "_confusion_matrix.png")
plt.savefig(cm_path)
print(f"📊 Confusion matrix saved as {cm_path}")
