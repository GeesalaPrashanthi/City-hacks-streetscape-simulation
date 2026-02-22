import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from pathlib import Path
import json

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

DATA_DIR = Path("dataset_masked")
MODEL_OUT = Path("sidewalk_classifier.pt")
BATCH_SIZE = 32
EPOCHS = 20
EARLY_STOP_PATIENCE = 5
LR = 1e-4
IMG_SIZE = 224

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATA_DIR)
classes = full_dataset.classes
print(f"Classes: {classes}")
print(f"Total images: {len(full_dataset)}")

# 70/15/15 split
total = len(full_dataset)
test_size = int(0.15 * total)
val_size = int(0.15 * total)
train_size = total - val_size - test_size

train_split, val_split, test_split = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=train_tf), train_split.indices)
val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tf), val_split.indices)
test_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tf), test_split.indices)

print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# EfficientNet-B2
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model = model.to(DEVICE)

# Class weights
class_counts = [0] * len(classes)
for label in full_dataset.targets:
    class_counts[label] += 1
print(f"Class distribution: {dict(zip(classes, class_counts))}")
weights = torch.tensor([1.0/c for c in class_counts]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = -float("inf")
epochs_without_improvement = 0
for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_correct += (logits.argmax(1) == labels).sum().item()

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            val_correct += (model(imgs).argmax(1) == labels).sum().item()

    train_acc = train_correct / train_size
    val_acc = val_correct / val_size
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save({"model_state": model.state_dict(), "classes": classes, "val_acc": val_acc}, MODEL_OUT)
        print(f"  → Best model saved (val={val_acc:.3f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"  → Early stopping at epoch {epoch+1} "
                f"(no val improvement for {EARLY_STOP_PATIENCE} epochs)"
            )
            break

# Final test evaluation
print("\n--- Test Set Evaluation ---")
checkpoint = torch.load(MODEL_OUT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

test_correct = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(1)
        test_correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

test_acc = test_correct / test_size
print(f"Test Accuracy: {test_acc:.3f} ({test_correct}/{test_size})")

# Per-class accuracy
for i, cls in enumerate(classes):
    cls_mask = [l == i for l in all_labels]
    cls_correct = sum(p == l for p, l in zip(all_preds, all_labels) if l == i)
    cls_total = sum(cls_mask)
    print(f"  {cls}: {cls_correct}/{cls_total} ({cls_correct/cls_total:.3f})")

results = {
    "classes": classes,
    "best_val_acc": best_val_acc,
    "test_acc": test_acc,
    "train_size": train_size,
    "val_size": val_size,
    "test_size": test_size
}
with open("training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Model: {MODEL_OUT}")
print(f"Results saved to training_results.json")
