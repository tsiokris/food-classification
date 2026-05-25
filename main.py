import csv
import os
import torch
from torch import nn
from src.data.dataset import FoodDataset
from src.data.dataloader import get_dataloader
from src.data.transforms import train_transforms, val_transforms
from src.models.model import FoodClassifier
from src.training.train import train, validate


# ----------------------- CONFIG --------------------------
PATH = "data/annotations/food_classification_dataset.csv"
LOSS_FN = nn.CrossEntropyLoss()
NUM_CLASSES = 42
BACKBONE = "resnet50"
FREEZE_BACKBONE = True
UNFREEZE_LAST_BLOCK = True
EPOCHS = 10
RESULTS_DIR = "results"


# ----------------------- MAIN --------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

train_dataset = FoodDataset(food_dataset_csv=PATH, transform=train_transforms, split=['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'])
val_dataset = FoodDataset(food_dataset_csv=PATH, transform=val_transforms, split=['holdout_val', 'holdout_test'])

train_dataloader = get_dataloader(dataset=train_dataset)
val_dataloader = get_dataloader(dataset=val_dataset, shuffle=False)

model = FoodClassifier(num_classes=NUM_CLASSES, backbone=BACKBONE, freeze_backbone=FREEZE_BACKBONE, unfreeze_last_block=UNFREEZE_LAST_BLOCK).to(device)
optimizer = torch.optim.Adam([
    {"params": model.model.layer4.parameters(), "lr": 1e-4},
    {"params": model.model.fc.parameters(),     "lr": 1e-3},
])

metrics = []
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    train_loss, train_acc = train(model=model, dataloader=train_dataloader, loss_fn=LOSS_FN, optimizer=optimizer, device=device)
    val_loss, val_acc = validate(model=model, dataloader=val_dataloader, loss_fn=LOSS_FN, device=device)
    print(f"  train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
    print(f"  val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")
    metrics.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })

with open(os.path.join(RESULTS_DIR, "metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    writer.writeheader()
    writer.writerows(metrics)

print(f"\nMetrics saved to {RESULTS_DIR}/metrics.csv")
torch.save(model.state_dict(), "models/model.pth")
