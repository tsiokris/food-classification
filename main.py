import csv
import os
import torch
from torch import nn

from src.data.config import VIPPSTAR, UNIFIED_FOLD_0, UNIFIED_FOLD_1, UNIFIED_FOLD_2
from src.data.dataset import FoodDataset
from src.data.dataloader import get_dataloader
from src.data.transforms import train_transforms, val_transforms
from src.models.model import FoodClassifier, ARCH_CONFIGS, _get_submodule
from src.training.train import train, validate


# ----------------------------- DATASET CONFIG --------------------------------
# Pick one — everything else (num_classes, CSV columns, image roots) is derived.
DATA_CONFIG = VIPPSTAR
# DATA_CONFIG = UNIFIED_FOLD_0
# DATA_CONFIG = UNIFIED_FOLD_1
# DATA_CONFIG = UNIFIED_FOLD_2
# -----------------------------------------------------------------------------


# --------------------------------- CONFIG ------------------------------------
LOSS_FN = nn.CrossEntropyLoss()
FREEZE_BACKBONE = True
UNFREEZE_LAST_BLOCK = True
EPOCHS = 10
RESULTS_DIR = "results"
BACKBONES = ["resnet50", "efficientnet_b3", "mobilenet_v3_large", "densenet121"]
# -----------------------------------------------------------------------------


os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dataset : {DATA_CONFIG.name}  ({DATA_CONFIG.num_classes} classes)")
print(f"Device  : {device}")

train_dataset = FoodDataset(config=DATA_CONFIG, split="train", transform=train_transforms)
val_dataset   = FoodDataset(config=DATA_CONFIG, split="val",   transform=val_transforms)

train_dataloader = get_dataloader(dataset=train_dataset)
val_dataloader   = get_dataloader(dataset=val_dataset, shuffle=False)


for backbone in BACKBONES:
    print(f"\n{'=' * 60}\nTraining: {backbone}\n{'=' * 60}")

    model = FoodClassifier(
        num_classes=DATA_CONFIG.num_classes,
        backbone=backbone,
        freeze_backbone=FREEZE_BACKBONE,
        unfreeze_last_block=UNFREEZE_LAST_BLOCK,
    ).to(device)

    cfg = ARCH_CONFIGS[backbone]
    specs = cfg["last_block"] if isinstance(cfg["last_block"], list) else [cfg["last_block"]]
    last_block_params = [p for spec in specs for p in _get_submodule(model.model, spec).parameters()]
    optimizer = torch.optim.Adam([
        {"params": last_block_params,                                         "lr": 1e-4},
        {"params": _get_submodule(model.model, cfg["head"]).parameters(),     "lr": 1e-3},
    ])

    metrics = []
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train(model=model, dataloader=train_dataloader, loss_fn=LOSS_FN, optimizer=optimizer, device=device)
        val_loss, val_acc     = validate(model=model, dataloader=val_dataloader, loss_fn=LOSS_FN, device=device)
        print(f"  train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")
        metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    metrics_path = os.path.join(RESULTS_DIR, f"{DATA_CONFIG.name}_{backbone}_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Metrics saved to {metrics_path}")

    torch.save(model.state_dict(), f"models/{DATA_CONFIG.name}_{backbone}.pth")
    print(f"Model saved to models/{DATA_CONFIG.name}_{backbone}.pth")
