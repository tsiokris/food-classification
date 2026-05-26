import csv
import os
from collections import defaultdict

import torch
from torch import nn

from src.data.config import VIPPSTAR, UNIFIED_FOLD_0, UNIFIED_FOLD_1, UNIFIED_FOLD_2
from src.data.dataset import FoodDataset
from src.data.dataloader import get_dataloader
from src.data.transforms import train_transforms, val_transforms
from src.models.model import FoodClassifier, ARCH_CONFIGS, _get_submodule
from src.training.train import train, validate


# ----------------------------- DATASET CONFIG --------------------------------
# Single config → one training run (fast iteration / vippstar baseline)
DATA_CONFIGS = [VIPPSTAR]

# Single unified fold → one run on the 3-source dataset
# DATA_CONFIGS = [UNIFIED_FOLD_0]

# All 3 folds → full cross-validation; prints averaged results at the end
# DATA_CONFIGS = [UNIFIED_FOLD_0, UNIFIED_FOLD_1, UNIFIED_FOLD_2]
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
print(f"Device: {device}")

# fold → backbone → best val acc  (used for cross-val summary)
cv_results = defaultdict(dict)

for data_config in DATA_CONFIGS:
    print(f"\n{'#' * 60}")
    print(f"Dataset: {data_config.name}  ({data_config.num_classes} classes)")
    print(f"{'#' * 60}")

    train_dataset = FoodDataset(config=data_config, split="train", transform=train_transforms)
    val_dataset   = FoodDataset(config=data_config, split="val",   transform=val_transforms)
    train_dataloader = get_dataloader(dataset=train_dataset)
    val_dataloader   = get_dataloader(dataset=val_dataset, shuffle=False)

    print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")

    for backbone in BACKBONES:
        print(f"\n{'=' * 60}\nTraining: {backbone}\n{'=' * 60}")

        model = FoodClassifier(
            num_classes=data_config.num_classes,
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

        metrics_path = os.path.join(RESULTS_DIR, f"{data_config.name}_{backbone}_metrics.csv")
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Metrics saved to {metrics_path}")

        torch.save(model.state_dict(), f"models/{data_config.name}_{backbone}.pth")
        print(f"Model saved to models/{data_config.name}_{backbone}.pth")

        best_val_acc = max(m["val_acc"] for m in metrics)
        cv_results[backbone][data_config.name] = best_val_acc


# ----------------------------- CROSS-VAL SUMMARY ----------------------------
if len(DATA_CONFIGS) > 1:
    print(f"\n{'=' * 60}")
    print("Cross-validation summary (best val acc per fold)")
    print(f"{'=' * 60}")
    header = f"{'Backbone':<25}" + "".join(f"{cfg.name:<22}" for cfg in DATA_CONFIGS) + "Average"
    print(header)
    print("-" * len(header))
    for backbone in BACKBONES:
        fold_accs = [cv_results[backbone][cfg.name] for cfg in DATA_CONFIGS]
        avg = sum(fold_accs) / len(fold_accs)
        row = f"{backbone:<25}" + "".join(f"{acc * 100:<22.2f}" for acc in fold_accs) + f"{avg * 100:.2f}%"
        print(row)
