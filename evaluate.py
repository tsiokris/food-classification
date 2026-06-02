import os

import torch
from torch import nn

from src.data.config import UNIFIED_FOLD_0, UNIFIED_FOLD_1, UNIFIED_FOLD_2
from src.data.dataset import FoodDataset
from src.data.dataloader import get_dataloader
from src.data.transforms import val_transforms
from src.models.model import FoodClassifier
from src.training.train import validate


DATA_CONFIGS = [UNIFIED_FOLD_0, UNIFIED_FOLD_1, UNIFIED_FOLD_2]
BACKBONES = ["resnet50", "efficientnet_b3", "mobilenet_v3_large", "densenet121"]
LOSS_FN = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

for data_config in DATA_CONFIGS:
    print(f"{'#' * 60}")
    print(f"Dataset: {data_config.name}  ({data_config.num_classes} classes)")
    print(f"{'#' * 60}")

    val_dataset = FoodDataset(config=data_config, split="val", transform=val_transforms)
    val_dataloader = get_dataloader(dataset=val_dataset, shuffle=False)
    print(f"Val samples: {len(val_dataset)}\n")

    for backbone in BACKBONES:
        model_path = f"models/{data_config.name}_{backbone}.pth"
        if not os.path.exists(model_path):
            print(f"  [{backbone}] no saved model at {model_path}, skipping")
            continue

        model = FoodClassifier(num_classes=data_config.num_classes, backbone=backbone, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        val_loss, top1, top3, top5 = validate(model=model, dataloader=val_dataloader, loss_fn=LOSS_FN, device=device)
        print(f"  {backbone:<25}  loss: {val_loss:.4f}  top1: {top1*100:.2f}%  top3: {top3*100:.2f}%  top5: {top5*100:.2f}%")

    print()
