import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import nn

from src.data.config import ROTATION_0, ROTATION_1, ROTATION_2
from src.data.dataset import FoodDataset
from src.data.dataloader import get_dataloader
from src.data.transforms import val_transforms
from src.models.model import FoodClassifier
from src.training.train import validate


ROTATIONS = [ROTATION_0, ROTATION_1, ROTATION_2]
BACKBONES = ["resnet50", "efficientnet_b3", "mobilenet_v3_large", "densenet121"]
LOSS_FN = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

for i, config in enumerate(ROTATIONS):
    print(f"{'#' * 60}")
    print(f"{config.name}")
    print(f"{'#' * 60}")

    test_dataset = FoodDataset(config=config, split="test", transform=val_transforms)
    test_loader = get_dataloader(dataset=test_dataset, shuffle=False)
    print(f"Test samples: {len(test_dataset)}\n")

    for backbone in BACKBONES:
        model_path = f"models/cv_fold{i}_{backbone}.pth"
        if not os.path.exists(model_path):
            print(f"  [{backbone}] no saved model at {model_path}, skipping")
            continue

        model = FoodClassifier(
            num_classes=config.num_classes,
            backbone=backbone,
            freeze_backbone=False,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_loss, top1, top3, top5 = validate(model=model, dataloader=test_loader, loss_fn=LOSS_FN, device=device)
        print(f"  {backbone:<25}  loss: {test_loss:.4f}  top1: {top1*100:.2f}%  top3: {top3*100:.2f}%  top5: {top5*100:.2f}%")

    print()
