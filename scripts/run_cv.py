import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import nn
from torch.optim import Adam

from src.data.config import ROTATION_0, ROTATION_1, ROTATION_2
from src.data.dataset import FoodDataset
from src.data.dataloader import get_dataloader
from src.data.transforms import train_transforms, val_transforms
from src.models.model import FoodClassifier
from src.training.train import train, validate


ROTATIONS = [ROTATION_0, ROTATION_1, ROTATION_2]
ALL_BACKBONES = ["resnet50", "efficientnet_b3", "mobilenet_v3_large", "densenet121"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=ALL_BACKBONES, choices=ALL_BACKBONES)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--freeze", action="store_true", default=True)
    p.add_argument("--unfreeze-last", action="store_true", default=False)
    return p.parse_args()


def run_fold(i, config, backbone, args, loss_fn, device):
    print(f"\n{'=' * 60}")
    print(f"[{backbone}]  {config.name}")
    print(f"{'=' * 60}")

    train_dataset = FoodDataset(config=config, split="train", transform=train_transforms)
    val_dataset = FoodDataset(config=config, split="val", transform=val_transforms)
    test_dataset = FoodDataset(config=config, split="test", transform=val_transforms)
    print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}  test: {len(test_dataset)}")

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = FoodClassifier(
        num_classes=config.num_classes,
        backbone=backbone,
        freeze_backbone=args.freeze,
        unfreeze_last_block=args.unfreeze_last,
    ).to(device)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_loss = float("inf")
    model_path = f"models/cv_fold{i}_{backbone}.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\n  Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device)
        val_loss, top1, top3, top5 = validate(model, val_loader, loss_fn, device)

        print(f"  train  loss: {train_loss:.4f}  acc: {train_acc * 100:.2f}%")
        print(f"  val    loss: {val_loss:.4f}  top1: {top1 * 100:.2f}%  top3: {top3 * 100:.2f}%  top5: {top5 * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> saved best model")

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, top1, top3, top5 = validate(model, test_loader, loss_fn, device)

    print(f"\n  Test (best model):")
    print(f"  loss: {test_loss:.4f}  top1: {top1 * 100:.2f}%  top3: {top3 * 100:.2f}%  top5: {top5 * 100:.2f}%")

    return top1, top3, top5


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbones: {args.backbones}  epochs: {args.epochs}  lr: {args.lr}  batch: {args.batch_size}")

    loss_fn = nn.CrossEntropyLoss()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results_path = "results/cv_results.csv"
    csv_file = open(results_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["backbone", "fold", "top1", "top3", "top5"])
    writer.writeheader()

    all_results = {}

    for backbone in args.backbones:
        fold_results = []
        for i, config in enumerate(ROTATIONS):
            top1, top3, top5 = run_fold(i, config, backbone, args, loss_fn, device)
            fold_results.append((top1, top3, top5))
            writer.writerow({"backbone": backbone, "fold": i, "top1": round(top1, 4), "top3": round(top3, 4), "top5": round(top5, 4)})
            csv_file.flush()
        avg = [sum(r[k] for r in fold_results) / 3 for k in range(3)]
        writer.writerow({"backbone": backbone, "fold": "avg", "top1": round(avg[0], 4), "top3": round(avg[1], 4), "top5": round(avg[2], 4)})
        csv_file.flush()
        all_results[backbone] = fold_results

    csv_file.close()
    print(f"\nResults saved to {results_path}")

    print(f"\n{'=' * 60}")
    print("3-Fold CV Summary")
    print(f"{'=' * 60}")
    for backbone, fold_results in all_results.items():
        print(f"\n  {backbone}")
        for i, (top1, top3, top5) in enumerate(fold_results):
            print(f"    Fold {i}:  top1: {top1 * 100:.2f}%  top3: {top3 * 100:.2f}%  top5: {top5 * 100:.2f}%")
        avg = [sum(r[k] for r in fold_results) / 3 for k in range(3)]
        print(f"    Avg:    top1: {avg[0] * 100:.2f}%  top3: {avg[1] * 100:.2f}%  top5: {avg[2] * 100:.2f}%")


if __name__ == "__main__":
    main()
